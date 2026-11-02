# %% [markdown]
# # Generate a dataset of solved scenarios

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version

from dataclasses import fields

import numpy as np
import pandas as pd
import yaml

from ta import ROOT
from ta.edits import build_superset
from ta.network import Network, solve_scenario
from ta.scenarios import ScenarioConfig, sample_scenario
from ta.solver import aon_assign
from ta.virtual import names as virtual_names

ARRAY_FIELDS = ["demand", "closed", "flows", "times"]
COMPACT_ARRAYS = ["demand_prod", "demand_attr", "aon", "aon_open"]
OPTIONAL_ARRAYS = ["capacity_factor", "fft_factor"]
CONFIG_FIELDS = {f.name for f in fields(ScenarioConfig)}
SCALAR_FIELDS = ["closure_set", "variant", "demand_scale", "closure_target", "attempts", "num_closed", "rgap",
                 "rgap_reported", "iterations", "runtime", "converged", "ok", "pruned_links", "conservation_error",
                 "zero_flow_share", "overcapacity_share", "max_vc"]

# %% [markdown]
# ## One scenario (runs inside a worker process)

# %%
WORKER = {}


def load_network(network, superset_cfg=None):
    net = Network.from_tntp(network)
    if not superset_cfg:
        return net, None
    sup = build_superset(net, **superset_cfg)
    return sup.net, sup


def init_worker(network, superset_cfg=None):
    WORKER["net"], WORKER["superset"] = load_network(network, superset_cfg)


def compact(res, net, sc):
    res["demand_prod"], res["demand_attr"] = sc.demand.sum(1), sc.demand.sum(0)
    res["aon"] = np.zeros(net.num_links)
    res["aon"][~sc.closed] = aon_assign(net.links[~sc.closed].reset_index(drop=True), sc.demand, net.block_centroids)
    res["aon_open"] = aon_assign(net.links, sc.demand, net.block_centroids)
    del res["demand"]
    return res


def solve_one(closure_set, variant, cfg, seed, rgap_target, max_iter, compact_storage=False):
    net = WORKER["net"]
    try:
        sc = sample_scenario(net, closure_set, variant, cfg, seed, WORKER.get("superset"))
        sol, rep = solve_scenario(net, sc.demand, sc.closed, rgap_target=rgap_target, max_iter=max_iter,
                                  capacity_factor=sc.capacity_factor, fft_factor=sc.fft_factor)
    except Exception as e:
        return {"closure_set": closure_set, "variant": variant, "error": f"{type(e).__name__}: {e}"}
    res = {
        "closure_set": closure_set, "variant": variant, "demand": sc.demand, "closed": sc.closed,
        "flows": sol.flows, "times": sol.times, "demand_scale": sc.demand_scale, "closure_target": sc.closure_target,
        "attempts": sc.attempts, "num_closed": sc.num_closed, "rgap": sol.rgap, "rgap_reported": sol.rgap_reported,
        "iterations": sol.iterations, "runtime": sol.runtime, "converged": sol.converged, "ok": rep["ok"],
        "failed": ",".join(rep["failed"]), "pruned_links": sol.pruned_links,
        "conservation_error": rep["conservation_error"], "zero_flow_share": rep["zero_flow_share"],
        "overcapacity_share": rep["overcapacity_share"], "max_vc": rep["max_vc"],
    }
    if sc.capacity_factor is not None:
        res["capacity_factor"], res["fft_factor"] = sc.capacity_factor, sc.fft_factor
    if sc.edits:
        res["edits"] = json.dumps(list(sc.edits))
    return compact(res, net, sc) if compact_storage else res


# %% [markdown]
# ## Chunks on disk


# %%
def save_chunk(path, results):
    arrays = {k: np.stack([r[k] for r in results]) for k in ARRAY_FIELDS + COMPACT_ARRAYS if k in results[0]}
    for k in OPTIONAL_ARRAYS:
        if any(k in r for r in results):
            arrays[k] = np.stack([r.get(k, np.ones(len(r["flows"]))) for r in results])
    arrays.update({k: np.array([r[k] for r in results]) for k in SCALAR_FIELDS})
    arrays["failed"] = np.array([r["failed"] for r in results])
    arrays["edits"] = np.array([r.get("edits", "[]") for r in results])
    np.savez_compressed(path, **arrays)


def save_links(path, net, superset):
    table = net.links.copy()
    table["kind"] = superset.kind if superset is not None else "base"
    table["candidate"], table["is_connector"], table["street"] = net.candidate, net.is_connector, net.street
    table.to_csv(path, index_label="link")


def load_npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def merge(parts):
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((data["variant"], data["closure_set"]))
    return {k: v[order] for k, v in data.items()}


def done_keys(parts_dir):
    keys = set()
    for p in sorted(parts_dir.glob("part_*.npz")):
        d = load_npz(p)
        keys |= set(zip(d["closure_set"].tolist(), d["variant"].tolist()))
    return keys


# %% [markdown]
# ## One set


# %%
def git_commit():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def scenario_config(set_cfg, config):
    extra = {k: (tuple(v) if isinstance(v, list) else v) for k, v in set_cfg.items() if k in CONFIG_FIELDS}
    return ScenarioConfig(**config["demand"], **extra)


def generate_set(network, set_name, set_cfg, config, out_dir, workers, chunk=100, log_every=50, shard=None):
    final, meta_path, parts_dir = out_dir / f"{set_name}.npz", out_dir / f"{set_name}.meta.json", out_dir / f"{set_name}_parts"
    links_path = out_dir / f"{set_name}.links.csv"
    superset_cfg = set_cfg.get("superset")
    if not links_path.exists():
        save_links(links_path, *load_network(network, superset_cfg))
    if final.exists():
        print(f"[{set_name}] already complete: {final}")
        return final
    parts_dir.mkdir(parents=True, exist_ok=True)

    cfg = scenario_config(set_cfg, config)
    first = set_cfg["first_closure_set"]
    wanted = [(c, v) for c in range(first, first + set_cfg["closure_sets"]) for v in range(set_cfg["variants"])]
    done = done_keys(parts_dir)
    todo = [key for key in wanted if key not in done and (shard is None or key[0] % shard[1] == shard[0])]
    print(f"[{set_name}] {len(wanted)} scenarios, {len(done)} already done, {len(todo)} to solve with {workers} workers"
          + (f" (shard {shard[0]}/{shard[1]})" if shard else ""))

    solver = config["solver"]
    buffer, errors, n_done, t0 = [], [], 0, time.perf_counter()
    prefix = f"part_s{shard[0]}_" if shard else "part_"
    next_part = len(list(parts_dir.glob(f"{prefix}[0-9]*.npz")))
    compact_storage = set_cfg.get("compact", False)
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(network, superset_cfg)) as pool:
        futures = [pool.submit(solve_one, c, v, cfg, config["seed"], solver["rgap_target"], solver["max_iter"], compact_storage)
                   for c, v in todo]
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if "error" in res:
                errors.append(res)
            else:
                buffer.append(res)
            if len(buffer) >= chunk:
                save_chunk(parts_dir / f"{prefix}{next_part:05d}.npz", buffer)
                next_part, buffer = next_part + 1, []
            if n_done % log_every == 0 or n_done == len(todo):
                rate = n_done / (time.perf_counter() - t0)
                print(f"[{set_name}] {n_done}/{len(todo)} | {rate:.2f} scen/s | ETA {(len(todo) - n_done) / rate / 60:.1f} min | errors {len(errors)}", flush=True)
    if buffer:
        save_chunk(parts_dir / f"{prefix}{next_part:05d}.npz", buffer)

    for e in errors:
        print(f"[{set_name}] ERROR closure_set={e['closure_set']} variant={e['variant']}: {e['error']}")
    if errors:
        print(f"[{set_name}] incomplete ({len(errors)} errors) — fix and re-run; finished scenarios are kept")
        return None
    if shard:
        print(f"[{set_name}] shard {shard[0]}/{shard[1]} done; run without --shard to merge")
        return None

    data = merge([load_npz(p) for p in sorted(parts_dir.glob("part_*.npz"))])
    assert len(data["closure_set"]) == len(wanted)
    np.savez_compressed(final, **data)
    meta = {
        "network": network, "set": set_name, "set_config": set_cfg, "scenario_config": cfg.__dict__,
        "seed": config["seed"], "solver": solver, "num_scenarios": len(wanted),
        "num_ok": int(data["ok"].sum()), "num_converged": int(data["converged"].sum()),
        "created": datetime.now().isoformat(timespec="seconds"), "git": git_commit(),
        "aequilibrae": version("aequilibrae"), "network_files": json.loads((ROOT / "networks" / "manifest.json").read_text())["networks"][network.split("@v")[0]],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    for p in parts_dir.glob("part_*.npz"):
        p.unlink()
    parts_dir.rmdir()
    print(f"[{set_name}] saved {final} ({final.stat().st_size / 1e6:.1f} MB), ok {meta['num_ok']}/{len(wanted)}, converged {meta['num_converged']}/{len(wanted)}")
    return final


# %% [markdown]
# ## Networks of a config


# %%
def config_networks(config):
    v = config.get("virtual")
    return [config["network"]] if not v else virtual_names(config["network"], range(v["first"], v["first"] + v["count"]))


# %% [markdown]
# ## Remove failed scenarios


# %%
def remove_failed(out_dir, set_name):
    path, meta_path = out_dir / f"{set_name}.npz", out_dir / f"{set_name}.meta.json"
    data = load_npz(path)
    bad = ~data["ok"]
    if not bad.any():
        print(f"[{set_name}] no failed scenarios")
        return 0
    removed = [{"closure_set": int(data["closure_set"][i]), "variant": int(data["variant"][i]), "failed": str(data["failed"][i]),
                "rgap": float(data["rgap"][i]), "iterations": int(data["iterations"][i])} for i in np.flatnonzero(bad)]
    np.savez_compressed(path, **{k: v[~bad] for k, v in data.items()})
    meta = json.loads(meta_path.read_text())
    meta["removed_failed"] = meta.get("removed_failed", []) + removed
    meta["num_scenarios"] = int((~bad).sum())
    meta["num_ok"], meta["num_converged"] = int(data["ok"][~bad].sum()), int(data["converged"][~bad].sum())
    meta_path.write_text(json.dumps(meta, indent=2))
    for stale in [out_dir / f"{set_name}.aon.npz", out_dir / f"{set_name}.splits.csv"]:
        stale.unlink(missing_ok=True)
    print(f"[{set_name}] removed {len(removed)} failed scenarios: {[(r['closure_set'], r['variant']) for r in removed]}")
    return len(removed)


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/data_berlin.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--sets", nargs="+")
    parser.add_argument("--workers", type=int, default=min(28, os.cpu_count()))
    parser.add_argument("--remove-failed", action="store_true")
    parser.add_argument("--shard", help="i/n: solve only closure sets with index %% n == i, do not merge")
    args = parser.parse_known_args()[0]

    config = yaml.safe_load((ROOT / args.config).read_text())
    networks = config_networks(config)
    if not args.full:
        networks = networks[:1]
    for network in networks:
        out_dir = ROOT / "data" / (network if args.full else f"{network}_pilot")
        out_dir.mkdir(parents=True, exist_ok=True)
        for set_name in args.sets or list(config["sets"]):
            set_cfg = dict(config["sets"][set_name])
            if not args.full:
                set_cfg.update(config["pilot"])
            shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
            generate_set(network, set_name, set_cfg, config, out_dir, args.workers, shard=shard)
            if args.remove_failed and not shard:
                remove_failed(out_dir, set_name)

# %% [markdown]
# ## Inspect what was written

# %%
if __name__ == "__main__":
    for set_name in args.sets or list(config["sets"]):
        out_dir = ROOT / "data" / (networks[0] if args.full else f"{networks[0]}_pilot")
        path = out_dir / f"{set_name}.npz"
        if not path.exists():
            continue
        d = load_npz(path)
        print(f"== {path.relative_to(ROOT)}")
        print({k: v.shape for k, v in d.items() if k in ARRAY_FIELDS})
        print(f"scenarios {len(d['closure_set'])} | closure sets {len(np.unique(d['closure_set']))} | ok {d['ok'].mean():.0%} | "
              f"converged {d['converged'].mean():.0%} | closed links {d['num_closed'].min()}-{d['num_closed'].max()} | "
              f"runtime median {np.median(d['runtime']):.1f}s | max rgap {d['rgap'].max():.1e}")
