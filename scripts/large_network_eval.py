# %% [markdown]
# # Large network (Exp 3): baselines, solver curve, timing and model scores on unseen closures

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")
os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import importlib
import time
from dataclasses import fields
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch_geometric.data import Batch

from ta import ROOT
from ta.baselines import PerLinkMean
from ta.dataset import GraphDataset
from ta.features import closure_set_aon
from ta.metrics import evaluate, flow_metrics
from ta.network import Network, solve_scenario
from ta.scenarios import ScenarioConfig, demand_rng, sample_demand
from ta.solver import aon_assign, prepare_links, prune_dead_ends, run_aequilibrae
from ta.splits import load_set, split_closure_sets

# %% [markdown]
# ## Setup

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config", default="configs/data_sydney.yaml")
parser.add_argument("--set", default="main")
parser.add_argument("--runs", nargs="*", default=[], help="name=run_dir of trained models (their predictions.npz)")
parser.add_argument("--workers", type=int, default=os.cpu_count())
parser.add_argument("--iterations", type=int, nargs="*", default=[1, 2, 3, 5, 10, 20])
parser.add_argument("--curve-variant", type=int, default=5)
parser.add_argument("--near-hops", type=int, default=2)
parser.add_argument("--time-scenarios", type=int, default=10)
parser.add_argument("--no-closure-only", action="store_true")
args = parser.parse_known_args()[0]
config = yaml.safe_load((ROOT / args.config).read_text())
set_cfg = config["sets"][args.set]
NETWORK, SEED, rgap_target = config["network"], config["seed"], config["solver"]["rgap_target"]
names = {f.name for f in fields(ScenarioConfig)}
cfg = ScenarioConfig(**config["demand"], **{k: tuple(v) if isinstance(v, list) else v for k, v in set_cfg.items() if k in names})
OUT = ROOT / "runs" / "results" / NETWORK / "exp3"
OUT.mkdir(parents=True, exist_ok=True)
net = Network.from_tntp(NETWORK)
first = set_cfg["first_closure_set"]
labels = split_closure_sets(np.arange(first, first + set_cfg["closure_sets"]))
test_keys = [(c, v) for c in sorted(c for c, s in labels.items() if s == "test") for v in range(set_cfg["variants"])]


def scenario_demand(closure_set, variant):
    return sample_demand(demand_rng(net.name, SEED, int(closure_set), int(variant)), net.demand, cfg)[0]


# %% [markdown]
# ## Per-scenario work in worker processes (one core each)

# %%
WORKER = {}


def init_worker():
    WORKER["net"] = Network.from_tntp(NETWORK)


def no_closure_one(key):
    n = WORKER["net"]
    sol, _ = solve_scenario(n, scenario_demand(*key), n.default_closed, rgap_target=rgap_target)
    return key, sol.flows, sol.runtime


def curve_one(r):
    n, closed = WORKER["net"], data["closed"][r]
    od = scenario_demand(data["closure_set"][r], data["variant"][r])
    np.fill_diagonal(od, 0.0)
    out = {}
    t0 = time.perf_counter()
    a = aon_assign(n.links[~closed].reset_index(drop=True), od, n.block_centroids)
    out["aon"] = (time.perf_counter() - t0, a)
    prep = prepare_links(n.links[~closed].reset_index(drop=True))
    for k in args.iterations:
        t0 = time.perf_counter()
        keep = prune_dead_ends(prep, n.num_zones)
        active = prep[keep].reset_index(drop=True)
        active["link_id"] = np.arange(1, len(active) + 1)
        f, _ = run_aequilibrae(active, od, n.block_centroids, 1e-12, k, "bfw", 1)
        flows = np.zeros(len(prep))
        flows[keep] = f
        out[f"solver_{k}"] = (time.perf_counter() - t0, flows)
    full = {}
    for name, (secs, f) in out.items():
        g = np.zeros(n.num_links)
        g[~closed] = f
        full[name] = (secs, g)
    return r, full


def cached(path, compute):
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    out = compute()
    np.savez_compressed(path, **out)
    return out


# %% [markdown]
# ## No-closure equilibrium of every test scenario (its demand, no closures)

# %%
def compute_no_closure():
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        res = sorted(pool.map(no_closure_one, test_keys), key=lambda x: x[0])
    return {"closure_set": np.array([k[0] for k, _, _ in res]), "variant": np.array([k[1] for k, _, _ in res]),
            "flows": np.stack([f for _, f, _ in res]), "runtime": np.array([t for _, _, t in res])}


t0 = time.perf_counter()
nc = cached(OUT / "no_closure_test.npz", compute_no_closure)
print(f"no-closure equilibria: {len(nc['flows'])} test scenarios ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
if args.no_closure_only:
    raise SystemExit

# %% [markdown]
# ## Generated data, test rows

# %%
data, links, splits = load_set(NETWORK, args.set)
real = ~links.is_connector.to_numpy()
grid = splits.gridlock.to_numpy()
test = np.flatnonzero((splits.split == "test").to_numpy())
train = np.flatnonzero(((splits.split == "train") & ~splits.holdout_variant).to_numpy())
curve_rows = test[data["variant"][test] == args.curve_variant]
nc_index = {k: i for i, k in enumerate(zip(nc["closure_set"].tolist(), nc["variant"].tolist()))}
nc_pos = np.array([nc_index[k] for k in zip(data["closure_set"][test].tolist(), data["variant"][test].tolist())])
nc["flows"], nc["runtime"] = nc["flows"][nc_pos], nc["runtime"][nc_pos]
print(f"{NETWORK}/{args.set}: {len(data['flows'])} scenarios, test {len(test)}, solver-curve sample {len(curve_rows)}", flush=True)


# %% [markdown]
# ## Solver curve and single-core timing on one variant per test closure set

# %%
def compute_curve():
    with ProcessPoolExecutor(min(args.workers, len(curve_rows)), initializer=init_worker) as pool:
        res = sorted(pool.map(curve_one, curve_rows), key=lambda x: x[0])
    names = list(res[0][1])
    out = {"rows": np.array([r for r, _ in res])}
    for name in names:
        out[f"{name}_flows"] = np.stack([full[name][1] for _, full in res])
        out[f"{name}_secs"] = np.array([full[name][0] for _, full in res])
    return out


t0 = time.perf_counter()
curve = cached(OUT / "solver_curve.npz", compute_curve)
assert (curve["rows"] == curve_rows).all()
print(f"solver curve: {len(curve_rows)} scenarios ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

# %% [markdown]
# ## Links near the closures (within `near_hops` undirected hops of a closed link)

# %%
a_idx, b_idx = links.a_node.to_numpy(), links.b_node.to_numpy()
n_nodes = int(max(a_idx.max(), b_idx.max())) + 1
street_graph = csr_matrix((np.ones(real.sum()), (a_idx[real], b_idx[real])), shape=(n_nodes, n_nodes))


def near_mask(closed):
    ends = np.unique(np.r_[a_idx[closed], b_idx[closed]])
    if len(ends) == 0:
        return np.zeros(len(closed), dtype=bool)
    hops = dijkstra(street_graph, directed=False, indices=ends, unweighted=True, limit=args.near_hops, min_only=True)
    return np.isfinite(hops[a_idx]) | np.isfinite(hops[b_idx])


# %% [markdown]
# ## Scores

# %%
def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def scores(pred, rows):
    pos = np.searchsorted(test, rows)
    y, closed, base = data["flows"][rows], data["closed"][rows], nc["flows"][pos]
    m = ~closed & real[None, :]
    near = np.stack([near_mask(c) for c in closed]) & m
    t = evaluate(pred, data, links, links.is_connector.to_numpy(), rows, grid).loc[("all", "real")]
    return {"r2": t.r2, "mae": t.mae, "conservation": t.conservation,
            "r2_change": r2((y - base)[m], (pred - base)[m]), "mae_change": float(np.abs((y - pred)[m]).mean()),
            "r2_near": r2(y[near], pred[near]), "r2_change_near": r2((y - base)[near], (pred - base)[near]),
            "near_links": float(near.sum(1).mean())}


aon_cs, aon_cs_index = closure_set_aon(NETWORK, args.set, data, workers=args.workers)
aon_cs = aon_cs[aon_cs_index[test]]
base = np.asarray(net.demand)
demand_ratio = (data["demand_prod"][test].sum(1) / (base.sum() - np.trace(base)))[:, None]
preds = {"per-link mean": PerLinkMean().fit(data, train).predict(data, test),
         "per-link mean x demand": PerLinkMean(scale=True).fit(data, train).predict(data, test),
         "no-closure equilibrium": np.where(data["closed"][test], 0.0, nc["flows"]),
         "all-or-nothing": data["aon"][test],
         "AON on no-closure paths": np.where(data["closed"][test], 0.0, data["aon_open"][test]),
         "AON per closure set (base demand)": aon_cs,
         "AON per closure set x demand": aon_cs * demand_ratio}
for spec in args.runs:
    name, run = spec.split("=", 1)
    with np.load(ROOT / run / "predictions.npz") as f:
        assert (f["test_rows"] == test).all(), f"{run}: other test rows"
        preds[name] = f["test"]

rows_all = []
for name, p in preds.items():
    rows_all.append({"method": name, "sample": "test", **scores(p, test)})
    print(f"{name:28} test  R² {rows_all[-1]['r2']:.3f} | change R² {rows_all[-1]['r2_change']:.3f} | near R² {rows_all[-1]['r2_near']:.3f}", flush=True)
pos = np.searchsorted(test, curve_rows)
curve_preds = {name: p[pos] for name, p in preds.items()}
curve_preds |= {f"solver, {k} iteration{'s' if k > 1 else ''}": curve[f"solver_{k}_flows"] for k in args.iterations}
for name, p in curve_preds.items():
    rows_all.append({"method": name, "sample": "curve", **scores(p, curve_rows)})
table = pd.DataFrame(rows_all)
table.to_csv(OUT / "scores.csv", index=False)

# %% [markdown]
# ## Time per scenario on one Grace core

# %%
timing = [{"method": "all-or-nothing (AequilibraE)", "sec_median": float(np.median(curve["aon_secs"]))}]
timing += [{"method": f"solver, {k} iterations", "sec_median": float(np.median(curve[f"solver_{k}_secs"]))} for k in args.iterations]
timing += [{"method": "solver, converged (data generation)", "sec_median": float(np.median(data["runtime"][test]))},
           {"method": "no-closure equilibrium", "sec_median": float(np.median(nc["runtime"]))}]


def load_model(run, ds):
    cfg = yaml.safe_load((ROOT / run / "config.yaml").read_text())
    model = importlib.import_module(f"models.{cfg['model']}").Model(num_zones=net.num_zones, **cfg.get("model_args", {}))
    model.fit_normalizer(ds, ds.rows("train", holdout=False))
    model.load_state_dict(torch.load(ROOT / run / "best.pt", map_location="cpu"))
    return model.eval()


if args.runs:
    torch.set_num_threads(1)
    ds = GraphDataset(NETWORK, args.set, with_closure_set_aon=True)
    with torch.inference_mode():
        for spec in args.runs:
            name, run = spec.split("=", 1)
            model, secs = load_model(run, ds), []
            for r in curve_rows[:args.time_scenarios]:
                t0 = time.perf_counter()
                model.predict_flows(Batch.from_data_list([ds[r]]))
                secs.append(time.perf_counter() - t0)
            timing.append({"method": f"{name}, forward pass", "sec_median": float(np.median(secs))})
timing = pd.DataFrame(timing)
timing.to_csv(OUT / "timing_single_core.csv", index=False)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(timing.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
