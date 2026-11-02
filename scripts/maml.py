# %% [markdown]
# # MAML meta-training

# %%
import argparse
import importlib
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from ta import ROOT
from ta.dataset import GraphDataset
from ta.fewshot import eval_episodes, evaluate_fewshot, meta_loss, network_episode, network_tasks, summary, train_tasks
from ta.virtual import expand

# %% [markdown]
# ## Model and run folder


# %%
def build_model(cfg, tasks):
    by_ds = {}
    for ds, rows in tasks.values():
        by_ds.setdefault(id(ds), (ds, []))[1].extend(rows)
    dss, rows = zip(*[(d, np.sort(r)) for d, r in by_ds.values()])
    model = importlib.import_module(f"models.{cfg['model']}").Model(num_zones=dss[0].net.num_zones, **cfg.get("model_args", {}))
    if len(dss) == 1:
        model.fit_normalizer(dss[0], rows[0])
    else:
        model.fit_normalizer(list(dss), list(rows))
    if cfg.get("init"):
        model.load_state_dict(torch.load(ROOT / cfg["init"] / "best.pt", map_location="cpu"))
    return model


def setup_tasks(cfg, limit=None):
    if cfg.get("task", "closure_set") == "closure_set":
        ds = GraphDataset(cfg["network"], cfg["dataset"])
        val = [(ds, eval_episodes(ds, "val", cfg["seed"], limit))]
        test = [(ds, eval_episodes(ds, "test", cfg["seed"], limit))]
        return train_tasks(ds, cfg.get("train_closure_sets"), cfg["seed"]), val, test
    load = lambda key: [GraphDataset(n, cfg["dataset"]) for n in expand(cfg.get(key, []))[:limit]]
    episodes = lambda key: [(ds, [network_episode(ds, cfg["seed"])]) for ds in load(key)]
    return network_tasks(load("train_networks")), episodes("val_networks"), episodes("eval_networks")


def fewshot_test(model, groups, ks, inner):
    rows = []
    for ds, eps in groups:
        for k in ks:
            if k > min(len(ep["support"]) for ep in eps):
                continue
            table = evaluate_fewshot(model, ds, eps, k, "adapt", **inner)[2]
            rows.append(summary(table, network=ds.network, method="maml_adapt" if k else "zero_shot", k=k, solves_per_task=k))
    return pd.DataFrame(rows)


def inner_settings(mc, device):
    return {"steps": mc["inner_steps"], "lr": mc["inner_lr"], "scope": mc["scope"], "device": device}


def evaluate_saved(run_dir, ks, device="cpu", threads=4):
    torch.set_num_threads(threads)
    device = torch.device(device)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg.get("task") == "network", "extra k values are for networks as tasks"
    model = importlib.import_module(f"models.{cfg['model']}").Model(num_zones=None, **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    groups = [(ds, [network_episode(ds, cfg["seed"])]) for ds in [GraphDataset(n, cfg["dataset"]) for n in expand(cfg["eval_networks"])]]
    out = fewshot_test(model.to(device), groups, ks, inner_settings(cfg["maml"], device))
    out.to_csv(run_dir / "fewshot_test_extra.csv", index=False)
    return out


def fewshot_scores(model, groups, k, inner):
    tables = [evaluate_fewshot(model, ds, eps, k, "adapt", **inner)[2].loc[("all", "real")] for ds, eps in groups]
    return pd.DataFrame(tables).mean()


def new_run_dir(cfg, smoke=False):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{'smoke_' if smoke else ''}{stamp}_{cfg.get('name', 'maml')}"
    parent = ROOT / "runs" / cfg["model"]
    parent.mkdir(parents=True, exist_ok=True)
    for k in range(100):
        run_dir = parent / (base if k == 0 else f"{base}_{k}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a run folder for {base}")


# %% [markdown]
# ## Meta-training loop


# %%
def meta_train(cfg, run_dir, smoke=False, resume=False):
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    device = torch.device(cfg.get("device", "cpu"))
    mc = cfg["maml"]
    steps, limit = (2, 4) if smoke else (cfg["meta_steps"], None)
    if smoke:
        mc = mc | {"tasks_per_step": min(mc["tasks_per_step"], 2)}
    tasks, val, test = setup_tasks(cfg, limit)
    model = build_model(cfg, tasks).to(device)
    val_every = max(1, steps // cfg.get("val_points", 30))
    inner = inner_settings(mc, device)
    print(f"training tasks {len(tasks)} | k {mc['k']} m {mc['m']} | {mc['tasks_per_step']} tasks × {steps} outer steps | "
          f"inner {mc['inner_steps']} × lr {mc['inner_lr']} ({mc['scope']}, {'first' if mc['first_order'] else 'second'} order)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3)
    rng = np.random.default_rng([cfg["seed"], 3])
    history, best, start, losses, skipped = [], np.inf, 1, [], 0
    if resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        history, best, start, skipped = state["history"], state["best"], state["step"] + 1, state.get("skipped", 0)
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed at step {start}")

    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        model.train()
        opt.zero_grad()
        loss = meta_loss(model, tasks, rng, mc["k"], mc["m"], mc["tasks_per_step"], mc["inner_steps"], mc["inner_lr"],
                         mc["scope"], mc["first_order"], device)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(loss.item())
        else:
            skipped += 1
        sched.step()
        if step % val_every and step != steps:
            continue
        v = fewshot_scores(model, val, mc["k"], inner)
        history.append({"step": step, "meta_loss": float(np.mean(losses)) if losses else np.nan, "val_r2": v.r2, "val_mae": v.mae,
                        "skipped": skipped, "lr": sched.get_last_lr()[0], "seconds": time.perf_counter() - t0})
        losses, t0 = [], time.perf_counter()
        if v.mae < best:
            best = v.mae
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "history": history, "best": best, "skipped": skipped, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()},
                   run_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        h = history[-1]
        print(f"step {step}/{steps} | meta loss {h['meta_loss']:.5f} | val k={mc['k']} R² {h['val_r2']:.4f} MAE {h['val_mae']:.1f} "
              f"| skipped {skipped} | {h['seconds']:.0f}s", flush=True)

    model.load_state_dict(torch.load(run_dir / "best.pt"))
    out = fewshot_test(model, test, cfg.get("eval_k", [0, 1, 2, 4]), inner)
    out.to_csv(run_dir / "fewshot_test.csv", index=False)
    print(out.to_string(float_format=lambda x: f"{x:.4g}"))
    return run_dir


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/maml_aon_gnn_main.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--evaluate", nargs="+")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 50])
    args = parser.parse_known_args()[0]
    overrides = {k: v for k, v in [("device", args.device), ("threads", args.threads)] if v is not None}
    if args.evaluate:
        for run in args.evaluate:
            out = evaluate_saved(ROOT / run, args.k, args.device or "cpu", args.threads or 4)
            print(run, out[["network", "k", "r2"]].to_string(float_format=lambda x: f"{x:.4g}"), sep="\n", flush=True)
        raise SystemExit
    if args.resume:
        run_dir = ROOT / args.resume
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) | overrides
    else:
        cfg = yaml.safe_load((ROOT / args.config).read_text()) | overrides
        run_dir = new_run_dir(cfg, smoke=not args.full)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"run dir: {run_dir.relative_to(ROOT)} | pid {os.getpid()}")
    meta_train(cfg, run_dir, smoke=not (args.full or args.resume), resume=bool(args.resume))
