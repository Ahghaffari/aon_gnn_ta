# %% [markdown]
# # Few-shot comparison table

# %%
import argparse
import importlib

import numpy as np
import pandas as pd
import torch
import yaml

from ta import ROOT
from ta.dataset import GraphDataset
from ta.fewshot import fewshot_table, train_tasks

LRS = [0.01, 0.03, 0.1, 0.3]

# %% [markdown]
# ## Models


# %%
def load_model(cfg, ds, run_dir=None):
    module = importlib.import_module(f"models.{cfg['model']}")
    model = module.Model(num_zones=ds.net.num_zones, **cfg.get("model_args", {}))
    rows = np.array([r for _, v in train_tasks(ds, cfg.get("train_closure_sets"), cfg["seed"]).values() for r in v])
    model.fit_normalizer(ds, np.sort(rows))
    if run_dir is not None:
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model


def evaluate_runs(run_dirs, ks, lrs, steps, device, full, scratch=False, scratch_steps=50):
    limit = None if full else 4
    lrs = lrs if full else lrs[1:3]
    out, cache = [], {}
    for run_dir in run_dirs:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        key = (cfg["network"], cfg["dataset"])
        ds = cache.setdefault(key, GraphDataset(*key))
        model = load_model(cfg, ds, run_dir).to(device)
        table = fewshot_table(model, ds, ks, lrs, steps, device=device, limit=limit, label=run_dir.name, seed=cfg["seed"])
        out.append(table.assign(network=cfg["network"], dataset=cfg["dataset"], train_closure_sets=cfg.get("train_closure_sets", "all")))
        if scratch:
            fresh = load_model(cfg, ds).to(device)
            table = fewshot_table(fresh, ds, ks, lrs, scratch_steps, scopes=("all",), methods=(), device=device, limit=limit,
                                  label="scratch", seed=cfg["seed"])
            out.append(table.assign(method=table.method.replace({"adapt_all": "scratch", "zero_shot": "scratch_aon"}),
                                    network=cfg["network"], dataset=cfg["dataset"], train_closure_sets=0))
            scratch = False  # the scratch arm does not depend on the trained run: once is enough
    return pd.concat(out, ignore_index=True)


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--runs", nargs="+", default=["runs/aon_gnn/20261002_143538_main"])
    parser.add_argument("--k", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--lrs", nargs="+", type=float, default=LRS)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--scratch-steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_known_args()[0]
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    runs = [ROOT / r for r in args.runs]
    table = evaluate_runs(runs, args.k, args.lrs, args.steps, torch.device(args.device), args.full, args.scratch, args.scratch_steps)
    print(table.to_string(float_format=lambda x: f"{x:.4g}"))
    if args.full:
        net = table.network.iloc[0]
        out = ROOT / "runs" / "results" / net / "fewshot"
        out.mkdir(parents=True, exist_ok=True)
        name = runs[0].name if len(runs) == 1 else f"{runs[0].name}_and_{len(runs) - 1}_more"
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"saved {out / name}.csv")
