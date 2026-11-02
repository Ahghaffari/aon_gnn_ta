# %% [markdown]
# # Large network (Exp 3): time per scenario, one process on one core, one scenario at a time

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import importlib
import time
from dataclasses import fields

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from ta import ROOT
from ta.dataset import GraphDataset
from ta.network import Network
from ta.scenarios import ScenarioConfig, demand_rng, sample_demand
from ta.solver import aon_assign, prepare_links, prune_dead_ends, run_aequilibrae, solve
from ta.splits import load_set

# %% [markdown]
# ## Setup

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config", default="configs/data_sydney.yaml")
parser.add_argument("--set", default="main")
parser.add_argument("--index", type=int, default=0, help="rank among the test scenarios nearest the median stored solve time")
parser.add_argument("--iterations", type=int, nargs="*", default=[1, 2, 3, 5, 10, 20])
parser.add_argument("--no-closure-gap", type=float, default=1e-4)
parser.add_argument("--runs", nargs="*", default=[], help="name=run_dir of trained models")
args = parser.parse_known_args()[0]
torch.set_num_threads(1)

config = yaml.safe_load((ROOT / args.config).read_text())
set_cfg = config["sets"][args.set]
NETWORK, SEED, rgap_target, max_iter = config["network"], config["seed"], config["solver"]["rgap_target"], config["solver"]["max_iter"]
names = {f.name for f in fields(ScenarioConfig)}
cfg = ScenarioConfig(**config["demand"], **{k: v for k, v in set_cfg.items() if k in names})
OUT = ROOT / "runs" / "results" / NETWORK / "exp3" / "timing_alone"
OUT.mkdir(parents=True, exist_ok=True)

net = Network.from_tntp(NETWORK)
data, links, splits = load_set(NETWORK, args.set)
test = np.flatnonzero((splits.split == "test").to_numpy())
order = test[np.argsort(np.abs(data["runtime"][test] - np.median(data["runtime"][test])))]
r = int(order[args.index])
closed = data["closed"][r]
od = sample_demand(demand_rng(net.name, SEED, int(data["closure_set"][r]), int(data["variant"][r])), net.demand, cfg)[0]
np.fill_diagonal(od, 0.0)
open_links = net.links[~closed].reset_index(drop=True)
rows = []


def record(method, secs, **extra):
    rows.append({"method": method, "sec": secs, "row": r, **extra})
    print(f"{method:44} {secs:9.2f} s {extra}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"timing_{args.index}.csv", index=False)


# %% [markdown]
# ## AON and truncated solves

# %%
t0 = time.perf_counter()
aon_assign(open_links, od, net.block_centroids)
record("all-or-nothing", time.perf_counter() - t0)

prep = prepare_links(open_links)
for k in args.iterations:
    t0 = time.perf_counter()
    keep = prune_dead_ends(prep, net.num_zones)
    active = prep[keep].reset_index(drop=True)
    active["link_id"] = np.arange(1, len(active) + 1)
    run_aequilibrae(active, od, net.block_centroids, 1e-12, k, "bfw", 1)
    record(f"solver, {k} iterations", time.perf_counter() - t0)

# %% [markdown]
# ## Forward pass of the trained models

# %%
if args.runs:
    ds = GraphDataset(NETWORK, args.set, with_closure_set_aon=True)
    with torch.inference_mode():
        for spec in args.runs:
            name, run = spec.split("=", 1)
            run_cfg = yaml.safe_load((ROOT / run / "config.yaml").read_text())
            model = importlib.import_module(f"models.{run_cfg['model']}").Model(num_zones=net.num_zones, **run_cfg.get("model_args", {}))
            model.fit_normalizer(ds, ds.rows("train", holdout=False))
            model.load_state_dict(torch.load(ROOT / run / "best.pt", map_location="cpu"))
            model.eval()
            secs = []
            for _ in range(3):
                t0 = time.perf_counter()
                model.predict_flows(Batch.from_data_list([ds[r]]))
                secs.append(time.perf_counter() - t0)
            record(f"{name}, forward pass", float(np.median(secs)))

# %% [markdown]
# ## Converged solves (gap check and re-solves included, as for the labels)

# %%
sol = solve(open_links, od, net.block_centroids, rgap_target=rgap_target, max_iter=max_iter)
record("solver, converged", sol.runtime, iterations=sol.iterations, rgap=sol.rgap, stored_sec=float(data["runtime"][r]))
sol = solve(net.links, od, net.block_centroids, rgap_target=args.no_closure_gap, max_iter=max_iter)
record("no-closure equilibrium", sol.runtime, iterations=sol.iterations, rgap=sol.rgap)
