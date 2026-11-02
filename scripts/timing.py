# %% [markdown]
# # Time per scenario of every method

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import importlib
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from ta import ROOT
from ta.baselines import aon_flows
from ta.dataset import GraphDataset
from ta.features import LinkFeatures, aon_cache
from ta.network import solve_scenario

NETWORK, SET = "Berlin-Friedrichshain", "main"
GNN_RUN = ROOT / "runs" / "aon_gnn" / "20261002_143538_main"
REF_RUN = ROOT / "runs" / "ref_gatedgcn" / "20260929_193510"
OUT = ROOT / "runs" / "results" / NETWORK / "timing_single_core.csv"

# %% [markdown]
# ## Setup

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--every", type=int, default=7)
args = parser.parse_known_args()[0]
torch.set_num_threads(1)
ds = GraphDataset(NETWORK, SET)
net, data, links = ds.net, ds.data, ds.links
real = ~links.is_connector.to_numpy()
test = ds.rows("test")
sample = test[:: args.every]
cache = aon_cache(NETWORK, SET, data)
print(f"timing on {len(sample)} test scenarios, one process, one thread", flush=True)


def r2(pred, rows):
    m = ~data["closed"][rows] & real[None, :]
    y, p = data["flows"][rows][m], np.asarray(pred)[m]
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(run):
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model = importlib.import_module(f"models.{cfg['model']}").Model(num_zones=net.num_zones, **cfg.get("model_args", {}))
    model.fit_normalizer(ds, ds.rows("train", holdout=False))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    return model.eval()


rows, preds = [], {}


def record(name, secs, pred):
    secs = np.asarray(secs)
    rows.append({"method": name, "sec_median": float(np.median(secs)), "sec_mean": float(secs.mean()), "r2_sample": r2(pred, sample)})
    print(f"{name:28} median {1e3 * np.median(secs):9.1f} ms | mean {1e3 * secs.mean():9.1f} ms | R² {rows[-1]['r2_sample']:.3f}", flush=True)


# %% [markdown]
# ## All-or-nothing, AON-GNN, HetGatedGCN

# %%
secs, out = [], []
for r in sample:
    t0 = time.perf_counter()
    f = aon_flows(net, data["demand"][r], data["closed"][r])
    secs.append(time.perf_counter() - t0)
    out.append(np.where(data["closed"][r], 0.0, f))
aon_secs = np.array(secs)
record("all-or-nothing", aon_secs, out)

with torch.inference_mode():
    for name, run, with_aon in [("AON-GNN", GNN_RUN, True), ("HetGatedGCN", REF_RUN, False)]:
        model = load(run)
        secs, out = [], []
        for i, r in enumerate(sample):
            t0 = time.perf_counter()
            item = ds[r]
            if with_aon:
                item.aon = torch.as_tensor(aon_flows(net, data["demand"][r], data["closed"][r]), dtype=torch.float32)
            pred = model.predict_flows(Batch.from_data_list([item]))
            secs.append(time.perf_counter() - t0)
            out.append(pred.numpy())
        record(name, secs, out)

# %% [markdown]
# ## Boosted trees (refit once with the settings of `scripts/plot_results.py`)

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

feats = LinkFeatures(net)
train = ds.rows("train", holdout=False)
x = feats.matrix(data, cache, train)
keep = ~data["closed"][train].reshape(-1)
trees = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                      validation_fraction=0.1, random_state=42).fit(x[keep], data["flows"][train].reshape(-1)[keep])
secs, out = [], []
for r in sample:
    t0 = time.perf_counter()
    aon_flows(net, data["demand"][r], data["closed"][r])
    aon_flows(net, data["demand"][r], net.default_closed)
    p = np.clip(trees.predict(feats.matrix(data, cache, [r])), 0, None)
    secs.append(time.perf_counter() - t0)
    out.append(np.where(data["closed"][r], 0.0, p))
record("boosted trees", secs, out)

# %% [markdown]
# ## Equilibrium solver: truncated and converged

# %%
for n in [1, 2, 3, 5, 10, 20]:
    secs, out = [], []
    for r in sample:
        t0 = time.perf_counter()
        sol, _ = solve_scenario(net, data["demand"][r], data["closed"][r], rgap_target=1e-12, max_iter=n, retries=1)
        secs.append(time.perf_counter() - t0)
        out.append(sol.flows)
    record(f"solver, {n} iterations", secs, out)
secs, out, iters = [], [], []
for r in sample:
    t0 = time.perf_counter()
    sol, _ = solve_scenario(net, data["demand"][r], data["closed"][r], rgap_target=1e-5, max_iter=2000)
    secs.append(time.perf_counter() - t0)
    out.append(sol.flows)
    iters.append(sol.iterations)
record("solver, converged", secs, out)
print(f"converged solves: median {np.median(iters):.0f} iterations", flush=True)

# %% [markdown]
# ## Save

# %%
table = pd.DataFrame(rows)
table.to_csv(OUT, index=False)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
