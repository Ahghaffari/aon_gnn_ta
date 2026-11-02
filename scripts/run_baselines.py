# %% [markdown]
# # Score the no-learning baselines

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ta import ROOT
from ta.baselines import NearestClosureSets, PerLinkMean, SameClosureSet, aon_flows, no_closure_flows
from ta.metrics import evaluate, select
from ta.network import Network
from ta.splits import load_set

NETWORK = "Berlin-Friedrichshain"
K_GRID = [1, 3, 5, 10, 20, 40, 80, 160]

# %% [markdown]
# ## Helpers

# %%
WORKER = {}


def init_worker(network):
    WORKER["net"] = Network.from_tntp(network)


def no_closure_one(args):
    demand, closed = args
    return no_closure_flows(WORKER["net"], demand, closed)


def timed(fn, n):
    t0 = time.perf_counter()
    pred = fn()
    return pred, (time.perf_counter() - t0) / n


def choose_k(data, train, val, real):
    scores = {}
    for k in K_GRID:
        pred = NearestClosureSets(k, real).fit(data, train).predict(data, val)
        y, p = data["flows"][val], pred
        m = ~data["closed"][val] & real[None, :]
        scores[k] = 1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    return max(scores, key=scores.get), scores


# %% [markdown]
# ## Run one dataset


# %%
def run_set(name, workers):
    net = Network.from_tntp(NETWORK)
    data, links, splits = load_set(NETWORK, name)
    real, conn, grid = ~links.is_connector.to_numpy(), links.is_connector.to_numpy(), splits.gridlock.to_numpy()
    train = np.flatnonzero(((splits.split == "train") & ~splits.holdout_variant).to_numpy())
    val, test, holdout = select(splits, "val"), select(splits, "test"), select(splits, "train", holdout=True)

    k, k_scores = choose_k(data, train, val, real)
    print(f"[{name}] k chosen on val: {k} (val R² per k: " + ", ".join(f"{kk}: {v:.3f}" for kk, v in k_scores.items()) + ")")

    stat = {"per-link mean": PerLinkMean(), "per-link mean x demand": PerLinkMean(scale=True),
            f"nearest closure sets (k={k})": NearestClosureSets(k, real), "same closure set": SameClosureSet()}
    for b in stat.values():
        b.fit(data, train)

    results = []

    def score(label, evaluation, rows, pred, sec):
        table = evaluate(pred, data, links, conn, rows, grid).reset_index()
        table.insert(0, "baseline", label)
        table.insert(0, "evaluation", evaluation)
        table["sec_per_scenario"] = sec
        results.append(table)

    for evaluation, rows in [("test (unseen closures)", test), ("holdout (seen closures, new demand)", holdout)]:
        for label, b in stat.items():
            if label == "same closure set" and evaluation.startswith("test"):
                continue
            score(label, evaluation, rows, *timed(lambda: b.predict(data, rows), len(rows)))
        score("all-or-nothing", evaluation, rows,
              *timed(lambda: np.stack([aon_flows(net, data["demand"][r], data["closed"][r]) for r in rows]), len(rows)))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(NETWORK,)) as pool:
        pred = np.stack(list(pool.map(no_closure_one, [(data["demand"][r], data["closed"][r]) for r in test], chunksize=4)))
    cpu_sec = (time.perf_counter() - t0) * workers / len(test)
    score("no-closure equilibrium (reference)", "test (unseen closures)", test, pred, cpu_sec)

    out = pd.concat(results, ignore_index=True)
    path = ROOT / "runs" / "baselines" / NETWORK / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out, path


def summary(out):
    real = out[out.links == "real"]
    wide = real.pivot_table(index=["evaluation", "baseline"], columns="subset", values="r2", sort=False)
    wide.columns = [f"R² {c}" for c in wide.columns]
    alls = real[real.subset == "all"].set_index(["evaluation", "baseline"])
    wide["MAE"], wide["wMAPE"] = alls["mae"], alls["wmape"]
    wide["conservation"], wide["s/scenario"] = alls["conservation"], alls["sec_per_scenario"]
    return wide[["R² all", "R² no_gridlock", "R² gridlock", "MAE", "wMAPE", "conservation", "s/scenario"]]


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sets", nargs="+", default=["main", "old_like"])
    parser.add_argument("--workers", type=int, default=min(28, os.cpu_count()))
    args = parser.parse_known_args()[0]
    tables = {}
    for name in args.sets:
        tables[name], path = run_set(name, args.workers)
        print(f"[{name}] saved {path.relative_to(ROOT)}")
        print(summary(tables[name]).to_string(float_format=lambda v: f"{v:.3g}"))
