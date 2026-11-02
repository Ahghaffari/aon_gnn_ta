# %% [markdown]
# # ML baselines on per-link features

# %%
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ta import ROOT
from ta.baselines import PerLinkMean, aon_flows
from ta.features import FEATURES, LinkFeatures, aon_cache
from ta.metrics import evaluate, select
from ta.network import Network
from ta.splits import load_set

NETWORK = "Berlin-Friedrichshain"
OUT = ROOT / "runs" / "baselines" / NETWORK


def models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "boosted trees": lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=63,
                                                               early_stopping=True, validation_fraction=0.1, random_state=42),
    }


# %% [markdown]
# ## Helpers


# %%
def design(feats, data, cache, rows, link_mean):
    x = feats.matrix(data, cache, rows, link_mean)
    open_m = ~data["closed"][rows].reshape(-1)
    return x, open_m


def predict(model, feats, data, cache, rows, link_mean):
    x, open_m = design(feats, data, cache, rows, link_mean)
    out = np.zeros(len(x))
    out[open_m] = np.clip(model.predict(x[open_m]), 0.0, None)
    return out.reshape(len(rows), -1)


def feature_seconds(net, data, rows, n=50):
    t0 = time.perf_counter()
    for r in rows[:n]:
        aon_flows(net, data["demand"][r], data["closed"][r])
        aon_flows(net, data["demand"][r], net.default_closed)
    return (time.perf_counter() - t0) / min(n, len(rows))


# %% [markdown]
# ## Run one dataset


# %%
def run_set(name):
    net = Network.from_tntp(NETWORK)
    data, links, splits = load_set(NETWORK, name)
    conn, grid = links.is_connector.to_numpy(), splits.gridlock.to_numpy()
    cache = aon_cache(NETWORK, name, data)
    feats = LinkFeatures(net)
    train = np.flatnonzero(((splits.split == "train") & ~splits.holdout_variant).to_numpy())
    val, test, holdout = select(splits, "val"), select(splits, "test"), select(splits, "train", holdout=True)
    link_mean = PerLinkMean().fit(data, train).mean
    feat_sec = feature_seconds(net, data, test)

    results = []
    for variant, lm in [("transferable", None), ("+ link mean", link_mean)]:
        x, open_m = design(feats, data, cache, train, lm)
        y = data["flows"][train].reshape(-1)
        for model_name, make in models().items():
            t0 = time.perf_counter()
            model = make().fit(x[open_m], y[open_m])
            fit_s = time.perf_counter() - t0
            val_pred = predict(model, feats, data, cache, val, lm)
            val_r2 = evaluate(val_pred, data, links, conn, val).loc[("all", "real"), "r2"]
            print(f"[{name}] {model_name} ({variant}): fit {fit_s:.0f}s, val R² {val_r2:.3f}")
            for evaluation, rows in [("test (unseen closures)", test), ("holdout (seen closures, new demand)", holdout)]:
                t0 = time.perf_counter()
                pred = predict(model, feats, data, cache, rows, lm)
                sec = feat_sec + (time.perf_counter() - t0) / len(rows)
                table = evaluate(pred, data, links, conn, rows, grid).reset_index()
                table.insert(0, "baseline", f"{model_name} ({variant})")
                table.insert(0, "evaluation", evaluation)
                table["sec_per_scenario"] = sec
                results.append(table)
    out = pd.concat(results, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"ml_{name}.csv", index=False)
    return out


def summary(name, out):
    prev = pd.read_csv(OUT / f"{name}.csv")
    prev = prev[prev.baseline.isin(["per-link mean x demand", "all-or-nothing", "same closure set"])]
    both = pd.concat([prev, out], ignore_index=True)
    real = both[both.links == "real"]
    wide = real.pivot_table(index=["evaluation", "baseline"], columns="subset", values="r2", sort=False)
    wide.columns = [f"R² {c}" for c in wide.columns]
    alls = real[real.subset == "all"].set_index(["evaluation", "baseline"])
    for col, src in [("MAE", "mae"), ("wMAPE", "wmape"), ("conservation", "conservation"), ("s/scenario", "sec_per_scenario")]:
        wide[col] = alls[src]
    return wide[["R² all", "R² no_gridlock", "R² gridlock", "MAE", "wMAPE", "conservation", "s/scenario"]]


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sets", nargs="+", default=["main", "old_like"])
    args = parser.parse_known_args()[0]
    for name in args.sets:
        out = run_set(name)
        print(f"== {name}")
        print(summary(name, out).to_string(float_format=lambda v: f"{v:.3g}"))
