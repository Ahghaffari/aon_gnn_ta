# %% [markdown]
# # Error vs disruption per test scenario

# %%
import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ta import ROOT
from ta.disruption import METRICS, load_disruption
from ta.features import aon_cache
from ta.splits import load_set

NETWORK, SET = "Berlin-Friedrichshain", "main"
OUT, FIG = ROOT / "runs" / "results" / NETWORK, ROOT / "runs" / "figures"
REFERENCE_RUNS = {"AON-GNN": ROOT / "runs" / "aon_gnn" / "20260930_061736", "HGNN": ROOT / "runs" / "ref_gatedgcn" / "20260929_193510"}
COLORS = {"AON-GNN": "#1f77b4", "boosted trees": "#2ca02c", "HGNN": "#d62728", "all-or-nothing": "#ff7f0e"}

# %% [markdown]
# ## Predictions of every method on the test scenarios


# %%
data, links, splits = load_set(NETWORK, SET)
test = np.flatnonzero(splits.split.to_numpy() == "test")
real = ~links.is_connector.to_numpy()


def run_predictions(run):
    with np.load(run / "predictions.npz") as f:
        assert np.array_equal(f["test_rows"], test)
        return f["test"]


def ablation_runs():
    runs = {}
    for run in sorted((ROOT / "runs" / "aon_gnn").glob("*_ablation_*")):
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        if (run / "predictions.npz").exists() and cfg.get("dataset") == SET:
            runs[cfg["name"].removeprefix("ablation_")] = run
    return runs


preds = {"all-or-nothing": aon_cache(NETWORK, SET, data)["aon"][test]}
preds["boosted trees"] = np.load(OUT / "trees_test_predictions.npy")
preds |= {name: run_predictions(run) for name, run in REFERENCE_RUNS.items()}
ablations = ablation_runs()
preds |= {f"ablation: {name}": run_predictions(run) for name, run in ablations.items()}
print("methods:", list(preds))

# %% [markdown]
# ## Per-scenario errors (open real links)


# %%
mask = ~data["closed"][test] & real[None, :]
y = data["flows"][test]


def per_scenario(p):
    err = np.abs(p - y)
    mae = np.array([err[i, mask[i]].mean() for i in range(len(test))])
    r2 = np.array([1 - ((p[i, m] - y[i, m]) ** 2).sum() / ((y[i, m] - y[i, m].mean()) ** 2).sum() for i, m in enumerate(mask)])
    return mae, r2


dis = load_disruption(NETWORK, SET).iloc[test].reset_index(drop=True)
table = dis.copy()
table["gridlock"] = splits.gridlock.to_numpy()[test]
for name, p in preds.items():
    table[f"mae: {name}"], table[f"r2: {name}"] = per_scenario(p)
OUT.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT / "disruption_errors.csv", index=False)
print(table[[f"mae: {n}" for n in preds]].describe().T[["mean", "50%", "max"]].to_string(float_format=lambda v: f"{v:.1f}"))

# %% [markdown]
# ## Spearman correlation of each metric with each method's MAE, plain and partial on `num_closed`


# %%
def ranks(x):
    return pd.Series(x).rank().to_numpy()


def partial_spearman(x, yv, z):
    rx, ry, rz = ranks(x), ranks(yv), ranks(z)
    A = np.column_stack([np.ones_like(rz), rz])
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return np.corrcoef(res(rx), res(ry))[0, 1]


rows = []
for metric in METRICS:
    for name in preds:
        mae = table[f"mae: {name}"].to_numpy()
        rows.append({"metric": metric, "method": name, "spearman": np.corrcoef(ranks(table[metric]), ranks(mae))[0, 1],
                     "partial_on_num_closed": np.nan if metric == "num_closed" else partial_spearman(table[metric], mae, table.num_closed)})
corr = pd.DataFrame(rows)
corr.to_csv(OUT / "disruption_correlations.csv", index=False)
for col in ["spearman", "partial_on_num_closed"]:
    print(f"\n{col} (metric × method MAE):")
    print(corr.pivot(index="metric", columns="method", values=col).loc[METRICS, list(preds)].to_string(float_format=lambda v: f"{v:.2f}"))

# %% [markdown]
# ## Figure: mean per-scenario MAE per metric quintile (main methods)

# %%
fig, axes = plt.subplots(2, 5, figsize=(22, 8), facecolor="white")
for ax, metric in zip(axes.flat, METRICS):
    bins = pd.qcut(table[metric].rank(method="first"), 5, labels=False)
    centers = table.groupby(bins)[metric].median()
    for name, color in COLORS.items():
        ax.plot(range(5), table.groupby(bins)[f"mae: {name}"].mean(), "o-", color=color, label=name)
    ax.set_xticks(range(5), [f"{v:.3g}" for v in centers], fontsize=8)
    ax.set_title(metric); ax.set_xlabel("quintile (median value)", fontsize=8); ax.grid(alpha=0.3)
for ax in axes[:, 0]:
    ax.set_ylabel("mean MAE per scenario (veh/h)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Test scenarios (unseen closures): error vs disruption", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "disruption_errors.png", dpi=130, bbox_inches="tight")
print("saved", (FIG / "disruption_errors.png").relative_to(ROOT))
