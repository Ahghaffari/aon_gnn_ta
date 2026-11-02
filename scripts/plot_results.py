# %% [markdown]
# # Results plots

# %%
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.collections import LineCollection
from torch_geometric.loader import DataLoader

from ta import ROOT
from ta.baselines import aon_flows
from ta.dataset import GraphDataset
from ta.features import LinkFeatures, aon_cache
from ta.metrics import evaluate
from ta.network import Network

NETWORK, SET = "Berlin-Friedrichshain", "main"
FIG = ROOT / "runs" / "figures"
OUT = ROOT / "runs" / "results" / NETWORK
COLORS = {"AON-GNN": "#1f77b4", "boosted trees": "#2ca02c", "HGNN": "#d62728", "all-or-nothing": "#ff7f0e"}


def latest(model):
    runs = sorted(p for p in (ROOT / "runs" / model).iterdir() if p.is_dir() and not p.name.startswith("smoke") and (p / "predictions.npz").exists())
    return runs[-1]


# %% [markdown]
# ## Load data, runs and predictions

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--gnn-run")
parser.add_argument("--ref-run")
args = parser.parse_known_args()[0]
gnn_run = ROOT / args.gnn_run if args.gnn_run else latest("aon_gnn")
ref_run = ROOT / args.ref_run if args.ref_run else latest("ref_gatedgcn")
print("AON-GNN run:", gnn_run.relative_to(ROOT), "| HGNN run:", ref_run.relative_to(ROOT))

torch.set_num_threads(4)
ds = GraphDataset(NETWORK, SET)
net = ds.net
data, links, splits = ds.data, ds.links, ds.splits
real, conn = ~links.is_connector.to_numpy(), links.is_connector.to_numpy()
test = ds.rows("test")
cache = aon_cache(NETWORK, SET, data)


def run_predictions(run):
    with np.load(run / "predictions.npz") as f:
        assert np.array_equal(f["test_rows"], test)
        return f["test"]


preds = {"AON-GNN": run_predictions(gnn_run), "HGNN": run_predictions(ref_run),
         "all-or-nothing": np.where(data["closed"][test], 0.0, cache["aon"][test])}

# %% [markdown]
# ## Boosted trees (step 2.4, transferable features) — refit once, predictions cached

# %%
OUT.mkdir(parents=True, exist_ok=True)
trees_path = OUT / "trees_test_predictions.npy"
feats = LinkFeatures(net)
if trees_path.exists():
    preds["boosted trees"] = np.load(trees_path)
    trees = None
else:
    from sklearn.ensemble import HistGradientBoostingRegressor

    train = ds.rows("train", holdout=False)
    x = feats.matrix(data, cache, train)
    keep = ~data["closed"][train].reshape(-1)
    t0 = time.perf_counter()
    trees = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                          validation_fraction=0.1, random_state=42).fit(x[keep], data["flows"][train].reshape(-1)[keep])
    print(f"trees refit in {time.perf_counter() - t0:.0f} s")
    xt = feats.matrix(data, cache, test)
    p = np.clip(trees.predict(xt), 0, None).reshape(len(test), -1)
    preds["boosted trees"] = np.where(data["closed"][test], 0.0, p)
    np.save(trees_path, preds["boosted trees"])

# %% [markdown]
# ## Metrics table (test, real links)

# %%
grid = splits.gridlock.to_numpy()
rows = []
for name, p in preds.items():
    t = evaluate(p, data, links, conn, test, grid)
    rows.append({"method": name, "R² all": t.loc[("all", "real"), "r2"], "R² no_gridlock": t.loc[("no_gridlock", "real"), "r2"],
                 "R² gridlock": t.loc[("gridlock", "real"), "r2"], "MAE": t.loc[("all", "real"), "mae"],
                 "wMAPE": t.loc[("all", "real"), "wmape"], "conservation": t.loc[("all", "real"), "conservation"]})
table = pd.DataFrame(rows).set_index("method")
print(table.to_string(float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ## 1. Predicted vs true

# %%
open_real = ~data["closed"][test] & real[None, :]
y = data["flows"][test][open_real]
order = ["AON-GNN", "boosted trees", "HGNN", "all-or-nothing"]
fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
lim = np.percentile(y, 99.9)
for ax, name in zip(axes, order):
    p = preds[name][open_real]
    ax.hexbin(y, p, gridsize=70, extent=(0, lim, 0, lim), bins="log", cmap="viridis", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_title(f"{name}: R² {table.loc[name, 'R² all']:.3f}, MAE {table.loc[name, 'MAE']:.0f}")
    ax.set_xlabel("true flow (veh/h)"); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
axes[0].set_ylabel("predicted flow (veh/h)")
fig.suptitle(f"Test scenarios (unseen closures), open real links, n = {len(y):,} (colour = log count)")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "results_scatter.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 2. R² per scenario

# %%
def per_scenario_r2(p):
    out = []
    for i in range(len(test)):
        m = open_real[i]
        yt, pt = data["flows"][test[i]][m], p[i][m]
        out.append(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    return np.array(out)


r2s = {name: per_scenario_r2(preds[name]) for name in order}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="white")
a1.boxplot([r2s[n] for n in order], showfliers=False)
a1.set_xticks(range(1, 5), order)
a1.set_ylabel("R² of one scenario"); a1.set_title("R² per test scenario (box = 25–75 %, line = median)"); a1.grid(alpha=0.3)
closed_n = data["num_closed"][test]
bins = [0, 10, 20, 30, 40, 50]
centers = [(a + b) / 2 for a, b in zip(bins, bins[1:])]
for n in order:
    med = [np.median(r2s[n][(closed_n >= a) & (closed_n < b)]) for a, b in zip(bins, bins[1:])]
    a2.plot(centers, med, "o-", color=COLORS[n], label=n)
a2.set_xlabel("closed links in the scenario"); a2.set_ylabel("median R² per scenario"); a2.set_title("Harder with more closures")
a2.legend(); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "results_per_scenario.png", dpi=120)
print(pd.DataFrame({n: pd.Series(r2s[n]).describe()[["25%", "50%", "75%"]] for n in order}).T.to_string(float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ## 3. Where the errors are: flow size and distance to the nearest closure

# %%
hop1 = feats.one > 0
dist = np.full((len(test), len(real)), 99)
for i, r in enumerate(test):
    frontier = data["closed"][r].copy()
    seen = frontier.copy()
    for d in range(0, 6):
        nxt = hop1[frontier].any(0) & ~seen
        dist[i][nxt] = d
        seen |= nxt
        frontier = nxt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="white")
flow_bins = [0, 1, 250, 500, 1000, 2000, 1e9]
labels = ["0", "1–250", "250–500", "500–1k", "1k–2k", ">2k"]
for n in order:
    err = np.abs(preds[n] - data["flows"][test])
    ytrue = data["flows"][test]
    a1.plot(labels, [err[open_real & (ytrue >= a) & (ytrue < b)].mean() for a, b in zip(flow_bins, flow_bins[1:])], "o-", color=COLORS[n], label=n)
    a2.plot(["0", "1", "2", "3", "4", "≥5"], [err[open_real & (dist == d)].mean() if d < 5 else err[open_real & (dist >= 5)].mean()
                                              for d in range(6)], "o-", color=COLORS[n], label=n)
a1.set_xlabel("true flow (veh/h)"); a1.set_ylabel("mean absolute error (veh/h)"); a1.set_title("Error by flow size"); a1.legend(); a1.grid(alpha=0.3)
a2.set_xlabel("links to the nearest closure"); a2.set_ylabel("mean absolute error (veh/h)"); a2.set_title("Error by distance to a closure"); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "results_errors.png", dpi=120)

# %% [markdown]
# ## 4–5. On the map: GNN error per link, and one example scenario

# %%
xy = net.nodes.set_index("node")[["x", "y"]]
segs = np.array([[xy.loc[a].to_numpy(), xy.loc[b].to_numpy()] for a, b in zip(links.a_node, links.b_node)])
shift = 0.006 * np.array([[0, 1], [-1, 0]])


def offset(s):
    d = s[:, 1] - s[:, 0]
    nrm = d @ shift / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return s + nrm[:, None, :]


def draw(ax, values, vmax, cmap, title, closed=None, width=None):
    s = offset(segs)[real]
    v = values[real]
    lc = LineCollection(s, array=v, cmap=cmap, norm=plt.Normalize(0, vmax), linewidths=width[real] if width is not None else 2.0)
    ax.add_collection(lc)
    if closed is not None:
        ax.add_collection(LineCollection(offset(segs)[closed & real], colors="red", linewidths=3, linestyles="--"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return lc


gnn_err = np.where(open_real, np.abs(preds["AON-GNN"] - data["flows"][test]), np.nan)
link_mae = np.nanmean(gnn_err, axis=0)
fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
lc = draw(ax, np.nan_to_num(link_mae), np.nanpercentile(link_mae, 98), "magma_r", "AON-GNN: mean absolute error per link over the test scenarios (veh/h)")
fig.colorbar(lc, ax=ax, shrink=0.7)
fig.tight_layout(); fig.savefig(FIG / "results_map_error.png", dpi=120, bbox_inches="tight")

k = int(np.argsort(np.abs(r2s["AON-GNN"] - np.median(r2s["AON-GNN"])))[0])
r = test[k]
vmax = np.percentile(data["flows"][r][real], 99)
fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
width = lambda f: 0.5 + 5 * np.clip(f / vmax, 0, 1)
for ax, (label, f) in zip(axes, [("true equilibrium", data["flows"][r]), (f"AON-GNN (scenario R² {r2s['AON-GNN'][k]:.3f})", preds["AON-GNN"][k]),
                                 (f"all-or-nothing (scenario R² {r2s['all-or-nothing'][k]:.3f})", preds["all-or-nothing"][k])]):
    lc = draw(ax, f, vmax, "viridis", f"{label} — closed links red", closed=data["closed"][r], width=width(f))
fig.colorbar(lc, ax=axes, shrink=0.6, label="flow (veh/h)")
fig.suptitle(f"Test scenario {r} (median-quality AON-GNN scenario): {int(data['num_closed'][r])} closed links, demand ×{data['demand_scale'][r]:.2f}")
fig.savefig(FIG / "results_map_example.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## 6. Time vs accuracy (one CPU core)

# %%
torch.set_num_threads(1)
sample = test[:100]


def time_aon():
    t0 = time.perf_counter()
    for r in sample:
        aon_flows(net, data["demand"][r], data["closed"][r])
        aon_flows(net, data["demand"][r], net.default_closed)
    return (time.perf_counter() - t0) / len(sample)


def load_model(run):
    import importlib

    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model = importlib.import_module(f"models.{cfg['model']}").Model(num_zones=net.num_zones, **cfg.get("model_args", {}))
    model.fit_normalizer(ds, ds.rows("train", holdout=False))
    model.load_state_dict(torch.load(run / "best.pt"))
    return model.eval()


def time_model(model, batch_size):
    loader = DataLoader(ds.subset(sample), batch_size=batch_size)
    batches = list(loader)
    t0 = time.perf_counter()
    for b in batches:
        model.predict_flows(b)
    return (time.perf_counter() - t0) / len(sample)


aon_one = time_aon() / 2
timing = {"all-or-nothing": aon_one}
gnn, ref = load_model(gnn_run), load_model(ref_run)
timing["AON-GNN (one at a time)"] = aon_one + time_model(gnn, 1)
timing["AON-GNN (batches of 64)"] = aon_one + time_model(gnn, 64)
timing["HGNN (one at a time)"] = time_model(ref, 1)
if trees is not None:
    x1 = feats.matrix(data, cache, sample)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        trees.predict(x1[i * len(real):(i + 1) * len(real)])
    timing["boosted trees"] = 2 * aon_one + (time.perf_counter() - t0) / len(sample)
else:
    base = pd.read_csv(ROOT / "runs" / "baselines" / NETWORK / "ml_main.csv")
    timing["boosted trees"] = base[(base.baseline == "boosted trees (transferable)") & base.evaluation.str.startswith("test")].sec_per_scenario.iloc[0]
curve = pd.read_csv(ROOT / "runs" / "baselines" / NETWORK / "solver_curve_main.csv")
full_solve = float(np.median(data["runtime"][test]))

speed = pd.DataFrame([
    {"method": "all-or-nothing", "sec": timing["all-or-nothing"], "r2": table.loc["all-or-nothing", "R² all"]},
    {"method": "boosted trees", "sec": timing["boosted trees"], "r2": table.loc["boosted trees", "R² all"]},
    {"method": "HGNN", "sec": timing["HGNN (one at a time)"], "r2": table.loc["HGNN", "R² all"]},
    {"method": "AON-GNN (one at a time)", "sec": timing["AON-GNN (one at a time)"], "r2": table.loc["AON-GNN", "R² all"]},
    {"method": "AON-GNN (batches of 64)", "sec": timing["AON-GNN (batches of 64)"], "r2": table.loc["AON-GNN", "R² all"]},
    {"method": "full equilibrium solve", "sec": full_solve, "r2": 1.0},
])
print(speed.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
speed.to_csv(OUT / "speed.csv", index=False)
table.to_csv(OUT / "test_metrics.csv")

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
ax.plot(curve.sec_median, curve.r2_all, "o-", color="gray", label="solver stopped after k iterations")
for _, rr in curve.iterrows():
    ax.annotate(f"k={int(rr.iterations)}", (rr.sec_median, rr.r2_all), xytext=(4, -10), textcoords="offset points", fontsize=7, color="gray")
marks = {"all-or-nothing": ("s", COLORS["all-or-nothing"]), "boosted trees": ("D", COLORS["boosted trees"]), "HGNN": ("v", COLORS["HGNN"]),
         "AON-GNN (one at a time)": ("*", COLORS["AON-GNN"]), "AON-GNN (batches of 64)": ("P", COLORS["AON-GNN"]), "full equilibrium solve": ("o", "black")}
for _, rr in speed.iterrows():
    m, c = marks[rr.method]
    ax.scatter(rr.sec, rr.r2, marker=m, s=140 if m == "*" else 70, color=c, zorder=5, label=f"{rr.method}: {rr.sec * 1000:.1f} ms")
ax.set_xscale("log"); ax.set_xlabel("seconds per scenario, one CPU core (log)"); ax.set_ylabel("R² on test (real links)")
ax.set_ylim(0.55, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Accuracy vs time per scenario (Berlin-Friedrichshain, unseen closures)")
fig.tight_layout(); fig.savefig(FIG / "results_speed.png", dpi=120)
print("figures:", sorted(p.name for p in FIG.glob("results_*.png")))
