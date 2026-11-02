# %% [markdown]
# # Paper figures and table numbers

# %%
import importlib.util
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from ta import ROOT
from ta.features import aon_cache
from ta.metrics import evaluate
from ta.network import Network
from ta.splits import load_set

FIG = ROOT / "runs" / "figures" / "paper"
PAPER = ROOT / "paper" / "figures"
NUM = ROOT / "runs" / "results" / "paper"
for d in (FIG, PAPER, NUM):
    d.mkdir(parents=True, exist_ok=True)

NETS = {"F": "Berlin-Friedrichshain", "P": "Berlin-Prenzlauerberg-Center", "T": "Berlin-Tiergarten",
        "M": "Berlin-Mitte-Center", "MPF": "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center"}
KEYS = {"F": "friedrichshain", "P": "prenzlauerberg", "T": "tiergarten", "M": "mitte"}
LABEL = {"F": "Friedrichshain", "P": "Prenzlauerberg", "T": "Tiergarten", "M": "Mitte", "MPF": "Berlin-MPF"}

# %% [markdown]
# ## Style

# %%
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})
FULL, HALF = 7.0, 3.4


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    shutil.copy(FIG / f"{name}.pdf", PAPER / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {name}.pdf")


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def titled(ax, letter, text, size=7):
    ax.set_title(f"({letter}) {text}", fontsize=size)


def bare(ax):
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


# %% [markdown]
# ## Run folders and metrics


# %%
def run_dirs(name, model="aon_gnn"):
    out = {}
    for p in sorted((ROOT / "runs" / model).iterdir()):
        m = re.fullmatch(rf"\d{{8}}_\d{{6}}_{re.escape(name)}(?:_s(4[34])|_seed(4[34]))?", p.name)
        if m and any(p.glob("test*.csv")):
            out[int(m.group(1) or m.group(2) or 42)] = p
    return out


def metric(run, net=None, value="r2", file="test"):
    t = pd.read_csv(run / (f"{file}_{net}.csv" if net else f"{file}.csv"), index_col=[0, 1])
    return float(t.loc[("all", "real"), value])


def over_seeds(name, net=None, value="r2", file="test"):
    v = np.array([metric(r, net, value, file) for r in run_dirs(name).values()])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# %% [markdown]
# ## Headline numbers (Table: AON-GNN over seeds)

# %%
rows = []
for name, label in [("main", "AON-GNN, full-range set"), ("heavy_closures", "AON-GNN, heavy-closure set"),
                    ("up50k", "AON-GNN + 50,000 extended scenarios"), ("lc005", "AON-GNN, 5 closure sets"),
                    ("lc020", "AON-GNN, 20 closure sets"), ("lc050", "AON-GNN, 50 closure sets")]:
    r = {"run": name, "label": label}
    for value in ["r2", "mae", "wmape", "conservation"]:
        m, s, n = over_seeds(name, value=value)
        r[value], r[f"{value}_sd"], r["seeds"] = m, s, n
    m, s, _ = over_seeds(name, file="holdout")
    r["holdout_r2"], r["holdout_r2_sd"] = m, s
    rows.append(r)
headline = pd.DataFrame(rows)
headline.to_csv(NUM / "headline.csv", index=False)
print(headline.to_string(float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ## Figure: the five Berlin networks

# %%
geo = {}
for key, name in NETS.items():
    net = Network.from_tntp(name)
    xy = net.nodes.set_index("node")[["x", "y"]]
    seg = np.stack([xy.loc[net.links.a_node].to_numpy(), xy.loc[net.links.b_node].to_numpy()], 1)
    geo[key] = (net, xy, seg)
ext = {k: np.ptp(g[2][:, :, 0]) for k, g in geo.items()}
height = max(np.ptp(g[2][:, :, 1]) for g in geo.values())
fig, axes = plt.subplots(1, 5, figsize=(FULL, 2.1), gridspec_kw={"width_ratios": [ext[k] + 0.3 for k in NETS], "wspace": 0.08})
for ax, key, letter in zip(axes, NETS, "abcde"):
    net, xy, seg = geo[key]
    conn = net.is_connector
    ax.add_collection(LineCollection(seg[conn], colors="#c9c8c2", linewidths=0.3))
    ax.add_collection(LineCollection(seg[~conn], colors=SLOT[0], linewidths=0.6))
    z = xy.loc[np.arange(1, net.num_zones + 1)].to_numpy()
    ax.scatter(z[:, 0], z[:, 1], s=3, color=SLOT[1], zorder=3, linewidths=0)
    bare(ax)
    x0, y0 = seg[:, :, 0].min(), seg[:, :, 1].min()
    ax.set_xlim(x0 - 0.15, x0 + ext[key] + 0.15); ax.set_ylim(y0 - 0.1, y0 + height + 0.1)
    ax.set_title(f"({letter}) {LABEL[key]}", fontsize=7)
handles = [plt.Line2D([], [], color=SLOT[0], lw=1, label="street link"), plt.Line2D([], [], color="#c9c8c2", lw=1, label="zone connector"),
           plt.Line2D([], [], color=SLOT[1], marker="o", lw=0, ms=3, label="zone centroid")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
save(fig, "fig_networks")

# %% [markdown]
# ## Figure: the scenarios (closures, demand level, congestion)

# %%
sets = {"full-range closure set": "main", "heavy-closure set": "old_like"}
fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0))
base_total = np.asarray(Network.from_tntp(NETS["F"]).demand).sum()
for i, (label, s) in enumerate(sets.items()):
    d, _, sp = load_set(NETS["F"], s)
    first = d["variant"] == 0
    style = {"histtype": "step", "color": SLOT[i], "linewidth": 1.2, "label": label}
    axes[0].hist(d["num_closed"][first], bins=np.arange(0, 51, 2), **style)
    axes[1].hist(d["demand"].sum((1, 2)) / base_total, bins=np.linspace(0.4, 1.7, 27), **style)
    axes[2].hist(np.log10(sp.att_ratio), bins=np.linspace(-0.4, 2.2, 27), **style)
axes[0].set_xlabel("closed links per closure set"); axes[0].set_ylabel("closure sets")
axes[1].set_xlabel("total demand / base demand"); axes[1].set_ylabel("scenarios")
axes[2].axvline(np.log10(5), color=INK2, lw=0.8, ls="--")
axes[2].text(np.log10(5) + 0.04, axes[2].get_ylim()[1] * 0.9, "near-gridlock\nthreshold (5×)", fontsize=6.5, color=INK2, va="top")
axes[2].set_xlabel("average trip time / training median (log$_{10}$)"); axes[2].set_ylabel("scenarios")
axes[2].set_yscale("log")
for ax, letter in zip(axes, "abc"):
    panel(ax, letter)
axes[0].legend(loc="upper left", handles=[plt.Line2D([], [], color=SLOT[i], lw=1.2, label=l) for i, l in enumerate(sets)])
fig.tight_layout()
save(fig, "fig_data")

# %% [markdown]
# ## Test predictions of the four main methods (full-range set, seed 42)

# %%
data, links, splits = load_set(NETS["F"], "main")
test = np.flatnonzero((splits.split == "test").to_numpy())
real = ~links.is_connector.to_numpy()
conn = links.is_connector.to_numpy()
cache = aon_cache(NETS["F"], "main", data)


def predictions(run):
    with np.load(run / "predictions.npz") as f:
        assert np.array_equal(f["test_rows"], test)
        return f["test"]


GNN_RUN = run_dirs("main")[42]
preds = {"AON-GNN": predictions(GNN_RUN),
         "boosted trees": np.load(ROOT / "runs" / "results" / NETS["F"] / "trees_test_predictions.npy"),
         "HetGatedGCN": predictions(ROOT / "runs" / "ref_gatedgcn" / "20260929_193510"),
         "all-or-nothing": np.where(data["closed"][test], 0.0, cache["aon"][test])}
COLOR = {"AON-GNN": SLOT[0], "all-or-nothing": SLOT[1], "boosted trees": SLOT[2], "HetGatedGCN": SLOT[3], "GNN without AON input": SLOT[4]}
MARKER = {"AON-GNN": "o", "all-or-nothing": "s", "boosted trees": "^", "HetGatedGCN": "D", "GNN without AON input": "v"}
scores = {n: evaluate(p, data, links, conn, test).loc[("all", "real")] for n, p in preds.items()}

open_real = ~data["closed"][test] & real[None, :]
y = data["flows"][test][open_real]
lim = float(np.percentile(y, 99.9))
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.35), sharey=True, layout="constrained")
for ax, name, letter in zip(axes, ["all-or-nothing", "HetGatedGCN", "boosted trees", "AON-GNN"], "abcd"):
    hb = ax.hexbin(y, preds[name][open_real], gridsize=55, extent=(0, lim, 0, lim), cmap=BLUES, norm=LogNorm(), mincnt=1,
                   linewidths=0.1, edgecolors="face")
    ax.plot([0, lim], [0, lim], color=INK, lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(False)
    titled(ax, letter, f"{name}\n$R^2$ {scores[name].r2:.3f}, MAE {scores[name].mae:.0f} veh/h", size=6.8)
axes[0].set_ylabel("predicted flow (veh/h)")
fig.supxlabel("equilibrium flow (veh/h)", fontsize=8)
cb = fig.colorbar(hb, ax=axes, shrink=0.85, pad=0.01)
cb.set_label("link-scenario pairs per cell", fontsize=7); cb.outline.set_visible(False)
save(fig, "fig_scatter")

# %% [markdown]
# ## Figure: accuracy vs time per scenario

# %%
timing = pd.read_csv(ROOT / "runs" / "results" / NETS["F"] / "timing_single_core.csv").set_index("method")
node = pd.read_csv(ROOT / "runs" / "results" / NETS["F"] / "throughput_node.csv").set_index("method")
curve = pd.read_csv(ROOT / "runs" / "baselines" / NETS["F"] / "solver_curve_main.csv").set_index("iterations")
up50 = headline.set_index("run").loc["up50k", "r2"]
its = [1, 2, 3, 5, 10, 20]
fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
x = [timing.loc[f"solver, {n} iterations", "sec_median"] for n in its] + [timing.loc["solver, converged", "sec_median"]]
y = [curve.loc[n, "r2_all"] for n in its] + [1.0]
ax.plot(x, y, "-", color=INK2, lw=1.0, marker="o", ms=2.5, label="solver stopped after $n$ iterations")
for n, off in [(1, (5, -2)), (3, (5, -6)), (5, (5, -6)), (10, (4, -9))]:
    ax.annotate(f"$n$={n}", (timing.loc[f"solver, {n} iterations", "sec_median"], curve.loc[n, "r2_all"]), xytext=off,
                textcoords="offset points", fontsize=6, color=INK2)
points = [("all-or-nothing", "all-or-nothing", scores["all-or-nothing"].r2), ("boosted trees", "boosted trees", scores["boosted trees"].r2),
          ("HetGatedGCN", "HetGatedGCN", scores["HetGatedGCN"].r2), ("AON-GNN", "AON-GNN", scores["AON-GNN"].r2)]
for name, key, r in points:
    ax.scatter(timing.loc[key, "sec_median"], r, color=COLOR[name], marker=MARKER[name], s=22, zorder=4, edgecolors="white", linewidths=0.6, label=name)
ax.scatter(timing.loc["AON-GNN", "sec_median"], up50, color=COLOR["AON-GNN"], marker="*", s=55, zorder=4, edgecolors="white",
           linewidths=0.6, label="AON-GNN + extended data")
ax.scatter(timing.loc["solver, converged", "sec_median"], 1.0, color=INK, marker="o", s=18, zorder=4, label="converged solve (gap $10^{-5}$)")
ax.set_xscale("log"); ax.set_ylim(0.58, 1.01)
ax.set_xlabel("seconds per scenario, one CPU core"); ax.set_ylabel("test $R^2$ (street links)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.8)
panel(ax, "a")
ax = axes[1]
bars = [("AON-GNN, GPU, batch 512", "AON-GNN pipeline (AON + GPU forward), batch 512", SLOT[0]),
        ("AON-GNN, GPU, batch 64", "AON-GNN pipeline (AON + GPU forward), batch 64", SLOT[0]),
        ("AON-GNN, GPU, batch 1", "AON-GNN pipeline (AON + GPU forward), batch 1", SLOT[0]),
        ("solver, 5 iterations, 72 cores", "solver, 5 iterations, 72 CPU cores", INK2),
        ("solver, converged, 72 cores", "solver, converged, 72 CPU cores", INK2)]
vals = [node.loc[k, "scenarios_per_s"] for _, k, _ in bars]
ax.barh(range(len(bars)), vals, color=[c for _, _, c in bars], height=0.6)
for i, v in enumerate(vals):
    ax.text(v * 1.1, i, f"{v:,.0f}", va="center", fontsize=6.5, color=INK)
ax.set_yticks(range(len(bars)), [b for b, _, _ in bars], fontsize=6.5)
ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(10, 6e4); ax.grid(axis="y", visible=False)
ax.set_xlabel("scenarios per second, one GH200 node")
panel(ax, "b")
fig.tight_layout()
save(fig, "fig_speed")

# %% [markdown]
# ## Figure: data-size curve (fewer closure sets, and the extended training set)

# %%
points = [("lc005", 50), ("lc010", 100), ("lc020", 200), ("lc050", 500), ("lc100", 1000), ("main", 2050),
          ("up01k", 3050), ("up02k", 4050), ("up05k", 7050), ("up10k", 12050), ("up20k", 22050), ("up50k", 52050)]
curve_rows = []
for name, n in points:
    for value in ["r2", "mae"]:
        m, s, k = over_seeds(name, value=value)
        curve_rows.append({"run": name, "train_scenarios": n, "value": value, "mean": m, "sd": s, "seeds": k})
lc = pd.DataFrame(curve_rows)
lc.to_csv(NUM / "data_size_curve.csv", index=False)
aon_r2, trees_r2 = scores["all-or-nothing"].r2, scores["boosted trees"].r2
fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.72, 2.2))
for ax, value, ylabel in [(axes[0], "r2", "test $R^2$ (street links)"), (axes[1], "mae", "test MAE (veh/h)")]:
    d = lc[lc.value == value]
    ax.errorbar(d.train_scenarios, d["mean"], yerr=d["sd"].fillna(0), color=SLOT[0], marker="o", ms=3, capsize=2, lw=1.2)
    ax.axvline(2050, color=INK2, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("training scenarios"); ax.set_ylabel(ylabel)
ref = {"r2": [(aon_r2, "all-or-nothing"), (trees_r2, "boosted trees (2,050 scenarios)")],
       "mae": [(scores["all-or-nothing"].mae, "all-or-nothing"), (scores["boosted trees"].mae, "boosted trees (2,050 scenarios)")]}
for ax, value in [(axes[0], "r2"), (axes[1], "mae")]:
    for v, label in ref[value]:
        ax.axhline(v, color=INK2, lw=0.6)
        ax.text(5.5e4, v, label, fontsize=6, color=INK2, va="top" if value == "r2" and "trees" in label else "bottom", ha="right")
axes[0].text(2150, 0.77, "205 closure sets\n(full training split)", fontsize=6, color=INK2)
for ax, letter in zip(axes, "ab"):
    panel(ax, letter)
fig.tight_layout()
save(fig, "fig_learning")

# %% [markdown]
# ## Figure: error vs disruption strength

# %%
dis = pd.read_csv(ROOT / "runs" / "results" / NETS["F"] / "disruption_errors.csv")
methods = {"all-or-nothing": "mae: all-or-nothing", "HetGatedGCN": "mae: HGNN", "boosted trees": "mae: boosted trees",
           "GNN without AON input": "mae: ablation: no_aon", "AON-GNN": "mae: AON-GNN"}
fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.8, 2.4), sharey=True)
for ax, col, xlabel in [(axes[0], "base_flow_cut", "flow on the closed links, undisturbed (veh/h)"),
                        (axes[1], "detour_mean", "demand-weighted free-flow detour")]:
    q = pd.qcut(dis[col], 5, labels=False)
    x = dis.groupby(q)[col].median()
    for name, mcol in methods.items():
        ax.plot(x, dis.groupby(q)[mcol].mean(), color=COLOR[name], marker=MARKER[name], ms=3.5, label=name)
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("MAE per scenario (veh/h)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=6.2)
for ax, letter in zip(axes, "ab"):
    panel(ax, letter)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "fig_disruption")

# %% [markdown]
# ## Figure: where the errors are (map of mean absolute error per link)

# %%
net = Network.from_tntp(NETS["F"])
xy = net.nodes.set_index("node")[["x", "y"]]
seg = np.stack([xy.loc[links.a_node].to_numpy(), xy.loc[links.b_node].to_numpy()], 1)
d = seg[:, 1] - seg[:, 0]
normal = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
seg_off = seg + 0.012 * normal[:, None, :]
link_mae = {n: np.nanmean(np.where(open_real, np.abs(preds[n] - data["flows"][test]), np.nan), axis=0)
            for n in ["all-or-nothing", "AON-GNN"]}
vmax = float(np.nanpercentile(link_mae["all-or-nothing"][real], 98))
fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.8, 3.0), layout="constrained")
for ax, name, letter in zip(axes, ["all-or-nothing", "AON-GNN"], "ab"):
    lc_ = LineCollection(seg_off[real], array=np.nan_to_num(link_mae[name][real]), cmap=BLUES, norm=plt.Normalize(0, vmax),
                         linewidths=1.3)
    ax.add_collection(lc_)
    bare(ax)
    titled(ax, letter, f"{name}\nmean over street links {np.nanmean(link_mae[name][real]):.0f} veh/h")
cb = fig.colorbar(lc_, ax=axes, shrink=0.75, pad=0.01)
cb.set_label("MAE over test scenarios (veh/h)", fontsize=7); cb.outline.set_visible(False)
save(fig, "fig_error_map")

# %% [markdown]
# ## Experiment 1: adaptation to new closure sets

# %%
sm = load_script("stage_m_tables")
m1 = sm.m1_rows()
agg = m1.groupby(["size", "arm", "k"]).r2.agg(["mean", "std", "count"]).reset_index()
agg.to_csv(NUM / "fewshot_closures.csv", index=False)
ARMS1 = [("fine-tune all", "zero-shot", "fine-tuning (5 SGD steps)"), ("shift", "zero-shot", "residual shift (no training)"),
         ("MAML 2nd order", "MAML 2nd order (start)", "MAML"), ("MAML 1st order", "MAML 1st order (start)", "first-order MAML"),
         ("MAML head only", "MAML head only (start)", "ANIL (head-only MAML)")]
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.45), sharey=False)
for ax, size, letter in zip(axes, [5, 20, 50, 205], "abcd"):
    a = agg[agg["size"] == size]
    for i, (arm, start, label) in enumerate(ARMS1):
        s = a[a.arm == start].sort_values("k")
        p = a[a.arm == arm].sort_values("k")
        k = np.r_[s.k.to_numpy(), p.k.to_numpy()]
        m = np.r_[s["mean"].to_numpy(), p["mean"].to_numpy()]
        sd = np.r_[s["std"].to_numpy(), p["std"].to_numpy()]
        ax.plot(k, m, color=SLOT[i], marker=MARK[i], ms=3.2, label=label)
        ax.fill_between(k, m - sd, m + sd, color=SLOT[i], alpha=0.12, linewidth=0)
    ax.set_xticks([0, 1, 2, 4])
    titled(ax, letter, f"trained on {size} closure sets")
axes[0].set_ylabel("query $R^2$ (street links)")
handles, labels = axes[0].get_legend_handles_labels()
fig.supxlabel("solved scenarios of the new closure set ($k$)", fontsize=8, y=0.1)
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))
save(fig, "fig_fewshot_closures")

# %% [markdown]
# ## Zero-shot transfer between networks

# %%
transfer = []
aon_row = {}
for key, name in NETS.items():
    d_, l_, s_ = load_set(name, "main")
    t_ = np.flatnonzero((s_.split == "test").to_numpy())
    a_ = aon_cache(name, "main", d_)["aon"][t_]
    a_[d_["closed"][t_]] = 0.0
    aon_row[key] = evaluate(a_, d_, l_, l_.is_connector.to_numpy(), t_).loc[("all", "real")].r2
    transfer.append({"train": "none (all-or-nothing)", "test": key, "mean": aon_row[key], "sd": np.nan, "seeds": 1})
for tr in KEYS:
    for te in NETS:
        m, s, n = over_seeds(f"m2_own_{KEYS[tr]}", NETS[te])
        transfer.append({"train": f"{LABEL[tr]} only", "test": te, "mean": m, "sd": s, "seeds": n})
for te in KEYS:
    for prefix, label in [("m2_loo", "the other three"), ("m23_erm", "the other three + 150 synthetic")]:
        m, s, n = over_seeds(f"{prefix}_{KEYS[te]}", NETS[te])
        transfer.append({"train": label, "test": te, "mean": m, "sd": s, "seeds": n})
transfer = pd.DataFrame(transfer)
transfer.to_csv(NUM / "transfer.csv", index=False)
mpf = []
for prefix, label in [("m2_loo", "the other three"), ("m23_erm", "the other three + 150 synthetic")]:
    for te in KEYS:
        m, s, n = over_seeds(f"{prefix}_{KEYS[te]}", NETS["MPF"])
        mpf.append({"train": label, "held_out": te, "mean": m, "sd": s, "seeds": n})
m, s, n = over_seeds("m2_all4", NETS["MPF"])
mpf.append({"train": "all four", "held_out": "-", "mean": m, "sd": s, "seeds": n})
pd.DataFrame(mpf).to_csv(NUM / "mpf.csv", index=False)

rows_ = ["none (all-or-nothing)"] + [f"{LABEL[k]} only" for k in KEYS] + ["the other three", "the other three + 150 synthetic"]
cols_ = ["F", "P", "T", "M"]
mat = transfer[transfer.test.isin(cols_)].pivot(index="train", columns="test", values="mean").loc[rows_, cols_]
fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.7))
im = ax.imshow(mat.to_numpy(), cmap=BLUES, vmin=0.68, vmax=0.94, aspect="auto")
for i in range(len(rows_)):
    for j in range(len(cols_)):
        v = mat.iat[i, j]
        if np.isnan(v):
            continue
        diag = rows_[i] == f"{LABEL[cols_[j]]} only"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.86 else INK,
                fontweight="bold" if diag else "normal")
ax.set_xticks(range(len(cols_)), [LABEL[c] for c in cols_], rotation=20)
ax.set_yticks(range(len(rows_)), ["all-or-nothing"] +[f"trained on {LABEL[k]}" for k in KEYS]
              + ["trained on the other three", "+ 150 synthetic variants"])
ax.set_xlabel("test network"); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(0.5, color="white", lw=1.5); ax.axhline(4.5, color="white", lw=1.5)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("test $R^2$ (street links)", fontsize=7); cb.outline.set_visible(False)
save(fig, "fig_transfer")

# %% [markdown]
# ## Experiment 2: adaptation to a new network

# %%
erm = sm.network_rows("m23_erm")
maml = sm.maml_network_rows()
net_rows = pd.concat([erm, maml], ignore_index=True)
net_rows = net_rows[net_rows.kind == "held-out network"]
nagg = net_rows.groupby(["held_out", "method", "k"]).r2.agg(["mean", "std", "count"]).reset_index()
nagg.to_csv(NUM / "fewshot_networks.csv", index=False)
KS = [0, 1, 2, 5, 10, 20, 50]
POS = {k: i for i, k in enumerate(KS)}
ARMS2 = [("adapt_all", "fine-tuning (short, SGD)"), ("shift", "residual shift"), ("MAML 2nd order", "MAML"),
         ("MAML 1st order", "first-order MAML"), ("MAML head only", "ANIL (head-only MAML)"),
         ("train_on_k", "trained on the $k$ solves only"), ("finetune_long", "fine-tuning (long, Adam)")]
SLOT2 = {"adapt_all": 0, "shift": 1, "MAML 2nd order": 2, "MAML 1st order": 3, "MAML head only": 4, "train_on_k": 7, "finetune_long": 6}
ZOOM = ["adapt_all", "MAML 2nd order", "MAML 1st order", "MAML head only"]


def network_arm(ax, a, i, method, label):
    p = a[(a.method == method) & (a.k > 0)].sort_values("k")
    if method.startswith("MAML"):
        start = a[(a.method == method) & (a.k == 0)]
    elif method == "train_on_k":
        start = None
    else:
        start = a[(a.method == "zero_shot") & (a.k == 0)]
    if start is not None:
        p = pd.concat([start, p])
    x = [POS[k] for k in p.k]
    c = SLOT[SLOT2[method]]
    ax.plot(x, p["mean"], color=c, marker=MARK[i], ms=3.2, label=label)
    ax.fill_between(x, p["mean"] - p["std"], p["mean"] + p["std"], color=c, alpha=0.12, linewidth=0)
    return p


fig, axes = plt.subplots(2, 4, figsize=(FULL, 4.5), height_ratios=[1.2, 1])
for j, (key, short) in enumerate(KEYS.items()):
    ax, zoom = axes[0, j], axes[1, j]
    a = nagg[nagg.held_out == short]
    near = []
    for i, (method, label) in enumerate(ARMS2):
        network_arm(ax, a, i, method, label)
        if method in ZOOM:
            near.append(network_arm(zoom, a, i, method, label))
    own = over_seeds(f"m2_own_{short}", NETS[key])[0]
    ax.axhline(own, color=INK2, lw=0.8, ls="--")
    ax.text(len(KS) - 1, own - 0.004, "trained on this network", fontsize=6, color=INK2, va="top", ha="right")
    titled(ax, "abcd"[j], f"held out: {LABEL[key]}")
    titled(zoom, "efgh"[j], f"zoom: {LABEL[key]}")
    shown = a[a.method.isin([m for m, _ in ARMS2] + ["zero_shot"])]
    ax.set_ylim(shown["mean"].min() - 0.03, own + 0.015)
    near = pd.concat(near)
    zoom.set_ylim((near["mean"] - near["std"]).min() - 0.002, (near["mean"] + near["std"]).max() + 0.002)
    for one in (ax, zoom):
        one.set_xticks(range(len(KS)), KS)
axes[0, 0].set_ylabel("test $R^2$ (street links)")
axes[1, 0].set_ylabel("test $R^2$ (street links)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.supxlabel("solved scenarios of the new network ($k$)", fontsize=8, y=0.085)
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.085, 1, 1))
save(fig, "fig_fewshot_networks")

