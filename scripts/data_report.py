# %% [markdown]
# # Dataset report

# %%
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ta import ROOT
from ta.checks import check_flows
from ta.network import Network

DATA = ROOT / "data" / "Berlin-Friedrichshain"
OLD = ROOT / "old" / "datageneration" / "Berlin-Friedrichshain"
FIG = ROOT / "runs" / "figures"
SETS = ["main", "old_like"]

# %% [markdown]
# ## Load

# %%

net = Network.from_tntp("Berlin-Friedrichshain")
real = ~net.is_connector
cap = net.links.capacity.to_numpy()
data = {}
for s in SETS:
    with np.load(DATA / f"{s}.npz") as f:
        data[s] = {k: f[k] for k in f.files}
    links = pd.read_csv(DATA / f"{s}.links.csv")
    assert (links.is_connector.to_numpy() == net.is_connector).all()


def load_old():
    tasks = pd.read_csv(OLD / "tasks.csv")
    flows, closed, demand = [], [], []
    for tid, cut in zip(tasks.task_id, tasks.cut_edge_ids):
        f = pd.read_csv(OLD / "tasks" / tid / "flows.csv").sort_values("edge_id")
        assert (f.a_node.to_numpy() == net.links.a_node.to_numpy()).all() and (f.b_node.to_numpy() == net.links.b_node.to_numpy()).all()
        flows.append(f.flow.to_numpy())
        closed.append(f.edge_id.isin(json.loads(cut)).to_numpy())
        demand.append(pd.read_csv(OLD / "tasks" / tid / "perturbed_od_matrix.csv", index_col=0).to_numpy())
    return {"flows": np.array(flows), "closed": np.array(closed), "demand": np.array(demand)}


old = load_old() if OLD.exists() else None
print({s: len(d["flows"]) for s, d in data.items()}, "| old data:", None if old is None else len(old["flows"]))

# %% [markdown]
# ## 1. Overview

# %%
base_total = net.demand.sum() - np.trace(net.demand)
rows = {}
for s, d in data.items():
    tot = d["demand"].sum(axis=(1, 2)) / base_total
    rows[s] = {
        "scenarios": len(d["flows"]), "closure sets": len(np.unique(d["closure_set"])), "variants": len(np.unique(d["variant"])),
        "closed links (min/med/max)": f"{d['num_closed'].min()}/{np.median(d['num_closed']):.0f}/{d['num_closed'].max()}",
        "demand vs base (min/med/max)": f"{tot.min():.2f}/{np.median(tot):.2f}/{tot.max():.2f}",
        "all checks ok": f"{d['ok'].mean():.0%}", "converged": f"{d['converged'].mean():.0%}",
        "gap (median/max)": f"{np.median(d['rgap']):.1e}/{d['rgap'].max():.1e}",
        "solve s (median/max)": f"{np.median(d['runtime']):.1f}/{d['runtime'].max():.1f}",
        "iterations (median/max)": f"{np.median(d['iterations']):.0f}/{d['iterations'].max()}",
    }
if old is not None:
    tot = old["demand"].sum(axis=(1, 2)) / base_total
    n_closed = old["closed"].sum(1)
    rows["old data (GNN set)"] = {"scenarios": len(old["flows"]), "closure sets": len(old["flows"]), "variants": 1,
                               "closed links (min/med/max)": f"{n_closed.min()}/{np.median(n_closed):.0f}/{n_closed.max()}",
                               "demand vs base (min/med/max)": f"{tot.min():.2f}/{np.median(tot):.2f}/{tot.max():.2f}"}
print(pd.DataFrame(rows).fillna("-").to_string())

# %% [markdown]
# ## 2. Flows on real links


# %%
def open_real(d):
    return (~d["closed"]) & real[None, :]


def flow_summary(d):
    m = open_real(d)
    f = d["flows"][m]
    vc = (d["flows"] / cap[None, :])[m]
    ever_open = m.any(0) & real
    never_used = ever_open & ~((d["flows"] > 1e-6) & m).any(0)
    return {
        "mean flow": f.mean(), "median flow": np.median(f), "p90 flow": np.percentile(f, 90), "max flow": f.max(),
        "zero-flow share": (f < 1e-6).mean(), "v/c > 1 share": (vc > 1).mean(), "v/c p99": np.percentile(vc, 99),
        "real links never used": int(never_used.sum()),
    }


table = {s: flow_summary(d) for s, d in data.items()}
if old is not None:
    table["old data (GNN set)"] = flow_summary(old)
print(pd.DataFrame(table).to_string(float_format=lambda v: f"{v:.3g}"))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")
bins = np.linspace(0, 3200, 65)
for s, d in {**data, **({"old data (GNN set)": old} if old is not None else {})}.items():
    f = d["flows"][open_real(d)]
    axes[0].hist(f, bins=bins, histtype="step", density=True, lw=1.5, label=s)
    vc = (d["flows"] / cap[None, :])[open_real(d)]
    axes[1].hist(vc, bins=np.linspace(0, 3, 61), histtype="step", density=True, lw=1.5, label=s)
axes[0].set_yscale("log"); axes[0].set_xlabel("flow on open real links (veh/h)"); axes[0].set_ylabel("density (log)")
axes[1].set_yscale("log"); axes[1].set_xlabel("volume / capacity"); axes[1].axvline(1, color="k", lw=0.8, ls=":")
axes[0].legend(); axes[0].set_title("Flow distribution"); axes[1].set_title("Congestion (v/c)")
fig.tight_layout(); fig.savefig(FIG / "data_report_flows.png", dpi=130)

# %% [markdown]
# ## 3. Where the variation comes from


# %%
def variance_split(d):
    f = d["flows"][:, real]
    groups = [np.flatnonzero(d["closure_set"] == c) for c in np.unique(d["closure_set"])]
    total = f.var(0).sum()
    within = np.mean([f[g].var(0).sum() for g in groups])
    return {"between closure sets": 1 - within / total, "within (demand variants)": within / total}


print(pd.DataFrame({s: variance_split(d) for s, d in data.items()}).to_string(float_format=lambda v: f"{v:.1%}"))

# %% [markdown]
# ## 4. Difficulty preview: no-learning baselines on held-out closure sets


# %%
def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def baselines(d, holdout_share=0.15):
    sets = np.unique(d["closure_set"])
    test = np.isin(d["closure_set"], sets[int(len(sets) * (1 - holdout_share)):])
    m_open = open_real(d)
    f, tot = d["flows"], d["demand"].sum(axis=(1, 2))
    train_open = m_open[~test]
    mean_open = np.where(train_open.any(0), (f[~test] * train_open).sum(0) / np.maximum(train_open.sum(0), 1), 0.0)
    y = f[test][m_open[test]]
    p_mean = np.broadcast_to(mean_open, f[test].shape)[m_open[test]]
    ratio = (tot[test] / tot[~test].mean())[:, None]
    p_scaled = (mean_open[None, :] * ratio)[m_open[test]]
    return {"test scenarios": int(test.sum()), "R² per-link mean": r2(y, p_mean), "R² per-link mean × demand ratio": r2(y, p_scaled),
            "MAE per-link mean": np.abs(y - p_mean).mean(), "mean flow": y.mean()}


print(pd.DataFrame({s: baselines(d) for s, d in data.items()}).to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## 5. The old data under our physics checks (summary)

# %%
if old is not None:
    fails = pd.Series([f for i in range(len(old["flows"]))
                       for f in check_flows(net.links, old["demand"][i], old["flows"][i], True, closed=old["closed"][i])["failed"]])
    ours = {s: int((~d["ok"]).sum()) for s, d in data.items()}
    print(f"old data: {len(old['flows'])} scenarios, failures per check:\n{fails.value_counts().to_string()}")
    print(f"ours: scenarios failing any check: {ours}")

# %% [markdown]
# ## 6. Closures and congestion


# %%
def avg_trip_time(d):
    return np.nansum(d["flows"] * d["times"], axis=1) / d["demand"].sum(axis=(1, 2))


rows = {}
for s, d in data.items():
    att = avg_trip_time(d)
    med = np.median(att)
    rows[s] = {"avg trip time median": med, "p99": np.percentile(att, 99), "max": att.max(),
               "> 2x median": (att > 2 * med).mean(), "> 5x median": (att > 5 * med).mean(), "> 20x median": (att > 20 * med).mean(),
               "closure sets with a >5x variant": len(np.unique(d["closure_set"][att > 5 * med]))}
print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3g}"))

# %%
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
for s, d in data.items():
    ax.scatter(d["num_closed"], avg_trip_time(d), s=4, alpha=0.3, label=s)
ax.set_yscale("log")
ax.set_xlabel("closed links"); ax.set_ylabel("average trip time (log)"); ax.legend(markerscale=4)
ax.set_title("More closures → longer trips, and a tail of near-gridlock scenarios")
fig.tight_layout(); fig.savefig(FIG / "data_report_closures.png", dpi=130)
print("figures:", sorted(p.name for p in FIG.glob("data_report_*.png")))
