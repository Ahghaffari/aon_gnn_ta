# %% [markdown]
# # Metrics

# %%
import numpy as np
import pandas as pd

MAPE_MIN_FLOW = 10.0

# %% [markdown]
# ## Flow errors


# %%
def flow_metrics(y, p, capacity=None):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    big = y >= MAPE_MIN_FLOW
    out = {
        "n": len(y),
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err**2).mean()),
        "r2": 1.0 - (err**2).sum() / sst if sst > 0 else np.nan,
        "wmape": np.abs(err).sum() / y.sum() if y.sum() > 0 else np.nan,
        "bias": err.mean(),
        "mape": (np.abs(err[big]) / y[big]).mean() if big.any() else np.nan,
    }
    if capacity is not None:
        vc = err / np.asarray(capacity, dtype=np.float64)
        out |= {"vc_mae": np.abs(vc).mean(), "vc_rmse": np.sqrt((vc**2).mean())}
    return out


# %% [markdown]
# ## Conservation of predicted flows


# %%
def zone_balance_violation(links, flows, zone_prod, zone_attr, block_centroids=True):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    out = np.empty(len(flows))
    for i, (f, zp, za) in enumerate(zip(flows, zone_prod, zone_attr)):
        z = len(zp)
        inflow, outflow = np.bincount(b, f, n), np.bincount(a, f, n)
        resid = np.abs(inflow - outflow)
        attr, prod = np.zeros(n), np.zeros(n)
        attr[1 : z + 1], prod[1 : z + 1] = za, zp
        if block_centroids:
            resid[1 : z + 1] = np.abs(inflow - attr)[1 : z + 1] + np.abs(outflow - prod)[1 : z + 1]
        else:
            resid = np.abs(inflow - outflow - (attr - prod))
        out[i] = resid.sum() / (2.0 * np.sum(zp))
    return out


def conservation_violation(links, flows, demand, block_centroids=True):
    od = np.array(demand, dtype=np.float64)
    idx = np.arange(od.shape[1])
    od[:, idx, idx] = 0.0
    return zone_balance_violation(links, flows, od.sum(2), od.sum(1), block_centroids)


# %% [markdown]
# ## Relative gap of predicted flows


# %%
def predicted_gap(links, flows, demand, closed, block_centroids=True):
    from ta.solver import link_times, prepare_links, zone_to_zone_costs

    net = prepare_links(links)
    out = np.empty(len(flows))
    for i, (f, od, c) in enumerate(zip(flows, demand, closed)):
        od = od.copy()
        np.fill_diagonal(od, 0.0)
        sub, x = net[~c], np.clip(f[~c], 0.0, None)
        t = link_times(sub, x)
        sp = zone_to_zone_costs(sub, t, od.shape[0], block_centroids)
        m = od > 0
        out[i] = 1.0 - (od[m] * sp[m]).sum() / (x * t).sum()
    return out


# %% [markdown]
# ## Scenario selection


# %%
def select(splits, split="test", holdout=None):
    m = splits.split == split
    if holdout is not None:
        m &= splits.holdout_variant == holdout
    return np.flatnonzero(m.to_numpy())


# %% [markdown]
# ## Evaluate


# %%
def evaluate(pred, data, links, is_connector, rows, gridlock=None, block_centroids=True, gap=False):
    rows = np.asarray(rows)
    pred = np.asarray(pred, dtype=np.float64)
    assert pred.shape == (len(rows), data["flows"].shape[1]), "one prediction row per selected scenario"
    y, closed = data["flows"][rows], data["closed"][rows]
    demand = data["demand"][rows] if "demand" in data else None
    cap = links.capacity.to_numpy()[None, :] * (data["capacity_factor"][rows] if "capacity_factor" in data else 1.0)
    cap = np.broadcast_to(cap, y.shape)
    if demand is not None:
        cons = conservation_violation(links, pred, demand, block_centroids)
    else:
        cons = zone_balance_violation(links, pred, data["demand_prod"][rows], data["demand_attr"][rows], block_centroids)
    gaps = predicted_gap(links, pred, demand, closed, block_centroids) if gap and demand is not None else None
    subsets = {"all": np.ones(len(rows), dtype=bool)}
    if gridlock is not None:
        g = np.asarray(gridlock)[rows]
        subsets |= {"no_gridlock": ~g, "gridlock": g}

    out = []
    for name, s in subsets.items():
        if not s.any():
            continue
        for group, lm in [("real", ~is_connector), ("connector", is_connector)]:
            open_m = ~closed[s] & lm[None, :]
            closed_m = closed[s] & lm[None, :]
            row = {"subset": name, "links": group, "scenarios": int(s.sum()),
                   **flow_metrics(y[s][open_m], pred[s][open_m], cap[s][open_m])}
            row["closed_mean_abs"] = np.abs(pred[s][closed_m]).mean() if closed_m.any() else np.nan
            row["conservation"] = cons[s].mean()
            if gaps is not None:
                row["gap"] = gaps[s].mean()
            out.append(row)
    return pd.DataFrame(out).set_index(["subset", "links"])


def fmt(table):
    return table.to_string(float_format=lambda v: f"{v:.4g}")


# %% [markdown]
# ## Inspect: sanity checks on the Berlin `main` test split

# %%
if __name__ == "__main__":
    from ta.splits import load_set

    data, links, splits = load_set("Berlin-Friedrichshain", "main")
    is_conn = links.is_connector.to_numpy()
    rows = select(splits, "test")
    truth = data["flows"][rows]
    rng = np.random.default_rng(0)
    leak = np.where(data["closed"][rows], data["flows"].mean(0)[None, :], truth)
    candidates = {"truth": truth, "zeros": np.zeros_like(truth),
                  "truth + 10% noise": truth * rng.lognormal(0, 0.1, truth.shape), "truth, flow on closed links": leak}
    for name, pred in candidates.items():
        print(f"== {name}")
        print(fmt(evaluate(pred, data, links, is_conn, rows, splits.gridlock.to_numpy(), gap=True)))
