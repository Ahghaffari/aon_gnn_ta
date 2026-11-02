# %% [markdown]
# # Physics checks

# %%
import numpy as np
import pandas as pd

from ta import FOCUS_NETWORKS
from ta.solver import conservation_error, link_times, prepare_links, prune_dead_ends, zone_to_zone_costs

# %% [markdown]
# ## Connectors


# %%
def zones_are_separate(links, num_zones):
    return not ((links.a_node <= num_zones) & (links.b_node <= num_zones)).any()


def connector_mask(links, num_zones):
    if not zones_are_separate(links, num_zones):
        return np.zeros(len(links), dtype=bool)
    return ((links.a_node <= num_zones) | (links.b_node <= num_zones)).to_numpy()


# %% [markdown]
# ## The report


# %%
def zone_balance(links, flows, num_zones):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    into = np.bincount(b[b <= num_zones] - 1, flows[b <= num_zones], num_zones)
    out = np.bincount(a[a <= num_zones] - 1, flows[a <= num_zones], num_zones)
    return into, out


def dead_end_mask(links, num_zones, closed):
    open_idx = np.flatnonzero(~closed)
    keep_open = prune_dead_ends(links.iloc[open_idx], num_zones)
    dead = np.zeros(len(links), dtype=bool)
    dead[open_idx[~keep_open]] = True
    return dead


def gap_and_reachability(links, od, flows, block_centroids):
    net = prepare_links(links)
    times = link_times(net, flows)
    sp = zone_to_zone_costs(net, times, od.shape[0], block_centroids)
    lost = (od > 0) & ~np.isfinite(sp)
    if lost.any():
        return np.nan, od[lost].sum()
    return 1.0 - (od[od > 0] * sp[od > 0]).sum() / (flows * times).sum(), 0.0


def check_flows(links, demand, flows, block_centroids, closed=None, rgap_target=None, rel_tol=1e-6):
    od = np.array(demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    flows = np.asarray(flows, dtype=np.float64)
    total = od.sum()
    tol = rel_tol * max(1.0, total)
    closed = np.zeros(len(links), dtype=bool) if closed is None else np.asarray(closed, dtype=bool)

    into, out = zone_balance(links, flows, od.shape[0])
    conn = connector_mask(links, od.shape[0])
    real = ~conn & ~closed
    vc = flows[real] / links.capacity.to_numpy()[real]
    r = {
        "min_flow": flows.min(),
        "conservation_error": conservation_error(links, flows, od),
        "zones_in_error": np.abs(into - od.sum(0)).max(),
        "zones_out_error": np.abs(out - od.sum(1)).max(),
        "through_zone_share": into.sum() / total - 1.0,
        "closed_flow": flows[closed].sum(),
        "dead_end_flow": flows[dead_end_mask(links, od.shape[0], closed)].sum(),
        "zero_flow_share": (flows[real] < 1e-6).mean(),
        "overcapacity_share": (vc > 1).mean(),
        "max_vc": vc.max(),
    }
    r["rgap"], r["unreachable_demand"] = gap_and_reachability(links[~closed].reset_index(drop=True), od, flows[~closed], block_centroids)

    passed = {
        "nonnegative": r["min_flow"] >= -tol,
        "conservation": r["conservation_error"] <= tol,
        "closed_links_empty": r["closed_flow"] <= tol,
        "dead_ends_empty": r["dead_end_flow"] <= tol,
        "reachable": r["unreachable_demand"] == 0.0,
    }
    if block_centroids:
        passed["zones_in"] = r["zones_in_error"] <= tol
        passed["zones_out"] = r["zones_out_error"] <= tol
    if rgap_target is not None:
        passed["gap"] = 0.0 <= r["rgap"] <= rgap_target
    r["failed"] = [k for k, ok in passed.items() if not ok]
    r["ok"] = not r["failed"]
    return r


def check_solution(net, sol, rgap_target=None, closed=None):
    return check_flows(net.links, net.demand, sol.flows, net.blocks_centroid_flows, closed, rgap_target)


# %% [markdown]
# ## Inspect: our solutions on the focus networks, and the old data side by side

# %%
if __name__ == "__main__":
    from ta.io_tntp import load_network
    from ta.solver import solve

    rows = []
    for name in FOCUS_NETWORKS:
        net = load_network(name)
        for block in [True, False]:
            sol = solve(net.links, net.demand, block, rgap_target=1e-5)
            rep = check_flows(net.links, net.demand, sol.flows, net.blocks_centroid_flows, rgap_target=1e-5)
            rows.append({"network": name, "run": "blocked" if block else "not blocked", "ok": rep["ok"],
                         "failed": ",".join(rep["failed"]) or "-", "through_zone": rep["through_zone_share"],
                         "zero_flow": rep["zero_flow_share"], "v/c>1": rep["overcapacity_share"],
                         "max_vc": rep["max_vc"], "rgap": rep["rgap"], "cons_err": rep["conservation_error"]})
    print(pd.DataFrame(rows).set_index(["network", "run"]).to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Inspect: the same checks on the old Berlin data

# %%
if __name__ == "__main__":
    import json

    from ta import ROOT

    old = ROOT / "old" / "datageneration" / "Berlin-Friedrichshain"
    if old.exists():
        net = load_network("Berlin-Friedrichshain")
        tasks = pd.read_csv(old / "tasks.csv")
        rows = []
        for tid, cut in zip(tasks.task_id[:20], tasks.cut_edge_ids[:20]):
            f = pd.read_csv(old / "tasks" / tid / "flows.csv").sort_values("edge_id")
            od = pd.read_csv(old / "tasks" / tid / "perturbed_od_matrix.csv", index_col=0).to_numpy()
            closed = f.edge_id.isin(json.loads(cut)).to_numpy()
            rep = check_flows(net.links, od, f.flow.to_numpy(), True, closed=closed)
            rows.append({"task": tid, "ok": rep["ok"], "failed": ",".join(rep["failed"]) or "-",
                         "through_zone": rep["through_zone_share"], "cons_err": rep["conservation_error"],
                         "unreachable": rep["unreachable_demand"], "zero_flow": rep["zero_flow_share"]})
        print(pd.DataFrame(rows).set_index("task").to_string(float_format=lambda v: f"{v:.3g}"))
