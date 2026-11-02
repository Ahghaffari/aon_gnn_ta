# %% [markdown]
# # User-equilibrium solver (AequilibraE wrapper)

# %%
import logging
import os
import time
import warnings
from dataclasses import dataclass

os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import numpy as np
import pandas as pd
from pandas.errors import ChainedAssignmentError
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

# false positive from pandas' refcount check inside AequilibraE's Cython graph builder
warnings.filterwarnings("ignore", category=ChainedAssignmentError)

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass

# AequilibraE logs "Desired RGap ... NOT reached" as an error; we compute and report our own gap instead
logging.getLogger("aequilibrae").setLevel(logging.CRITICAL)

from ta import FOCUS_NETWORKS

MIN_FREE_FLOW_TIME = 1e-6

# %% [markdown]
# ## Link costs


# %%
def bpr_time(flow, fft, capacity, b, power):
    return fft * (1.0 + b * np.power(flow / capacity, power))


def prepare_links(links, min_fft=MIN_FREE_FLOW_TIME):
    assert not ((links.power < 1) & (links.b > 0)).any(), "exponent < 1 on a congestible link"
    net = links[["a_node", "b_node", "free_flow_time", "capacity", "b", "power"]].copy()
    net.loc[net["free_flow_time"] <= 0, "free_flow_time"] = min_fft
    net.loc[net["power"] < 1, "power"] = 1.0
    net.insert(0, "link_id", np.arange(1, len(net) + 1))
    net["direction"] = 1
    return net


def link_times(net, flows):
    return bpr_time(flows, net.free_flow_time.to_numpy(), net.capacity.to_numpy(), net.b.to_numpy(), net.power.to_numpy())


# %% [markdown]
# ## Unusable links (dead ends and traps)


# %%
def reachable(a, b, starts, n):
    s = n
    rows = np.r_[a, np.full(len(starts), s)]
    cols = np.r_[b, starts]
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1))
    return breadth_first_order(graph, s, directed=True, return_predecessors=False)


def prune_dead_ends(links, num_zones):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    zones = np.arange(1, num_zones + 1)
    zones = zones[zones < n]
    forward = np.zeros(n + 1, dtype=bool)
    backward = np.zeros(n + 1, dtype=bool)
    forward[reachable(a, b, zones, n)] = True
    backward[reachable(b, a, zones, n)] = True
    return forward[a] & backward[b]


# %% [markdown]
# ## Flow conservation


# %%
def conservation_error(links, flows, od):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    balance = np.bincount(b, flows, n) - np.bincount(a, flows, n)
    expected = np.zeros(n)
    expected[1 : od.shape[0] + 1] = od.sum(0) - od.sum(1)
    return np.abs(balance - expected).max()


# %% [markdown]
# ## Independent relative gap


# %%
def zone_to_zone_costs(net, times, num_zones, block_centroids):
    a, b = net.a_node.to_numpy(), net.b_node.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    zones = np.arange(1, num_zones + 1)
    if block_centroids:
        b = np.where(b <= num_zones, n + b - 1, b)
        size, targets = n + num_zones, n + zones - 1
    else:
        size, targets = n, zones
    assert len(set(zip(a.tolist(), b.tolist()))) == len(a), "parallel links would be summed by csr_matrix"
    graph = csr_matrix((times, (a, b)), shape=(size, size))
    return dijkstra(graph, indices=zones)[:, targets]


def relative_gap(net, od, flows, block_centroids):
    times = link_times(net, flows)
    sp = zone_to_zone_costs(net, times, od.shape[0], block_centroids)
    assert np.isfinite(sp[od > 0]).all(), "demand between unconnected zones"
    return 1.0 - (od[od > 0] * sp[od > 0]).sum() / (flows * times).sum()


# %% [markdown]
# ## Solve


# %%
@dataclass
class Solution:
    flows: np.ndarray
    times: np.ndarray
    rgap: float
    rgap_reported: float
    iterations: int
    runtime: float
    converged: bool
    conservation_error: float
    pruned_links: int


def run_aequilibrae(net, od, block_centroids, rgap_target, max_iter, algorithm, cores):
    zones = np.arange(1, od.shape[0] + 1)
    g = Graph()
    g.network = net
    g.prepare_graph(zones)
    g.set_graph("free_flow_time")
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(block_centroids)

    mat = AequilibraeMatrix()
    mat.create_empty(zones=len(zones), matrix_names=["demand"], memory_only=True)
    mat.index[:] = zones
    mat.matrices[:, :, 0] = od
    mat.computational_view(["demand"])

    assig = TrafficAssignment()
    assig.set_classes([TrafficClass("car", g, mat)])
    assig.set_cores(cores)
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm(algorithm)
    assig.max_iter = max_iter
    assig.rgap_target = rgap_target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assig.execute()
    flows = assig.results()["demand_tot"].reindex(net["link_id"]).fillna(0.0).to_numpy()
    return flows, assig.report()


def solve(links, demand, block_centroids, rgap_target=1e-5, max_iter=2000, algorithm="bfw", cores=1, retries=3):
    net = prepare_links(links)
    od = np.array(demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    keep = prune_dead_ends(net, od.shape[0])
    active = net[keep].reset_index(drop=True)
    active["link_id"] = np.arange(1, len(active) + 1)

    t0, aeq_target = time.perf_counter(), rgap_target
    for _ in range(retries):
        active_flows, report = run_aequilibrae(active, od, block_centroids, aeq_target, max_iter, algorithm, cores)
        rgap_reported, iterations = float(report["rgap"].iloc[-1]), len(report)
        flows = np.zeros(len(net))
        flows[keep] = active_flows
        rgap = relative_gap(net, od, flows, block_centroids)
        if rgap <= rgap_target or iterations >= max_iter:
            break
        aeq_target /= 10
    runtime = time.perf_counter() - t0

    cons = conservation_error(net, flows, od)
    assert cons <= 1e-6 * max(1.0, od.sum()), f"flow not conserved (max node error {cons:.3g})"
    return Solution(flows, link_times(net, flows), rgap, rgap_reported, iterations, runtime,
                    rgap <= rgap_target, cons, int((~keep).sum()))


def aon_assign(links, demand, block_centroids, cores=1):
    net = prepare_links(links)
    od = np.array(demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    keep = prune_dead_ends(net, od.shape[0])
    active = net[keep].reset_index(drop=True)
    active["link_id"] = np.arange(1, len(active) + 1)
    flows = np.zeros(len(net))
    flows[keep] = run_aequilibrae(active, od, block_centroids, 1e-12, 1, "all-or-nothing", cores)[0]
    return flows


def solve_network(net, **kwargs):
    return solve(net.links, net.demand, net.blocks_centroid_flows, **kwargs)


# %% [markdown]
# ## Compare against a published solution


# %%
def compare_to_published(net, sol):
    pub = net.links[["a_node", "b_node"]].merge(net.flows, on=["a_node", "b_node"], how="left")
    assert pub["volume"].notna().all()
    x_pub, t_pub = pub["volume"].to_numpy(), pub["cost"].to_numpy()
    return {
        "flow_rel_l1": np.abs(sol.flows - x_pub).sum() / x_pub.sum(),
        "flow_max_abs": np.abs(sol.flows - x_pub).max(),
        "flow_corr": np.corrcoef(sol.flows, x_pub)[0, 1],
        "time_rel_l1": np.abs(sol.times - t_pub).sum() / t_pub.sum(),
    }


# %% [markdown]
# ## Inspect: why the gap is computed independently (Braess)

# %%
if __name__ == "__main__":
    from ta.io_tntp import load_network

    braess = load_network("Braess-Example")
    for algorithm in ["bfw", "fw", "msa"]:
        s = solve_network(braess, rgap_target=1e-8, max_iter=2000, algorithm=algorithm)
        print(f"{algorithm:4s} iters {s.iterations:3d} | reported gap {s.rgap_reported:.1e} | "
              f"independent gap {s.rgap:.1e} | flows {np.round(s.flows, 3)}")

# %% [markdown]
# ## Inspect: the focus networks

# %%
if __name__ == "__main__":
    rows, sols = [], {}
    for name in FOCUS_NETWORKS:
        net = load_network(name)
        sol = sols[name] = solve_network(net, rgap_target=1e-6, max_iter=5000)
        row = {"network": name, "links": net.num_links, "pruned": sol.pruned_links, "block": net.blocks_centroid_flows,
               "gap": sol.rgap, "gap_reported": sol.rgap_reported, "iters": sol.iterations, "sec": round(sol.runtime, 1),
               "cons_err": sol.conservation_error}
        if net.flows is not None:
            row.update(compare_to_published(net, sol))
        rows.append(row)
    print(pd.DataFrame(rows).set_index("network").to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Inspect: Berlin-Friedrichshain link by link

# %%
if __name__ == "__main__":
    net, sol = load_network("Berlin-Friedrichshain"), sols["Berlin-Friedrichshain"]
    detail = net.links[["a_node", "b_node", "capacity", "free_flow_time"]].copy()
    detail["flow"], detail["time"] = sol.flows, sol.times
    detail["v/c"] = detail["flow"] / detail["capacity"]
    real = detail["capacity"] < 999999
    print(f"real links: {real.sum()} | carrying zero flow: {(detail.flow[real] < 1e-6).mean():.1%} | "
          f"v/c > 1: {(detail['v/c'][real] > 1).mean():.1%}")
    print(detail[real].sort_values("flow", ascending=False).head(10).to_string(index=False))
