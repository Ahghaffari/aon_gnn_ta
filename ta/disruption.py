# %% [markdown]
# # Disruption metrics per scenario

# %%
import os

os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ta.features import aon_cache
from ta.network import Network, solve_scenario
from ta.solver import prepare_links, zone_to_zone_costs
from ta.splits import dataset_dir, load_set

METRICS = ["num_closed", "base_flow_cut", "base_vkt_cut", "aon_vkt_cut", "betweenness_cut", "efficiency_drop",
           "unreachable_pairs", "detour_mean", "detour_share_20", "novelty"]

# %% [markdown]
# ## Reference and metrics


# %%
class Disruption:
    def __init__(self, net):
        self.net = net
        self.fft = prepare_links(net.links).free_flow_time.to_numpy()
        self.length = net.links.length.to_numpy()
        self.real = ~net.is_connector & ~net.candidate
        self.a, self.b = net.links.a_node.to_numpy(), net.links.b_node.to_numpy()
        self.n = int(max(self.a.max(), self.b.max())) + 1
        self.nodes = np.unique(np.concatenate([self.a[self.real], self.b[self.real]]))
        base, _ = solve_scenario(net)
        self.base_flow = base.flows
        self.base_vkt = (base.flows * self.length)[self.real].sum()
        self.betweenness = self.edge_betweenness()
        self.base_efficiency, self.base_unreachable = self.efficiency(net.default_closed)
        self.base_costs = self.zone_costs(net.default_closed)

    def edge_betweenness(self):
        g = nx.DiGraph()
        idx = np.flatnonzero(self.real)
        g.add_weighted_edges_from((int(self.a[k]), int(self.b[k]), self.fft[k]) for k in idx)
        bc = nx.edge_betweenness_centrality(g, weight="weight", normalized=True)
        out = np.zeros(len(self.a))
        out[idx] = [bc[(int(self.a[k]), int(self.b[k]))] for k in idx]
        return out

    def efficiency(self, closed):
        m = self.real & ~closed
        g = csr_matrix((self.fft[m], (self.a[m], self.b[m])), shape=(self.n, self.n))
        d = dijkstra(g, indices=self.nodes)[:, self.nodes]
        off = ~np.eye(len(self.nodes), dtype=bool)
        return (1.0 / d[off]).mean(), (~np.isfinite(d[off])).mean()

    def zone_costs(self, closed):
        links = prepare_links(self.net.links[~closed].reset_index(drop=True))
        return zone_to_zone_costs(links, links.free_flow_time.to_numpy(), self.net.num_zones, self.net.block_centroids)

    def closure_metrics(self, closed):
        eff, unreachable = self.efficiency(closed)
        flow_cut = self.base_flow[closed].sum()
        return {"num_closed": int((closed & self.real).sum()), "base_flow_cut": flow_cut,
                "base_vkt_cut": (self.base_flow * self.length)[closed].sum() / self.base_vkt,
                "betweenness_cut": self.betweenness[closed].sum(), "efficiency_drop": 1.0 - eff / self.base_efficiency,
                "unreachable_pairs": unreachable - self.base_unreachable}, self.zone_costs(closed)

    def scenario_metrics(self, demand, costs, aon_open, closed):
        od = np.array(demand, dtype=np.float64)
        np.fill_diagonal(od, 0.0)
        pos = od > 0
        ratio = costs[pos] / self.base_costs[pos] - 1.0
        w = od[pos] / od[pos].sum()
        vkt = aon_open * self.length
        return {"aon_vkt_cut": vkt[closed].sum() / vkt[self.real].sum(), "detour_mean": float((w * ratio).sum()),
                "detour_share_20": float(w[ratio > 0.2].sum())}

# %% [markdown]
# ## Novelty: distance to the nearest training closure set


# %%
def novelty(closed_by_set, train_sets):
    sets = np.array(list(closed_by_set))
    m = np.stack([closed_by_set[c] for c in sets]).astype(np.float64)
    inter = m @ m[np.isin(sets, train_sets)].T
    sizes = m.sum(1)
    union = sizes[:, None] + sizes[np.isin(sets, train_sets)][None, :] - inter
    dist = np.where(union > 0, 1.0 - inter / np.maximum(union, 1), 0.0)
    dist[sets[:, None] == np.asarray(sets[np.isin(sets, train_sets)])[None, :]] = np.inf
    return dict(zip(sets.tolist(), dist.min(1)))

# %% [markdown]
# ## Table for a dataset


# %%
def disruption_table(network, name, dis=None):
    data, _, splits = load_set(network, name)
    dis = dis or Disruption(Network.from_tntp(network))
    aon_open = aon_cache(network, name, data)["aon_open"]
    sets, first = np.unique(data["closure_set"], return_index=True)
    closed_by_set, per_set = {}, {}
    for c, i in zip(sets.tolist(), first):
        closed = data["closed"][i]
        assert (data["closed"][data["closure_set"] == c] == closed).all(), f"closure set {c} varies across variants"
        closed_by_set[c] = closed & dis.real
        per_set[c] = dis.closure_metrics(closed)
    nov = novelty(closed_by_set, splits.closure_set[splits.split == "train"].unique())
    rows = []
    for i, c in enumerate(data["closure_set"].tolist()):
        graph, costs = per_set[c]
        rows.append({"closure_set": c, "variant": int(data["variant"][i]), **graph,
                     **dis.scenario_metrics(data["demand"][i], costs, aon_open[i], data["closed"][i]), "novelty": nov[c]})
    table = pd.DataFrame(rows)[["closure_set", "variant"] + METRICS]
    path = dataset_dir(network) / f"{name}.disruption.csv"
    table.to_csv(path, index=False)
    return path, table


def load_disruption(network, name):
    return pd.read_csv(dataset_dir(network) / f"{name}.disruption.csv")

# %% [markdown]
# ## Build the Berlin tables

# %%
if __name__ == "__main__":
    import time

    network = "Berlin-Friedrichshain"
    t0 = time.perf_counter()
    dis = Disruption(Network.from_tntp(network))
    print(f"reference ready in {time.perf_counter() - t0:.1f} s | base efficiency {dis.base_efficiency:.4f} | "
          f"betweenness max {dis.betweenness.max():.3f}")
    tables = {}
    for name in ["main", "old_like"]:
        t0 = time.perf_counter()
        path, tables[name] = disruption_table(network, name, dis)
        print(f"saved {path.relative_to(dataset_dir(network).parents[1])} in {time.perf_counter() - t0:.0f} s")

# %% [markdown]
# ## Inspect: distributions and how the metrics relate to each other (main)

# %%
if __name__ == "__main__":
    t = tables["main"]
    print(t[METRICS].describe(percentiles=[0.1, 0.5, 0.9]).T.to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nSpearman correlation between metrics (main):")
    print(t[METRICS].corr("spearman").to_string(float_format=lambda v: f"{v:.2f}"))
