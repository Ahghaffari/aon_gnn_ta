# %% [markdown]
# # No-learning baselines

# %%
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ta.network import solve_scenario
from ta.solver import prepare_links
from ta.splits import demand_totals

# %% [markdown]
# ## Statistical baselines


# %%
class PerLinkMean:
    def __init__(self, scale=False):
        self.scale = scale

    def fit(self, data, rows):
        open_m = ~data["closed"][rows]
        self.mean = (data["flows"][rows] * open_m).sum(0) / np.maximum(open_m.sum(0), 1)
        self.total = demand_totals(data)[rows].mean()
        return self

    def predict(self, data, rows):
        pred = np.broadcast_to(self.mean, (len(rows), len(self.mean))).copy()
        if self.scale:
            pred *= (demand_totals(data)[rows] / self.total)[:, None]
        return np.where(data["closed"][rows], 0.0, pred)


class NearestClosureSets:
    def __init__(self, k=5, real=None):
        self.k, self.real = k, real

    def fit(self, data, rows):
        real = np.ones(data["closed"].shape[1], dtype=bool) if self.real is None else self.real
        sets = data["closure_set"][rows]
        self.ids = np.unique(sets)
        first = np.array([rows[np.flatnonzero(sets == c)[0]] for c in self.ids])
        self.masks = data["closed"][first][:, real]
        tot = demand_totals(data)
        self.mean_flow = np.stack([data["flows"][rows[sets == c]].mean(0) for c in self.ids])
        self.mean_total = np.array([tot[rows[sets == c]].mean() for c in self.ids])
        self.real_mask = real
        return self

    def neighbours(self, closed_real):
        dist = (self.masks != closed_real[None, :]).sum(1)
        return np.argsort(dist, kind="stable")[: self.k]

    def predict(self, data, rows):
        tot = demand_totals(data)
        out = np.empty((len(rows), data["flows"].shape[1]))
        for i, r in enumerate(rows):
            nb = self.neighbours(data["closed"][r][self.real_mask])
            out[i] = self.mean_flow[nb].mean(0) * tot[r] / self.mean_total[nb].mean()
        return np.where(data["closed"][rows], 0.0, out)


class SameClosureSet(NearestClosureSets):
    def __init__(self):
        super().__init__(k=1)

    def predict(self, data, rows):
        tot = demand_totals(data)
        index = {int(c): i for i, c in enumerate(self.ids)}
        out = np.empty((len(rows), data["flows"].shape[1]))
        for i, r in enumerate(rows):
            j = index[int(data["closure_set"][r])]
            out[i] = self.mean_flow[j] * tot[r] / self.mean_total[j]
        return np.where(data["closed"][rows], 0.0, out)


# %% [markdown]
# ## All-or-nothing assignment


# %%
def aon_flows(net, demand, closed=None, times=None):
    closed = net.default_closed if closed is None else np.asarray(closed, dtype=bool)
    idx = np.flatnonzero(~closed)
    prep = prepare_links(net.links.iloc[idx].reset_index(drop=True))
    t = prep.free_flow_time.to_numpy() if times is None else np.asarray(times)[idx]
    a, b = prep.a_node.to_numpy(), prep.b_node.to_numpy()
    z = net.num_zones
    n = int(max(a.max(), b.max())) + 1
    if net.block_centroids:
        b_graph = np.where(b <= z, n + b - 1, b)
        size, target = n + z, lambda d: n + d - 1
    else:
        b_graph, size, target = b, n, lambda d: d
    edge = {(u, v): k for k, (u, v) in enumerate(zip(a.tolist(), b_graph.tolist()))}
    graph = csr_matrix((t, (a, b_graph)), shape=(size, size))
    _, pred = dijkstra(graph, indices=np.arange(1, z + 1), return_predecessors=True)

    od = np.array(demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    load = np.zeros(len(idx))
    for o in range(z):
        for d in np.flatnonzero(od[o]):
            v = target(d + 1)
            while v != o + 1:
                u = pred[o, v]
                assert u >= 0, f"zone {d + 1} unreachable from zone {o + 1}"
                load[edge[(u, v)]] += od[o, d]
                v = u
    flows = np.zeros(net.num_links)
    flows[idx] = load
    return flows


# %% [markdown]
# ## No-closure equilibrium


# %%
def no_closure_flows(net, demand, closed, **solve_kwargs):
    sol, _ = solve_scenario(net, demand, net.default_closed, **solve_kwargs)
    return np.where(closed, 0.0, sol.flows)


# %% [markdown]
# ## Inspect: AON on Braess and on one Berlin scenario

# %%
if __name__ == "__main__":
    from ta.checks import check_flows
    from ta.network import Network

    braess = Network.from_tntp("Braess-Example")
    print("Braess AON flows (all 6 trips on the free-flow shortest path 1-3-4-2):", aon_flows(braess, braess.demand))

    from ta.splits import load_set

    net = Network.from_tntp("Berlin-Friedrichshain")
    data, links, splits = load_set("Berlin-Friedrichshain", "main")
    r = int(np.flatnonzero(splits.split == "test")[0])
    f = aon_flows(net, data["demand"][r], data["closed"][r])
    rep = check_flows(net.links, data["demand"][r], f, True, closed=data["closed"][r])
    print(f"Berlin test scenario {r}: AON passes the flow checks except the gap: failed={rep['failed']}, "
          f"conservation error {rep['conservation_error']:.1e}, AON gap {rep['rgap']:.2f}")
