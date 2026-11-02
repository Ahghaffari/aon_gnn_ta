# %% [markdown]
# # Network container, closures, reachability

# %%
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ta import FOCUS_NETWORKS
from ta.checks import check_flows, connector_mask
from ta.io_tntp import load_network
from ta.solver import Solution, prepare_links, solve, zone_to_zone_costs

# %% [markdown]
# ## Network


# %%
def street_ids(links):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    index = {(x, y): i for i, (x, y) in enumerate(zip(a.tolist(), b.tolist()))}
    street = np.full(len(links), -1, dtype=np.int64)
    next_id = 0
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        if street[i] >= 0:
            continue
        street[i] = next_id
        j = index.get((y, x))
        if j is not None:
            street[j] = next_id
        next_id += 1
    return street


@dataclass
class Network:
    name: str
    links: pd.DataFrame
    demand: np.ndarray
    num_zones: int
    block_centroids: bool
    nodes: pd.DataFrame | None = None
    candidate: np.ndarray | None = None
    is_connector: np.ndarray = field(init=False)
    street: np.ndarray = field(init=False)

    def __post_init__(self):
        self.links = self.links.reset_index(drop=True)
        self.candidate = np.zeros(len(self.links), dtype=bool) if self.candidate is None else np.asarray(self.candidate, dtype=bool)
        self.is_connector = connector_mask(self.links, self.num_zones)
        self.street = street_ids(self.links)

    @classmethod
    def from_tntp(cls, name):
        if "@v" in name:
            from ta.virtual import load

            return load(name)
        t = load_network(name)
        return cls(t.name, t.links, t.demand, t.num_zones, t.blocks_centroid_flows, t.nodes)

    @property
    def num_links(self):
        return len(self.links)

    @property
    def default_closed(self):
        return self.candidate.copy()

    def real_streets(self):
        return np.unique(self.street[~self.is_connector & ~self.candidate])

    def closure_mask(self, streets=(), links=()):
        closed = np.isin(self.street, np.asarray(streets, dtype=np.int64))
        closed[np.asarray(links, dtype=np.int64)] = True
        return closed


# %% [markdown]
# ## Reachability


# %%
def unreachable_demand(net, closed, demand=None):
    od = np.array(net.demand if demand is None else demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    open_links = prepare_links(net.links[~closed].reset_index(drop=True))
    sp = zone_to_zone_costs(open_links, open_links.free_flow_time.to_numpy(), net.num_zones, net.block_centroids)
    return od[(od > 0) & ~np.isfinite(sp)].sum()


# %% [markdown]
# ## Solve one scenario


# %%
def degraded_links(net, capacity_factor=None, fft_factor=None):
    links = net.links.copy()
    if capacity_factor is not None:
        assert (np.asarray(capacity_factor) > 0).all(), "use a closure instead of zero capacity"
        links["capacity"] = links["capacity"] * capacity_factor
    if fft_factor is not None:
        assert (np.asarray(fft_factor) >= 1).all(), "slow-downs only make links slower"
        links["free_flow_time"] = links["free_flow_time"] * fft_factor
    return links


def solve_scenario(net, demand=None, closed=None, rgap_target=1e-5, capacity_factor=None, fft_factor=None, **kwargs):
    od = net.demand if demand is None else demand
    closed = net.default_closed if closed is None else np.asarray(closed, dtype=bool)
    lost = unreachable_demand(net, closed, od)
    assert lost == 0.0, f"closures disconnect {lost:.1f} trips"

    links = degraded_links(net, capacity_factor, fft_factor)
    sol = solve(links[~closed].reset_index(drop=True), od, net.block_centroids, rgap_target=rgap_target, **kwargs)
    flows, times = np.zeros(net.num_links), np.full(net.num_links, np.nan)
    flows[~closed], times[~closed] = sol.flows, sol.times
    full = Solution(flows, times, sol.rgap, sol.rgap_reported, sol.iterations, sol.runtime, sol.converged,
                    sol.conservation_error, sol.pruned_links)
    report = check_flows(links, od, flows, net.block_centroids, closed=closed, rgap_target=rgap_target)
    return full, report


# %% [markdown]
# ## Statistics per network


# %%
def network_stats(net):
    real = net.links[~net.is_connector]
    od = net.demand.copy()
    np.fill_diagonal(od, 0.0)
    return {
        "links": net.num_links,
        "real_links": len(real),
        "connectors": int(net.is_connector.sum()),
        "streets_real": len(net.real_streets()),
        "two_way_real_share": float(pd.Series(net.street[~net.is_connector]).duplicated(keep=False).mean()),
        "zones": net.num_zones,
        "block_centroids": net.block_centroids,
        "median_capacity": float(real.capacity.median()),
        "median_fft": float(real.free_flow_time.median()),
        "total_demand": float(od.sum()),
    }


# %% [markdown]
# ## Inspect: the focus networks

# %%
if __name__ == "__main__":
    nets = {name: Network.from_tntp(name) for name in FOCUS_NETWORKS}
    print(pd.DataFrame({name: network_stats(n) for name, n in nets.items()}).to_string())

# %% [markdown]
# ## Inspect: close 10 random real streets and solve

# %%
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = []
    for name, net in nets.items():
        for attempt in range(1, 101):
            closed = net.closure_mask(streets=rng.choice(net.real_streets(), 10, replace=False))
            if unreachable_demand(net, closed) == 0.0:
                break
        sol, rep = solve_scenario(net, closed=closed)
        rows.append({"network": name, "attempts": attempt, "closed_links": int(closed.sum()),
                     "newly_pruned": sol.pruned_links - solve_scenario(net)[0].pruned_links,
                     "ok": rep["ok"], "rgap": sol.rgap, "iters": sol.iterations, "sec": round(sol.runtime, 1),
                     "zero_flow": rep["zero_flow_share"], "v/c>1": rep["overcapacity_share"]})
    print(pd.DataFrame(rows).set_index("network").to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Inspect: a closure set that isolates a zone is rejected before solving

# %%
if __name__ == "__main__":
    net = nets["Berlin-Friedrichshain"]
    closed = net.closure_mask(links=np.flatnonzero((net.links.a_node == 1).to_numpy()))
    print(f"closing all exits of zone 1 -> unreachable trips: {unreachable_demand(net, closed):.1f} "
          f"(zone 1 produces {net.demand[0].sum() - net.demand[0, 0]:.1f})")
