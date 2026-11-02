# %% [markdown]
# # Scenario sampler

# %%
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from ta import FOCUS_NETWORKS
from ta.network import Network, solve_scenario, unreachable_demand
from ta.solver import reachable

# %% [markdown]
# ## Configuration


# %%
@dataclass(frozen=True)
class ScenarioConfig:
    scale_min: float = 0.6
    scale_max: float = 1.4
    zone_sigma: float = 0.3
    cell_sigma: float = 0.2
    closure_mode: str = "street"
    closure_unit: str = "fraction"
    closure_min: float = 0.0
    closure_max: float = 0.10
    max_attempts: int = 1000
    repair: bool = False
    edits_min: int = 0
    edits_max: int = 0
    edit_types: tuple = ("reverse", "new_link", "split")
    slowdown_min: float = 0.0
    slowdown_max: float = 0.0
    speed_factor_min: float = 1.0
    speed_factor_max: float = 5.0
    capacity_factor_min: float = 0.3
    capacity_factor_max: float = 1.0


@dataclass
class Scenario:
    network: str
    seed: int
    closure_set: int
    variant: int
    demand: np.ndarray
    closed: np.ndarray
    demand_scale: float
    closure_target: float
    attempts: int
    edits: tuple = ()
    capacity_factor: np.ndarray | None = None
    fft_factor: np.ndarray | None = None

    @property
    def num_closed(self):
        return int(self.closed.sum())


def closure_rng(network, seed, closure_set):
    return np.random.default_rng([seed, zlib.crc32(network.encode()), closure_set, 0])


def demand_rng(network, seed, closure_set, variant):
    return np.random.default_rng([seed, zlib.crc32(network.encode()), closure_set, variant + 1])


def edit_rng(network, seed, closure_set):
    return np.random.default_rng([seed, zlib.crc32(network.encode()), closure_set, 10**6 + 1])


def slowdown_rng(network, seed, closure_set):
    return np.random.default_rng([seed, zlib.crc32(network.encode()), closure_set, 10**6 + 2])


# %% [markdown]
# ## Demand


# %%
def mean_one_lognormal(rng, sigma, size):
    return rng.lognormal(-0.5 * sigma**2, sigma, size)


def sample_demand(rng, base, cfg):
    z = base.shape[0]
    scale = rng.uniform(cfg.scale_min, cfg.scale_max)
    prod = rng.lognormal(0.0, cfg.zone_sigma, z)
    attr = rng.lognormal(0.0, cfg.zone_sigma, z)
    prod, attr = prod / prod.mean(), attr / attr.mean()
    noise = mean_one_lognormal(rng, cfg.cell_sigma, (z, z))
    demand = scale * prod[:, None] * attr[None, :] * noise * base
    np.fill_diagonal(demand, 0.0)
    return demand, scale


# %% [markdown]
# ## Closures


# %%
def closure_candidates(net, mode, exclude_streets=()):
    if mode == "street":
        streets = net.real_streets()
        return streets[~np.isin(streets, exclude_streets)] if len(exclude_streets) else streets
    if mode == "link":
        links = np.flatnonzero(~net.is_connector & ~net.candidate)
        return links[~np.isin(net.street[links], exclude_streets)] if len(exclude_streets) else links
    raise ValueError(f"unknown closure mode {mode!r}")


def draw_closure_target(rng, cfg, n_candidates, n_links=None):
    if cfg.closure_unit == "fraction":
        f = rng.uniform(cfg.closure_min, cfg.closure_max)
        return f, int(round(f * n_candidates))
    if cfg.closure_unit == "link_share":
        f = rng.uniform(cfg.closure_min, cfg.closure_max)
        return f, int(round(f * n_links))
    if cfg.closure_unit == "links":
        k = int(rng.integers(int(cfg.closure_min), int(cfg.closure_max) + 1))
        return k, k
    raise ValueError(f"unknown closure unit {cfg.closure_unit!r}")


def pick(rng, net, candidates, count, cfg):
    if cfg.closure_mode == "link":
        return net.closure_mask(links=rng.choice(candidates, count, replace=False))
    if cfg.closure_unit == "fraction":
        return net.closure_mask(streets=rng.choice(candidates, count, replace=False))
    sizes = np.bincount(net.street[~net.candidate], minlength=net.street.max() + 1)
    order = rng.permutation(candidates)
    upto = np.searchsorted(np.cumsum(sizes[order]), count) + 1 if count > 0 else 0
    return net.closure_mask(streets=order[:upto])


def core_reach(net, closed):
    a, b = net.links.a_node.to_numpy(), net.links.b_node.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    street = ~net.is_connector & ~closed
    graph = csr_matrix((np.ones(street.sum()), (a[street], b[street])), shape=(n, n))
    _, label = connected_components(graph, directed=True, connection="strong")
    core = np.flatnonzero(label == np.bincount(label).argmax())
    fwd, bwd = np.zeros(n + 1, dtype=bool), np.zeros(n + 1, dtype=bool)
    fwd[reachable(a[street], b[street], core, n)] = True
    bwd[reachable(b[street], a[street], core, n)] = True
    return fwd[:n], bwd[:n]


def reopen_paths(net, closed, start, targets, reverse):
    real = ~net.is_connector & ~net.candidate
    a, b = net.links.a_node.to_numpy(), net.links.b_node.to_numpy()
    if reverse:
        a, b = b, a
    n = len(start)
    w = np.where(closed[real], 1.0, 1e-6)
    graph = csr_matrix((w, (a[real], b[real])), shape=(n, n))
    dist, pred, _ = dijkstra(graph, indices=np.flatnonzero(start), min_only=True, return_predecessors=True)
    index = {(u, v): i for i, u, v in zip(np.flatnonzero(real).tolist(), a[real].tolist(), b[real].tolist())}
    closed = closed.copy()
    for group in targets:
        v = group[np.argmin(dist[group])]
        assert np.isfinite(dist[v]), "zone unreachable even with every street open"
        while pred[v] >= 0:
            closed[index[(int(pred[v]), int(v))]] = False
            v = pred[v]
    return closed


def reconnect(net, closed, rounds=100):
    a, b, z = net.links.a_node.to_numpy(), net.links.b_node.to_numpy(), net.num_zones
    od = np.array(net.demand, dtype=np.float64)
    np.fill_diagonal(od, 0.0)
    into, outof = net.is_connector & (b <= z), net.is_connector & (a <= z)
    for _ in range(rounds):
        fwd, bwd = core_reach(net, closed)
        reached, reaching = np.zeros(z + 1, dtype=bool), np.zeros(z + 1, dtype=bool)
        reached[b[into & fwd[a]]] = True
        reaching[a[outof & bwd[b]]] = True
        cut_sinks = np.flatnonzero((od.sum(0) > 0) & ~reached[1:]) + 1
        cut_sources = np.flatnonzero((od.sum(1) > 0) & ~reaching[1:]) + 1
        if not len(cut_sinks) and not len(cut_sources):
            return closed
        if len(cut_sinks):
            closed = reopen_paths(net, closed, fwd, [a[into & (b == d)] for d in cut_sinks], reverse=False)
        if len(cut_sources):
            closed = reopen_paths(net, closed, bwd, [b[outof & (a == o)] for o in cut_sources], reverse=True)
    raise RuntimeError("closures not repaired")


def repair_closures(rng, net, cfg, candidates, count, closed, rounds=50, tolerance=0.01):
    real = ~net.is_connector & ~net.candidate
    for _ in range(rounds):
        closed = reconnect(net, closed)
        deficit = count - int((closed & real).sum())
        if deficit <= tolerance * count:
            break
        free = candidates[~np.isin(candidates, net.street[closed])]
        closed = closed | pick(rng, net, free, deficit, cfg)
    return closed


def sample_closures(rng, net, cfg, base_closed=None, exclude_streets=()):
    base_closed = np.zeros(net.num_links, dtype=bool) if base_closed is None else base_closed
    candidates = closure_candidates(net, cfg.closure_mode, exclude_streets)
    n_links = int((~net.is_connector & ~net.candidate).sum())
    target, count = draw_closure_target(rng, cfg, len(candidates), n_links)
    if cfg.repair:
        closed = repair_closures(rng, net, cfg, candidates, count, base_closed | pick(rng, net, candidates, count, cfg))
        assert unreachable_demand(net, closed) == 0.0, "repaired closures still cut off trips"
        return closed, target, 1
    for attempt in range(1, cfg.max_attempts + 1):
        closed = base_closed | pick(rng, net, candidates, count, cfg)
        if unreachable_demand(net, closed) == 0.0:
            return closed, target, attempt
    raise RuntimeError(f"no reachable closure set of size {count} after {cfg.max_attempts} attempts on {net.name}")


# %% [markdown]
# ## Edits and slow-downs


# %%
def sample_edits(rng, superset, cfg):
    pool = superset.edits.index[superset.edits.type.isin(cfg.edit_types)].to_numpy()
    k = int(rng.integers(cfg.edits_min, cfg.edits_max + 1))
    return tuple(sorted(int(e) for e in rng.choice(pool, min(k, len(pool)), replace=False)))


def sample_slowdowns(rng, net, closed, cfg):
    open_streets = np.unique(net.street[~net.is_connector & ~closed])
    k = int(round(rng.uniform(cfg.slowdown_min, cfg.slowdown_max) * len(open_streets)))
    chosen = rng.choice(open_streets, k, replace=False)
    fft_f, cap_f = np.ones(net.num_links), np.ones(net.num_links)
    for s in chosen:
        on = (net.street == s) & ~closed
        fft_f[on] = rng.uniform(cfg.speed_factor_min, cfg.speed_factor_max)
        cap_f[on] = rng.uniform(cfg.capacity_factor_min, cfg.capacity_factor_max)
    return cap_f, fft_f


# %% [markdown]
# ## One scenario


# %%
def sample_scenario(net, closure_set, variant=0, cfg=ScenarioConfig(), seed=42, superset=None):
    edits, base_closed, touched = (), None, ()
    if cfg.edits_max > 0:
        assert superset is not None and superset.net is net, "edits need the superset network"
        edits = sample_edits(edit_rng(net.name, seed, closure_set), superset, cfg)
        base_closed, touched = superset.edit_closed(edits), superset.touched_base_streets(edits)
    elif net.candidate.any():
        base_closed = net.default_closed
    closed, target, attempts = sample_closures(closure_rng(net.name, seed, closure_set), net, cfg, base_closed, touched)
    cap_f = fft_f = None
    if cfg.slowdown_max > 0:
        cap_f, fft_f = sample_slowdowns(slowdown_rng(net.name, seed, closure_set), net, closed, cfg)
    demand, scale = sample_demand(demand_rng(net.name, seed, closure_set, variant), net.demand, cfg)
    return Scenario(net.name, seed, closure_set, variant, demand, closed, scale, target, attempts, edits, cap_f, fft_f)


# %% [markdown]
# ## Inspect: 200 closure sets per focus network (sampling only, no solving)

# %%
if __name__ == "__main__":
    nets = {name: Network.from_tntp(name) for name in FOCUS_NETWORKS}
    configs = {
        "default (0-10% streets)": ScenarioConfig(),
        "main (0-47 links)": ScenarioConfig(closure_unit="links", closure_min=0, closure_max=47),
        "old_like (26-47 links)": ScenarioConfig(closure_unit="links", closure_min=26, closure_max=47),
    }
    rows = []
    for name, net in nets.items():
        base_total = net.demand.sum() - np.trace(net.demand)
        for label, cfg in configs.items():
            for c in range(200):
                sc = sample_scenario(net, c, 0, cfg)
                rows.append({"network": name, "config": label, "closed_links": sc.num_closed, "attempts": sc.attempts,
                             "demand_ratio": sc.demand.sum() / base_total})
    samples = pd.DataFrame(rows)
    summary = samples.groupby(["network", "config"]).agg(
        closed_min=("closed_links", "min"), closed_median=("closed_links", "median"), closed_max=("closed_links", "max"),
        attempts_median=("attempts", "median"), attempts_max=("attempts", "max"),
        demand_min=("demand_ratio", "min"), demand_max=("demand_ratio", "max"))
    print(summary.to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Inspect: variants share closures, differ in demand; ids are reproducible

# %%
if __name__ == "__main__":
    net = nets["Berlin-Friedrichshain"]
    cfg = configs["main (0-47 links)"]
    a, b, a2 = sample_scenario(net, 7, 0, cfg), sample_scenario(net, 7, 1, cfg), sample_scenario(net, 7, 0, cfg)
    print("same closure set -> same closures:", np.array_equal(a.closed, b.closed))
    print("different variant -> different demand:", not np.array_equal(a.demand, b.demand))
    print("same id reproduced exactly:", np.array_equal(a.demand, a2.demand) and np.array_equal(a.closed, a2.closed))
    print(f"closure set 7: target {a.closure_target} links, closed {a.num_closed}: {np.flatnonzero(a.closed).tolist()}")

# %% [markdown]
# ## Inspect: solve a few scenarios and run the physics checks

# %%
if __name__ == "__main__":
    rows = []
    for c in range(2):
        for v in range(2):
            sc = sample_scenario(net, c, v, cfg)
            sol, rep = solve_scenario(net, sc.demand, sc.closed)
            rows.append({"closure_set": c, "variant": v, "scale": sc.demand_scale, "closed": sc.num_closed,
                         "ok": rep["ok"], "rgap": sol.rgap, "iters": sol.iterations, "sec": round(sol.runtime, 1),
                         "zero_flow": rep["zero_flow_share"], "v/c>1": rep["overcapacity_share"]})
    print(pd.DataFrame(rows).set_index(["closure_set", "variant"]).to_string(float_format=lambda v: f"{v:.3g}"))
