# %% [markdown]
# # Per-link features

# %%
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ta.baselines import aon_flows
from ta.network import Network
from ta.solver import aon_assign
from ta.splits import dataset_dir

FEATURES = ["aon", "aon_open", "aon_vc", "log_capacity", "fft", "length", "b", "is_connector", "demand_ratio",
            "num_closed", "closed_1hop", "closed_2hop", "displaced_1hop", "displaced_2hop", "zone_production", "zone_attraction"]

# %% [markdown]
# ## AON cache

# %%
WORKER = {}


def init_worker(network):
    WORKER["net"] = Network.from_tntp(network)


def aon_pair(args):
    demand, closed = args
    net = WORKER["net"]
    return aon_flows(net, demand, closed), aon_flows(net, demand, net.default_closed)


def aon_cache(network, name, data, workers=None):
    if "aon" in data:
        return {"aon": data["aon"], "aon_open": data["aon_open"]}
    path = dataset_dir(network) / f"{name}.aon.npz"
    if path.exists():
        with np.load(path) as f:
            cached = {k: f[k] for k in f.files}
        if len(cached["aon"]) == len(data["flows"]):
            return cached
    workers = workers or min(28, os.cpu_count())
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(network,)) as pool:
        pairs = list(pool.map(aon_pair, zip(data["demand"], data["closed"]), chunksize=16))
    cached = {"aon": np.stack([p[0] for p in pairs]), "aon_open": np.stack([p[1] for p in pairs])}
    np.savez_compressed(path, **cached)
    return cached


def base_demand_aon(closed):
    net = WORKER["net"]
    return aon_flows(net, net.demand, closed)


def base_demand_aon_assign(closed):
    net = WORKER["net"]
    flows = np.zeros(net.num_links)
    flows[~closed] = aon_assign(net.links[~closed].reset_index(drop=True), net.demand, net.block_centroids)
    return flows


def closure_set_aon(network, name, data, workers=None):
    sets, first, inverse = np.unique(data["closure_set"], return_index=True, return_inverse=True)
    closed = data["closed"][first]
    assert (data["closed"] == closed[inverse]).all(), "closures differ within a closure set"
    path = dataset_dir(network) / f"{name}.aon_closure_set.npz"
    if path.exists():
        with np.load(path) as f:
            cached = {k: f[k] for k in f.files}
        if np.array_equal(cached["closure_set"], sets):
            return cached["aon"], inverse
    workers = workers or min(28, os.cpu_count())
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(network,)) as pool:
        aon = np.stack(list(pool.map(base_demand_aon_assign if "aon" in data else base_demand_aon, closed))).astype(np.float32)
    np.savez_compressed(path, closure_set=sets, aon=aon)
    return aon, inverse


# %% [markdown]
# ## Neighbourhoods of links


# %%
def link_neighbourhoods(links):
    a, b = links.a_node.to_numpy(), links.b_node.to_numpy()
    ends = np.stack([a, b], 1)
    share = (ends[:, None, :, None] == ends[None, :, None, :]).any(axis=(2, 3))
    np.fill_diagonal(share, False)
    one = share.astype(np.float64)
    two = ((one @ one + one) > 0).astype(np.float64)
    np.fill_diagonal(two, 0.0)
    return one, two


# %% [markdown]
# ## Feature matrix


# %%
class LinkFeatures:
    def __init__(self, net):
        self.net = net
        links = net.links
        self.one, self.two = link_neighbourhoods(links)
        z = net.num_zones
        self.static = np.stack([np.log1p(links.capacity.to_numpy()), links.free_flow_time.to_numpy(), links.length.to_numpy(),
                                links.b.to_numpy(), net.is_connector.astype(float)], 1)
        self.capacity = links.capacity.to_numpy()
        self.from_zone = np.where(links.a_node.to_numpy() <= z, links.a_node.to_numpy() - 1, -1)
        self.to_zone = np.where(links.b_node.to_numpy() <= z, links.b_node.to_numpy() - 1, -1)
        base = net.demand.copy()
        np.fill_diagonal(base, 0.0)
        self.base_total = base.sum()

    def scenario(self, demand, closed, aon, aon_open):
        e = len(closed)
        od = demand.copy()
        np.fill_diagonal(od, 0.0)
        prod, attr = od.sum(1), od.sum(0)
        c = closed.astype(float)
        displaced = c * aon_open
        cols = [aon, aon_open, aon / self.capacity, *self.static.T,
                np.full(e, od.sum() / self.base_total), np.full(e, c.sum()),
                self.one @ c, self.two @ c, self.one @ displaced, self.two @ displaced,
                np.where(self.from_zone >= 0, prod[self.from_zone], 0.0), np.where(self.to_zone >= 0, attr[self.to_zone], 0.0)]
        return np.stack(cols, 1)

    def matrix(self, data, cache, rows, link_mean=None):
        blocks = [self.scenario(data["demand"][r], data["closed"][r], cache["aon"][r], cache["aon_open"][r]) for r in rows]
        x = np.concatenate(blocks)
        if link_mean is not None:
            x = np.concatenate([x, np.tile(link_mean, len(rows))[:, None]], 1)
        return x


# %% [markdown]
# ## Inspect: features of one Berlin test scenario

# %%
if __name__ == "__main__":
    import pandas as pd

    from ta.splits import load_set

    data, links, splits = load_set("Berlin-Friedrichshain", "main")
    cache = aon_cache("Berlin-Friedrichshain", "main", data)
    feats = LinkFeatures(Network.from_tntp("Berlin-Friedrichshain"))
    r = int(np.flatnonzero((splits.split == "test").to_numpy() & (data["num_closed"] > 20))[0])
    x = pd.DataFrame(feats.matrix(data, cache, [r]), columns=FEATURES)
    x["true_flow"], x["closed"] = data["flows"][r], data["closed"][r]
    print(f"scenario {r}: {int(data['closed'][r].sum())} closed links")
    print(x[~x.closed].sort_values("displaced_1hop", ascending=False).head(8).round(2).to_string())
    print("\ncorrelation with the true flow (open real links):")
    real_open = (~x.closed & (x.is_connector == 0))
    print(x[real_open][FEATURES].corrwith(x.true_flow[real_open]).round(2).to_string())
