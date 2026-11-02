# %% [markdown]
# # Graph dataset

# %%
import numpy as np
import torch
from torch_geometric.data import Data

from ta.features import aon_cache, closure_set_aon
from ta.network import Network
from ta.solver import aon_assign
from ta.splits import load_set

# %% [markdown]
# ## Static graph


# %%
def static_graph(net):
    links = net.links
    nodes = np.unique(np.concatenate([links.a_node, links.b_node, np.arange(1, net.num_zones + 1)]))
    index = {int(n): i for i, n in enumerate(nodes)}
    edge_index = np.stack([links.a_node.map(index).to_numpy(), links.b_node.map(index).to_numpy()])
    xy = np.zeros((len(nodes), 2))
    if net.nodes is not None:
        coords = net.nodes.set_index("node")[["x", "y"]]
        known = [i for i, n in enumerate(nodes) if n in coords.index]
        xy[known] = coords.loc[nodes[known]].to_numpy()
    node_static = np.column_stack([nodes <= net.num_zones, xy]).astype(np.float32)
    edge_static = np.column_stack([links.capacity, links.free_flow_time, links.length, links.b, links.power,
                                   net.is_connector]).astype(np.float32)
    zone_node = np.array([index[z] for z in range(1, net.num_zones + 1)])
    scale = [float(np.median(links.capacity.to_numpy()[~net.is_connector])), float(np.asarray(net.demand).sum())]
    open_links = ~net.default_closed
    aon_base = np.zeros(len(links))
    aon_base[open_links] = aon_assign(links[open_links].reset_index(drop=True), net.demand, net.block_centroids)
    return {"graph_scale": torch.tensor([scale], dtype=torch.float32),
            "nodes": nodes, "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
            "node_static": torch.as_tensor(node_static), "edge_static": torch.as_tensor(edge_static),
            "zone_node": torch.as_tensor(zone_node, dtype=torch.long), "aon_base": torch.as_tensor(aon_base, dtype=torch.float32)}


# %% [markdown]
# ## Scenario graphs


# %%
class ScenarioData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "zone_node":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, network, name, with_closure_set_aon=False):
        self.network, self.name = network, name
        self.net = Network.from_tntp(network)
        self.data, self.links, self.splits = load_set(network, name)
        cache = aon_cache(network, name, self.data)
        self.aon, self.aon_open = cache["aon"], cache["aon_open"]
        self.aon_cs, self.aon_cs_index = closure_set_aon(network, name, self.data) if with_closure_set_aon else (None, None)
        self.graph = static_graph(self.net)
        e = self.data["flows"].shape[1]
        assert e == self.graph["edge_index"].shape[1], "link table and network disagree"
        self.ones = np.ones(e, dtype=np.float32)

    def __len__(self):
        return len(self.data["flows"])

    def rows(self, split, holdout=None):
        m = (self.splits.split == split).to_numpy()
        if holdout is not None:
            m = m & (self.splits.holdout_variant == holdout).to_numpy()
        return np.flatnonzero(m)

    def __getitem__(self, i):
        d, g = self.data, self.graph
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
        extra = {} if self.aon_cs is None else {"aon_cs": f32(self.aon_cs[self.aon_cs_index[i]])}
        demand = ({"demand": f32(d["demand"][i])[None]} if "demand" in d
                  else {"prod": f32(d["demand_prod"][i])[None], "attr": f32(d["demand_attr"][i])[None]})
        return ScenarioData(
            edge_index=g["edge_index"], node_static=g["node_static"], edge_static=g["edge_static"],
            zone_node=g["zone_node"], num_nodes=len(g["nodes"]),
            **demand, closed=torch.as_tensor(d["closed"][i]), aon=f32(self.aon[i]), aon_open=f32(self.aon_open[i]), aon_base=g["aon_base"],
            capacity_factor=f32(d["capacity_factor"][i]) if "capacity_factor" in d else f32(self.ones),
            fft_factor=f32(d["fft_factor"][i]) if "fft_factor" in d else f32(self.ones),
            y=f32(d["flows"][i]), y_time=f32(d["times"][i]),
            row=torch.tensor([i]), gridlock=torch.tensor([bool(self.splits.gridlock.iloc[i])]), graph_scale=g["graph_scale"], **extra,
        )

    def subset(self, rows):
        return torch.utils.data.Subset(self, rows)


# %% [markdown]
# ## Inspect: one graph and one batch

# %%
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    ds = GraphDataset("Berlin-Friedrichshain", "main")
    print(f"{len(ds)} scenarios | train {len(ds.rows('train', holdout=False))}, val {len(ds.rows('val'))}, "
          f"test {len(ds.rows('test'))}, holdout {len(ds.rows('train', holdout=True))}")
    g = ds[0]
    print(g)
    batch = next(iter(DataLoader(ds.subset(ds.rows("test")), batch_size=4)))
    print(batch)
    offset = batch.ptr[2].item()
    print("zone nodes of graph 2, minus its node offset:", (batch.zone_node[2 * 23:3 * 23] - offset).tolist()[:5], "== graph 0:", g.zone_node.tolist()[:5])
