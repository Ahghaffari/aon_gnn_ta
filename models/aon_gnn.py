# %% [markdown]
# # AON-GNN: message-passing correction of all-or-nothing

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ta.splits import demand_totals

DEFAULTS = {"hidden": 128, "layers": 8, "dropout": 0.0}

# %% [markdown]
# ## Inputs


# %%
class Normalizer(nn.Module):
    def __init__(self, scale="fit"):
        super().__init__()
        assert scale in ("fit", "graph")
        self.scale = scale
        for name in ["c", "cap_mean", "cap_std", "fft_mean", "fft_std", "len_mean", "len_std", "demand_mean"]:
            self.register_buffer(name, torch.tensor(1.0))

    def fit(self, datasets, rows):
        parts = list(zip(datasets, rows)) if isinstance(datasets, (list, tuple)) else [(datasets, rows)]
        real = [d.graph["edge_static"][d.graph["edge_static"][:, 5] < 0.5] for d, _ in parts]
        self.c.fill_(torch.cat([e[:, 0] for e in real]).median().item())
        cap = [torch.log(e[:, 0] / d.graph["graph_scale"][0, 0]) if self.scale == "graph" else torch.log1p(e[:, 0])
               for e, (d, _) in zip(real, parts)]
        for name, values in [("cap", cap), ("fft", [torch.log1p(e[:, 1]) for e in real]), ("len", [torch.log1p(e[:, 2]) for e in real])]:
            v = torch.cat(values)
            getattr(self, f"{name}_mean").fill_(v.mean().item())
            getattr(self, f"{name}_std").fill_(max(v.std().item(), 1e-6))
        self.demand_mean.fill_(float(np.concatenate([demand_totals(d.data)[r] for d, r in parts]).mean()))
        return self

    def z(self, v, name):
        return (v - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

    def node_graph(self, batch):
        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.batch
        return torch.zeros(batch.num_nodes, dtype=torch.long, device=self.c.device)

    def edge_scale(self, batch):
        if self.scale == "fit":
            return self.c
        return batch.graph_scale[:, 0][self.node_graph(batch)[batch.edge_index[0]]]

    def inputs(self, batch, aon, displaced=False):
        e, c_e = batch.edge_static, self.edge_scale(batch)
        cap = e[:, 0] * batch.capacity_factor
        fft = e[:, 1] * batch.fft_factor
        aon_s = aon / c_e
        cap_in = torch.log(cap / c_e) if self.scale == "graph" else torch.log1p(cap)
        edge = torch.stack([aon_s, torch.log1p(aon_s), aon / cap, self.z(cap_in, "cap"), self.z(torch.log1p(fft), "fft"),
                            self.z(torch.log1p(e[:, 2]), "len"), e[:, 3], e[:, 5], batch.capacity_factor, batch.fft_factor], 1)
        n, graph = batch.num_nodes, self.node_graph(batch)
        c_n = self.c if self.scale == "fit" else batch.graph_scale[:, 0][graph]
        prod, attr = torch.zeros(n, device=e.device), torch.zeros(n, device=e.device)
        zone_prod, zone_attr = (batch.demand.sum(2), batch.demand.sum(1)) if "demand" in batch else (batch.prod, batch.attr)
        prod[batch.zone_node] = zone_prod.reshape(-1)
        attr[batch.zone_node] = zone_attr.reshape(-1)
        inflow = scatter(aon * (~batch.closed), batch.edge_index[1], dim=0, dim_size=n, reduce="sum")
        total = zone_prod.sum(1) / (self.demand_mean if self.scale == "fit" else batch.graph_scale[:, 1])
        cols = [batch.node_static[:, 0], torch.log1p(prod / c_n), torch.log1p(attr / c_n), torch.log1p(inflow / c_n), total[graph]]
        if displaced:
            cut = aon * batch.closed
            cols += [torch.log1p(scatter(cut, batch.edge_index[i], dim=0, dim_size=n, reduce="sum") / c_n) for i in (0, 1)]
        return torch.stack(cols, 1), edge


# %% [markdown]
# ## Layers


# %%
def mlp(n_in, hidden, n_out, dropout=0.0):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))


class TypedMLP(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.street, self.connector = mlp(*args), mlp(*args)

    def forward(self, x, conn):
        s, c = self.street(x[~conn]), self.connector(x[conn])
        out = s.new_empty(x.shape[0], s.shape[1])
        out[~conn], out[conn] = s, c
        return out


def edge_mlp(typed, *args):
    return TypedMLP(*args) if typed else mlp(*args)


def run(m, x, conn):
    return m(x, conn) if isinstance(m, TypedMLP) else m(x)


class BiLayer(nn.Module):
    def __init__(self, hidden, dropout, bidirectional=True, typed=False):
        super().__init__()
        self.edge = edge_mlp(typed, 3 * hidden, hidden, hidden, dropout)
        self.msg_in = edge_mlp(typed, 2 * hidden, hidden, hidden, dropout)
        self.msg_out = edge_mlp(typed, 2 * hidden, hidden, hidden, dropout) if bidirectional else None
        self.node = mlp(3 * hidden, hidden, hidden, dropout)
        self.norm_e, self.norm_h = nn.LayerNorm(hidden), nn.LayerNorm(hidden)

    def forward(self, h, e, src, dst, conn):
        e = self.norm_e(e + run(self.edge, torch.cat([e, h[src], h[dst]], 1), conn))
        n = h.shape[0]
        from_up = scatter(run(self.msg_in, torch.cat([e, h[src]], 1), conn), dst, dim=0, dim_size=n, reduce="mean")
        if self.msg_out is not None:
            from_down = scatter(run(self.msg_out, torch.cat([e, h[dst]], 1), conn), src, dim=0, dim_size=n, reduce="mean")
        else:
            from_down = torch.zeros_like(from_up)
        h = self.norm_h(h + self.node(torch.cat([h, from_up, from_down], 1)))
        return h, e


# %% [markdown]
# ## Model


# %%
class AonGNN(nn.Module):
    def __init__(self, hidden=128, layers=8, dropout=0.0, use_aon=True, residual=True, bidirectional=True, remove_closed=True,
                 typed_edges=False, scale="fit", aon_input="scenario", **_):
        super().__init__()
        assert use_aon or not residual, "the residual needs the AON flows"
        assert aon_input in ("scenario", "open", "base", "closure_set")
        self.aon_input = aon_input
        self.use_aon, self.residual, self.remove_closed = use_aon, residual, remove_closed
        self.norm = Normalizer(scale)
        self.embed_node = mlp(7 if aon_input in ("open", "base") else 5, hidden, hidden)
        self.embed_edge = edge_mlp(typed_edges, 10 if remove_closed else 11, hidden, hidden)
        self.layers = nn.ModuleList([BiLayer(hidden, dropout, bidirectional, typed_edges) for _ in range(layers)])
        self.readout = edge_mlp(typed_edges, 3 * hidden, hidden, 1)
        for head in [self.readout.street, self.readout.connector] if typed_edges else [self.readout]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def fit_normalizer(self, dataset, rows):
        self.norm.fit(dataset, rows)
        return self

    def aon_of(self, batch):
        return getattr(batch, {"scenario": "aon", "open": "aon_open", "base": "aon_base", "closure_set": "aon_cs"}[self.aon_input])

    def forward(self, batch):
        aon = self.aon_of(batch)
        node, edge = self.norm.inputs(batch, aon, displaced=self.aon_input in ("open", "base"))
        if not self.use_aon:
            edge, node = edge.clone(), node.clone()
            edge[:, :3], node[:, 3] = 0.0, 0.0
        if self.remove_closed:
            mp = ~batch.closed
        else:
            mp = torch.ones_like(batch.closed)
            edge = torch.cat([edge, (~batch.closed).float()[:, None]], 1)
        src, dst = batch.edge_index[:, mp]
        conn = batch.edge_static[mp, 5] > 0.5
        h, e = self.embed_node(node), run(self.embed_edge, edge[mp], conn)
        for layer in self.layers:
            h, e = layer(h, e, src, dst, conn)
        delta = run(self.readout, torch.cat([e, h[src], h[dst]], 1), conn).squeeze(1)
        out = torch.zeros(batch.num_edges, device=node.device)
        c_e = self.norm.edge_scale(batch)
        out[mp] = ((aon / c_e)[mp] if self.residual else 0.0) + delta
        return out.masked_fill(batch.closed, 0.0)

    def loss(self, batch):
        pred = self(batch)
        keep = ~batch.closed
        return F.mse_loss(pred[keep], (batch.y / self.norm.edge_scale(batch))[keep])

    @torch.no_grad()
    def predict_flows(self, batch):
        return torch.relu(self(batch)) * self.norm.edge_scale(batch)


Model = AonGNN


# %% [markdown]
# ## Inspect: build, check that the untrained model equals AON, one training step, CPU cost

# %%
if __name__ == "__main__":
    import os
    import time

    from torch_geometric.loader import DataLoader

    from ta.dataset import GraphDataset

    os.nice(10)
    torch.set_num_threads(12)
    torch.manual_seed(42)
    ds = GraphDataset("Berlin-Friedrichshain", "main")
    train = ds.rows("train", holdout=False)
    model = AonGNN(**DEFAULTS).fit_normalizer(ds, train)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,} | c = {model.norm.c.item():.0f} veh/h")
    batch = next(iter(DataLoader(ds.subset(train), batch_size=32, shuffle=True)))
    model.eval()
    print("untrained prediction == AON (open links):", torch.allclose(model.predict_flows(batch)[~batch.closed], batch.aon[~batch.closed], atol=1e-3))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for step in range(3):
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        opt.step()
        print(f"step {step}: loss {loss.item():.4f}")
    sec = (time.perf_counter() - t0) / 3
    print(f"{sec:.2f} s per batch of 32 on 12 CPU threads → ~{sec * len(train) / 32 / 60:.1f} min per epoch")
