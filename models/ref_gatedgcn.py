# %% [markdown]
# # Reference model: heterogeneous GatedGCN

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

DEFAULTS = {"hidden": 320, "layers": 6, "dropout": 0.05, "aux_zero_weight": 0.05, "lr": 5e-4, "weight_decay": 1e-4,
            "batch_size": 128, "epochs": 500, "patience": 30}

# %% [markdown]
# ## Inputs


# %%
class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ["od_mean", "od_std", "cap_mean", "cap_std", "sentinel"]:
            self.register_buffer(name, torch.tensor(0.0))

    def fit(self, dataset, rows):
        d = dataset.data
        num_nodes = len(dataset.graph["nodes"])
        od = torch.as_tensor(d["demand"][rows], dtype=torch.float64)
        cells = od.numel() / od.shape[1] * num_nodes
        mean = od.sum() / cells
        var = (od**2).sum() / cells - mean**2
        e = dataset.graph["edge_static"]
        conn = e[:, 5] > 0.5
        sentinel = 5.0 * e[~conn, 0].max()
        log_cap = torch.log1p(torch.where(conn, sentinel, e[:, 0]))
        self.od_mean.fill_(mean.item())
        self.od_std.fill_(max(var.sqrt().item(), 1e-6))
        self.cap_mean.fill_(log_cap[~conn].mean().item())
        self.cap_std.fill_(max(log_cap[~conn].std().item(), 1e-6))
        self.sentinel.fill_(sentinel.item())
        return self

    def norm_cap(self, batch):
        cap, conn = batch.edge_static[:, 0], batch.edge_static[:, 5] > 0.5
        return torch.where(conn, self.sentinel, cap)

    def inputs(self, batch):
        n = batch.num_nodes
        src, dst = batch.edge_index
        ones = torch.ones(src.numel(), device=src.device)
        deg_out = scatter(ones, src, dim=0, dim_size=n, reduce="sum")
        deg_in = scatter(ones, dst, dim=0, dim_size=n, reduce="sum")
        z = batch.demand.shape[1]
        od_rows = torch.zeros(n, z, device=src.device)
        od_rows[batch.zone_node] = batch.demand.reshape(-1, z)
        od_rows = (od_rows - self.od_mean) / self.od_std
        x = torch.cat([deg_in[:, None], deg_out[:, None], (deg_in + deg_out)[:, None], od_rows], 1)
        conn = batch.edge_static[:, 5]
        cap = (torch.log1p(self.norm_cap(batch)) - self.cap_mean) / self.cap_std
        present = (~batch.closed).float()
        edge_attr = torch.stack([cap, conn, present], 1)
        return x, edge_attr, (conn > 0.5).long()


# %% [markdown]
# ## Layers


# %%
class TypedLinear(nn.Module):
    def __init__(self, n_in, n_out, types=2):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(n_in, n_out) for _ in range(types)])

    def forward(self, x, masks):
        return sum(m(x) * mask for m, mask in zip(self.maps, masks))


class HGatedGCNLayer(nn.Module):
    def __init__(self, in_node, in_edge, hidden, dropout):
        super().__init__()
        self.lin_x_in = nn.Linear(in_node, hidden)
        self.lin_self = nn.Linear(hidden, hidden)
        self.lin_e_in = TypedLinear(in_edge, hidden)
        names = ["msg", "gate_e", "gate_dst", "gate_src", "e_res", "e_dst", "e_src"]
        self.typed = nn.ModuleDict({k: TypedLinear(hidden, hidden) for k in names})
        self.bn_x, self.bn_e = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_type):
        t = self.typed
        masks = [(edge_type == k).float()[:, None] for k in range(2)]
        xh = self.lin_x_in(x)
        src, dst = edge_index
        h_src, h_dst = xh[src], xh[dst]
        eh = self.lin_e_in(edge_attr, masks)
        gates = torch.sigmoid(t["gate_e"](eh, masks) + t["gate_dst"](h_dst, masks) + t["gate_src"](h_src, masks))
        msg = t["msg"](h_src, masks) * gates
        n = x.shape[0]
        agg = scatter(msg, dst, dim=0, dim_size=n, reduce="sum") / (scatter(gates, dst, dim=0, dim_size=n, reduce="sum") + 1e-6)
        x_out = self.dropout(F.relu(self.bn_x(self.lin_self(xh) + agg)))
        e_out = eh + t["e_res"](eh, masks) + t["e_dst"](h_dst, masks) + t["e_src"](h_src, masks)
        e_out = self.dropout(F.relu(self.bn_e(e_out)))
        return x_out, e_out


def readout(n_in, hidden, dropout):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))


# %% [markdown]
# ## Model


# %%
class RefGatedGCN(nn.Module):
    def __init__(self, num_zones, hidden=320, layers=6, dropout=0.05, aux_zero_weight=0.05, **_):
        super().__init__()
        in_node, in_edge = 3 + num_zones, 3
        self.norm = Normalizer()
        self.layers = nn.ModuleList([HGatedGCNLayer(in_node if i == 0 else hidden, in_edge if i == 0 else hidden, hidden, dropout)
                                     for i in range(layers)])
        self.readout_real = readout(3 * hidden, hidden, dropout)
        self.readout_connector = readout(3 * hidden + in_edge, hidden, dropout)
        self.zero_head = readout(3 * hidden + in_edge, hidden, dropout)
        self.aux_zero_weight = aux_zero_weight

    def fit_normalizer(self, dataset, rows):
        self.norm.fit(dataset, rows)
        return self

    def forward(self, batch):
        x, edge_attr, edge_type = self.norm.inputs(batch)
        raw = edge_attr
        for layer in self.layers:
            x, edge_attr = layer(x, batch.edge_index, edge_attr, edge_type)
        src, dst = batch.edge_index
        z = torch.cat([x[src], x[dst], edge_attr], 1)
        z_conn = torch.cat([z, raw], 1)
        zero_logit = self.zero_head(z_conn).squeeze(1)
        y_conn = self.readout_connector(z_conn).squeeze(1) * torch.sigmoid(-zero_logit) * raw[:, 2]
        is_conn = edge_type.float()
        y_hat = self.readout_real(z).squeeze(1) * (1 - is_conn) + y_conn * is_conn
        return y_hat, zero_logit

    def target(self, batch):
        return torch.log1p(batch.y / self.norm.norm_cap(batch))

    def loss(self, batch):
        y_hat, zero_logit = self(batch)
        loss = F.smooth_l1_loss(y_hat, self.target(batch))
        open_conn = (batch.edge_static[:, 5] > 0.5) & ~batch.closed
        if open_conn.any():
            loss = loss + self.aux_zero_weight * F.binary_cross_entropy_with_logits(zero_logit[open_conn], (batch.y[open_conn] == 0).float())
        return loss

    @torch.no_grad()
    def predict_flows(self, batch):
        y_hat, _ = self(batch)
        return torch.expm1(y_hat) * self.norm.norm_cap(batch)


Model = RefGatedGCN


# %% [markdown]
# ## Inspect: build, one forward / backward pass on the CPU, and the cost per batch

# %%
if __name__ == "__main__":
    import os
    import time

    from torch_geometric.loader import DataLoader

    from ta.dataset import GraphDataset

    os.nice(10)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ds = GraphDataset("Berlin-Friedrichshain", "main")
    train = ds.rows("train", holdout=False)
    model = RefGatedGCN(num_zones=23, **DEFAULTS).fit_normalizer(ds, train)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print({k: round(v.item(), 4) for k, v in model.norm.state_dict().items()})
    batch = next(iter(DataLoader(ds.subset(train), batch_size=DEFAULTS["batch_size"], shuffle=True)))
    opt = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["lr"], weight_decay=DEFAULTS["weight_decay"])
    t0 = time.perf_counter()
    for step in range(3):
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        opt.step()
        print(f"step {step}: loss {loss.item():.4f}")
    sec = (time.perf_counter() - t0) / 3
    print(f"{sec:.2f} s per training batch of {DEFAULTS['batch_size']} graphs on 4 CPU threads "
          f"→ ~{sec * len(train) / DEFAULTS['batch_size'] / 60:.1f} min per epoch")
    model.eval()
    flows = model.predict_flows(batch)
    print("predicted flows:", tuple(flows.shape), "| on closed links (not forced to 0, as in the old model):",
          f"{flows[batch.closed].abs().mean():.1f} veh/h mean")
