# %% [markdown]
# # Network edits: edge and node addition

# %%
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta.network import Network

EDIT_TYPES = ("reverse", "new_link", "split")

# %% [markdown]
# ## Geometry and attribute estimates


# %%
@dataclass
class Geometry:
    net: Network

    def __post_init__(self):
        real = self.net.links[~self.net.is_connector & ~self.net.candidate]
        self.real = real
        self.xy = None if self.net.nodes is None else self.net.nodes.set_index("node")[["x", "y"]]
        if self.xy is not None:
            d = self.dist(real.a_node.to_numpy(), real.b_node.to_numpy())
            ok = d > 0
            self.fft_per_unit = np.median(real.free_flow_time.to_numpy()[ok] / d[ok])
            self.len_per_unit = np.median(real.length.to_numpy()[ok] / d[ok])
        self.b, self.power = real.b.median(), real.power.median()

    def dist(self, a, b):
        pa, pb = self.xy.loc[a].to_numpy(), self.xy.loc[b].to_numpy()
        return np.hypot(*(pa - pb).T)

    def capacity_near(self, *nodes):
        at = self.real[self.real.a_node.isin(nodes) | self.real.b_node.isin(nodes)]
        return float(at.capacity.median()) if len(at) else float(self.real.capacity.median())

    def new_link_rows(self, a, b, via_fft=None, via_len=None):
        if self.xy is not None:
            d = float(self.dist(np.array([a]), np.array([b]))[0])
            fft, length = d * self.fft_per_unit, d * self.len_per_unit
        else:
            fft, length = 0.8 * via_fft, 0.8 * via_len
        cap = self.capacity_near(a, b)
        row = {"capacity": cap, "length": length, "free_flow_time": fft, "b": self.b, "power": self.power,
               "speed": 0.0, "toll": 0.0, "link_type": 0}
        return [{"a_node": a, "b_node": b, **row}, {"a_node": b, "b_node": a, **row}]


# %% [markdown]
# ## Candidate generators


# %%
def has_parallel_opposite(geo, links, a, b, others):
    if geo.xy is None:
        return False
    tol = 0.5 * float(geo.dist(np.array([a]), np.array([b]))[0])
    c, d = others[:, 0], others[:, 1]
    near = (geo.dist(np.full(len(c), b), c) <= tol) & (geo.dist(np.full(len(d), a), d) <= tol)
    return bool(near.any())


def reverse_edits(net, geo):
    links, real = net.links, ~net.is_connector & ~net.candidate
    pairs = set(zip(links.a_node.tolist(), links.b_node.tolist()))
    others = links.loc[real, ["a_node", "b_node"]].to_numpy()
    edits = []
    for i in np.flatnonzero(real):
        r = links.loc[i]
        if (r.b_node, r.a_node) in pairs or has_parallel_opposite(geo, links, int(r.a_node), int(r.b_node), others):
            continue
        row = r.to_dict() | {"a_node": int(r.b_node), "b_node": int(r.a_node)}
        edits.append({"type": "reverse", "rows": [row], "closes": [], "node": None})
    return edits


def nearby_pairs(net, geo):
    links = net.links
    z = net.num_zones
    adjacent = set(zip(links.a_node.tolist(), links.b_node.tolist()))
    adjacent |= {(b, a) for a, b in adjacent}
    real = geo.real
    nodes = np.unique(np.concatenate([real.a_node, real.b_node]))
    nodes = nodes[nodes > z]
    if geo.xy is not None:
        limit = 1.5 * np.median(geo.dist(real.a_node.to_numpy(), real.b_node.to_numpy()))
        xy = geo.xy.loc[nodes].to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        i, j = np.nonzero(np.triu(d <= limit, k=1))
        return [(int(nodes[p]), int(nodes[q]), None, None) for p, q in zip(i, j) if (nodes[p], nodes[q]) not in adjacent]
    out = {}
    by_a = real.groupby("a_node")
    for x, grp in by_a:
        into = real[real.b_node == x]
        for _, r1 in into.iterrows():
            for _, r2 in grp.iterrows():
                a, b = int(r1.a_node), int(r2.b_node)
                if a != b and a > z and b > z and (a, b) not in adjacent:
                    key = (min(a, b), max(a, b))
                    fft, length = r1.free_flow_time + r2.free_flow_time, r1.length + r2.length
                    if key not in out or fft < out[key][2]:
                        out[key] = (key[0], key[1], fft, length)
    return list(out.values())


def new_link_edits(net, geo, count, rng):
    pairs = nearby_pairs(net, geo)
    chosen = rng.choice(len(pairs), min(count, len(pairs)), replace=False) if pairs else []
    return [{"type": "new_link", "rows": geo.new_link_rows(*pairs[k]), "closes": [], "node": None} for k in sorted(chosen)]


def split_edits(net, geo, count, rng, pairs):
    links = net.links
    real_streets = net.real_streets()
    chosen = rng.choice(real_streets, min(count, len(real_streets)), replace=False)
    next_node = int(max(links.a_node.max(), links.b_node.max(), net.nodes.node.max() if net.nodes is not None else 0)) + 1
    by_node = {}
    for a, b, fft, length in pairs:
        by_node.setdefault(a, []).append((b, fft, length))
        by_node.setdefault(b, []).append((a, fft, length))
    edits = []
    for s in chosen:
        idx = np.flatnonzero(net.street == s)
        first = links.loc[idx[0]]
        a, b, n = int(first.a_node), int(first.b_node), next_node
        next_node += 1
        rows = []
        for i in idx:
            r = links.loc[i].to_dict()
            half = {"free_flow_time": r["free_flow_time"] / 2, "length": r["length"] / 2}
            rows += [r | half | {"a_node": int(r["a_node"]), "b_node": n}, r | half | {"a_node": n, "b_node": int(r["b_node"])}]
        options = [(c, f, l) for c, f, l in by_node.get(a, []) + by_node.get(b, []) if c not in (a, b)]
        xy = None
        if geo.xy is not None:
            xy = (geo.xy.loc[a].to_numpy() + geo.xy.loc[b].to_numpy()) / 2
            geo.xy.loc[n] = xy
        if options:
            c, f, l = options[int(rng.integers(len(options)))]
            rows += geo.new_link_rows(n, c, via_fft=f, via_len=l)
        edits.append({"type": "split", "rows": rows, "closes": idx.tolist(), "node": n, "xy": xy})
    return edits


# %% [markdown]
# ## Superset


# %%
@dataclass
class Superset:
    net: Network
    num_base_links: int
    kind: np.ndarray
    edits: pd.DataFrame

    def edit_closed(self, edit_ids, closed=None):
        closed = self.net.default_closed if closed is None else closed.copy()
        for e in self.edits.loc[list(edit_ids)].itertuples():
            closed[e.opens] = False
            closed[e.closes] = True
        return closed

    def touched_base_streets(self, edit_ids):
        links = [i for e in self.edits.loc[list(edit_ids)].itertuples() for i in e.closes + e.opens]
        return np.unique(self.net.street[links]) if links else np.array([], dtype=np.int64)


def build_superset(net, reverse=True, new_links=50, splits=20, seed=0):
    rng = np.random.default_rng([seed, zlib.crc32(net.name.encode()), 7])
    geo = Geometry(net)
    pairs = nearby_pairs(net, geo)
    edits = (reverse_edits(net, geo) if reverse else []) + new_link_edits(net, geo, new_links, rng) + split_edits(net, geo, splits, rng, pairs)

    rows, kind, table = [], [], []
    base_n = net.num_links
    for eid, e in enumerate(edits):
        start = base_n + len(rows)
        rows += e["rows"]
        k = {"reverse": ["reverse"], "new_link": ["new_link", "new_link"]}.get(e["type"])
        kind += k if k else ["split_part"] * (2 * len(e["closes"])) + ["spur"] * (len(e["rows"]) - 2 * len(e["closes"]))
        table.append({"edit_id": eid, "type": e["type"], "opens": list(range(start, base_n + len(rows))),
                      "closes": e["closes"], "node": e["node"]})

    links = pd.concat([net.links, pd.DataFrame(rows, columns=net.links.columns)], ignore_index=True)
    links[["a_node", "b_node", "link_type"]] = links[["a_node", "b_node", "link_type"]].astype(np.int64)
    nodes = net.nodes
    if nodes is not None:
        new_nodes = [{"node": e["node"], "x": e["xy"][0], "y": e["xy"][1]} for e in edits if e["type"] == "split" and e["xy"] is not None]
        nodes = pd.concat([nodes, pd.DataFrame(new_nodes, columns=nodes.columns)], ignore_index=True)
    candidate = np.r_[np.zeros(base_n, dtype=bool), np.ones(len(rows), dtype=bool)]
    ext = Network(net.name, links, net.demand, net.num_zones, net.block_centroids, nodes, candidate)
    return Superset(ext, base_n, np.array(["base"] * base_n + kind), pd.DataFrame(table).set_index("edit_id"))


# %% [markdown]
# ## Inspect: the Berlin-Friedrichshain superset

# %%
if __name__ == "__main__":
    from ta.network import network_stats, solve_scenario

    base = Network.from_tntp("Berlin-Friedrichshain")
    sup = build_superset(base)
    print(f"base links {sup.num_base_links} -> superset links {sup.net.num_links}")
    print(sup.edits.type.value_counts().to_string())
    print(pd.Series(sup.kind).value_counts().to_string())

# %% [markdown]
# ## Inspect: solve with one edit of each type and compare to the base network

# %%
if __name__ == "__main__":
    base_sol, _ = solve_scenario(sup.net)
    rows = []
    for t in EDIT_TYPES:
        eid = int(sup.edits.index[sup.edits.type == t][0])
        closed = sup.edit_closed([eid])
        sol, rep = solve_scenario(sup.net, closed=closed)
        opened = sup.edits.loc[eid, "opens"]
        rows.append({"edit": f"{t} #{eid}", "opened_links": len(opened), "closed_links": len(sup.edits.loc[eid, "closes"]),
                     "flow_on_opened": round(sol.flows[opened].sum(), 1), "ok": rep["ok"], "rgap": sol.rgap,
                     "total_travel_time": round(np.nansum(sol.flows * sol.times), 0)})
    print(f"base network total travel time: {np.nansum(base_sol.flows * base_sol.times):.0f}")
    print(pd.DataFrame(rows).set_index("edit").to_string())

# %% [markdown]
# ## Inspect: a street slow-down

# %%
if __name__ == "__main__":
    busiest = int(np.argmax(np.where(base.is_connector, 0, base_sol.flows[: base.num_links])))
    fft_f, cap_f = np.ones(sup.net.num_links), np.ones(sup.net.num_links)
    fft_f[busiest], cap_f[busiest] = 3.0, 0.5
    slow, rep = solve_scenario(sup.net, capacity_factor=cap_f, fft_factor=fft_f)
    print(f"link {busiest} ({base.links.a_node[busiest]}->{base.links.b_node[busiest]}): flow {base_sol.flows[busiest]:.0f} -> {slow.flows[busiest]:.0f}, ok={rep['ok']}")

# %% [markdown]
# ## Inspect: where the candidates are (figure)

# %%
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ta import ROOT

    xy = sup.net.nodes.set_index("node")
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    colors = {"base": "#999", "reverse": "#2e86c1", "new_link": "#27ae60", "split_part": "#c0392b", "spur": "#c0392b"}
    for i, r in sup.net.links.iterrows():
        if sup.net.is_connector[i]:
            continue
        k = sup.kind[i]
        (x1, y1), (x2, y2) = xy.loc[r.a_node, ["x", "y"]], xy.loc[r.b_node, ["x", "y"]]
        ax.plot([x1, x2], [y1, y2], color=colors[k], lw=0.7 if k == "base" else 1.6, ls="--" if k == "spur" else "-", zorder=2 if k == "base" else 3)
    new = sup.edits.node.dropna().astype(int)
    ax.scatter(xy.loc[new, "x"], xy.loc[new, "y"], s=30, color="#c0392b", zorder=4)
    ax.set_title("Candidates: reverse of one-way (blue), new links (green), new intersections + side roads (red)", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    out = ROOT / "runs" / "figures" / "berlin_friedrichshain_edits.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
