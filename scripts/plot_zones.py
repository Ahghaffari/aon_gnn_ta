# %% [markdown]
# # Figure: zones, connectors and dead ends

# %%
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ta import ROOT
from ta.checks import connector_mask
from ta.io_tntp import load_network
from ta.solver import prune_dead_ends

# %% [markdown]
# ## Drawing


# %%
def draw(ax, net, zoom_nodes=None, pad=0.12):
    links, xy, z = net.links, net.nodes.set_index("node")[["x", "y"]], net.num_zones
    conn = connector_mask(links, z)
    keep = prune_dead_ends(links, z)
    cap_scale = links.capacity[~conn].max() / 3
    for i, r in links.iterrows():
        (x1, y1), (x2, y2) = xy.loc[r.a_node], xy.loc[r.b_node]
        if conn[i]:
            ax.plot([x1, x2], [y1, y2], color="#e07b39", lw=0.9, ls="--", alpha=0.8, zorder=1)
        elif not keep[i]:
            ax.plot([x1, x2], [y1, y2], color="#c0392b", lw=2.5, zorder=3)
        else:
            ax.plot([x1, x2], [y1, y2], color="#555", lw=0.8 + r.capacity / cap_scale, zorder=2)
    real = xy.loc[xy.index > z]
    ax.scatter(real.x, real.y, s=6, color="#555", zorder=4)
    zones = xy.loc[1:z]
    ax.scatter(zones.x, zones.y, s=110, color="#e07b39", edgecolor="black", zorder=5)
    for node, (x, y) in zones.iterrows():
        ax.annotate(str(node), (x, y), ha="center", va="center", fontsize=6.5, zorder=6)
    if zoom_nodes:
        dead_nodes = set(links.a_node[~keep]) | set(links.b_node[~keep])
        for node in zoom_nodes:
            color = "#c0392b" if node in dead_nodes and node not in links.a_node[keep].values else "black"
            ax.annotate(str(node), xy.loc[node], xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
        c = xy.loc[zoom_nodes]
        ax.set_xlim(c.x.min() - pad, c.x.max() + pad)
        ax.set_ylim(c.y.min() - pad, c.y.max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_zones(name, zoom_nodes=None, out=None):
    net = load_network(name)
    assert net.nodes is not None, f"{name} has no node coordinates"
    fig, axes = plt.subplots(1, 2 if zoom_nodes else 1, figsize=(15 if zoom_nodes else 8, 7.5), facecolor="white", squeeze=False)
    draw(axes[0, 0], net)
    axes[0, 0].set_title(f"{name}: streets (grey, width ~ capacity), zones (orange), connectors (dashed)", fontsize=10)
    if zoom_nodes:
        draw(axes[0, 1], net, zoom_nodes)
        axes[0, 1].set_title(f"Zoom around nodes {zoom_nodes} (red = pruned dead-end links)", fontsize=10)
    fig.tight_layout()
    out = out or ROOT / "runs" / "figures" / f"{name.lower().replace('-', '_')}_zones.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    return fig, out


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--network", default="Berlin-Friedrichshain")
    parser.add_argument("--zoom", nargs="*", type=int, default=[2, 31, 83, 84, 85, 215, 216, 218, 40])
    parser.add_argument("--out", default=None)
    args = parser.parse_known_args()[0]
    fig, out = plot_zones(args.network, args.zoom or None, args.out and ROOT / args.out)
    print(f"saved {out}")
