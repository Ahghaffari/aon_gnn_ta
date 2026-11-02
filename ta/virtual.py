# %% [markdown]
# # Virtual networks

# %%
import zlib

import numpy as np

SEP = "@v"

# %% [markdown]
# ## Names and draws


# %%
def is_virtual(name):
    return SEP in name


def split_name(name):
    base, vid = name.split(SEP)
    return base, int(vid)


def names(base, ids):
    return [f"{base}{SEP}{i}" for i in ids]


def expand(entries):
    out = []
    for e in entries:
        if SEP in e and ":" in e:
            base, ids = e.split(SEP)
            first, stop = (int(x) for x in ids.split(":"))
            out += names(base, range(first, stop))
        else:
            out.append(e)
    return out


def character(net, vid):
    rng = np.random.default_rng([zlib.crc32(net.name.encode()), vid, 23])
    loguni = lambda lo, hi, size=None: np.exp(rng.uniform(np.log(lo), np.log(hi), size))
    real = ~net.is_connector
    caps = net.links.capacity.to_numpy()
    classes, cls = np.unique(caps[real], return_inverse=True)
    return {
        "classes": classes,
        "class_index": cls,
        "cap_class": loguni(0.7, 1.4, len(classes)),
        "cap_link": rng.lognormal(0.0, 0.1, int(real.sum())),
        "fft_all": loguni(0.8, 1.25),
        "fft_class": loguni(0.9, 1.1, len(classes)),
        "b": rng.uniform(0.5, 1.5),
        "power": float(rng.choice([3.0, 4.0, 5.0])),
        "prod": rng.lognormal(0.0, 0.4, net.num_zones),
        "attr": rng.lognormal(0.0, 0.4, net.num_zones),
    }


# %% [markdown]
# ## Build


# %%
def make_virtual(base_net, vid):
    from ta.network import Network

    ch = character(base_net, vid)
    links = base_net.links.copy()
    real = ~base_net.is_connector
    idx = np.flatnonzero(real)
    cls = ch["class_index"]
    links.loc[idx, "capacity"] = links.capacity.to_numpy()[idx] * ch["cap_class"][cls] * ch["cap_link"]
    links.loc[idx, "free_flow_time"] = links.free_flow_time.to_numpy()[idx] * ch["fft_all"] * ch["fft_class"][cls]
    links.loc[idx, "b"] = ch["b"]
    links.loc[idx, "power"] = ch["power"]
    demand = np.asarray(base_net.demand, dtype=np.float64)
    shaped = demand * ch["prod"][:, None] * ch["attr"][None, :]
    shaped *= demand.sum() / shaped.sum()
    return Network(f"{base_net.name}{SEP}{vid}", links, shaped, base_net.num_zones, base_net.block_centroids, base_net.nodes)


def load(name):
    from ta.network import Network

    base, vid = split_name(name)
    return make_virtual(Network.from_tntp(base), vid)


# %% [markdown]
# ## Inspect: a few virtual versions of Tiergarten

# %%
if __name__ == "__main__":
    from ta.network import Network

    base = Network.from_tntp("Berlin-Tiergarten")
    real = ~base.is_connector
    for name in names(base.name, range(4)):
        v = Network.from_tntp(name)
        ratio = v.links.capacity.to_numpy()[real] / base.links.capacity.to_numpy()[real]
        print(f"{name}: capacity × {ratio.min():.2f}–{ratio.max():.2f}, fft × {np.median(v.links.free_flow_time.to_numpy()[real] / base.links.free_flow_time.to_numpy()[real]):.2f}, "
              f"b {v.links.b[real].iloc[0]:.2f}, power {v.links.power[real].iloc[0]:.0f}, demand total {v.demand.sum():.0f} (base {base.demand.sum():.0f})")
