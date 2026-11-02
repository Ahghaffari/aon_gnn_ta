# %% [markdown]
# # TNTP parser

# %%
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ta import ROOT

NET_COLUMNS = ["a_node", "b_node", "capacity", "length", "free_flow_time", "b", "power", "speed", "toll", "link_type"]
FILE_KINDS = {"net": "net", "trips": "trips", "node": "node", "flow": "flow"}

# %% [markdown]
# ## Metadata and body


# %%
def read_text(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".tntp") and not n.startswith("__MACOSX"))
            raw = zf.read(member)
    else:
        raw = path.read_bytes()
    return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")


def split_metadata(text):
    head, sep, body = text.partition("<END OF METADATA>")
    if not sep:
        head, body = "", text
    meta = {}
    for key, value in re.findall(r"<([^>]+)>([^\n<]*)", head):
        meta[key.strip().upper()] = value.strip()
    return meta, body


def meta_int(meta, key):
    value = meta.get(key, "")
    return int(float(value)) if value else None


# %% [markdown]
# ## Network file


# %%
def header_columns(line):
    names = line.lstrip("~").rstrip(";").split()
    return NET_COLUMNS[:8] + names[8:len(NET_COLUMNS)] if len(names) >= len(NET_COLUMNS) else NET_COLUMNS


def read_net(path):
    meta, body = split_metadata(read_text(path))
    rows, columns = [], NET_COLUMNS
    for line in body.split("\n"):
        line = line.strip().rstrip(";").strip()
        if line.startswith("~"):
            columns = header_columns(line)
            continue
        if not line:
            continue
        rows.append([float(tok) for tok in line.split()[: len(NET_COLUMNS)]])
    links = pd.DataFrame(rows, columns=columns)
    links = links.reindex(columns=NET_COLUMNS, fill_value=0.0)
    links[["a_node", "b_node", "link_type"]] = links[["a_node", "b_node", "link_type"]].astype(np.int64)
    return meta, links


# %% [markdown]
# ## Trips file

# %%
TRIPS_TOKEN = re.compile(r"Origin\s+(\d+)|(\d+)\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def read_trips(path, num_zones=None):
    meta, body = split_metadata(read_text(path))
    num_zones = num_zones or meta_int(meta, "NUMBER OF ZONES")
    demand = np.zeros((num_zones, num_zones), dtype=np.float64)
    origin = None
    for m in TRIPS_TOKEN.finditer(body):
        if m.group(1) is not None:
            origin = int(m.group(1))
        else:
            assert origin is not None, "destination value before any 'Origin' line"
            demand[origin - 1, int(m.group(2)) - 1] += float(m.group(3))
    return meta, demand


# %% [markdown]
# ## Node and flow files (optional)


# %%
def read_numeric_rows(path, ncols):
    rows = []
    for line in read_text(path).split("\n"):
        toks = line.strip().rstrip(";").split()
        try:
            rows.append([float(t) for t in toks[:ncols]])
        except ValueError:
            continue
    return [r for r in rows if len(r) == ncols]


def read_nodes(path):
    nodes = pd.DataFrame(read_numeric_rows(path, 3), columns=["node", "x", "y"])
    nodes["node"] = nodes["node"].astype(np.int64)
    return nodes


def read_flows(path):
    flows = pd.DataFrame(read_numeric_rows(path, 4), columns=["a_node", "b_node", "volume", "cost"])
    flows[["a_node", "b_node"]] = flows[["a_node", "b_node"]].astype(np.int64)
    return flows


# %% [markdown]
# ## Whole network


# %%
@dataclass
class TNTPNetwork:
    name: str
    links: pd.DataFrame
    demand: np.ndarray
    num_zones: int
    num_nodes: int
    first_thru_node: int
    total_od_flow: float
    nodes: pd.DataFrame | None = None
    flows: pd.DataFrame | None = None

    @property
    def num_links(self):
        return len(self.links)

    @property
    def blocks_centroid_flows(self):
        return self.first_thru_node > 1


def find_files(folder):
    found = {}
    for path in sorted(Path(folder).iterdir()):
        m = re.search(r"_(net|trips|node|flow)\.tntp(\.zip)?$", path.name, re.IGNORECASE)
        if m:
            kind = m.group(1).lower()
            assert kind not in found, f"two {kind} files in {folder}"
            found[kind] = path
    assert {"net", "trips"} <= found.keys(), f"{folder} needs a net and a trips file, found {sorted(found)}"
    return found


def load_network(name_or_path):
    folder = Path(name_or_path)
    if not folder.is_dir():
        folder = ROOT / "networks" / name_or_path
    files = find_files(folder)
    net_meta, links = read_net(files["net"])
    num_zones = meta_int(net_meta, "NUMBER OF ZONES")
    trips_meta, demand = read_trips(files["trips"], num_zones)
    return TNTPNetwork(
        name=folder.name,
        links=links,
        demand=demand,
        num_zones=num_zones,
        num_nodes=meta_int(net_meta, "NUMBER OF NODES"),
        first_thru_node=meta_int(net_meta, "FIRST THRU NODE") or 1,
        total_od_flow=float(trips_meta.get("TOTAL OD FLOW") or "nan"),
        nodes=read_nodes(files["node"]) if "node" in files else None,
        flows=read_flows(files["flow"]) if "flow" in files else None,
    )


# %% [markdown]
# ## Inspect: every downloaded network against its own header

# %%
if __name__ == "__main__":
    names = sorted(p.name for p in (ROOT / "networks").iterdir() if p.is_dir())
    rows = []
    for name in names:
        net = load_network(name)
        rows.append({
            "network": name,
            "zones": net.num_zones,
            "nodes(hdr)": net.num_nodes,
            "nodes(links)": len(np.union1d(net.links.a_node, net.links.b_node)),
            "links": net.num_links,
            "first_thru": net.first_thru_node,
            "od_total": round(net.demand.sum(), 2),
            "od_total(hdr)": net.total_od_flow,
            "coords": net.nodes is not None,
            "flows": net.flows is not None,
        })
    summary = pd.DataFrame(rows).set_index("network")
    print(summary.to_string())

# %% [markdown]
# ## Inspect: one network in detail

# %%
if __name__ == "__main__":
    net = load_network("Berlin-Friedrichshain")
    print(net.links.head(20))
    print("\ndemand (first 10 zones):\n", net.demand[:10, :10])

# %%
if __name__ == "__main__":
    print(len(net.demand))
