# %% [markdown]
# # Environment check

# %%
import os
import sys
import warnings
from importlib.metadata import version

os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

# %% [markdown]
# ## 1. PyTorch on the GPUs


# %%
def check_torch():
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA not available"
    for i in range(torch.cuda.device_count()):
        arch = "sm_{}{}".format(*torch.cuda.get_device_capability(i))
        assert arch in torch.cuda.get_arch_list(), f"this torch build has no {arch} kernels"
        a = torch.randn(1024, 1024, device=f"cuda:{i}")
        err = (a @ a.T - (a.cpu() @ a.cpu().T).to(a.device)).abs().max().item()
        assert err < 1e-2, f"GPU {i} matmul mismatch {err}"
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)} matmul ok (max err {err:.1e})")


if __name__ == "__main__":
    check_torch()

# %% [markdown]
# ## 2. PyTorch Geometric on the GPU


# %%
def check_pyg():
    import torch_geometric
    from torch_geometric.nn import GCNConv

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device="cuda")
    x = torch.randn(4, 8, device="cuda")
    conv = GCNConv(8, 4).cuda()
    conv(x, edge_index).sum().backward()
    assert conv.lin.weight.grad is not None
    print(f"torch_geometric {torch_geometric.__version__} | GCNConv forward+backward on GPU ok")


if __name__ == "__main__":
    check_pyg()

# %% [markdown]
# ## 3. AequilibraE against an analytic equilibrium


# %%
def check_aequilibrae():
    from pandas.errors import ChainedAssignmentError

    # false positive from pandas' refcount check inside AequilibraE's Cython graph builder
    warnings.filterwarnings("ignore", category=ChainedAssignmentError)
    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass

    net = pd.DataFrame({
        "link_id": [1, 2, 3, 4],
        "a_node": [1, 3, 1, 4],
        "b_node": [3, 2, 4, 2],
        "direction": 1,
        "free_flow_time": [5.0, 5.0, 7.5, 7.5],
        "capacity": [100.0] * 4,
        "b": 0.15,
        "power": 4.0,
    })
    demand = 200.0

    g = Graph()
    g.network = net
    g.prepare_graph(np.array([1, 2]))
    g.set_graph("free_flow_time")
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(True)

    mat = AequilibraeMatrix()
    mat.create_empty(zones=2, matrix_names=["demand"], memory_only=True)
    mat.index[:] = np.array([1, 2])
    mat.matrices[:, :, 0] = np.array([[0.0, demand], [0.0, 0.0]])
    mat.computational_view(["demand"])

    assig = TrafficAssignment()
    assig.set_classes([TrafficClass("car", g, mat)])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("bfw")
    assig.max_iter = 500
    assig.rgap_target = 1e-8
    assig.execute()

    flows = assig.results()["demand_tot"].reindex([1, 2, 3, 4]).to_numpy()

    def route_time(fft, x):
        return fft * (1 + 0.15 * (x / 100) ** 4)

    x_a = brentq(lambda a: route_time(10, a) - route_time(15, demand - a), 0, demand)
    expected = np.array([x_a, x_a, demand - x_a, demand - x_a])
    err = np.abs(flows - expected).max()
    print(f"aequilibrae {version('aequilibrae')} | 2-route UE: got A={flows[0]:.3f} B={flows[2]:.3f}, "
          f"exact A={x_a:.3f} B={demand - x_a:.3f}, max err {err:.2e}")
    assert err < 1e-2, "AequilibraE equilibrium does not match the analytic solution"


if __name__ == "__main__":
    check_aequilibrae()
    print("ALL CHECKS PASSED")
