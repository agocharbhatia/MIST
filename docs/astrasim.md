# ASTRA-sim network backend

By default, MIST times each inter-client transfer (for example, the KV cache
moving from a prefill client to a decode client) with the analytical link
model in `mist.Global_Network`: latency + size / bandwidth, with no
congestion. The coordinator can instead send these transfers through the
congestion-aware analytical backend of
[ASTRA-sim](https://github.com/astra-sim/astra-network-analytical), so
concurrent transfers share links.

## Build the backend

The backend is a git submodule at
`extern/network_backend/astra-network-analytical`. It points to the
`mist-integration` branch of a fork that adds Python bindings
(`simulation_py_congestion_aware`). The compiled module is not committed.

Prerequisites: a C++17 compiler, CMake ≥ 3.14 and `pip install pybind11`.

```sh
git submodule update --init --recursive
cd extern/network_backend/astra-network-analytical
mkdir -p build && cd build
cmake .. -DPython_EXECUTABLE="$(which python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
make -j simulation_py_congestion_aware
```

This writes `simulation_py_congestion_aware.*.so` to
`extern/network_backend/astra-network-analytical/lib/`. MIST looks there by
default. To use a module built elsewhere, set `MIST_ASTRASIM_LIB` to its
directory.

## Use it

Pass an ASTRA-sim topology file to any coordinator:

```python
from mist.Coordinator import MISTCoordinatorDisagg
from mist.Global_Network.astrasim_network import ASTRA_INPUT_DIR

coordinator = MISTCoordinatorDisagg(
    ...,
    astrasim_topology=f"{ASTRA_INPUT_DIR}/Ring.yml",
)
```

The topology file sets the topology, NPU count, per-dimension bandwidth
(GB/s) and latency (ns). Example files are in the submodule's `input/`
directory.

- Client `i` maps to NPU `i`. The topology must have at least as many NPUs as
  the coordinator has clients.
- Transfers between different clients go through ASTRA-sim. A request that
  stays on the same client moves in zero time, as before.
- `request.metrics.kv_transfer_time` holds the ASTRA-sim transfer time for the
  prefill-to-decode move.

## Timing

ASTRA-sim keeps time in ns and MIST in ms. The coordinator keeps the two
clocks equal: before it starts a transfer at MIST time `t`, it runs ASTRA-sim
up to `t`, and it schedules a step at the time of every pending ASTRA-sim
event. Transfers therefore start when the request leaves its source client and
finish at the time ASTRA-sim computes, including queueing behind earlier
transfers on shared links. This uses the `get_next_event_time` and
`advance_to` bindings, which the submodule commit pinned here provides.

## Verify

`python scripts/verify_astrasim.py` runs MIST with and without ASTRA-sim and
checks that isolated transfers take latency + size / bandwidth, that queueing
on a shared link matches a hand-computed FIFO link, that no event is scheduled
in the past, and that runs without ASTRA-sim are unchanged.
