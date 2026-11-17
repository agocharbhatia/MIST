"""ASTRA-sim congestion-aware network backend for inter-client transfers.

Wraps the ``simulation_py_congestion_aware`` pybind11 module built from the
``extern/network_backend/astra-network-analytical`` submodule (see
``docs/astrasim.md``). Client ``i`` maps to NPU ``i`` of the topology.

ASTRA-sim keeps time in ns; MIST keeps time in ms. ASTRA-sim has its own
clock, so a transfer's duration is measured on that clock and added to the
time the request left its source client.
"""
import importlib
import os
import sys

_ASTRA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "extern", "network_backend", "astra-network-analytical"))
ASTRA_LIB_DIR = os.environ.get("MIST_ASTRASIM_LIB", os.path.join(_ASTRA_DIR, "lib"))
ASTRA_INPUT_DIR = os.path.join(_ASTRA_DIR, "input")

NS_PER_MS = 1e6


def _load_backend():
    """Import the compiled backend on first use, so `import mist` never needs it."""
    if ASTRA_LIB_DIR not in sys.path:
        sys.path.append(ASTRA_LIB_DIR)
    try:
        return importlib.import_module("simulation_py_congestion_aware")
    except ImportError as e:
        raise ImportError(
            "Could not import 'simulation_py_congestion_aware'. Build the ASTRA-sim "
            f"network backend first (see docs/astrasim.md); expected under {ASTRA_LIB_DIR} "
            "or the directory in $MIST_ASTRASIM_LIB."
        ) from e


class AstraSimNetwork:
    """One ASTRA-sim topology shared by all clients of a coordinator."""

    def __init__(self, topology_file: str):
        if not os.path.isfile(topology_file):
            raise FileNotFoundError(f"ASTRA-sim topology file not found: {topology_file}")
        self._sim = _load_backend()
        self.event_queue = self._sim.EventQueue()
        self._sim.Topology.set_event_queue(self.event_queue)
        self.topology = self._sim.construct_topology(self._sim.NetworkParser(topology_file))

    @property
    def npus_count(self) -> int:
        return self.topology.get_npus_count()

    def now_ns(self) -> float:
        return self.event_queue.get_current_time()

    def send(self, size_bytes: int, src: int, dst: int, tag: int):
        """Start a transfer of `size_bytes` from NPU `src` to NPU `dst`, tagged with `tag`."""
        chunk = self._sim.Chunk.create_with_event_queue(
            size_bytes, src, dst, tag, self.topology, self.event_queue)
        self.topology.send_python(chunk)

    def idle(self) -> bool:
        return self.event_queue.finished()

    def step(self):
        """Process the next ASTRA-sim event. Returns (ns advanced, tags of finished transfers)."""
        before = self.now_ns()
        self.event_queue.proceed()
        return self.now_ns() - before, list(self.event_queue.get_and_clear_arrivals())
