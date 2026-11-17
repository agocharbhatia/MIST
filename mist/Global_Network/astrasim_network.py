"""ASTRA-sim congestion-aware network backend for inter-client transfers.

Wraps the ``simulation_py_congestion_aware`` pybind11 module built from the
``extern/network_backend/astra-network-analytical`` submodule (see
``docs/astrasim.md``). Client ``i`` maps to NPU ``i`` of the topology.

ASTRA-sim keeps time in ns and MIST in ms; both clocks start at 0 and the
coordinator keeps them equal. This needs the `get_next_event_time` and
`advance_to` bindings of the submodule's `mist-integration` branch.
"""
import importlib
import math
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
        self._activate()
        self.topology = self._sim.construct_topology(self._sim.NetworkParser(topology_file))

    def _activate(self):
        # ASTRA-sim keeps one event queue for all links in a static member. Point it at
        # this instance's queue before every operation, so several coordinators (each
        # with its own AstraSimNetwork) do not schedule into each other's queues.
        self._sim.Topology.set_event_queue(self.event_queue)

    @property
    def npus_count(self) -> int:
        return self.topology.get_npus_count()

    def send(self, now_ms: float, size_bytes: int, src: int, dst: int, tag: int):
        """Start a transfer of `size_bytes` from NPU `src` to NPU `dst` at MIST time `now_ms`.

        Call advance(now_ms) first, so ASTRA-sim has no pending event before `now_ms`."""
        self._activate()
        t = _to_ns(now_ms)
        if not self.event_queue.finished() and self.event_queue.get_next_event_time() <= t:
            raise RuntimeError("ASTRA-sim has pending events before the send time; call advance() first")
        if t > self.event_queue.get_current_time():
            self.event_queue.advance_to(t)
        chunk = self._sim.Chunk.create_with_event_queue(
            size_bytes, src, dst, tag, self.topology, self.event_queue)
        self.topology.send_python(chunk)

    def next_event_ms(self):
        """MIST time of the next pending ASTRA-sim event, or None when idle."""
        if self.event_queue.finished():
            return None
        return self.event_queue.get_next_event_time() / NS_PER_MS

    def advance(self, now_ms: float):
        """Process every ASTRA-sim event up to MIST time `now_ms`.

        Returns [(finish time ms, tag)] for the transfers that finished."""
        self._activate()
        t, done = _to_ns(now_ms), []
        while not self.event_queue.finished() and self.event_queue.get_next_event_time() <= t:
            self.event_queue.proceed()
            done += [(self.event_queue.get_current_time() / NS_PER_MS, tag)
                     for tag in self.event_queue.get_and_clear_arrivals()]
        return done


def _to_ns(ms: float) -> int:
    # Round up: a transfer may start up to 1 ns late, never early, so no ASTRA-sim
    # event (even a zero-length serialization) lands before the MIST time that caused it.
    return math.ceil(ms * NS_PER_MS)
