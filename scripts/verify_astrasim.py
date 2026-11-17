"""End-to-end check of MIST with and without the ASTRA-sim network backend.

Needs the compiled backend (docs/astrasim.md). Run from the repo root:
    python scripts/verify_astrasim.py
Each check prints PASS/FAIL; the exit code is the number of failures.
"""
import math
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import mist
from mist import ClusterConfig, LengthVariables, PoissonDistribution, SchedulerConfig, BatchingMethod, TraceIngestion
from mist.Client import ClientType, LLMClient
from mist.Coordinator import MISTCoordinator, MISTCoordinatorDisagg, MISTCoordinatorDisagg_DistServe
from mist.Global_Network.astrasim_network import ASTRA_INPUT_DIR, AstraSimNetwork

MODEL = "meta-llama/meta-llama-3.1-8b"
TMP = Path(tempfile.mkdtemp(prefix="mist-astra-"))
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def topo(kind, npus, bw_gbps, lat_ns):
    p = TMP / f"{kind}_{npus}_{bw_gbps}_{lat_ns}.yml"
    p.write_text(f"topology: [ {kind} ]\nnpus_count: [ {npus} ]\nbandwidth: [ {bw_gbps} ]\nlatency: [ {lat_ns} ]\n")
    return str(p)


def queue(rps, sim_time=3000, seed=0):
    return PoissonDistribution(rps=rps, sim_time=sim_time, input_vars=LengthVariables(1024, 256),
                               output_vars=LengthVariables(64, 16), rand_seed=seed).request_queue


def splitwise(q, n_p=1, n_d=1, **kw):
    c = MISTCoordinatorDisagg(request_queue_distr=deepcopy(q), model=MODEL,
                              cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL),
                              num_llm_clients=n_p + n_d, num_prefill_clients=n_p, num_decode_clients=n_d,
                              logging_file=None, **kw)
    c.run_sim()
    return c


def kv(c):
    # Requests a mixed client served end to end never move, so they have no KV transfer time.
    return {r.request_id: r.metrics.kv_transfer_time for r in c.completed_requests
            if r.metrics.kv_transfer_time is not None}


def kv_bytes(c):
    """Bytes moved prefill->decode, recomputed from the finished request."""
    from GenZ.Models import get_configs
    cfg = get_configs(MODEL)
    return {r.request_id: math.ceil(r.input_len * cfg.get_kv_size()) for r in c.completed_requests}


# 1. Isolated transfers are timed exactly: latency + bytes / bandwidth (ASTRA-sim reads GB/s as GiB/s).
BW, LAT = 25, 500
q = queue(rps=0.5, sim_time=10000)
c = splitwise(q, astrasim_topology=topo("Ring", 4, BW, LAT))
ref = splitwise(q)
got = kv(c)
exp = {i: (LAT + b / (BW * 2**30 / 1e9)) / 1e6 for i, b in kv_bytes(c).items()}
err = max(abs(got[i] - exp[i]) / exp[i] for i in got)
check("isolated transfer = latency + size/BW", c.request_serviced == len(q) and err < 1e-3,
      f"n={len(q)} max_rel_err={err:.2e} mean_kv_ms={sum(got.values())/len(got):.4f}")
check("prefill side unchanged (TTFT)", abs(c.get_global_stats().TTFT - ref.get_global_stats().TTFT) < 1e-9,
      f"TTFT astra={c.get_global_stats().TTFT:.3f} analytical={ref.get_global_stats().TTFT:.3f}")
check("no transfer left in flight", not c._in_flight and c.network.next_event_ms() is None)

# Causality probe: no coordinator event may be pushed earlier than the latest one popped.
# (Coordinator heap entries are (time, EventType, items); other heaps are ignored.)
import heapq as _hq
import mist.Coordinator.MIST_Coordinator as _M
_late, _now = [], [0.0]
_push, _pop = _hq.heappush, _hq.heappop
_is_event = lambda e: isinstance(e, tuple) and len(e) == 3 and isinstance(e[1], _M.EventType)
def _p(h, e):
    if _is_event(e) and e[0] < _now[0] - 1e-9:
        _late.append((_now[0] - e[0], e[1]))
    _push(h, e)
def _o(h):
    e = _pop(h)
    if _is_event(e):
        _now[0] = max(_now[0], e[0])
    return e
_M.heapq.heappush, _M.heapq.heappop = _p, _o

# 2. Contention: same trace through a thin link -> KV time grows with load; never below isolated time.
thin = topo("Ring", 4, 2, LAT)
prev = 0
for rps in (1, 10, 40):
    q = queue(rps=rps)
    _late.clear(); _now[0] = 0.0
    a = splitwise(q, astrasim_topology=thin)
    t = kv(a)
    iso = {i: (LAT + b / (2 * 2**30 / 1e9)) / 1e6 for i, b in kv_bytes(a).items()}
    worst_below = min(t[i] / iso[i] for i in t)
    mean = sum(t.values()) / len(t)
    check(f"causal rps={rps}", not _late, f"events pushed into the past: {len(_late)}")
    _late.clear(); _now[0] = 0.0
    check(f"contention rps={rps}", a.request_serviced == len(q) and worst_below > 1 - 1e-3 and mean >= prev * 0.999,
          f"served={a.request_serviced}/{len(q)} mean_kv_ms={mean:.3f} max_slowdown={max(t[i]/iso[i] for i in t):.2f} "
          f"p99_e2e={a.get_global_stats().T99_latency:.1f}")
    prev = mean

# 2b. Contention is exact: on one Ring hop (0 -> 1), ASTRA-sim is a FIFO link
#     (serialize at BW, then add latency); each direction is its own link. Recompute
#     every finish time by hand from the send times MIST used and compare.
sends = []
_orig_send = _M.MISTCoordinator._network_send
def _rec(self, now, req, src, dst, size, record_kv):
    sends.append((now, size, req.request_id, record_kv, (src, dst)))
    _orig_send(self, now, req, src, dst, size, record_kv)
_M.MISTCoordinator._network_send = _rec
a = splitwise(queue(rps=40), astrasim_topology=thin)
_M.MISTCoordinator._network_send = _orig_send
bw_bytes_per_ms = 2 * 2**30 / 1e3
free, worst = {}, 0.0
got = {r.request_id: r.metrics.kv_transfer_time for r in a.completed_requests}
for sent, size, rid, record_kv, link in sends:  # MIST order; already time-sorted
    start = max(sent, free.get(link, 0.0))
    free[link] = start + size / bw_bytes_per_ms
    if record_kv:
        worst = max(worst, abs(got[rid] - (free[link] + LAT / 1e6 - sent)))
check("contention matches FIFO link by hand", worst < 1e-4, f"transfers={len(sends)} links={sorted(free)} max_abs_err_ms={worst:.2e}")

# 3. Without ASTRA-sim the analytical path still runs and differs only in KV time.
q = queue(rps=10)
an, na = splitwise(q, n_p=2, n_d=2), splitwise(q, n_p=2, n_d=2, astrasim_topology=topo("Switch", 4, 25, LAT))
check("2P+2D Switch: all served both ways", an.request_serviced == na.request_serviced == len(q),
      f"analytical kv={sum(kv(an).values())/len(kv(an)):.3f}ms astra kv={sum(kv(na).values())/len(kv(na)):.3f}ms "
      f"TPOT {an.get_global_stats().TPOT:.3f}/{na.get_global_stats().TPOT:.3f}")

# 4. Determinism.
b1, b2 = (splitwise(q, astrasim_topology=thin) for _ in range(2))
check("deterministic", kv(b1) == kv(b2) and b1.get_global_stats() == b2.get_global_stats())

# 5. Too few NPUs is an error, not a silent wrong mapping.
try:
    splitwise(queue(rps=1, sim_time=500), n_p=3, n_d=3, astrasim_topology=topo("Ring", 4, 25, LAT))
    check("NPU count < clients raises", False)
except ValueError as e:
    check("NPU count < clients raises", True, str(e)[:60])

# 6. Aggregated serving (no inter-client move) is identical with and without ASTRA-sim.
def aggregated(q, **kw):
    c = MISTCoordinator(deepcopy(q), logging_file=None, **kw)
    types = [ClientType.PREFILL, ClientType.DECODE]
    c.add_client(LLMClient(model=MODEL, client_types=types,
                           scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512),
                           cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL)), types)
    c.run_sim()
    return c
q = queue(rps=5)
aggregated(q)  # warm the latency cache: cold and warm runs can differ in the last digits (README)
x, y = aggregated(q), aggregated(q, astrasim_topology=f"{ASTRA_INPUT_DIR}/Ring.yml")
check("aggregated identical", x.get_global_stats() == y.get_global_stats(), f"TTFT={x.get_global_stats().TTFT:.3f}")

# 7. DistServe coordinator: same outcome with and without ASTRA-sim.
def distserve(q, **kw):
    try:
        d = MISTCoordinatorDisagg_DistServe(request_queue_distr=deepcopy(q), model=MODEL,
                                            cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL), **kw)
        d.run_sim()
        return f"served {d.request_serviced}/{len(q)}"
    except Exception as e:
        return repr(e)[:80]
q = queue(rps=5)
r0, r1 = distserve(q), distserve(q, astrasim_topology=f"{ASTRA_INPUT_DIR}/Ring.yml")
if r0.startswith("served"):
    check("DistServe with ASTRA-sim", r1 == r0, f"analytical: {r0} | astra: {r1}")
else:
    # DistServe raises KeyError on main as well, before any transfer; not an ASTRA-sim issue.
    print(f"SKIP  DistServe fails without ASTRA-sim too  analytical: {r0} | astra: {r1}")

# 7b. Two coordinators with their own ASTRA-sim instances do not share event queues.
q = queue(rps=10)
solo = kv(splitwise(q, astrasim_topology=thin))
first = MISTCoordinatorDisagg(request_queue_distr=deepcopy(q), model=MODEL,
                              cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL),
                              num_llm_clients=2, num_prefill_clients=1, num_decode_clients=1,
                              logging_file=None, astrasim_topology=thin)
second = splitwise(q, astrasim_topology=thin)   # built and run after `first` was built
first.run_sim()
check("two ASTRA-sim instances stay separate", kv(first) == solo == kv(second)
      and first.request_serviced == len(q), f"served={first.request_serviced}/{len(q)}")

# 7c. Sub-ns send times and tiny payloads never create an event before the send.
from mist.Global_Network.astrasim_network import AstraSimNetwork
n = AstraSimNetwork(f"{ASTRA_INPUT_DIR}/Ring.yml")
ok = True
for i, t in enumerate((1.0000004, 1.0000004, 2.5, 2.5000001)):
    n.advance(t)
    n.send(t, 1, 0, 1, i)
    ok &= n.next_event_ms() >= t
n.advance(10)
check("sub-ns sends stay causal", ok and n.next_event_ms() is None)

# 8. Bundled trace, paper-style 70B 2P+2D, with and without ASTRA-sim.
M70 = "meta-llama/Llama-2-70B"
trace = Path(mist.__file__).parent / "Input_requests" / "Example_Traces" / "rr_code_10.csv"
reqs = TraceIngestion(ingestion_trace_file=str(trace)).request_queue
res = {}
for name, kw in (("analytical", {}), ("astra", {"astrasim_topology": f"{ASTRA_INPUT_DIR}/Ring.yml"})):
    c = MISTCoordinatorDisagg(request_queue_distr=deepcopy(reqs), model=M70,
                              cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=8, model=M70),
                              num_llm_clients=4, num_prefill_clients=2, num_decode_clients=2, logging_file=None, **kw)
    c.run_sim()
    s = c.get_global_stats()
    res[name] = c
    print(f"      trace {name}: served {c.request_serviced}/{len(reqs)} TTFT={s.TTFT:.1f} TPOT={s.TPOT:.2f} "
          f"p99={s.T99_latency:.1f} kv={sum(kv(c).values())/max(1,len(kv(c))):.3f}ms")
check("trace run both ways", all(c.request_serviced == len(reqs) for c in res.values()))

print(f"\n{failures} failure(s)")
sys.exit(failures)
