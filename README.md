<div align="center">

# MIST

<h4>Multi-stage AI Inference Simulation Toolkit</h4>

*Co-design LLM serving across hardware, schedulers and networks — in seconds, on a laptop.*

Install with `pip install -e .` · one-file quickstart · plug in your own cost model or router by overriding one method

[![tests](https://github.com/astra-sim/MIST/actions/workflows/tests.yml/badge.svg)](https://github.com/astra-sim/MIST/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](#install)
[![paper](https://img.shields.io/badge/paper-SC%20'26-8a2be2)](#citation)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

## Latest news

- **ASTRA-sim network backend.** Inter-client transfers (for example, the KV
  cache moving from prefill to decode) can now go through ASTRA-sim's
  congestion-aware network model, so concurrent transfers share links. Turn it
  on with one argument; see [Network backends](#network-backends).
- **Paper naming.** Code names now match the SC '26 paper (*client*,
  *cluster*, *Stage-Push*, *Client-Step*). Old names still work for one release;
  see [CHANGELOG.md](CHANGELOG.md).
- **SC '26.** MIST is the artifact for *"MIST: A Co-Design Framework for
  Heterogeneous, Multi-Stage LLM Inference."*

## What is MIST?

MIST is an event-driven simulator for LLM inference systems. It answers
questions like:

> *What TTFT, TPOT and throughput would this model get on 2×H100 prefill +
> 4×L40S decode under this trace?*

You describe the requests, the replicas and the hardware. MIST replays the
trace through a coordinator, per-replica schedulers adapted from vLLM, and a
step-cost model for each piece of hardware, and reports latency and
throughput. A run takes seconds on a laptop CPU.

<p align="center"><img src="docs/images/mist_overview.png" width="860" alt="MIST overview: coordinator, per-client schedulers, step-cost sources and outputs"></p>
<p align="center"><em>Dashed boxes are external tools.</em></p>

### Key features

- **Multi-stage requests.** Each request is a pipeline of stages: RAG,
  KV-cache retrieval, prefill, decode. Each stage runs on the client that
  serves it.
- **Aggregated and disaggregated serving.** One replica doing prefill and
  decode, or Splitwise/DistServe-style prefill and decode pools with KV-cache
  transfer between them.
- **Heterogeneous hardware.** Mix GPU types and vendors per pool, each with
  its own tensor/pipeline parallelism and precision.
- **Pluggable step costs.** Regressors trained on profiled vLLM runs, the
  analytical [ECHELON](https://github.com/astra-sim/GenZ-LLM-Analyzer) model
  for hardware that does not exist yet, or interpolated performance traces.
- **Batching policies.** Static, continuous, mixed, chunked and disaggregated
  batching with token, batch-size and KV-cache budgets.
- **Network modelling.** A fast analytical link model by default, or
  [ASTRA-sim](https://github.com/astra-sim/astra-network-analytical) for
  congestion on shared links.
- **Chrome traces.** Every request and client step can be written to a trace
  you open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

### Accuracy (from the paper)

| Metric | Result |
| --- | --- |
| End-to-end latency vs. real deployed systems and Splitwise-sim | within 6% |
| Mean absolute error per stage | 4.2% prefill · 1.9% decode · 3.2% chunked prefill · 5.7% KV retrieval · 4.0% inter-client communication |
| Profiled step-cost predictor | 2.5% average error, 20–50× faster than real hardware |

### What people use it for

| Question | Where to start |
| --- | --- |
| Which hardware mix gives the most tokens per dollar? Under the paper's pricing and hardware assumptions, multi-vendor PD-disaggregated deployments give up to 49.3% more tokens/$ than single-vendor PD disaggregation. | `disaggregated_prefill_decode.py` |
| How does the KV-cache memory hierarchy (HBM, DRAM, SSD) shape tail latency? | `kv_cache_retrieval.py` |
| Chunked or continuous batching for this workload? | `aggregated_serving.py` |
| How much does network congestion slow KV-cache transfer under load? | [Network backends](#network-backends) |

## Install

Requires Python ≥ 3.10 and [Git LFS](https://git-lfs.com) (the profiled vLLM
runtime tables, ~220 MB, are stored in LFS).

```sh
git lfs install
git clone https://github.com/astra-sim/MIST.git
cd MIST
pip install -e ".[test]"
pytest            # ~10 s
```

This installs ECHELON (package `genz_llm`) from its GitHub repository; the
PyPI `genz_llm` release is too old for MIST.

If you only use the analytical `ClusterConfig`, you can skip the large tables
with `GIT_LFS_SKIP_SMUDGE=1 git clone ...` and later fetch what you need with
`git lfs pull --include "mist/Clusters/vllm_runtime_data/<file>.csv"`.

## Quickstart

```python
from mist import BatchingMethod, LengthVariables, ClusterConfig, PoissonDistribution, SchedulerConfig
from mist.Coordinator import MISTCoordinator
from mist.Client import ClientType, LLMClient

model = "meta-llama/meta-llama-3.1-8b"

# 4 req/s for 10 s; all times in MIST are milliseconds.
requests = PoissonDistribution(
    rps=4, sim_time=10_000, rand_seed=0,
    input_vars=LengthVariables(1024, 256),   # mean, std-dev
    output_vars=LengthVariables(128, 32),
).request_queue

coordinator = MISTCoordinator(requests, logging_file=None)
types = [ClientType.PREFILL, ClientType.DECODE]
coordinator.add_client(
    LLMClient(
        model=model,
        client_types=types,
        scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512),
        cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=model, bits="bf16"),
    ),
    types,
)
coordinator.run_sim()

stats = coordinator.get_global_stats()
print(f"TTFT {stats.TTFT:.1f} ms, TPOT {stats.TPOT:.1f} ms, {stats.output_throughput:.0f} output tok/s")
```

Pass `logging_file="trace.json"` to write a Chrome trace.

### Examples

| Script | Shows |
| --- | --- |
| [`aggregated_serving.py`](examples/aggregated_serving.py) | One replica doing prefill + decode; chunked vs continuous batching |
| [`disaggregated_prefill_decode.py`](examples/disaggregated_prefill_decode.py) | Splitwise-style prefill/decode pools with KV transfer, replaying a trace |
| [`kv_cache_retrieval.py`](examples/kv_cache_retrieval.py) | A cache-retrieval stage loading prefix KV from a DRAM/SSD hierarchy |
| [`custom_extensions.py`](examples/custom_extensions.py) | Your own step-cost model and routing policy, in ~30 lines |

## Network backends

Requests move between clients, for example the KV cache from a prefill
replica to a decode replica. MIST times these transfers with one of two
backends. Both are in `main`; you choose per coordinator.

| | Analytical (default) | ASTRA-sim |
| --- | --- | --- |
| Model | latency + size / bandwidth per link | congestion-aware: concurrent transfers queue on shared links |
| Setup | none | build the submodule once ([docs/astrasim.md](docs/astrasim.md)) |
| Turn on | — | `astrasim_topology="<topology>.yml"` |

```python
from mist.Coordinator import MISTCoordinatorDisagg
from mist.Global_Network.astrasim_network import ASTRA_INPUT_DIR

coordinator = MISTCoordinatorDisagg(
    ...,                                               # as before
    astrasim_topology=f"{ASTRA_INPUT_DIR}/Ring.yml",   # omit (or None) for the analytical model
)
```

Without `astrasim_topology`, results are identical to earlier releases.
`python scripts/verify_astrasim.py` checks the ASTRA-sim path end to end.

## Extend MIST

MIST is built to be extended without touching its core. Each layer has one
plug-in point: subclass a class and override one or two methods.

| To change | Subclass | Override |
| --- | --- | --- |
| How long a batch takes (new hardware, your own cost model) | `ClusterConfig` | `get_chunked_time(prefills, decodes) -> (latency_ms, energy)`, `get_max_kv_tokens()` |
| Which replica serves a stage (routing / load balancing) | `MISTCoordinator` | `_determine_dst_client(client_type, request) -> client_id` |
| How inter-client transfers are timed | `MISTCoordinator` | `_get_move_request_time(request, src, dst) -> ms`, or pass `astrasim_topology=` |
| A new kind of client (RAG, KV retrieval, a tool call, ...) | `MISTClient` | `get_machine_runtimes(requests) -> [ms per request]` (simple), or `step(current_time)` (batched); see `KVRetrievalClient` |

A complete cost model is a few lines:

```python
from mist import ClusterConfig

class LinearCost(ClusterConfig):
    def get_chunked_time(self, prefills, decodes):
        prefill_tokens = sum(r.current_scheduled_prefill for r in prefills)
        return 1.0 + 0.02 * prefill_tokens + 0.05 * len(decodes), 0.0   # (ms, energy)

    def get_max_kv_tokens(self):
        return 200_000

# then: LLMClient(..., cluster=LinearCost(device="H100_GPU", model=model))
```

[`examples/custom_extensions.py`](examples/custom_extensions.py) runs this
cost model together with a custom router. Built-in variants to copy from:
`vLLMClusterConfig` and `TraceClusterConfig` (cost models),
`MISTCoordinatorDisagg` (routing), `KVRetrievalClient` (client).

## Architecture

The paper describes MIST in three layers: the **AI Workload layer** (request
traces, each request a pipeline of stages), the **System & Software layer** (a
coordinator plus one scheduler per client) and the **Hardware layer** (one
logical cluster per client, costed by a step-cost source).

```
 Coordinator   event loop (heap of STAGE_PUSH / CLIENT_STEP / NETWORK_* events),
     │         request routing, inter-client transfers
     ▼
 Client        one model instance: LLMClient (prefill/decode),
     │         KVRetrievalClient, RAGClient, host
     ▼
 Scheduler     per-client batching, adapted from vLLM: token / batch-size /
     │         KV budgets; STATIC, CONTINUOUS, MIXED, CHUNKED, DISAGGREGATED
     ▼
 Cluster       cost model: how long a batch takes
               ClusterConfig        analytical (ECHELON)
               vLLMClusterConfig    RandomForest over profiled vLLM step times
               TraceClusterConfig   interpolated performance traces
               MemoryCacheConfig    tiered KV-cache retrieval
```

<details>
<summary><b>Paper terms → code</b></summary>

| Paper term | Code |
| --- | --- |
| Coordinator (load balancer, model mapper, global communicator) | `mist.Coordinator` (`MISTCoordinator`, `global_router`) |
| Stage-Push / Client-Step events | `EventType.STAGE_PUSH` / `EventType.CLIENT_STEP` |
| Client: LLM client, KV client, RAG client | `mist.Client`: `LLMClient`, `KVRetrievalClient`, `RAGClient` |
| Scheduler (request queuer, batch manager, KV manager) | `mist.Scheduler` |
| HW cluster and its runtime method (empirical, analytical, external simulator) | `mist.Clusters`: `vLLMClusterConfig`, `ClusterConfig` (ECHELON), `TraceClusterConfig`, `MemoryCacheConfig` |
| Inter-client network model | `mist.Global_Network` (analytical links, `AstraSimNetwork`) |

Code from versions before 0.2 used *Engine* for client and *Platform* for
cluster (`mist.Engine`, `LLMEngine`, `PlatformConfig`, `engine_types=`,
`platform=`, `coordinator.add_engine`, ...). These names still work for one
release and emit a `DeprecationWarning`; see [CHANGELOG.md](CHANGELOG.md).

</details>

<details>
<summary><b>Package map</b></summary>

| Package | Contents |
| --- | --- |
| `mist.Coordinator` | `MISTCoordinator`, `MISTCoordinatorDisagg` (Splitwise), `MISTCoordinatorDisagg_DistServe`, routers |
| `mist.Client` | `MISTClient` base, `LLMClient`, `KVRetrievalClient`, `ClientMetrics` |
| `mist.Scheduler` | `Scheduler`, `SchedulerConfig`, `BatchingMethod` |
| `mist.Clusters` | cost models above; bundled vLLM profiles in `Clusters/vllm_runtime_data/` |
| `mist.Request`, `mist.Input_requests` | request lifecycle; Uniform/Poisson/Normal/Bursty generators, trace replay |
| `mist.Global_Network` | GPU interconnect specs and the ASTRA-sim backend for transfer cost |
| `mist.Tracing` | Chrome trace JSON writer |

</details>

### Things worth knowing

- **Units:** all times are milliseconds, including replayed trace arrivals.
- **Precision:** `ClusterConfig(bits=...)` defaults to `"bf16"`. Set it to what
  the deployment actually runs (`"fp8"`, `"int8"`, ...).
- **Latency cache:** `ClusterConfig` keeps computed step latencies in
  `~/.cache/mist` (override with `MIST_CACHE_DIR`) and reuses the nearest entry
  within a tolerance. Delete the directory for a cold, bit-reproducible run.
- **Reproducibility:** every request generator takes `rand_seed` and uses its
  own RNG, so parallel sweeps are safe.
- **Logging:** diagnostics go through the standard `logging` module under the
  `mist` logger, e.g. `logging.getLogger("mist").setLevel(logging.DEBUG)`.

## Reproducing the paper

Paper figures and tables are produced by
[MIST-Simulator/SC_Paper_Charts](https://github.com/MIST-Simulator/SC_Paper_Charts),
which installs this package.

> **Status: research code.** The core simulation loop is tested and used for
> the paper, but APIs are not stable and some modules are experimental (see
> [Known limitations](#known-limitations)). Results from older, pre-release
> versions can differ; see [CHANGELOG.md](CHANGELOG.md).

## Roadmap

These are described in the paper or known gaps, and are good places to
contribute:

- Speculative decoding with an EAGLE-style draft stage.
- 3- and 4-way disaggregation.
- DistServe coordinator: it currently raises `KeyError` on the first request.
- Preemption in the scheduler (today, requests whose KV cache can never fit
  are dropped).

## Contributing

Bug reports, fixes and new cluster or client models are welcome. A good
first PR:

- **Plug in a model:** a new cost model, router or client (see
  [Extend MIST](#extend-mist)), with an example.
- **Add a hardware profile:** drop a profiled vLLM table into
  `mist/Clusters/vllm_runtime_data/` (named `<Model>_NVIDIA_<GPU>_<TP>.csv`).
- **Add an example** for a question MIST answers that is not covered above.
- **Pick an item** from the [Roadmap](#roadmap) or the
  [Known limitations](#known-limitations).

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and PR rules (tests,
milliseconds everywhere, flag result-changing PRs in the changelog). Open an
[issue](https://github.com/astra-sim/MIST/issues) for questions.

## Known limitations

- **GenZ version skew.** Current GenZ `db.py` refers to `*QuantMode.bfloat16`
  names that its vendored `aiconfigurator` renamed; constructing an ECHELON
  `NPU` for the profiled-ops devices (`h100_sxm`, `tpu_v6e`, ...) raises
  `AttributeError` until that is fixed upstream. The table-driven devices
  (`H100_GPU`, `A100_40GB_GPU`, `L40S`, ...) are unaffected.
- **Latency cache history.** Because the cache matches within a tolerance,
  results can differ in the last few significant digits between a cold and a
  warm cache.
- **Over-long prompts with non-chunked batching.** A prompt longer than
  `max_num_batched_tokens` is never scheduled and blocks the head of the
  waiting queue. Use `CHUNKED` batching or raise the budget.
- The scheduler has no preemption. Requests whose KV cache can never fit on
  their client are dropped (`FINISHED_IGNORED`) and are not counted as served.
- `vLLMClusterConfig` falls back to the analytical model outside the profiled
  (model, GPU, TP) combinations and warns when it does.
- The RAG client and the DistServe coordinator see far less use than the rest
  of the code; treat their results as experimental.
- The ASTRA-sim backend needs a compiled module; see
  [docs/astrasim.md](docs/astrasim.md).

## Acknowledgments

MIST builds on these projects; thank you to their developers:

- [**vLLM**](https://github.com/vllm-project/vllm): the per-client scheduler is adapted from its scheduler.
- [**ECHELON / GenZ**](https://github.com/astra-sim/GenZ-LLM-Analyzer): analytical step-cost model.
- [**ASTRA-sim**](https://github.com/astra-sim/astra-network-analytical): congestion-aware network backend.

## Citation

If you use MIST, please cite the SC 2026 paper:

```bibtex
@inproceedings{mist_sc26,
  title     = {{MIST}: A Co-Design Framework for Heterogeneous,
               Multi-Stage {LLM} Inference},
  author    = {Bambhaniya, Abhimanyu Rajeshkumar and Wu, Hanjiang and
               Subramanian, Suvinay and Srinivasan, Sudarshan and
               Kundu, Souvik and Yazdanbakhsh, Amir and
               Elavazhagan, Midhilesh and Kumar, Madhu and Yu, Minlan and
               Raychowdhury, Arijit and Krishna, Tushar},
  booktitle = {Proceedings of the International Conference for High
               Performance Computing, Networking, Storage and Analysis
               (SC '26)},
  address   = {Chicago, IL, USA},
  year      = {2026}
}
```

If you use the analytical step-cost backend, also cite ECHELON:

```bibtex
@inproceedings{echelon_iccad26,
  title     = {Analyzing the Impact of Emerging Technology on {AI} Datacenter
               Serving Distributed {LLM} Inference using {ECHELON}},
  author    = {Bambhaniya, Abhimanyu Rajeshkumar and Raj, Ritik and
               Jeong, Geonhwa and Kundu, Souvik and Srinivasan, Sudarshan and
               Subramanian, Suvinay and Elavazhagan, Midhilesh and
               Kumar, Madhu and Krishna, Tushar},
  booktitle = {IEEE/ACM International Conference on Computer-Aided Design
               (ICCAD '26)},
  address   = {San Jose, CA, USA},
  year      = {2026},
  doi       = {10.1145/3831252.3834151}
}
```

## License

MIT; see [LICENSE](LICENSE).
