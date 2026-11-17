"""Plug your own models into MIST: a step-cost model and a routing policy.

1. A step-cost model is a ClusterConfig subclass with two methods:
   get_chunked_time(prefills, decodes) -> (latency_ms, energy) and
   get_max_kv_tokens() -> int. Here: a toy linear model.
2. A routing policy is a MISTCoordinator subclass that overrides
   _determine_dst_client(client_type, request) -> client id. Here: long prompts
   go to one replica, short prompts to the other.
"""
from mist import BatchingMethod, ClusterConfig, LengthVariables, PoissonDistribution, SchedulerConfig
from mist.Client import ClientType, LLMClient
from mist.Coordinator import MISTCoordinator

MODEL = "meta-llama/meta-llama-3.1-8b"


class LinearCost(ClusterConfig):
    """Batch time = fixed overhead + per-token cost. Replace with your own model."""

    def __init__(self, ms_per_prefill_token=0.02, ms_per_decode_token=0.05, overhead_ms=1.0, **kw):
        super().__init__(**kw)
        self.ms_per_prefill_token = ms_per_prefill_token
        self.ms_per_decode_token = ms_per_decode_token
        self.overhead_ms = overhead_ms

    def get_chunked_time(self, prefills, decodes):
        prefill_tokens = sum(r.current_scheduled_prefill for r in prefills)
        latency_ms = (self.overhead_ms + prefill_tokens * self.ms_per_prefill_token
                      + len(decodes) * self.ms_per_decode_token)
        return latency_ms, 0.0          # (latency in ms, energy)

    def get_max_kv_tokens(self):
        return 200_000                  # KV-cache capacity of one replica, in tokens


class SplitByPromptLength(MISTCoordinator):
    """Prompts of 1024+ tokens go to the first client, shorter ones to the second."""

    def _determine_dst_client(self, client_type, request=-1):
        long_client, short_client = self.client_matcher[client_type][:2]
        return long_client if request.input_len >= 1024 else short_client


requests = PoissonDistribution(
    rps=8, sim_time=5_000, rand_seed=0,
    input_vars=LengthVariables(1024, 512), output_vars=LengthVariables(64, 16),
).request_queue

coordinator = SplitByPromptLength(requests, logging_file=None)
types = [ClientType.PREFILL, ClientType.DECODE]
for _ in range(2):
    coordinator.add_client(
        LLMClient(
            model=MODEL, client_types=types,
            scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512),
            cluster=LinearCost(device="H100_GPU", model=MODEL),
        ),
        types,
    )
coordinator.run_sim()

s = coordinator.get_global_stats()
print(f"served {coordinator.request_serviced}/{coordinator.request_accepted} requests")
print(f"TTFT={s.TTFT:.1f} ms  TPOT={s.TPOT:.2f} ms  p99={s.T99_latency:.1f} ms")
