#!/usr/bin/env python3
"""Launch Ray Serve LLM P/D 4p4d using the public build_pd_openai_app builder.

The decode stage reuses the prefill stage's prompt tokenization (tokenize-once,
ray-project/ray#64049). It is on by default (PD_TOKENIZE_ONCE, set to 0 to
disable) and enabled through both gates: the env var forwarded to the decode
replicas and experimental_configs["pd_tokenize_once"]. One engine config is
used across all concurrencies.
"""

from __future__ import annotations

import json
import os
import random
import signal
import socket
import sys
import threading
from typing import Optional

os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)

import ray
from ray import serve
from ray.serve.config import ControllerOptions, RequestRouterConfig
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_pd_openai_app


POLICY_TO_CLASS = {
    "round_robin": "ray.serve.experimental.round_robin_router.RoundRobinRouter",
    "consistent_hash": "ray.serve.experimental.consistent_hash_router.ConsistentHashRouter",
    "pow2": "ray.serve._private.request_router.pow_2_router.PowerOfTwoChoicesRequestRouter",
    "prefix_aware": "ray.serve.llm.request_router.PrefixCacheAffinityRouter",
}

DEFAULT_MODEL = "microsoft/Phi-tiny-MoE-instruct"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


MAX_MODEL_LEN = env_int("MAX_MODEL_LEN", 81920)
GPU_MEMORY_UTILIZATION = env_float("GPU_MEM_UTIL", 0.90)
MAX_NUM_SEQS = env_int("MAX_NUM_SEQS", 128)
MAX_NUM_BATCHED_TOKENS = env_int("MAX_NUM_BATCHED_TOKENS", 16384)
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")
TOKENIZER_MODE = os.environ.get("TOKENIZER_MODE", "auto")
ENABLE_PREFIX_CACHING = env_bool("ENABLE_PREFIX_CACHING", True)
ENABLE_EXPERT_PARALLEL = env_bool("ENABLE_EXPERT_PARALLEL", True)

PREFILL_GROUPS = env_int("PREFILL_GROUPS", env_int("PREFILL_REPLICAS", 1))
PREFILL_DP_SIZE = env_int("PREFILL_DP", 4)
PREFILL_MAX_ONGOING = env_int("PREFILL_MAX_ONGOING", 256)
PREFILL_KV_ROLE = os.environ.get("PREFILL_KV_ROLE", "kv_producer")

DECODE_GROUPS = env_int("DECODE_GROUPS", 1)
DECODE_DP_SIZE = env_int("DECODE_DP", 4)
DECODE_MAX_ONGOING = env_int("DECODE_MAX_ONGOING", 10000)
DECODE_KV_ROLE = os.environ.get("DECODE_KV_ROLE", "kv_consumer")

CHASH_VIRTUAL_NODES = env_int("CHASH_VIRTUAL_NODES", 100)
CHASH_FALLBACK_REPLICAS = env_int("CHASH_FALLBACK_REPLICAS", 2)
ROUTER_INITIAL_BACKOFF_S = env_float(
    "ROUTER_INITIAL_BACKOFF_S", env_float("RAY_LLM_INITIAL_BACKOFF_S", 0.001)
)
ROUTER_MAX_BACKOFF_S = env_float(
    "ROUTER_MAX_BACKOFF_S", env_float("RAY_LLM_MAX_BACKOFF_S", 0.005)
)
ROUTER_BACKOFF_MULTIPLIER = env_float(
    "ROUTER_BACKOFF_MULTIPLIER", env_float("RAY_LLM_BACKOFF_MULT", 1.2)
)

PD_PREWARM = env_bool("PD_PREWARM", False)
# Tokenize-once (ray-project/ray#64049): decode reuses prefill's prompt token ids.
# On by default; drives both gates -- the env var (forwarded to replicas) and
# experimental_configs["pd_tokenize_once"].
PD_TOKENIZE_ONCE = env_bool("PD_TOKENIZE_ONCE", True)


def request_router_config(policy: str) -> RequestRouterConfig:
    kwargs = {}
    if policy == "consistent_hash":
        kwargs = {
            "num_virtual_nodes": CHASH_VIRTUAL_NODES,
            "num_fallback_replicas": CHASH_FALLBACK_REPLICAS,
        }
    return RequestRouterConfig(
        request_router_class=POLICY_TO_CLASS[policy],
        request_router_kwargs=kwargs,
        initial_backoff_s=ROUTER_INITIAL_BACKOFF_S,
        max_backoff_s=ROUTER_MAX_BACKOFF_S,
        backoff_multiplier=ROUTER_BACKOFF_MULTIPLIER,
    )


def deployment_config(
    groups: int, max_ongoing: int, router_config: RequestRouterConfig
) -> dict:
    return {
        "autoscaling_config": {
            "min_replicas": groups,
            "max_replicas": groups,
        },
        "max_ongoing_requests": max_ongoing,
        "request_router_config": router_config,
    }


def runtime_env() -> dict:
    os.environ.setdefault(
        "NIXL_PORT_ALLOC_RUN_ID",
        f"{socket.gethostname()}-{os.getpid()}-{random.randint(0, 10_000_000)}",
    )
    env_vars = {
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "NIXL_PORT_ALLOC_RUN_ID": os.environ["NIXL_PORT_ALLOC_RUN_ID"],
        "RAY_RUNTIME_ENV_HOOK": "ray._private.runtime_env.uv_runtime_env_hook.hook",
        # ray-project/ray#64049: forwarded so the env gate sees it on the decode
        # replicas; on by default (see PD_TOKENIZE_ONCE).
        "PD_TOKENIZE_ONCE": "1" if PD_TOKENIZE_ONCE else "0",
    }
    for key in (
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND",
        "VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "NCCL_DEBUG",
    ):
        if os.environ.get(key):
            env_vars[key] = os.environ[key]
    return {"env_vars": env_vars}


def engine_kwargs(role: str, dp_size: int) -> dict:
    return {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": dp_size,
        "distributed_executor_backend": "ray",
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "tokenizer_mode": TOKENIZER_MODE,
        "enable_prefix_caching": ENABLE_PREFIX_CACHING,
        "trust_remote_code": True,
        "hf_overrides": {
            "max_position_embeddings": MAX_MODEL_LEN,
            "model_max_length": MAX_MODEL_LEN,
        },
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": role,
        },
    }


def controller_options() -> Optional[ControllerOptions]:
    if os.environ.get("RAY_SERVE_CONTROLLER_ENV_MODE", "all") == "none":
        return None
    env_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("RAY_SERVE_")
    }
    if not env_vars:
        return None
    return ControllerOptions(runtime_env={"env_vars": env_vars})


def make_config(
    *,
    groups: int,
    data_parallel_size: int,
    kv_role: str,
    max_ongoing: int,
    router_config: RequestRouterConfig,
) -> LLMConfig:
    model = os.environ.get("MODEL", DEFAULT_MODEL)
    return LLMConfig(
        model_loading_config=ModelLoadingConfig(model_id=model, model_source=model),
        engine_kwargs=engine_kwargs(kv_role, data_parallel_size),
        deployment_config=deployment_config(groups, max_ongoing, router_config),
        runtime_env=runtime_env(),
        experimental_configs={
            "stream_batching_interval_ms": 0,
            "_prewarm_prefill_decode": PD_PREWARM,
            # ray-project/ray#64049 config gate; on by default.
            "pd_tokenize_once": PD_TOKENIZE_ONCE,
        },
    )


def connect_ray_without_runtime_env() -> None:
    if ray.is_initialized():
        return
    ray.init(
        "auto",
        namespace="serve",
        ignore_reinit_error=True,
        runtime_env={},
        logging_level="ERROR",
    )


def main() -> int:
    policy = os.environ.get("ROUTING_POLICY", "consistent_hash")
    if policy not in POLICY_TO_CLASS:
        raise SystemExit(
            f"Unknown ROUTING_POLICY={policy!r}; choices={sorted(POLICY_TO_CLASS)}"
        )

    router_config = request_router_config(policy)
    prefill_config = make_config(
        groups=PREFILL_GROUPS,
        data_parallel_size=PREFILL_DP_SIZE,
        kv_role=PREFILL_KV_ROLE,
        max_ongoing=PREFILL_MAX_ONGOING,
        router_config=router_config,
    )
    decode_config = make_config(
        groups=DECODE_GROUPS,
        data_parallel_size=DECODE_DP_SIZE,
        kv_role=DECODE_KV_ROLE,
        max_ongoing=DECODE_MAX_ONGOING,
        router_config=router_config,
    )

    config_summary = {
        "model": os.environ.get("MODEL", DEFAULT_MODEL),
        "policy": policy,
        "prefill_groups": PREFILL_GROUPS,
        "prefill_dp": PREFILL_DP_SIZE,
        "decode_groups": DECODE_GROUPS,
        "decode_dp": DECODE_DP_SIZE,
        "prefill_max_ongoing": PREFILL_MAX_ONGOING,
        "decode_max_ongoing": DECODE_MAX_ONGOING,
        "prefill_kv_role": PREFILL_KV_ROLE,
        "decode_kv_role": DECODE_KV_ROLE,
        "pd_prewarm": PD_PREWARM,
    }

    if os.environ.get("DRY_RUN") == "1":
        print(
            json.dumps(
                {
                    "summary": config_summary,
                    "prefill_config": prefill_config.model_dump(mode="python"),
                    "decode_config": decode_config.model_dump(mode="python"),
                    "controller_env_keys": sorted(
                        (controller_options() or ControllerOptions())
                        .runtime_env.get("env_vars", {})
                        .keys()
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    connect_ray_without_runtime_env()
    app = build_pd_openai_app(
        {
            "prefill_config": prefill_config,
            "decode_config": decode_config,
        }
    )

    print(f"[launch-pd] {json.dumps(config_summary, sort_keys=True)}", flush=True)
    print(f"[launch-pd] prefill_engine_kwargs={prefill_config.engine_kwargs}", flush=True)
    print(f"[launch-pd] decode_engine_kwargs={decode_config.engine_kwargs}", flush=True)
    options = controller_options()
    controller_env_keys = sorted(
        options.runtime_env.get("env_vars", {}) if options else {}
    )
    print(f"[launch-pd] controller_env_keys={controller_env_keys}", flush=True)

    serve.run(app, controller_options=options)

    stop = threading.Event()

    def signal_handler(*_):
        stop.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    while not stop.wait(5):
        pass
    serve.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
