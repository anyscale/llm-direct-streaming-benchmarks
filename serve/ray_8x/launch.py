#!/usr/bin/env python3
"""Launch an 8x Ray Serve LLM deployment with direct streaming."""

from __future__ import annotations

import json
import os
import signal
import time

from ray import serve
from ray.serve.config import ControllerOptions, RequestRouterConfig
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_openai_app


POLICY_TO_CLASS = {
    "round_robin": "ray.serve.experimental.round_robin_router.RoundRobinRouter",
    "consistent_hash": "ray.serve.experimental.consistent_hash_router.ConsistentHashRouter",
    "pow2": "ray.serve._private.request_router.pow_2_router.PowerOfTwoChoicesRequestRouter",
    "prefix_aware": "ray.serve.llm.request_router.PrefixCacheAffinityRouter",
}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def build_config() -> LLMConfig:
    model = os.environ.get("MODEL", "Qwen/Qwen3-0.6B-FP8")
    policy = os.environ.get("ROUTING_POLICY", "round_robin")
    if policy not in POLICY_TO_CLASS:
        raise SystemExit(f"Unknown ROUTING_POLICY={policy!r}; choices={sorted(POLICY_TO_CLASS)}")

    max_model_len = env_int("MAX_MODEL_LEN", 81920)
    hf_overrides_raw = os.environ.get("HF_OVERRIDES")
    if hf_overrides_raw:
        hf_overrides = json.loads(hf_overrides_raw)
    elif max_model_len > 40960:
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 40960,
            },
            "max_position_embeddings": max_model_len,
        }
    else:
        hf_overrides = None

    router_config = RequestRouterConfig(
        request_router_class=POLICY_TO_CLASS[policy],
        initial_backoff_s=env_float("RAY_LLM_INITIAL_BACKOFF_S", 0.001),
        max_backoff_s=env_float("RAY_LLM_MAX_BACKOFF_S", 0.005),
        backoff_multiplier=env_float("RAY_LLM_BACKOFF_MULT", 1.2),
    )

    engine_kwargs = {
        "tensor_parallel_size": 1,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": env_float("GPU_MEM_UTIL", 0.95),
        "max_num_seqs": env_int("MAX_NUM_SEQS", 128),
        "max_num_batched_tokens": env_int("MAX_NUM_BATCHED_TOKENS", 16384),
        "kv_cache_dtype": os.environ.get("KV_CACHE_DTYPE", "fp8"),
        "tokenizer_mode": os.environ.get("TOKENIZER_MODE", "auto"),
        "enable_prefix_caching": os.environ.get("ENABLE_PREFIX_CACHING", "1") != "0",
    }
    if hf_overrides is not None:
        engine_kwargs["hf_overrides"] = hf_overrides
    if os.environ.get("LOAD_FORMAT"):
        engine_kwargs["load_format"] = os.environ["LOAD_FORMAT"]

    return LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=model,
            model_source=model,
        ),
        engine_kwargs=engine_kwargs,
        deployment_config={
            "autoscaling_config": {
                "min_replicas": env_int("NUM_REPLICAS", 8),
                "max_replicas": env_int("NUM_REPLICAS", 8),
            },
            # Set high so the Serve layer admits requests straight to the engine's
            # continuous batching rather than queueing ahead of it.
            "max_ongoing_requests": env_int("RAY_LLM_MAX_ONGOING", 10000),
            "request_router_config": router_config,
        },
        runtime_env={
            "env_vars": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                # Native fastokens toggle (vLLM 0.22.0): swaps the HF fast-tokenizer
                # Rust BPE backend for a ~25x faster one. Passthrough for the runner.
                "VLLM_USE_FASTOKENS": os.environ.get("VLLM_USE_FASTOKENS", "0"),
            }
        },
        experimental_configs={"stream_batching_interval_ms": 0},
    )


def main() -> None:
    config = build_config()
    controller_env = {
        "RAY_SERVE_SESSION_ID_HEADER_KEY": os.environ.get(
            "RAY_SERVE_SESSION_ID_HEADER_KEY", "x-correlation-id"
        ),
        "RAY_SERVE_HAPROXY_TCP_NODELAY": "1",
    }
    if os.environ.get("RAY_SERVE_HAPROXY_NBTHREAD"):
        controller_env["RAY_SERVE_HAPROXY_NBTHREAD"] = os.environ["RAY_SERVE_HAPROXY_NBTHREAD"]

    if os.environ.get("DRY_RUN") == "1":
        print(json.dumps(config.model_dump(mode="python"), indent=2, default=str))
        print(json.dumps({"controller_env": controller_env}, indent=2))
        return

    app = build_openai_app({"llm_configs": [config]})
    serve.run(
        app,
        controller_options=ControllerOptions(runtime_env={"env_vars": controller_env}),
    )

    stop = False

    def handle_signal(signum, _frame):
        nonlocal stop
        print(f"received signal {signum}; shutting down")
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    while not stop:
        time.sleep(5)
    serve.shutdown()


if __name__ == "__main__":
    main()
