# Frozen Artifact Manifest

These artifacts are sufficient to recreate both Ray/vLLM figures without
rerunning benchmarks.

## synthetic_8x

Synthetic random-prompt 8x replica trials (max throughput, warm-up reading).
Regenerate with `bench/run_synthetic_8x.sh`.

- `ray/`: `ray+ha-opt+ds+async`
- `vllm/`: `vllm-native`
- Workloads: `isl8000_osl50`, `isl50_osl500`
- Concurrency levels: `64`, `128`, `256`

Each point is a `vllm bench serve` JSON file.

## agentic_8x

Agentic multi-turn 8x replica trials. Regenerate with `bench/run_agentic_8x.sh`.

- `ray/`: `ray-consistent_hash`
- `vllm/`: `vllm-consistent_hash`
- Concurrency levels: `64`, `128`, `256`

Each point includes `profile_export_aiperf.csv` and
`profile_export_aiperf.json`.

## agentic_pd4p4d

Model `microsoft/Phi-tiny-MoE-instruct`. Both backends use one config across all
concurrencies: `kv_producer`/`kv_consumer` roles, decode max-ongoing unbounded.
Regenerate with `bench/run_agentic_pd4p4d.sh`.

- Ray c64/c128/c256: public `ray.serve.llm.build_pd_openai_app` builder with the
  tokenize-once optimization (ray-project/ray#64049).
- vLLM c64/c128/c256: stock `--vllm-pd-disaggregation`.

Each point includes `profile_export_aiperf.csv` and
`profile_export_aiperf.json`.
