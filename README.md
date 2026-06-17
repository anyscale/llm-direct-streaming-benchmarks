# LLM Direct Streaming Benchmarks

This repo is a clean reproduction bundle for the two Ray/vLLM comparison figures:

- `Performance Comparison Across Workloads`
- `Agentic 4p4d P/D Wide-EP Comparison`

It includes the frozen result artifacts needed to replot immediately, plus the
minimal benchmark and serving code needed to rerun the Ray Serve LLM and
vLLM-router trials on a single-node 8x H100-class cluster.

## Quick Start

Recreate both figures from committed artifacts:

```bash
python plots/plot_ray_vllm.py --artifacts artifacts --out figures
```

Validate artifacts and scripts:

```bash
python scripts/validate_artifacts.py --artifacts artifacts
bash -n scripts/*.sh bench/*.sh serve/*/*.sh
python -m compileall scripts plots serve
```

If plotting or dataset generation dependencies are not already installed:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-plot.txt -r requirements-bench.txt
```

Generate or refresh the deterministic agentic dataset:

```bash
NUM_SESSIONS=2200 SEED=42 ./scripts/generate_agentic_dataset.sh
python scripts/validate_dataset.py datasets/agentic_loop_2200s_seed42/dataset.jsonl
```

Generate or refresh the reusable synthetic prompt corpus used by the `8000:50`
replica-scaling trials:

```bash
python scripts/generate_synthetic_dataset.py --workloads 8000:50 --force
```

The large synthetic JSONL corpus is generated on demand rather than committed.

## Benchmark Entry Points

Synthetic 8x replica trials:

```bash
./bench/run_synthetic_8x.sh --backend ray --workloads 8000:50 50:500 --concurrency 64 128 256
./bench/run_synthetic_8x.sh --backend vllm --workloads 8000:50 50:500 --concurrency 64 128 256
```

By default, `run_synthetic_8x.sh` reuses any matching custom JSONL under
`datasets/synthetic_8x/` and falls back to vLLM's random dataset otherwise.
Pass `--no-pregenerated` to force the per-run random prompt path.

Methodology: trials run at maximum throughput (`REQUEST_RATE=inf`) and use a
discard-first / take-second reading. The first run after a server starts pays a
one-time cold start (CUDA graph capture + cache warmup), so the runner sweeps
once to warm the engines (discarded) and then records the second reading on the
warm stack. Pass `--no-warmup` (or `WARMUP=0`) to skip the warm-up pass. Engine
config, routing, and dataset are identical across both stacks, so the only
variable is the serving layer.

Agentic 8x replica trials:

```bash
./bench/run_agentic_8x.sh --backend ray --concurrency 64 128 256
./bench/run_agentic_8x.sh --backend vllm --concurrency 64 128 256
```

These enable fastokens by default via the native `VLLM_USE_FASTOKENS=1` toggle
(vLLM 0.22 swaps in a faster Rust BPE backend). No tokenizer patch is required.

Agentic P/D 4-prefill/4-decode Wide-EP trials:

```bash
./bench/run_agentic_pd4p4d.sh --backend ray --concurrency 64 128 256
./bench/run_agentic_pd4p4d.sh --backend vllm --concurrency 64 128 256
```

Both backends use one engine config across all concurrencies. vLLM-router runs
`--vllm-pd-disaggregation`. Ray Serve LLM uses the public
`ray.serve.llm.build_pd_openai_app` builder with the tokenize-once optimization
(ray-project/ray#64049): the decode stage reuses the prefill stage's prompt
token ids instead of re-tokenizing. The wrapper sets `PD_TOKENIZE_ONCE=1` for
Ray and runs decode with no per-replica max-ongoing cap;
`serve/ray_pd4p4d/launch.py` builds the P/D app with the public
`build_pd_openai_app`.

All benchmark entry points support `--dry-run` to print the resolved serving
and benchmark commands without starting servers.

## Layout

- `artifacts/`: frozen JSON/CSV outputs used by the figures.
- `bench/`: benchmark sweep entry points.
- `configs/`: workload generator config.
- `datasets/`: deterministic generated agentic dataset.
- `plots/`: plotting code for the comparison figures.
- `scripts/`: dataset and validation helpers.
- `serve/`: minimal Ray Serve LLM and vLLM-router launchers.

## Notes

These figures compare Ray Serve LLM and vLLM-router.
