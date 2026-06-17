#!/usr/bin/env python3
"""Generate reusable vLLM custom JSONL datasets for synthetic sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vllm.benchmarks.datasets.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer


def parse_workload(value: str) -> tuple[int, int]:
    try:
        isl, osl = value.split(":", 1)
        return int(isl), int(osl)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"workload must be ISL:OSL, got {value!r}"
        ) from exc


def parse_range_ratio(value: str) -> float | dict[str, float]:
    value = value.strip()
    if value.startswith("{"):
        loaded: Any = json.loads(value)
        if not isinstance(loaded, dict):
            raise argparse.ArgumentTypeError("range ratio JSON must be an object")
        return {str(k): float(v) for k, v in loaded.items()}
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    *,
    out_dir: Path,
    model: str,
    tokenizer_name: str,
    tokenizer_mode: str,
    trust_remote_code: bool,
    workload: tuple[int, int],
    num_prompts: int,
    seed: int,
    range_ratio: float | dict[str, float],
    prefix_len: int,
    force: bool,
) -> None:
    isl, osl = workload
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"isl{isl}_osl{osl}_n{num_prompts}_seed{seed}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    manifest_path = out_dir / f"{stem}.manifest.json"

    if jsonl_path.exists() and not force:
        raise FileExistsError(f"{jsonl_path} already exists; pass --force to replace")

    tokenizer = get_tokenizer(
        tokenizer_name,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
    )
    dataset = RandomDataset(random_seed=seed)
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_prompts,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=isl,
        output_len=osl,
    )

    prompt_lens = [int(req.prompt_len) for req in requests]
    output_lens = [int(req.expected_output_len) for req in requests]
    records = [
        {
            "id": i,
            "prompt": req.prompt,
            "prompt_tokens": int(req.prompt_len),
            "output_tokens": int(req.expected_output_len),
        }
        for i, req in enumerate(requests)
    ]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "kind": "synthetic_random_vllm_custom",
        "model": model,
        "tokenizer": tokenizer_name,
        "tokenizer_mode": tokenizer_mode,
        "trust_remote_code": trust_remote_code,
        "num_prompts": num_prompts,
        "seed": seed,
        "range_ratio": range_ratio,
        "prefix_len": prefix_len,
        "workload": {"isl": isl, "osl": osl},
        "jsonl": jsonl_path.name,
        "sha256": sha256(jsonl_path),
        "rows": len(records),
        "prompt_tokens": {
            "min": min(prompt_lens),
            "max": max(prompt_lens),
            "total": sum(prompt_lens),
        },
        "output_tokens": {
            "min": min(output_lens),
            "max": max(output_lens),
            "total": sum(output_lens),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-FP8")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out-dir", default="datasets/synthetic_8x")
    parser.add_argument("--num-prompts", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument(
        "--range-ratio",
        type=parse_range_ratio,
        default=0.0,
        help="Float or JSON object with input/output keys. Default: 0.0",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        type=parse_workload,
        default=[(8000, 50)],
        help="One or more ISL:OSL workloads. Default: 8000:50",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tokenizer_name = args.tokenizer or args.model
    for workload in args.workloads:
        write_dataset(
            out_dir=Path(args.out_dir),
            model=args.model,
            tokenizer_name=tokenizer_name,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
            workload=workload,
            num_prompts=args.num_prompts,
            seed=args.seed,
            range_ratio=args.range_ratio,
            prefix_len=args.prefix_len,
            force=args.force,
        )


if __name__ == "__main__":
    main()
