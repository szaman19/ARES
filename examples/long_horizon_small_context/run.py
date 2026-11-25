"""CLI benchmark comparing ARES PDA vs a rolling-context baseline across horizons."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from ares.llm import OpenAICompatClient
from examples.long_horizon_small_context.machine import (
    BenchmarkSummary,
    MockSmallContextLLM,
    run_benchmark,
)


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 1:
            raise ValueError(f"Horizon must be >= 1, got {value}")
        horizons.append(value)
    if not horizons:
        raise ValueError("At least one horizon must be provided")
    return horizons


def _format_table(summaries: list[BenchmarkSummary], *, mode_label: str) -> str:
    header = (
        f"ARES Long-Horizon Small-Context Benchmark ({mode_label}, "
        f"cap={summaries[0].prompt_char_cap} chars)"
    )
    cols = (
        f"{'H':>4} | {'True Target':<10} | "
        f"{'ARES Acc':<8} | {'ARES Peak':>9} | {'Tasks':>5} | "
        f"{'Baseline Acc':<12} | {'Baseline Peak':>13} | {'Status'}"
    )
    sep = "-" * len(cols)
    lines = [header, sep, cols, sep]
    for s in summaries:
        ares_acc = "PASS" if s.ares_correct else "FAIL"
        base_acc = "PASS" if s.baseline_correct else ("OVERFLOW" if s.baseline_overflowed else "FAIL")
        base_status = "ContextLengthExceededError" if s.baseline_overflowed else "OK"
        lines.append(
            f"{s.hops:>4} | {s.true_candidate:<10} | "
            f"{ares_acc:<8} | {s.peak_prompt_chars:>9} | {s.total_tasks:>5} | "
            f"{base_acc:<12} | {s.baseline_peak_prompt_chars:>13} | {base_status}"
        )
    lines.append(sep)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare ARES O(1) PDA state machine vs rolling-context baseline."
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use deterministic offline small-context mock LLM (default: True unless --base-url is set).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL when running with --no-mock.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Model name for OpenAICompatClient when running with --no-mock.",
    )
    parser.add_argument(
        "--horizons",
        default="4,8,12",
        help="Comma-separated hop depths to benchmark (default: 4,8,12).",
    )
    parser.add_argument(
        "--context-cap",
        type=int,
        default=2500,
        help="Strict per-call prompt character limit (default: 2500).",
    )
    parser.add_argument(
        "--noise-lines",
        type=int,
        default=25,
        help="Log lines generated per hop (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for multi-hop case generation (default: 42).",
    )
    parser.add_argument(
        "--max-inline-bytes",
        type=int,
        default=600,
        help="ArtifactStore inline spill threshold in bytes (default: 600).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint running the horizon sweep and printing the comparison table."""
    parser = build_parser()
    args = parser.parse_args(argv)
    horizons = _parse_horizons(args.horizons)

    use_mock = (args.base_url is None) if args.mock is None else bool(args.mock)

    if use_mock:
        mode_label = "mock"
        llm_factory = lambda: MockSmallContextLLM()  # noqa: E731
    else:
        base_url = args.base_url or "http://localhost:11434/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
        mode_label = f"live:{args.model}@{base_url}"
        llm_factory = lambda: OpenAICompatClient(  # noqa: E731
            base_url=base_url,
            model=args.model,
            api_key=api_key,
        )

    summaries = [
        run_benchmark(
            hops=h,
            seed=args.seed,
            noise_lines=args.noise_lines,
            prompt_char_cap=args.context_cap,
            max_inline_bytes=args.max_inline_bytes,
            llm_factory=llm_factory,
        )
        for h in horizons
    ]
    print(_format_table(summaries, mode_label=mode_label))
    return 0 if all(s.ares_correct for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
