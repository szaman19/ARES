"""Benchmark runners and evaluation utilities for ARES."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.run_micro_bench import (
        build_baseline_board_for_scoring,
        run_stage1_micro_benchmark,
    )

__all__ = [
    "build_baseline_board_for_scoring",
    "run_stage1_micro_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from benchmark import run_micro_bench

        return getattr(run_micro_bench, name)
    raise AttributeError(f"module 'benchmark' has no attribute {name!r}")
