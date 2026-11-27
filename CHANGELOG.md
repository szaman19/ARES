# Changelog

All notable changes to `ares-sdk` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-26

### Added
- **Core State Machine SDK (`ares`)**:
  - Pushdown Automaton + Dependency DAG runtime (`StateMachine`, `TaskContext`) with `Spawn`, `Done`, `Suspend`, and `Fail` transitions, tail-call `then` continuations, `join="any"` competitive racing with recursive subtree cancellation, and `RunBudget` (`max_tasks`, `max_inline_bytes`, `max_tokens`, `max_cost_usd`).
  - Out-of-line `ArtifactStore` with auto-spilling and bounded slice reads (`read_slice`), plus atomic wave-level `CheckpointStore` for crash recovery and suspend/resume.
  - Typed `ares/__init__.py` exports, `__version__`, and PEP 561 `py.typed` marker.
- **CLI (`ares`)**:
  - `run`, `resume`, `inspect` (with `--tree` and `--tail`), `artifact` slice reader, and `--version`.
- **Optional Reusable Extensions (`ares.ext`)**:
  - `ares.ext.llm`: OpenAI-compatible client (`OpenAICompatClient`), schema-validated retries, and local-LLM observability/tracing (`ares.ext.llm.tracing`).
  - `ares.ext.git`: Isolated `GitWorktreeSandbox` helper for patch generation and verification.
  - `ares.ext.viz`: Self-contained interactive HTML slide-deck and DAG trace visualizer.
- **Examples**:
  - `examples/long_horizon_small_context/`, `examples/coarse_to_fine_search/`, `examples/codegen_verify_repair/`, and `examples/repo_fix/`.
