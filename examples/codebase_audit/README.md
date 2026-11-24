# Codebase Concurrency Audit Example (Non-SQL Custom Backend)

Demonstrates ARES orchestrating a search problem over a non-tabular JSON call/lock graph (`ASTGraphBackend`) with **zero DuckDB or SQL** and **zero external decision module**:

- **Custom `ExecutionBackend` (`ASTGraphBackend`)**: Validates `inspect:<property>` magnets, estimates scan cost, emits `UNCERTIFIED` exploratory `Note`s, and executes `TRUSTED` deterministic verifiers (`verify_reachability`, `verify_lock_inversion`, `verify_race_repro`).
- **Automatic Context Management (`ares.context`)**: Grounds certified/uncertified signals on a bounded `InvestigationBoard` with zero rebuild drift (`final_board_drift == 0.0`).
- **Built-In Evaluation (`DeterministicBoardEvaluator`)**: Separates true 3-of-3 bugs (`SessionPool`) from 2-of-3 decoys (`TokenCache`) and stops honestly (`honest_uncertified`) when deterministic race reproduction is missing (`OrderQueue`).

## Run Offline

```bash
uv run python examples/codebase_audit/run.py
```

Pass `--live` with `GEMINI_API_KEY` configured to run with live `GeminiModelProvider`.
