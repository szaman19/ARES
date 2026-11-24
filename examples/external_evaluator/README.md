# External Evaluator Example (`BoardEvaluator`)

Core ARES (`ares/`) intentionally excludes external decision-probing models and ships with `ares.context.DeterministicBoardEvaluator` by default.

This standalone example shows how a caller can attach any external evaluator satisfying `ares.contracts.BoardEvaluator`—here `DecisionProbeEvaluator` wrapping `DecisionProbeClient` (`gemini-decision-1` counterfactual ablation & Value-of-Information scoring)—without modifying `ares/`:

```python
from ares import LoopConfig, run_investigation_loop
from examples.external_evaluator.decision_probe import DecisionProbeEvaluator

result = run_investigation_loop(
    domain,
    workspace_dir,
    evaluator=DecisionProbeEvaluator(),
    config=LoopConfig(max_rounds=3),
)
```

## Run Offline (Mocked)

```bash
uv run python examples/external_evaluator/run_example.py --mock
```
