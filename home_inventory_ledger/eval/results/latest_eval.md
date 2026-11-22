# 5 Sigma Evaluation Results (`stranger_test_set`, n=5 unseen clips)

| System | Off-Market Prices >±15% (Wilson CI) | Uncited Point Prices | Duplicate Rows | Contradictions Caught | Item F1 | Verdict Sweep Acc |
|---|---|---|---|---|---|---|
| B0 · Single Gemini 4 Carbon call | 50/73 (68.5%, 95% CI [57%, 78%]) | 73/73 (100.0%, 95% CI [95%, 100%]) | 0/73 (0.0%, 95% CI [0%, 5%]) | 1/1 (100.0%, 95% CI [21%, 100%]) | 0.00 | 12/25 (48%) |
| B1 · Single call + top-1 Shop MCP | 49/73 (67.1%, 95% CI [56%, 77%]) | 0/73 (0.0%, 95% CI [0%, 5%]) | 0/73 (0.0%, 95% CI [0%, 5%]) | 1/1 (100.0%, 95% CI [21%, 100%]) | 0.00 | 11/25 (44%) |
| **Ledger (no asks)** | 0/0 (n/a) | 0/86 (0.0%, 95% CI [0%, 4%]) | 10/86 (11.6%, 95% CI [6%, 20%]) | 0/1 (0.0%, 95% CI [0%, 79%]) | 0.28 | 11/25 (44%) |
| **Ledger (+1 VoI ask round)** | 0/0 (n/a) | 0/86 (0.0%, 95% CI [0%, 4%]) | 10/86 (11.6%, 95% CI [6%, 20%]) | 0/1 (0.0%, 95% CI [0%, 79%]) | 0.28 | 11/25 (44%) |
