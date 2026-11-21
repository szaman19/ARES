## Multimodal Perception Addendum

### 1. Updated Note & Artifact Schema (`contracts.py`)
Add `ArtifactRef` and `artifacts` to `Note`:
- `ArtifactRef`: `artifact_id: str`, `kind: Literal["plot_png", "image", "table_slice"]`, `path: str`, `sha256: str`, `mime_type: str`, `produced_by_verifier: bool`
- `Note`: add `artifacts: list[ArtifactRef] = []` alongside existing `certification: Literal["CERTIFIED", "UNCERTIFIED"]` and `sources`.

### 2. Invariant Updates (Enforced in `fusion/patch.py`)
- **I1 (Provenance):** Any board field citing visual/multimodal evidence must reference a valid `note_id` and valid `artifact_id`s in `note_store`.
- **I2 (Weakest-Link Certification):** Visual interpretation by an LLM (`hydrogen` / `carbon`) is treated as `UNCERTIFIED` unless backed by a deterministic `TRUSTED` verifier check; merging with a `CERTIFIED` note still yields the weakest certification across all sources.
- **I4 (Verifier Separation):** Verifiers generate artifacts deterministically from Parquet data/shards; board claims never drive verifier pass/fail.
- **I7 (Budget & Size):** Keep multimodal artifacts small, bounded to surviving candidate slices (e.g., top 1–3 candidates per round, max 2–4 images per probe call), and well within `gemini-decision-1`'s 256K-token multimodal limit.

### 3. Probe & Demo Behavior
- **Counterfactual Probe with Multimodal Fields:** When probing a field backed by a multimodal note, run a separate `gemini-decision-1` call where that field's value/artifact is set to explicit `"unknown"` (one variant per call) to measure `influence` of the perceived visual evidence.
- **Live Demo Beat (0:40–2:30):** Show query magnets filtering the multi-GB Parquet haystack $\rightarrow$ deterministic chart/artifact generated for surviving candidates $\rightarrow$ multimodal perception + `gemini-decision-1` counterfactual probe showing which visual/tabular signal the decision relies on.

### 4. Schedule Adjustment (Protected Feature)
- Move **multimodal plot/artifact generation + `DecisionProbeClient` image support** from the cut list into **Wave 1/2 (Hours 2–6)**:
  - Person A's verifier track generates a deterministic PNG plot for surviving candidate slices.
  - Person B's model/probe track attaches the surviving candidate's PNG to `gemini-decision-1` calls (one variant per call) and renders the artifact + influence score in the UI.
- **Revised Cut List (only if behind at Hour 8):**
  1. Shapley random-subset sampling (keep single-field + group `"unknown"` ablation).
  2. Per-round full rebuild drift check (keep patch invariant assertions + end-of-run drift check).
  3. Multi-value VoI simulation across all unknown fields (keep probe `influence` + targeted follow-up task enqueueing).