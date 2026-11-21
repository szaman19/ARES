Below is a complete, execution-ready **ARES Hackathon Engineering Plan** formatted as a specification document (`ARES_PLAN.md`) you can drop directly into your repository and hand to your coding agents.

---

# ARES Hackathon Implementation Plan & Agent Spec

**Status:** Execution-ready specification for a 12-hour build by two human orchestrators managing parallel coding agents.

**Default Domain:** Software Incident Root-Cause Analysis (Parquet logs, metrics, and traces) — chosen at Hour 0 unless an astronomy dataset + domain verifier sanity check is pre-staged before Hour 4. The harness remains strictly domain-agnostic via the `Domain` protocol.

---

## 1. Core Thesis & Pitch

A single frontier model equipped with SQL tools can search large datasets iteratively—until hypotheses branch across multiple dimensions, plausible look-alike decoys appear, or high-stakes decisions require knowing **which facts are deterministically verified versus inferred, and what the conclusion actually hinges on**.

* **The Pitch:** The agents never read the haystack; they write the magnets. Parallel worker models compile hypotheses into structured queries executed by a deterministic engine over multi-GB local Parquet shards (1–3 GB, ~250×–750× a ~1M-token / ~4 MB context window). Results are certified deterministically, fused into a bounded investigation board, and stress-tested via counterfactual probing to measure decision reliance and select the highest value-of-information (VoI) follow-up.
* **Key Value Drivers over a Single Tool-Using Agent:**
1. **Parallel Hypothesis Breadth:** Simultaneous exploration of distinct failure families under a fixed wall-clock budget, with deterministic verifiers pruning dead ends without spending extra reasoning tokens.
2. **Decoy Resistance & Counterfactual Reliance:** Separating a true multi-condition cause from a convincing 2-of-3 look-alike decoy by probing how much the decision collapses when individual evidence fields are ablated or perturbed.
3. **Provenance & `UNCERTIFIED` Discipline:** Strict separation between `CERTIFIED` deterministic scan outputs and `UNCERTIFIED` inferences—surfacing incomplete or unverified conclusions honestly rather than hallucinating certainty.



---

## 2. Judging Criteria Alignment

| Criterion | Weight | Deliverable |
| --- | --- | --- |
| **Live Demo** | **35%** | Judge selects a needle (fault type, service, onset window, subtlety) from a live menu; harness appends a fresh Parquet shard containing the needle and executes end-to-end on stage (no cloud/network dependency). |
| **Proof It Works** | **25%** | 1) Fast 9-case micro-benchmark (go/no-go gate at Hour 3.5); 2) Full ~20-needle benchmark vs. an honest baseline (`gemini-4-carbon` + identical DuckDB SQL tool + identical wall-clock cap); 3) Influence vs. true information value correlation; 4) Decision 1 confidence calibration plot; 5) Honest `UNCERTIFIED` / decoy outcomes shown, not hidden. |
| **Creativity** | **20%** | Counterfactual probing of structured investigation state (`"unknown"` ablation + value perturbation) + VoI acquisition loop + end-to-end certification flow from raw Parquet scans to final decision. |
| **Gemini 4 + Decisions API** | **20%** | Three models in distinct roles (§4). `gemini-decision-1` used for calibrated closed-set judgments, candidate triage, and counterfactual board variants (one board variant per API call). |

---

## 3. Architecture & Strict Boundaries

### 3.1 Three-Layer Architecture

1. **ARES Library (Unchanged Core + General Extensions):**
Stateless processors, task DAG, propose–verify–refine execution, verifier registry, certification/taint tracking, immutable note store, and grep/query utilities. No global controller or domain logic.
2. **Fusion App (External Orchestration Layer):**
Maintains the bounded JSON investigation board (`board_version`), applies validated patches as notes arrive, enforces provenance invariants, and communicates with ARES exclusively through the narrow waist (`TaskSpec` $\rightarrow$ `solve()` $\rightarrow$ `RunReport` + `Note` store).
3. **Decision Probe (Inside Fusion App):**
Calls `gemini-decision-1` against the active board for closed-set answers and confidence, then runs counterfactual board variants (setting fields/groups to explicit `"unknown"` or perturbed values) to compute field `influence`, `uncertified_influence_share`, and `voi` for next-step task enqueueing.

### 3.2 Narrow-Waist Boundary Rule

> *Add functionality to ARES only if any generic calling harness would want it.*

| Lives in ARES Library | Lives in Fusion App |
| --- | --- |
| `solve()`, `TaskSpec`, `RunReport` | Investigation board schema & versioned state |
| Task DAG, scheduler, stateless processors | Carbon patch-merge, deduplication & archiving |
| `VerifierRegistry`, certification & taint | Decision 1 counterfactual probe & VoI picker |
| Immutable `Note` store & query tools | Mapping note certification $\rightarrow$ board field status |
| Callbacks, `EXTERNAL` hooks | `Domain` plug-ins, benchmark runner, UI |
| `Gemini` `ModelProvider` adapter | — |

* **Allowed ARES library work (Hours 0–2):** `Gemini` `ModelProvider` adapter, parallel dispatch (`max_concurrent`), streaming/progress callbacks, extending an existing graph via `solve(graph=...)`, and populated `UsageReport`.
* **Forbidden in ARES library:** Any state object, fusion hook, or processor that knows what an investigation board is.

### 3.3 Hard Invariants (Enforced Deterministically in `fusion/patch.py`)

Every board mutation must pass `validate_and_merge_patch(board, patch, note_store)`:

1. **I1 — Strict Provenance & Value Integrity:** Every non-unknown board field must cite $\ge 1$ existing `note_id` in `note_store`. A deterministic check verifies cited notes exist and certified values match note evidence. Fusion cannot invent ungrounded claims.
2. **I2 — Monotonic / Weakest-Link Certification:** Fusion cannot upgrade certification. A board field has `status = "certified"` **if and only if** all cited `sources` have `certification == "CERTIFIED"`. Merging/deduplicating fields takes the union of `sources` and the **weakest** certification among them.
3. **I3 — Notes Are Source of Truth:** Notes are immutable and append-only. `rebuild_board_from_notes(note_store)` deterministically reconstructs the board; `board_drift` between the patched board and a from-scratch rebuild is logged per run.
4. **I4 — Context vs. Verifier Separation:** Board state passed into a `TaskSpec` is proposal context only—never a verifier input. `TRUSTED` verifiers execute strictly against Parquet shards and deterministic rules.
5. **I5 — Staleness Tracking:** Every `TaskSpec` and resulting `Note` records `requested_under` (`board_version` at dispatch). Fusion detects stale arrivals when `current_board_version > requested_under`.
6. **I6 — Root Guards & Deduplication:** Automated tasks run under ARES root guards (query safety, scan byte/row caps) and cannot elevate permissions. Content-hash deduplication skips identical normalized hypothesis/query specs.
7. **I7 — Bounded Active Board:** Low-influence fields (`influence < 0.02` for $\ge 2$ rounds) are archived out of the active probe board (while remaining in `note_store`) to keep the active board compact (target $\le 12$ active fields, well under the 256K-token limit) and probe latency low.

---

## 4. Model Roles & API Constraints

| Role | Model ID | Responsibilities |
| --- | --- | --- |
| **Swarm Workers** | `gemini-4-hydrogen` | Parallel hypothesis generation inside ARES `solve` tasks; compiles hypotheses into constrained DuckDB SQL queries with estimated scan cost; summarizes surviving rows into structured notes. |
| **Fusion & Planner** | `gemini-4-carbon` | Merges new notes into JSON board patches, deduplicates findings, preserves citations, proposes new hypothesis branches, and serves as the single-agent SQL baseline. |
| **Decision Probe** | `gemini-decision-1` | Calibrated closed-set judgments (`0–1` confidence), batch candidate triage, counterfactual reliance probing, and value-of-information (VoI) scoring. |

**Decisions API Engineering Constraints:**

* Supports closed-set questions (yes/no, multiple choice, bounded score), up to 1,000 independent questions against one shared input, up to 256K tokens input, and `0–1` confidence per answer.
* **One board state per call:** Each counterfactual variant of the board is a separate API call. Do **not** pack multiple counterfactual board variants into a single input prompt.
* **Explicit `"unknown"`:** When ablating a field in counterfactual probing, set its value to `"unknown"` rather than deleting the JSON key so the model does not fill in priors silently.

---

## 5. Core Execution Loop & Data Contracts

### 5.1 Investigation Board Schema (`contracts.py`)

```json
{
  "board_version": 14,
  "decision": {
    "question_id": "root_cause",
    "question": "Which component is the root cause?",
    "answer": "payments-db",
    "confidence": 0.91,
    "certified_influence_share": 0.88,
    "uncertified_influence_share": 0.12,
    "unknown_voi_max": 0.14
  },
  "fields": {
    "first_conn_error_ts": {
      "value": "2026-11-21T14:03:07Z",
      "status": "certified",
      "sources": ["n_0042", "n_0057"],
      "influence": 0.38,
      "voi": 0.0,
      "requested_under": 9,
      "last_updated_version": 12,
      "stale_rounds": 0
    },
    "pool_saturation": {
      "value": "unknown",
      "status": "unknown",
      "sources": [],
      "influence": 0.0,
      "voi": 0.41,
      "requested_under": 13,
      "last_updated_version": 13,
      "stale_rounds": 0
    }
  },
  "archived_fields": []
}

```

### 5.2 Eight-Step Loop

1. **Seed:** `Domain` supplies `schema_description()`, `decision_questions()`, `hypothesis_families()`, and registers `TRUSTED` DuckDB verifiers. Initialize `board_version = 0`.
2. **Propose (Parallel, Bounded):** Dispatch $K = 4\text{–}6$ `gemini-4-hydrogen` workers in parallel ARES `solve` tasks. Each outputs a `HypothesisQuerySpec` (SQL query + target hypothesis family + expected signal). Validate SQL safety/read-only rules, estimate rows/bytes scanned via DuckDB `EXPLAIN`, and deduplicate by normalized query hash.
3. **Verify (Deterministic, Cheapest-First):** Sort valid queries by estimated scan cost ascending. Execute registered `TRUSTED` DuckDB verifiers over local Parquet shards. Emit immutable `Note` objects stamped `CERTIFIED` (passes deterministic verifier rules/thresholds) or `UNCERTIFIED` (raw/unverified summary or partial match).
4. **Fuse:** `gemini-4-carbon` proposes a JSON patch from new notes; `validate_and_merge_patch()` enforces Invariants I1–I7, merges duplicate facts (union of `sources`, weakest certification), increments `board_version`, and logs `board_drift`.
5. **Decide:** Call `gemini-decision-1` on the current active board for all closed-set decision questions + batch triage of surviving candidate entities/hypotheses.
6. **Probe (Adaptive Counterfactual Reliance):**
* Run **group-level and single-field `"unknown"` ablations** (setting target fields to `"unknown"`) and **value perturbations** as separate Decision 1 calls (2 shuffled field-order repeats when confidence is contested; single pass when clear).
* Compute normalized `influence` per field and `uncertified_influence_share`.


7. **Acquire (VoI Next-Step Selection):**
* For `"unknown"` or low-certainty fields, evaluate Decision 1 across 2–3 plausible values to compute expected decision swing (`voi`).
* Enqueue targeted ARES `solve` tasks aimed at resolving the highest-`voi` fields.


8. **Terminate:**
* **Stop (Certified Success):** `confidence >= 0.85` AND `uncertified_influence_share <= 0.15` AND `unknown_voi_max < 0.20`.
* **Stop (Honest `UNCERTIFIED` / Budget Cap):** Max rounds (`max_rounds = 6`) or wall-clock cap reached without clearing thresholds; output best answer alongside explicit `UNCERTIFIED` / ambiguous status and blocking fields.



---

## 6. Domain Plug-in Interface (`domain/protocol.py`)

```python
from typing import Protocol

class Domain(Protocol):
    name: str

    def generate(self, out_dir: str, size_gb: float, seed: int) -> None:
        """Write background haystack as partitioned Parquet shards."""

    def plant(self, out_dir: str, needle: "NeedleSpec") -> "GroundTruth":
        """Append a fresh Parquet shard containing the needle (used in benchmark & live demo)."""

    def needle_menu(self) -> list["NeedleSpec"]:
        """Return selectable needle definitions for the live judge menu."""

    def schema_description(self) -> str:
        """Tables, columns, join keys, and semantics for Hydrogen/Carbon prompts."""

    def register_verifiers(self, registry: "VerifierRegistry") -> None:
        """Register deterministic TRUSTED DuckDB verifiers (no LLM calls)."""

    def decision_questions(self) -> list["DecisionQuestion"]:
        """Closed-set questions the board must answer."""

    def hypothesis_families(self) -> list[str]:
        """Initial failure/signal families for parallel swarm exploration."""

    def score(self, board: dict, truth: "GroundTruth") -> dict:
        """Evaluate final board answer, certification accuracy, and candidate narrowing."""

```

### Primary Domain: Software Incident Root Cause (`domain/incident_logs.py`)

* **Haystack:** Partitioned Parquet tables (`service_logs`, `metrics`, `traces`) generated deterministically across shards (1–3 GB target scale).
* **Compound Needles:** Faults visible only by correlating **3 weak signals across services within a narrow time window** (e.g., upstream retry spike + auth/pool latency drift + downstream DB lock timeout on shared `trace_id`/`shard_id`), alongside **2-of-3 look-alike decoys** and **incomplete/under-determined incidents**.
* **Trusted Verifiers:** Deterministic DuckDB SQL verifiers checking cross-service join counts, time-window alignments, baseline z-score/threshold breaches, and minimum sample support.
* **Decision Questions:** (1) Root-cause component (multiple choice), (2) Recommended action (`rollback` / `restart` / `scale_pool` / `escalate`), (3) Is the root cause sufficiently verified (`yes` / `no`).

---

## 7. Two-Stage Benchmark & Proof Plan

### Stage 1: Minimal Pre-Demo Micro-Benchmark (Gate at Hour 3.5)

Run on a fast **9-case synthetic dataset** (~100K–250K rows across Parquet shards; executes in < 3 minutes total):

* **3× Multi-Clue Join Cases:** True needle requires aligning 3 weak signals across tables/windows.
* **3× Look-Alike Decoy Cases:** A high-volume decoy matches 2 of 3 clues; true needle is subtler.
* **3× Under-Determined Cases:** Missing/unverified third clue—must terminate as `UNCERTIFIED` / low confidence rather than false certainty.

**Go / No-Go Gate:**

* Zero invariant violations (`0` ungrounded fields, `0` illegal certification upgrades).
* Pivotal clues receive higher counterfactual `influence` than distractor fields.
* ARES outperforms or differentiates clearly from the single-agent SQL baseline on decoy resistance and false-certainty avoidance.

### Stage 2: Full Benchmark (~20 Planted Needles, Hours 7–11)

* **Honest Baseline:** `gemini-4-carbon` with the exact same DuckDB read-only SQL tool, schema prompt, and identical per-case wall-clock budget.
* **Logged Metrics:**
1. Needle accuracy & decoy-avoidance rate (ARES vs. Baseline).
2. Wall-clock time to correct conclusion, total tokens, model calls, and SQL queries executed.
3. Fraction of candidate hypotheses pruned by deterministic verifiers before further LLM calls.
4. Spearman/Pearson correlation between counterfactual `influence` and ground-truth information value (candidate-set reduction).
5. Reliability / calibration curve of `gemini-decision-1` confidences on benchmark tasks.
6. Patched-vs-rebuilt `board_drift` and final `uncertified_influence_share`.
7. At least one explicit `UNCERTIFIED` / look-alike failure case rendered transparently in the UI.



---

## 8. Agent-Orchestrated 12-Hour Schedule

Two humans act as **system architects, contract gatekeepers, and diff/test reviewers**, each managing **2–3 isolated Git worktree coding agents** in synchronized waves.

| Window | Wave / Gate | Person A (Data, ARES, Verifiers, Benchmark) | Person B (Adapters, Fusion, Probe, UI) |
| --- | --- | --- | --- |
| **0–1h** | **Wave 0: Contract Lock** | Author `contracts.py`, `Domain` protocol, and `pytest` skeleton; verify DuckDB Parquet scan speed. | Author board invariant rules, Gemini/Decisions API client interfaces; verify live model IDs & rate limits. |
| **1–3h** | **Wave 1: Parallel Build** | **Agent A1:** 9-case micro-benchmark Parquet generator + needle planter.<br>

<br>**Agent A2:** ARES parallel dispatch (`max_concurrent`), callbacks, `solve(graph=...)`, and DuckDB `TRUSTED` verifier runner. | **Agent B1:** `Gemini` `ModelProvider` (`carbon`, `hydrogen`) + single-variant `gemini-decision-1` probe client.<br>

<br>**Agent B2:** `validate_and_merge_patch()`, dedup, weakest-cert logic, `rebuild_board_from_notes()`, and unit tests. |
| **3–4h** | **Gate 1: Micro-Benchmark** | Run 9-case micro-benchmark + baseline; verify decoy resistance, probe sensitivity, and zero invariant failures. Lock domain. | Integrate loop controller (`seed -> propose -> verify -> fuse -> decide -> probe -> acquire -> stop`) on micro-benchmark. |
| **4–7h** | **Wave 2: Scale & Probe/UI** | **Agent A1:** Full Incident Logs Parquet generator (1–3 GB shards) + live judge needle menu.<br>

<br>**Agent A2:** Full `TRUSTED` SQL verifier suite + `score()` + baseline benchmark runner. | **Agent B1:** Adaptive counterfactual probe (`"unknown"` ablation + perturbation) + VoI task enqueueing.<br>

<br>**Agent B2:** Live demo UI (board table, `CERTIFIED`/`UNCERTIFIED` badges, influence heatmap, VoI pick, query/candidate counters, needle menu). |
| **7–9h** | **Wave 3: Full Benchmark** | Execute ~20-needle benchmark + Carbon SQL baseline; tune query concurrency and scan guards. | Wire live telemetry, active-board archiving (I7), and benchmark/calibration charts directly into the UI. |
| **9–11h** | **Metrics & Hardening** | Generate calibration plot, influence-vs-VoI correlation, and honest `UNCERTIFIED` failure case artifacts. | Polish UI transitions, shard-append trigger, and error/timeout handling for stage reliability. |
| **11–12h** | **Rehearsal & Freeze** | Run unseen judge-picked needle + shard-append drills; freeze backend code at Hour 11.5. | Run 3-minute timed demo rehearsals matching exact stage beats. |

### Strict Cut List (Top-Down if Behind Schedule)

1. Shapley random-subset sampling (keep group + single-field `"unknown"` ablation).
2. Per-round full rebuild drift check (keep patch invariant assertions + end-of-run drift check).
3. Multi-value VoI simulation (keep probe `influence` + priority dispatch over hypothesis families).
4. Multimodal attachments to Decision 1 (keep structured JSON board input).

* **Never Cut:** Honest Carbon + SQL baseline, `CERTIFIED` vs. `UNCERTIFIED` badges/logic on the board, and live judge-selected needle shard append.

---

## 9. Immediate Kickoff Prompts for Your Coding Agents (Wave 1)

Once `contracts.py` is committed in Hour 0–1, dispatch these four isolated agent tasks in parallel worktrees:

* **Agent A1 (`worktree-data`):** *"Implement `domain/micro_bench.py` conforming to `Domain` in `contracts.py`. Generate deterministic Parquet shards (`service_logs`, `metrics`, `traces`) for 9 test cases (3 multi-clue 3-way joins, 3 two-of-three look-alike decoys, 3 under-determined incidents) plus `plant()` shard appending and `score()`. Include `pytest` tests verifying ground truth isolation."*
* **Agent A2 (`worktree-ares-verifiers`):** *"Implement DuckDB `TRUSTED` query execution with `EXPLAIN` cost estimation, read-only SQL validation, content-hash deduplication, and `max_concurrent` worker dispatch in ARES `solve(graph=...)`. Emit immutable `Note` records with `CERTIFIED`/`UNCERTIFIED` status. Include `pytest` tests."*
* **Agent B1 (`worktree-models`):** *"Implement `GeminiModelProvider` for `gemini-4-hydrogen` and `gemini-4-carbon`, plus `DecisionProbeClient` for `gemini-decision-1` enforcing one board state per API call and explicit `'unknown'` field ablation. Include mocked and live smoke tests."*
* **Agent B2 (`worktree-fusion`):** *"Implement `fusion/patch.py` with `validate_and_merge_patch()`, weakest-link certification enforcement, citation checks against `note_store`, stale-version tracking (`requested_under`), active-board archiving (I7), and `rebuild_board_from_notes()` drift calculation. Achieve 100% unit test pass rate on Invariants I1–I7."*