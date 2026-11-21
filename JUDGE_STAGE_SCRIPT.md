# ARES × NASA TESS — 55-Second Interactive Judge Stage Script

## Pre-Stage Setup

```bash
.venv/bin/python app/observatory_server.py --port 8765
```

* Open `http://127.0.0.1:8765` on **Tab 1 (`Live Stage Demo (<30s)`)** at `110%–125%` browser zoom.
* Keep the wireless mouse/trackpad ready to hand to a judge at the scenario bar.

---

## Beat 1 — The Hook & Judge Handoff (`0:00 – 0:10`)

> *"This directory holds **2.23 million real 2-minute brightness measurements** from NASA's TESS space telescope, with all NASA labels stripped out. Nine out of ten textbook 'planet dips' in TESS are actually cosmic impostors or blended starfields that trick single-pass AI.*
>
> *Pick any blind NASA sector stream up top—an impostor battle, a crowded starfield trap, or any of the 9 cases in the dropdown—and hit **Run Live ARES**."*

**Stage Action:** Judge clicks Scenario `1`, `2`, `3`, or any dropdown case, then clicks **`▶ Run Live ARES (~18s)`**.

---

## Beat 2 — Narrating the Live 4-Stage Pipeline (`0:10 – 0:28`)

*(Point to the 4 pipeline pills lighting up left-to-right as the live SSE timer ticks.)*

> *"Notice our agents never read the 2.2-million-row haystack—they write the magnets:*
> 1. *`gemini-4-hydrogen` branches parallel SQL hypotheses,*
> 2. *Deterministic DuckDB verifiers check Kepler's 3rd Law, odd/even eclipses, and the raw 11×11 NASA camera pixels,*
> 3. *`gemini-4-carbon` fuses the board against a **4-gate self-check plan** with zero rebuild drift, and*
> 4. *`gemini-decision-1` ablates each field to `'unknown'`—one board variant per API call—to prove what the verdict actually relies on."*

---

## Beat 3 — The Visual Payoff (`0:28 – 0:43`)

### Branch A — If the Judge Picked an Impostor Battle (Scenario `1` or `3`) or Confirmed World
1. Point to the right **Calibrated Verdict** card (`4/4 CERTIFIED`, live runtime vs. Carbon SQL Baseline).
2. Click the **`✗ PRUNED/HOLD`** pill above the chart, then back to **`✓ PLANET`**:
   > *"Look at Panel 1 vs. Panel 2: the impostor actually had a **deeper** brightness dip—but on NASA's CCD chip, the dimming jumped off the target crosshair onto a neighbor star [or split odd/even depths]. Watch the checklist on the right: all 4 physical gates passed, and when we click **Ablate CCD Stamp**, `gemini-decision-1` quantifies exact spatial reliance."*
3. Click **`🧪 Ablate CCD Stamp ("unknown")`** to show the masked counterfactual variant and spatial reliance score, then click **`🔭 Reveal Official NASA Archive Label`**:
   > *"And opening NASA's official archive: ARES rejected NASA's vetted False Positive and locked onto the real Confirmed Planet."*

### Branch B — If the Judge Picked the Crowded Trap (Scenario `2` / `under_determined`)
1. Point to the amber **`HONEST UNCERTIFIED`** verdict and the **`Missing VoI`** bar on Gate #4 (`ccd_centroid_lock`):
   > *"Look at what happened: this star passed **3 out of 4** checks—clean dip, symmetric orbits, and Kepler's 3rd Law. A naive agent calls this a planet. ARES caught `CROWDSAP = 0.57`—43% blended neighbor starlight at TESS's 21-arcsecond pixel limit—refused to hallucinate discovery, and `gemini-decision-1` pinpointed `ccd_centroid_lock` (**VoI = 0.48**) to request ground spectroscopy."*
2. Click **`🔭 Reveal Official NASA Archive Label`**:
   > *"And NASA's official catalog agrees: astronomers hold `TOI-1832.01` as an unconfirmed `PC` awaiting spectroscopy."*

---

## Beat 4 — "Proof It Works (25%)" Closer (`0:43 – 0:55`)

**Stage Action:** Click **Tab 2: `Proof It Works (18-Case Evidence)`** in the top header.

> *"Beyond this live run, we benchmarked **18 cases across two domains**—Real NASA TESS and Software Incident Logs—against an honest single-agent `gemini-4-carbon` baseline with the exact same DuckDB tool: **18/18 accuracy** vs. 88.9% baseline action accuracy, **4× faster**, **2.84× fewer tokens**, **zero ungrounded claims**, and an explicit audit of where TESS's optical resolution hits physical limits."*

---

## Stage Safety Net (If Venue Wi-Fi Drops)

* Click **`⚡ Instant Scan (<1s)`** to run 100% locally offline over the 2.23M-row Parquet shards, render both full and ablated 2×2 NASA PNG cards, and execute the deterministic Invariant I1–I3 self-check in `< 4 seconds`.
* **Tab 2 (`Proof It Works`)** loads all 18 verified benchmark runs locally from disk.
