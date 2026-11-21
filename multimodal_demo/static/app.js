const state = {
  datasets: [],
  activeDatasetId: null,
  selectedRecordId: null,
  currentInvestigation: null,
};

function esc(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function activeDataset() {
  return state.datasets.find((d) => d.dataset_id === state.activeDatasetId) || null;
}

function useLiveGemini() {
  return Boolean(document.getElementById("liveGeminiToggle").checked);
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data;
}

function renderTabs() {
  const nav = document.getElementById("datasetTabs");
  nav.innerHTML = state.datasets
    .map((ds) => {
      const isAct = ds.dataset_id === state.activeDatasetId ? "active" : "";
      return `
        <button class="ds-tab ${isAct}" data-ds="${esc(ds.dataset_id)}">
          <span>${esc(ds.emoji)} ${esc(ds.short_label)}</span>
          <span class="count-pill">${Number(ds.row_count).toLocaleString()} rows</span>
        </button>
      `;
    })
    .join("");

  nav.querySelectorAll(".ds-tab").forEach((btn) => {
    btn.addEventListener("click", () => selectDataset(btn.dataset.ds));
  });
}

function renderDatasetHero() {
  const ds = activeDataset();
  if (!ds) return;

  document.getElementById("dsDomainBadge").textContent = `${ds.emoji} ${ds.domain}`;
  document.getElementById("dsTitle").textContent = ds.title;
  document.getElementById("dsDecoyDesc").innerHTML =
    `<strong>True Signal:</strong> ${esc(ds.positive_label_description)}<br/>` +
    `<strong>Look-Alike Decoy Trap:</strong> ${esc(ds.decoy_description)}`;
  document.getElementById("dsSourceLink").href = ds.source_url;

  document.getElementById("dsStatsRow").innerHTML = `
    <span class="stat-chip">Rows in DuckDB: <strong>${Number(ds.row_count).toLocaleString()}</strong></span>
    <span class="stat-chip">Verified Local Media: <strong>${ds.staged_media_files}</strong> (${esc(ds.primary_media_kind)})</span>
    <span class="stat-chip">Withheld Decoys: <strong>${Number(ds.decoy_count).toLocaleString()}</strong></span>
    <span class="stat-chip">Local Slice Size: <strong>${ds.size_mb} MB</strong></span>
  `;

  const presetWrap = document.getElementById("presetButtons");
  presetWrap.innerHTML = ds.presets
    .map(
      (p, idx) =>
        `<button class="preset-btn" data-idx="${idx}" title="${esc(p.description)}">✨ ${esc(p.label)}</button>`
    )
    .join("");

  presetWrap.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = ds.presets[Number(btn.dataset.idx)];
      document.getElementById("sqlInput").value = preset.sql;
      executeSearch();
    });
  });

  if (ds.presets.length > 0) {
    document.getElementById("sqlInput").value = ds.presets[0].sql;
  }
}

async function selectDataset(datasetId) {
  state.activeDatasetId = datasetId;
  state.selectedRecordId = null;
  document.getElementById("duelSection").classList.add("hidden");
  document.getElementById("inspectorBody").className = "inspector-empty";
  document.getElementById("inspectorBody").innerHTML =
    `Click any candidate card on the left (or click <strong>⚡ Run Real vs. Decoy Duel</strong>) to run <code>ares.solve()</code> + counterfactual probing.`;
  renderTabs();
  renderDatasetHero();
  await executeSearch();
}

async function executeSearch() {
  const ds = activeDataset();
  if (!ds) return;
  const sql = document.getElementById("sqlInput").value.trim();
  const errBox = document.getElementById("sqlErrorBox");
  const metaSpan = document.getElementById("sqlMeta");
  const gallery = document.getElementById("candidateGallery");

  errBox.classList.add("hidden");
  metaSpan.textContent = "Running DuckDB EXPLAIN & guarded query…";

  try {
    const res = await fetchJson("/api/search", {
      method: "POST",
      body: JSON.stringify({ dataset_id: ds.dataset_id, sql }),
    });
    metaSpan.textContent =
      `${res.row_count} rows in ${res.duration_ms}ms · est_scan=${Number(res.estimated_rows).toLocaleString()} rows · Note ${res.note.note_id} (${res.note.certification})`;
    renderCandidateGallery(ds, res.rows);
  } catch (err) {
    metaSpan.textContent = "";
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
    gallery.innerHTML = "";
  }
}

function renderCandidateGallery(ds, rows) {
  const gallery = document.getElementById("candidateGallery");
  if (!rows.length) {
    gallery.innerHTML = `<div class="muted">0 rows matched this SQL magnet.</div>`;
    return;
  }

  const cols = ds.preview_columns.filter((c) => c !== "record_id");
  gallery.innerHTML = rows
    .map((r) => {
      const rid = String(r.record_id ?? "");
      const selectedCls = rid === state.selectedRecordId ? "selected" : "";
      const thumbHtml = r.image_url
        ? `<img class="cand-thumb" src="${esc(r.image_url)}" alt="${esc(rid)}" loading="lazy" />`
        : `<div class="cand-thumb-placeholder">🌐 Uncached Row<br/>Click to safely fetch &amp; verify media on demand</div>`;

      const kvHtml = cols
        .slice(0, 4)
        .map((c) => {
          if (!(c in r)) return "";
          return `<div class="kv-line"><span>${esc(c)}</span><strong>${esc(r[c])}</strong></div>`;
        })
        .join("");

      const audioBadge = r.audio_url ? `<span class="badge badge-accent">🔊 .wav</span>` : "";

      return `
        <div class="cand-card ${selectedCls}" data-rid="${esc(rid)}">
          ${thumbHtml}
          <div class="cand-body">
            <div class="cand-id" title="${esc(rid)}">${audioBadge} ${esc(rid)}</div>
            ${kvHtml}
          </div>
        </div>
      `;
    })
    .join("");

  gallery.querySelectorAll(".cand-card").forEach((card) => {
    card.addEventListener("click", () => {
      const rid = card.dataset.rid;
      state.selectedRecordId = rid;
      gallery.querySelectorAll(".cand-card").forEach((c) => c.classList.toggle("selected", c.dataset.rid === rid));
      investigateCandidate(rid, []);
    });
  });
}

function renderInvestigationCardHtml(inv, { allowAblationControls = true, showGroundTruth = false } = {}) {
  const probe = inv.probe;
  const board = inv.board;
  const media = inv.media || {};
  const ablatedSet = new Set(inv.ablated_fields || []);
  const mediaAblated = ablatedSet.has("visual_media_evidence");

  const modeBadge =
    probe.mode_used === "live_gemini_decision_1"
      ? `<span class="badge badge-cert">Live gemini-decision-1 (${probe.api_calls} calls)</span>`
      : `<span class="badge badge-uncert">${esc(probe.mode_used)}</span>`;

  let mediaHtml = "";
  if (mediaAblated) {
    mediaHtml = `<div class="cand-thumb-placeholder" style="height:190px;">🙈 <strong>Visual Evidence Ablated (<code>"unknown"</code>)</strong><br/>Image was stripped from the Decision-1 call to measure counterfactual reliance!</div>`;
  } else if (media.image_url) {
    mediaHtml = `<img src="${esc(media.image_url)}" alt="${esc(inv.record_id)}" />`;
  }
  if (media.audio_url) {
    mediaHtml += `<audio controls preload="none" src="${esc(media.audio_url)}"></audio>`;
  }

  const probRows = Object.entries(probe.probabilities || {})
    .map(([opt, p]) => {
      const pct = Math.round(Number(p) * 100);
      return `
        <div class="bar-row">
          <span class="mono" title="${esc(opt)}">${esc(opt)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <span class="mono">${pct}%</span>
        </div>
      `;
    })
    .join("");

  const fieldsHtml = Object.entries(board.fields || {})
    .map(([fname, fobj]) => {
      const infPct = Math.round(Number(fobj.influence || 0) * 100);
      const snap = (probe.counterfactual_snapshots || {})[fname];
      const badgeCls = fobj.status === "certified" ? "badge-cert" : "badge-uncert";
      const checked = ablatedSet.has(fname) ? "checked" : "";
      const checkboxHtml = allowAblationControls
        ? `<label class="mono" style="cursor:pointer;font-size:0.77rem;"><input type="checkbox" class="ablate-cb" value="${esc(fname)}" ${checked}/> Ablate → "unknown"</label>`
        : "";
      const snapHtml = snap
        ? `<div class="muted" style="font-size:0.76rem;">If set to <code>"unknown"</code> → answer becomes <code>${esc(snap.answer_when_ablated)}</code> (TVD swing: <strong>${snap.tvd_swing}</strong>)</div>`
        : "";

      return `
        <div class="field-item">
          <div class="field-head">
            <div>
              <strong>${esc(fname)}</strong>
              <span class="badge ${badgeCls}">${esc(fobj.status)}</span>
              <span class="mono muted">sources=${esc((fobj.sources || []).join(",") || "none")}</span>
            </div>
            ${checkboxHtml}
          </div>
          <div class="field-val mono">${esc(fobj.value)}</div>
          <div class="bar-row">
            <span class="muted">Counterfactual Reliance</span>
            <div class="bar-track"><div class="bar-fill influence" style="width:${infPct}%"></div></div>
            <span class="mono">${infPct}%</span>
          </div>
          ${snapHtml}
        </div>
      `;
    })
    .join("");

  let gtHtml = "";
  if (showGroundTruth && inv.ground_truth) {
    const isTrue = Boolean(inv.ground_truth.is_true_event);
    gtHtml = `
      <div class="gt-box">
        <span class="badge ${isTrue ? "badge-cert" : "badge-decoy"}">
          ${isTrue ? "GROUND TRUTH: REAL POSITIVE" : "GROUND TRUTH: FALSE-ALARM DECOY"}
        </span>
        <div class="mono" style="margin-top:6px;">${esc(JSON.stringify(inv.ground_truth))}</div>
      </div>
    `;
  }

  return `
    <div class="media-stage">${mediaHtml}</div>
    <div class="decision-banner">
      <div class="decision-top">
        <div>
          <div class="muted" style="font-size:0.76rem;">Decision-1 Closed-Set Verdict · Record <code>${esc(inv.record_id)}</code></div>
          <div class="decision-answer">${esc(probe.primary_answer)} <span class="muted">(${Math.round(probe.primary_confidence * 100)}% conf)</span></div>
        </div>
        <div>${modeBadge}</div>
      </div>
      <div class="prob-bars">${probRows}</div>
      <div class="muted mono" style="margin-top:8px;font-size:0.75rem;">
        Verified Gate: <strong>${esc(probe.verified_gate_answer)}</strong> · Board Rebuild Drift (I3): <strong>${inv.invariants.board_drift}</strong> · Time: ${inv.duration_ms}ms
      </div>
    </div>
    <div class="fields-list">${fieldsHtml}</div>
    ${gtHtml}
  `;
}

async function investigateCandidate(recordId, ablateFields = []) {
  const ds = activeDataset();
  if (!ds) return;
  const body = document.getElementById("inspectorBody");
  body.className = "";
  body.innerHTML = `<div class="inspector-empty">Running <code>ares.solve()</code> (3 parallel <code>TRUSTED</code> verifiers) + <code>Decision-1</code> counterfactual ablations for <code>${esc(recordId)}</code>…</div>`;

  try {
    const inv = await fetchJson("/api/investigate", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: ds.dataset_id,
        record_id: recordId,
        use_live_gemini: useLiveGemini(),
        ablate_fields: ablateFields,
        reveal_ground_truth: false,
      }),
    });
    state.currentInvestigation = inv;

    body.innerHTML = `
      ${renderInvestigationCardHtml(inv, { allowAblationControls: true, showGroundTruth: false })}
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
        <button id="reprobeBtn" class="btn btn-primary">🔬 Re-Probe With Selected Ablations</button>
        <button id="resetAblateBtn" class="btn btn-ghost">↺ Restore All Evidence</button>
        <button id="revealGtBtn" class="btn btn-secondary">🎯 Reveal Withheld Ground Truth</button>
      </div>
      <div id="revealedGtContainer"></div>
    `;

    document.getElementById("reprobeBtn").addEventListener("click", () => {
      const checked = Array.from(body.querySelectorAll(".ablate-cb:checked")).map((el) => el.value);
      investigateCandidate(recordId, checked);
    });
    document.getElementById("resetAblateBtn").addEventListener("click", () => {
      investigateCandidate(recordId, []);
    });
    document.getElementById("revealGtBtn").addEventListener("click", async () => {
      const gtRes = await fetchJson(`/api/ground_truth/${encodeURIComponent(ds.dataset_id)}/${encodeURIComponent(recordId)}`);
      const isTrue = Boolean(gtRes.ground_truth.is_true_event);
      document.getElementById("revealedGtContainer").innerHTML = `
        <div class="gt-box">
          <span class="badge ${isTrue ? "badge-cert" : "badge-decoy"}">
            ${isTrue ? "✅ TRUE POSITIVE EVENT" : "🎭 LOOK-ALIKE FALSE-ALARM DECOY"}
          </span>
          <div class="mono" style="margin-top:6px;">${esc(JSON.stringify(gtRes.ground_truth))}</div>
        </div>
      `;
    });
  } catch (err) {
    body.innerHTML = `<div class="alert-error">${esc(err.message)}</div>`;
  }
}

async function runDuel() {
  const ds = activeDataset();
  if (!ds) return;
  const duelSec = document.getElementById("duelSection");
  const duelGrid = document.getElementById("duelGrid");
  const btn = document.getElementById("runDuelBtn");

  duelSec.classList.remove("hidden");
  btn.disabled = true;
  duelGrid.innerHTML = `<div class="inspector-empty" style="grid-column:1/-1;">Running parallel ARES investigations &amp; Decision-1 counterfactual probes on 1 Real Event vs. 1 High-Salience Decoy…</div>`;

  try {
    const res = await fetchJson("/api/duel", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: ds.dataset_id,
        use_live_gemini: useLiveGemini(),
      }),
    });
    duelGrid.innerHTML = `
      <div class="duel-col">
        <div style="margin-bottom:8px;"><span class="badge badge-cert">Candidate A · Real Positive</span></div>
        ${renderInvestigationCardHtml(res.real_candidate, { allowAblationControls: false, showGroundTruth: true })}
      </div>
      <div class="duel-col">
        <div style="margin-bottom:8px;"><span class="badge badge-decoy">Candidate B · Look-Alike Tabular Decoy</span></div>
        ${renderInvestigationCardHtml(res.decoy_candidate, { allowAblationControls: false, showGroundTruth: true })}
      </div>
    `;
  } catch (err) {
    duelGrid.innerHTML = `<div class="alert-error" style="grid-column:1/-1;">${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

async function init() {
  document.getElementById("runSqlBtn").addEventListener("click", executeSearch);
  document.getElementById("runDuelBtn").addEventListener("click", runDuel);
  document.getElementById("closeDuelBtn").addEventListener("click", () => {
    document.getElementById("duelSection").classList.add("hidden");
  });
  document.getElementById("testLeakGuardBtn").addEventListener("click", () => {
    document.getElementById("sqlInput").value = "SELECT * FROM read_json_auto('ground_truth.json') LIMIT 5";
    executeSearch();
  });

  const cfg = await fetchJson("/api/datasets");
  state.datasets = cfg.datasets || [];
  const toggle = document.getElementById("liveGeminiToggle");
  const statusLbl = document.getElementById("geminiStatusLabel");
  if (!cfg.gemini_key_present) {
    toggle.checked = false;
    statusLbl.textContent = "Offline Heuristic Mode (No GEMINI_API_KEY)";
  } else {
    statusLbl.textContent = "Live Gemini Decision-1 Ready";
  }

  if (state.datasets.length > 0) {
    await selectDataset(state.datasets[0].dataset_id);
  }
}

init();
