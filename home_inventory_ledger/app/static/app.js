/* Home Inventory Ledger — Interactive Multimodal Capture, Verdict & VoI Loop */

let currentSessionId = null;
let currentBoard = null;
let activeTopRequestId = null;
let previouslyUncertifiedIds = new Set();

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recTimerInterval = null;

const fmtUSD = (v) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v || 0);

async function initConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    const pill = document.getElementById("mode-pill");
    const replayToggle = document.getElementById("toggle-replay");
    const banner = document.getElementById("replay-banner");

    if (cfg.has_live_gemini_key) {
      pill.textContent = "● Live Gemini 4 + Shop MCP";
      pill.classList.add("live");
      replayToggle.checked = false;
      banner.classList.add("hidden");
    } else {
      pill.textContent = "● Replay Backup Mode (No API Key)";
      pill.classList.remove("live");
      replayToggle.checked = true;
      banner.classList.remove("hidden");
    }

    replayToggle.addEventListener("change", () => {
      banner.classList.toggle("hidden", !replayToggle.checked && cfg.has_live_gemini_key);
    });
  } catch (err) {
    console.error("Config check failed:", err);
  }
}

function setProgress(visible, title = "Running Gemini 4 + Decision-1 Pipeline…", activeStage = "ingest") {
  const panel = document.getElementById("progress-panel");
  panel.classList.toggle("hidden", !visible);
  document.getElementById("progress-title").textContent = title;
  document.querySelectorAll(".stage-pill").forEach((el) => {
    el.classList.toggle("active", el.dataset.stage === activeStage);
  });
}

function animateStagesDuringCall() {
  const stages = ["ingest", "understand", "extract", "price_verify", "decide_probe"];
  let i = 0;
  return setInterval(() => {
    i = Math.min(stages.length - 1, i + 1);
    document.querySelectorAll(".stage-pill").forEach((el) => {
      el.classList.toggle("active", el.dataset.stage === stages[i]);
    });
  }, 900);
}

// --- STEP 1: POLICY ---
async function parsePolicy(fileOrNull, useSample = false) {
  const fd = new FormData();
  if (fileOrNull) fd.append("file", fileOrNull);
  fd.append("use_sample", useSample ? "true" : "false");
  fd.append("replay", document.getElementById("toggle-replay").checked ? "true" : "false");

  const chip = document.getElementById("policy-status");
  chip.classList.remove("hidden");
  chip.textContent = "Gemini 4 Carbon reading declarations page…";

  try {
    const res = await fetch("/api/policy", { method: "POST", body: fd });
    const data = await res.json();
    if (data.policy && data.policy.coverage_c_limit) {
      document.getElementById("coverage-limit").value = data.policy.coverage_c_limit;
      if (data.policy.deductible != null) {
        document.getElementById("deductible").value = data.policy.deductible;
      }
      chip.textContent = `✓ CERTIFIED · policy_limit_parse passed: Coverage C = ${fmtUSD(
        data.policy.coverage_c_limit
      )} (Deductible ${fmtUSD(data.policy.deductible || 0)})`;
    } else {
      chip.textContent = "Could not verify Coverage C limit — using manual entry.";
    }
  } catch (e) {
    chip.textContent = `Policy parse error: ${e.message}`;
  }
}

// --- STEP 2: WALKTHROUGH RECORDING / UPLOAD / DEMO ---
async function startWalkthroughRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: true,
    });
    const vid = document.getElementById("live-video");
    vid.srcObject = mediaStream;
    vid.classList.remove("hidden");
    document.getElementById("cam-placeholder").classList.add("hidden");

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
      stopStreamTracks();
      await submitWalkthroughBlob(blob, "live_walkthrough.webm");
    };
    mediaRecorder.start(250);

    document.getElementById("btn-start-cam").classList.add("hidden");
    document.getElementById("btn-stop-cam").classList.remove("hidden");
    const ind = document.getElementById("rec-indicator");
    ind.classList.remove("hidden");

    let sec = 0;
    document.getElementById("rec-timer").textContent = "00:00";
    recTimerInterval = setInterval(() => {
      sec += 1;
      const m = String(Math.floor(sec / 60)).padStart(2, "0");
      const s = String(sec % 60).padStart(2, "0");
      document.getElementById("rec-timer").textContent = `${m}:${s}`;
    }, 1000);
  } catch (err) {
    alert("Camera/microphone permission needed for live capture: " + err.message);
  }
}

function stopStreamTracks() {
  if (recTimerInterval) clearInterval(recTimerInterval);
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  document.getElementById("live-video").classList.add("hidden");
  document.getElementById("cam-placeholder").classList.remove("hidden");
  document.getElementById("rec-indicator").classList.add("hidden");
  document.getElementById("btn-start-cam").classList.remove("hidden");
  document.getElementById("btn-stop-cam").classList.add("hidden");
}

async function submitWalkthroughBlob(fileOrBlob, filename) {
  const fd = new FormData();
  fd.append("video", fileOrBlob, filename);
  const polInput = document.getElementById("policy-file");
  if (polInput.files && polInput.files[0]) {
    fd.append("policy_image", polInput.files[0]);
  }
  fd.append("coverage_limit", document.getElementById("coverage-limit").value || "8000");
  fd.append("deductible", document.getElementById("deductible").value || "500");
  fd.append("replay", document.getElementById("toggle-replay").checked ? "true" : "false");

  setProgress(true, "Processing Multimodal Walkthrough Video + Audio…", "ingest");
  const anim = animateStagesDuringCall();
  try {
    const res = await fetch("/api/walkthrough", { method: "POST", body: fd });
    const board = await res.json();
    clearInterval(anim);
    setProgress(false);
    renderBoard(board);
  } catch (err) {
    clearInterval(anim);
    setProgress(false);
    alert("Walkthrough pipeline failed: " + err.message);
  }
}

async function runSampleWalkthrough() {
  const fd = new FormData();
  fd.append("coverage_limit", document.getElementById("coverage-limit").value || "8000");
  fd.append("replay", document.getElementById("toggle-replay").checked ? "true" : "false");

  setProgress(true, "Running Pipeline on Sample Desk Walkthrough + Policy…", "ingest");
  const anim = animateStagesDuringCall();
  try {
    const res = await fetch("/api/demo/walkthrough", { method: "POST", body: fd });
    const board = await res.json();
    clearInterval(anim);
    setProgress(false);
    renderBoard(board);
  } catch (err) {
    clearInterval(anim);
    setProgress(false);
    alert("Demo run failed: " + err.message);
  }
}

// --- STEP 4: FOLLOW-UP CLIP ---
async function recordFourSecondFollowup() {
  if (!currentSessionId || !activeTopRequestId) return;
  const wrap = document.getElementById("followup-cam-wrap");
  const vid = document.getElementById("followup-video");
  const btn = document.getElementById("btn-record-followup");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    vid.srcObject = stream;
    wrap.classList.remove("hidden");
    btn.disabled = true;
    btn.textContent = "● Filming Label (4s)…";

    const chunks = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      wrap.classList.add("hidden");
      btn.disabled = false;
      btn.textContent = "🎥 Film 4s Label Close-Up";
      const blob = new Blob(chunks, { type: rec.mimeType || "video/webm" });
      await submitFollowupFile(blob, "followup_label.webm");
    };
    rec.start(200);
    setTimeout(() => {
      if (rec.state !== "inactive") rec.stop();
    }, 4200);
  } catch (err) {
    alert("Camera access failed: " + err.message);
  }
}

async function submitFollowupFile(fileOrBlob, filename) {
  if (!currentSessionId) return;
  const fd = new FormData();
  fd.append("request_id", activeTopRequestId || "");
  fd.append("video", fileOrBlob, filename);
  fd.append("replay", document.getElementById("toggle-replay").checked ? "true" : "false");

  setProgress(true, "Incrementally Extracting Label & Running catalog_exact_match…", "price_verify");
  try {
    const res = await fetch(`/api/session/${currentSessionId}/followup`, { method: "POST", body: fd });
    const board = await res.json();
    setProgress(false);
    renderBoard(board, true);
  } catch (err) {
    setProgress(false);
    alert("Follow-up failed: " + err.message);
  }
}

async function runSampleFollowup() {
  if (!currentSessionId) return;
  const fd = new FormData();
  if (activeTopRequestId) fd.append("request_id", activeTopRequestId);

  setProgress(true, "Processing 4s Close-Up Clip (Model A2992) & Re-Probing Verdict…", "price_verify");
  try {
    const res = await fetch(`/api/session/${currentSessionId}/demo-followup`, { method: "POST", body: fd });
    const board = await res.json();
    setProgress(false);
    renderBoard(board, true);
  } catch (err) {
    setProgress(false);
    alert("Sample follow-up failed: " + err.message);
  }
}

// --- RENDER BOARD ---
function renderBoard(board, isFollowup = false) {
  currentSessionId = board.session_id;
  currentBoard = board;
  document.getElementById("results-section").classList.remove("hidden");
  document.getElementById("replay-banner").classList.toggle("hidden", board.telemetry.mode !== "replay");

  const dec = board.decision || {};
  const vTitle = document.getElementById("verdict-label");
  const niceVerdict = (dec.verdict || "unknown").replace(/_/g, " ");
  vTitle.textContent = niceVerdict.charAt(0).toUpperCase() + niceVerdict.slice(1);
  vTitle.className = "verdict-title " + (dec.verdict || "");

  document.getElementById("verdict-conf").textContent = `Decision-1 conf ${(dec.confidence || 0).toFixed(2)}`;
  document.getElementById("val-certified").textContent = fmtUSD(dec.certified_total);
  document.getElementById("val-range").textContent = `${fmtUSD(dec.estimate_low)} – ${fmtUSD(dec.estimate_high)}`;
  document.getElementById("val-cert-share").textContent = `${Math.round((dec.certified_share || 0) * 100)}%`;

  // Range bar scaling
  const limit = dec.coverage_c_limit || 8000;
  const maxAxis = Math.max(limit * 1.35, (dec.estimate_high || 0) * 1.12, 1000);
  const pct = (v) => Math.max(0, Math.min(100, ((v || 0) / maxAxis) * 100));

  document.getElementById("bar-certified").style.width = `${pct(dec.certified_total)}%`;
  document.getElementById("bar-uncertified").style.left = `${pct(dec.estimate_low)}%`;
  document.getElementById("bar-uncertified").style.width = `${Math.max(
    1.5,
    pct(dec.estimate_high) - pct(dec.estimate_low)
  )}%`;
  document.getElementById("limit-marker").style.left = `${pct(limit)}%`;
  document.getElementById("limit-marker-label").textContent = `Limit ${fmtUSD(limit)}`;

  // Hinges On pills
  const itemsMap = board.items || {};
  const hingesEl = document.getElementById("hinges-pills");
  hingesEl.innerHTML = "";
  (dec.hinges_on || []).forEach((iid) => {
    const it = itemsMap[iid];
    const span = document.createElement("span");
    span.className = "hinge-pill";
    span.textContent = it ? it.display_name : iid;
    hingesEl.appendChild(span);
  });

  // Calibrated probabilities
  const probsEl = document.getElementById("prob-bars");
  probsEl.innerHTML = "";
  Object.entries(dec.probabilities || {}).forEach(([k, v]) => {
    const div = document.createElement("div");
    div.textContent = `P(${k.replace(/_/g, " ")}) = ${(v * 100).toFixed(0)}%`;
    probsEl.appendChild(div);
  });

  // Contradiction banner
  const conflicts = Object.values(itemsMap).filter((it) => it.conflict_detail);
  const confBanner = document.getElementById("conflict-banner");
  if (conflicts.length > 0) {
    confBanner.classList.remove("hidden");
    document.getElementById("conflict-text").textContent = conflicts.map((c) => c.conflict_detail).join(" · ");
  } else {
    confBanner.classList.add("hidden");
  }

  // Active VoI Ask Card
  const asks = board.requests || [];
  const askCard = document.getElementById("ask-card");
  if (asks.length > 0) {
    askCard.classList.remove("hidden");
    const top = asks[0];
    activeTopRequestId = top.request_id;
    document.getElementById("ask-instruction").textContent = top.instruction;
    document.getElementById("ask-voi-sub").textContent = `Worth ±${fmtUSD(
      top.voi_usd
    )} of expected verdict uncertainty (range spread ${fmtUSD(top.uncertainty_width_usd)})`;

    const others = asks.slice(1);
    document.getElementById("other-asks").textContent =
      others.length > 0
        ? "Next queued asks: " + others.map((a) => `${a.item_name} (±${fmtUSD(a.voi_usd)})`).join(" · ")
        : "";
  } else {
    activeTopRequestId = null;
    document.getElementById("ask-instruction").textContent = "All pivotal items verified!";
    document.getElementById("ask-voi-sub").textContent = "Zero high-VoI unknowns remaining.";
  }

  // Evidence Ledger Table
  const tbody = document.getElementById("ledger-tbody");
  tbody.innerHTML = "";
  const sortedItems = Object.values(itemsMap).sort((a, b) => {
    const va = a.unit_price || a.price_high || 0;
    const vb = b.unit_price || b.price_high || 0;
    return vb * b.quantity - va * a.quantity;
  });

  const nextUncertified = new Set();
  sortedItems.forEach((it) => {
    if (it.price_status !== "CERTIFIED") nextUncertified.add(it.item_id);
    const tr = document.createElement("tr");
    if (isFollowup && it.price_status === "CERTIFIED" && previouslyUncertifiedIds.has(it.item_id)) {
      tr.classList.add("just-certified");
    }

    const thumbHtml = it.frame_url
      ? `<img class="thumb" src="${it.frame_url}" alt="${it.display_name}" />`
      : `<div class="thumb"></div>`;

    let priceHtml = "Needs Appraisal";
    if (it.price_status === "CERTIFIED" && it.unit_price != null) {
      priceHtml = `<strong>${fmtUSD(it.unit_price * it.quantity)}</strong>`;
    } else if (it.price_status === "UNCERTIFIED") {
      priceHtml = `${fmtUSD((it.price_low || 0) * it.quantity)} – ${fmtUSD((it.price_high || 0) * it.quantity)}`;
    }

    let citeHtml = `<span style="color:#64748b">No exact match yet</span>`;
    if (it.matched_listing) {
      citeHtml = `<a href="${it.matched_listing.url}" target="_blank" rel="noopener">${it.matched_listing.title.slice(
        0,
        42
      )}…</a><br/><small style="color:#64748b">catalog_exact_match · ${it.matched_listing.merchant || "Shop"}</small>`;
    } else if (it.candidate_listings && it.candidate_listings.length > 0) {
      citeHtml = `<small style="color:#64748b">${it.candidate_listings.length} Shop listings (25th–75th pct)</small>`;
    }

    const conflictHtml = it.conflict_detail ? `<div class="flag-inline">⚠ ${it.conflict_detail}</div>` : "";

    tr.innerHTML = `
      <td>${thumbHtml}<small style="color:#64748b">@${(it.first_seen_ms / 1000).toFixed(1)}s</small></td>
      <td><strong>${it.display_name}</strong><br/><small style="color:#64748b">${it.description || ""}</small>${conflictHtml}</td>
      <td>${it.quantity}</td>
      <td><span class="badge ${it.price_status}">${it.price_status}</span></td>
      <td>${priceHtml}</td>
      <td>${it.voi_usd > 0 ? "±" + fmtUSD(it.voi_usd) : "—"}</td>
      <td>${citeHtml}</td>
    `;
    tbody.appendChild(tr);
  });
  previouslyUncertifiedIds = nextUncertified;

  document.getElementById("drift-val").textContent = String(board.telemetry.board_drift_count || 0);
  document.getElementById("link-export-csv").href = `/api/session/${currentSessionId}/export.csv`;
  document.getElementById("link-export-html").href = `/api/session/${currentSessionId}/export.html`;

  const tel = board.telemetry || {};
  document.getElementById("telemetry-bar").innerHTML = `
    <span>Board v${board.board_version} (${tel.mode})</span>
    <span>Frames: ${tel.frames_kept_after_dedup}/${tel.frames_extracted} kept after dHash dedup</span>
    <span>Carbon calls: ${tel.carbon_calls}</span>
    <span>Hydrogen workers: ${tel.hydrogen_calls}</span>
    <span>Decision-1 calls: ${tel.decision_calls}</span>
    <span>Shop MCP calls: ${tel.shop_mcp_calls}</span>
    <span>Elapsed: ${tel.elapsed_seconds}s</span>
  `;
}

// --- SHOP CART ---
async function openShopCart() {
  if (!currentSessionId) return;
  try {
    const res = await fetch(`/api/session/${currentSessionId}/cart`, { method: "POST" });
    const cart = await res.json();
    const panel = document.getElementById("cart-panel");
    const linesEl = document.getElementById("cart-lines");
    linesEl.innerHTML = "";
    (cart.lines || []).forEach((ln) => {
      const div = document.createElement("div");
      div.className = "cart-line";
      div.innerHTML = `
        <div>
          <strong>${ln.display_name}</strong> ×${ln.quantity}
          <span class="badge ${ln.certification}" style="margin-left:8px">${ln.certification}</span><br/>
          <small style="color:#64748b">${ln.listing_title} · ${ln.merchant}</small>
        </div>
        <div style="text-align:right">
          <strong>${fmtUSD(ln.line_total)}</strong><br/>
          <a href="${ln.url}" target="_blank" rel="noopener" style="font-size:12px">View listing →</a>
        </div>
      `;
      linesEl.appendChild(div);
    });
    document.getElementById("cart-subtotal").textContent = fmtUSD(cart.subtotal_usd);
    document.getElementById("cart-checkout-link").href = cart.checkout_url || "https://shop.app";
    panel.classList.remove("hidden");
    panel.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    alert("Failed to build replacement cart: " + err.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initConfig();

  document.getElementById("policy-file").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) parsePolicy(e.target.files[0], false);
  });
  document.getElementById("btn-sample-policy").addEventListener("click", () => parsePolicy(null, true));

  document.getElementById("btn-start-cam").addEventListener("click", startWalkthroughRecording);
  document.getElementById("btn-stop-cam").addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  });
  document.getElementById("walkthrough-file").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) submitWalkthroughBlob(e.target.files[0], e.target.files[0].name);
  });
  document.getElementById("btn-demo-walkthrough").addEventListener("click", runSampleWalkthrough);

  document.getElementById("btn-record-followup").addEventListener("click", recordFourSecondFollowup);
  document.getElementById("followup-file").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) submitFollowupFile(e.target.files[0], e.target.files[0].name);
  });
  document.getElementById("btn-demo-followup").addEventListener("click", runSampleFollowup);

  document.getElementById("btn-shop-cart").addEventListener("click", openShopCart);
  document.getElementById("btn-close-cart").addEventListener("click", () => {
    document.getElementById("cart-panel").classList.add("hidden");
  });
});
