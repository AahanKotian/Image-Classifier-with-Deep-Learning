/**
 * classifier.js — Client logic for the Neural Vision drag-and-drop classifier.
 *
 * Handles:
 *  - Drag-and-drop / file input
 *  - Image preview
 *  - POST /predict API call
 *  - Top-K bar chart animation
 *  - Model metadata fetch (/classes, /health)
 */

"use strict";

// ── DOM refs ─────────────────────────────────
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const resultsArea   = document.getElementById("results-area");
const previewImg    = document.getElementById("preview-img");
const btnClear      = document.getElementById("btn-clear");
const statusEl      = document.getElementById("status");
const statusText    = document.getElementById("status-text");
const predBlock     = document.getElementById("prediction-block");
const predClass     = document.getElementById("pred-class");
const predConf      = document.getElementById("pred-conf");
const topkBlock     = document.getElementById("topk-block");
const topkBars      = document.getElementById("topk-bars");
const metaBlock     = document.getElementById("meta-block");
const metaTime      = document.getElementById("meta-time");
const metaDemo      = document.getElementById("meta-demo");
const modelBadge    = document.getElementById("model-name");
const archClasses   = document.getElementById("arch-classes");
const statAcc       = document.getElementById("stat-acc");
const statClasses   = document.getElementById("stat-classes");

const API_BASE = "";   // same origin

// ── On load: fetch model info ─────────────────
(async () => {
  try {
    const [healthRes, classRes] = await Promise.all([
      fetch(`${API_BASE}/health`),
      fetch(`${API_BASE}/classes`),
    ]);
    const health  = await healthRes.json();
    const classes = await classRes.json();

    if (health.model_loaded) {
      modelBadge.textContent = `${health.dataset?.toUpperCase() ?? "Model"} loaded`;
      archClasses.textContent = classes.num_classes;
      statClasses.textContent = classes.num_classes;
    } else {
      modelBadge.textContent = "Demo mode";
    }
  } catch (_) {
    modelBadge.textContent = "Server offline";
  }
})();

// ── Drag & Drop ───────────────────────────────

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

btnClear.addEventListener("click", resetUI);

// ── Paste support ─────────────────────────────
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find(i => i.type.startsWith("image/"));
  if (item) handleFile(item.getAsFile());
});

// ── Core flow ─────────────────────────────────

function handleFile(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;

  // Show results panel
  dropZone.hidden = true;
  resultsArea.removeAttribute("hidden");

  // Reset result panels
  showLoading("Classifying…");

  classify(file);
}

async function classify(file) {
  const form = new FormData();
  form.append("image", file);

  try {
    const t0  = performance.now();
    const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
    const elapsed = (performance.now() - t0).toFixed(1);

    if (!res.ok) {
      const err = await res.json();
      showError(err.error ?? "Server error");
      return;
    }

    const data = await res.json();
    showResults(data, elapsed);
  } catch (err) {
    showError("Network error — is the Flask server running?");
  }
}

// ── UI helpers ────────────────────────────────

function showLoading(msg) {
  statusEl.removeAttribute("hidden");
  statusText.textContent = msg;
  predBlock.hidden  = true;
  topkBlock.hidden  = true;
  metaBlock.hidden  = true;
  topkBars.innerHTML = "";
}

function showError(msg) {
  statusEl.removeAttribute("hidden");
  statusText.textContent = `⚠ ${msg}`;
  predBlock.hidden = true;
  topkBlock.hidden = true;
}

function showResults(data, clientElapsedMs) {
  statusEl.hidden = true;

  // Primary prediction
  predClass.textContent = data.prediction.toUpperCase();
  predConf.textContent  = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
  predBlock.removeAttribute("hidden");

  // Top-K bars
  if (data.top_k && data.top_k.length > 0) {
    topkBars.innerHTML = "";
    data.top_k.forEach((item, idx) => renderBar(item, idx === 0));
    topkBlock.removeAttribute("hidden");
  }

  // Meta row
  metaTime.textContent = `⚡ Server: ${data.inference_time_ms}ms  |  Client: ${clientElapsedMs}ms`;
  metaDemo.textContent = data.demo_mode ? "🟡 Demo mode — train a model for real results" : "";
  metaBlock.removeAttribute("hidden");

  // Update accuracy stat if present in response
  if (data.model_accuracy) {
    statAcc.textContent = `${(data.model_accuracy * 100).toFixed(1)}%`;
  }
}

function renderBar(item, isPrimary) {
  const pct = (item.confidence * 100).toFixed(1);

  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `
    <div class="bar-name">${escHtml(item.class)}</div>
    <div class="bar-track">
      <div class="bar-fill ${isPrimary ? "" : "secondary"}" style="width:0%"></div>
    </div>
    <div class="bar-pct">${pct}%</div>
  `;
  topkBars.appendChild(row);

  // Animate fill after a paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${pct}%`;
    });
  });
}

function resetUI() {
  dropZone.hidden = false;
  resultsArea.hidden = true;
  previewImg.src = "";
  fileInput.value = "";
  topkBars.innerHTML = "";
}

function escHtml(str) {
  return str.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
