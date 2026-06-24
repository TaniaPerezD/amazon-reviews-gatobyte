/* ═══════════════════════════════════════════════════
   GATOBYTE · Clasificador de Sentimiento — sentiment.js
   ═══════════════════════════════════════════════════ */

"use strict";

const sentState = {
  activeModel: "baseline",
  lastResult:  null,
};

const MOCK_RESPONSES = {
  baseline: {
    label: "positive",
    probabilities: { positive: 0.82, neutral: 0.11, negative: 0.07 },
    model_used: "baseline",
    model_display: "Baseline (LightGBM + TF-IDF)",
    latency_ms: 43,
  },
  transformer: {
    label: "positive",
    probabilities: { positive: 0.91, neutral: 0.06, negative: 0.03 },
    model_used: "transformer",
    model_display: "Transformer (DistilBERT + LoRA)",
    latency_ms: 187,
  },
};

const USE_MOCK = false;

document.addEventListener("DOMContentLoaded", () => {
  setupModelToggle();
  setupClassifier();
  setupExamples();
});

function setupModelToggle() {
  const btns = document.querySelectorAll(".model-toggle-btn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      sentState.activeModel = btn.dataset.model;
      const input = $("sentimentInput");
      if (sentState.lastResult && input && input.value.trim()) runClassifier();
    });
  });
}

function setupClassifier() {
  const btn   = $("classifyBtn");
  const input = $("sentimentInput");
  if (!btn || !input) return;
  btn.addEventListener("click", runClassifier);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) runClassifier();
  });
}

async function runClassifier() {
  const input = $("sentimentInput");
  const text  = input ? input.value.trim() : "";
  if (!text) {
    input && input.classList.add("input-shake");
    setTimeout(() => input && input.classList.remove("input-shake"), 400);
    return;
  }

  const btn = $("classifyBtn");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Clasificando…`;
  }

  const resultArea = $("classifyResult");
  showLoading(resultArea);

  try {
    let data;

    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 420));
      data = structuredClone(MOCK_RESPONSES[sentState.activeModel]);
      const lower = text.toLowerCase();
      if (lower.includes("terrible") || lower.includes("malo") || lower.includes("broken")) {
        data.label = "negative";
        data.probabilities = { positive: 0.08, neutral: 0.12, negative: 0.80 };
      } else if (lower.includes("okay") || lower.includes("regular") || lower.includes("normal")) {
        data.label = "neutral";
        data.probabilities = { positive: 0.28, neutral: 0.55, negative: 0.17 };
      }
    } else {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: sentState.activeModel }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
      if (data.error) throw new Error(data.error);
    }

    sentState.lastResult = data;
    renderClassifyResult(data, text);

  } catch (err) {
    showError(resultArea, "Error al clasificar.", err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="zap"></i> Clasificar`;
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
    }
  }
}

function renderClassifyResult(data, inputText) {
  const area = $("classifyResult");
  if (!area) return;

  const label         = (data.label || "neutral").toLowerCase();
  const probabilities = Object.fromEntries(
    Object.entries(data.probabilities || {}).map(([k, v]) => [k.toLowerCase(), v])
  );
  const model_display = data.model_display;
  const latency_ms    = data.latency_ms;
  const { cls }       = sentimentInfo(label);

  const colorMap = {
    positive: { bg: "var(--green-lt)",  border: "var(--green)",  text: "var(--green-dk)"  },
    negative: { bg: "var(--red-lt)",    border: "var(--red)",    text: "var(--red-dk)"    },
    neutral:  { bg: "var(--amber-lt)",  border: "var(--amber)",  text: "var(--amber-dk)"  },
    mixed:    { bg: "var(--purple-lt)", border: "var(--purple)", text: "var(--purple-dk)" },
  };
  const colors = colorMap[cls] || colorMap.neutral;

  const iconMap  = { positive: "smile", negative: "frown", neutral: "minus-circle", mixed: "meh" };
  const labelMap = { positive: "Positivo", negative: "Negativo", neutral: "Neutral", mixed: "Mixto" };

  const speedColor = latency_ms < 100
    ? "background:var(--green-lt);border:0.5px solid var(--green);color:var(--green-dk);"
    : latency_ms < 500
    ? "background:var(--amber-lt);border:0.5px solid var(--amber);color:var(--amber-dk);"
    : "background:var(--red-lt);border:0.5px solid var(--red);color:var(--red-dk);";

  const probOrder = ["positive", "neutral", "negative"];
  const barColorMap = {
    positive: "var(--green)",
    negative: "var(--red)",
    neutral:  "var(--amber)",
  };

  const probBars = probOrder.map((key) => {
    const val      = probabilities[key] ?? 0;
    const pct      = Math.round(val * 100);
    const { cls: barCls } = sentimentInfo(key);
    const isWinner = key === label;
    const barColor = isWinner ? (barColorMap[barCls] || "var(--amber)") : "var(--border-md)";

    return `
    <div class="prob-row ${isWinner ? "prob-winner" : ""}">
      <div class="prob-label">
        ${isWinner ? `<i data-lucide="${iconMap[barCls]}" style="width:13px;height:13px;flex-shrink:0;"></i>` : ""}
        <span>${labelMap[barCls] || key}</span>
        ${isWinner ? `<span class="prob-winner-tag">prediccion</span>` : ""}
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:${pct}%;background:${barColor};"></div>
      </div>
      <span class="prob-pct">${pct}%</span>
    </div>`;
  }).join("");

  area.innerHTML = `
  <div class="classify-result-wrap">
    <div class="verdict-card" style="background:${colors.bg};border:2px solid ${colors.border};">
      <div class="verdict-icon-wrap">
        <i data-lucide="${iconMap[cls]}" style="width:40px;height:40px;color:${colors.text};"></i>
      </div>
      <div class="verdict-body">
        <div class="verdict-label" style="color:${colors.text};">${labelMap[cls]}</div>
        <div class="verdict-conf" style="color:${colors.text};">
          ${Math.round((probabilities[label] ?? 0) * 100)}% de confianza
        </div>
        <div class="verdict-pills">
          <span class="vpill vpill-model">${icon("cpu", 10)} ${model_display || sentState.activeModel}</span>
          <span class="vpill vpill-speed" style="${speedColor}">${icon("zap", 10)} ${latency_ms} ms</span>
        </div>
      </div>
    </div>
    <div class="prob-section">
      <div class="prob-title">Distribucion de probabilidades</div>
      <div class="prob-bars">${probBars}</div>
    </div>
    <div class="analyzed-text">
      <div class="analyzed-label">${icon("file-text", 12)} Texto analizado</div>
      <div class="analyzed-body">"${escapeHtml(inputText.slice(0, 300))}${inputText.length > 300 ? "…" : ""}"</div>
    </div>
  </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [area] });
}

const EXAMPLES = [
  { label: "Positivo", cls: "positive", text: "Amazing product! Works exactly as described. The battery lasts all day and the sound quality is incredible. Best purchase I've made this year." },
  { label: "Negativo", cls: "negative", text: "Terrible quality. Broke after two weeks. The screen started flickering and customer support was completely useless. Total waste of money." },
  { label: "Neutral",  cls: "neutral",  text: "It's okay, nothing special. Does what it says but the packaging was damaged on arrival. Setup was straightforward. Average product for the price." },
];

function setupExamples() {
  const container = $("exampleChips");
  if (!container) return;
  EXAMPLES.forEach((ex) => {
    const btn = document.createElement("button");
    btn.className = `example-chip example-chip--${ex.cls}`;
    btn.textContent = ex.label;
    btn.title = ex.text.slice(0, 80) + "…";
    btn.addEventListener("click", () => {
      const input = $("sentimentInput");
      if (input) { input.value = ex.text; input.focus(); runClassifier(); }
    });
    container.appendChild(btn);
  });
}