/* ═══════════════════════════════════════════════════
   GATOBYTE · Dashboard de Metricas — dashboard.js
   ═══════════════════════════════════════════════════ */

"use strict";

const MOCK_METRICS = {
  baseline: {
    available: true,
    model_display: "Baseline",
    model_detail:  "LightGBM + TF-IDF (10k features)",
    metricas_val:  { f1_macro: 0.6953, f1_weighted: 0.8430, bal_accuracy: 0.7578, roc_auc: 0.9327 },
    metricas_test: { f1_macro: 0.6945, f1_weighted: 0.8422, bal_accuracy: 0.7575, roc_auc: 0.9320 },
    clases: ["negative", "neutral", "positive"],
    confusion_matrix: [
      [7823, 621,  456 ],
      [1102, 4301, 2197],
      [812,  1543, 41963],
    ],
  },
  transformer: {
    available: false,
    model_display: "Transformer",
    model_detail:  "DistilBERT + LoRA (PEFT)",
    metricas_val:  null,
    metricas_test: null,
    clases: [],
    confusion_matrix: null,
  },
};

const DASH_USE_MOCK = false;

document.addEventListener("DOMContentLoaded", () => {});

async function loadDashboard() {
  const area = $("dashboardContent");
  if (!area) return;
  showLoading(area);

  try {
    let metrics;

    if (DASH_USE_MOCK) {
      await new Promise((r) => setTimeout(r, 300));
      metrics = MOCK_METRICS;
    } else {
      const res = await fetch("/api/metrics");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      metrics = {
        baseline:    { ...MOCK_METRICS.baseline,    ...data.baseline    },
        transformer: { ...MOCK_METRICS.transformer, ...data.transformer },
      };
    }

    renderDashboard(metrics);
  } catch (err) {
    showError(area, "No se pudieron cargar las metricas.", err.message);
  }
}

function renderDashboard(metrics) {
  const area = $("dashboardContent");
  if (!area) return;

  const { baseline, transformer } = metrics;

  area.innerHTML = `
  <div class="dash-layout">
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("bar-chart-2", 15)} Comparativa de modelos
        ${DASH_USE_MOCK ? `<span class="mock-badge">datos de ejemplo</span>` : ""}
      </div>
      <div class="model-compare-grid">
        ${renderModelCard(baseline,    "baseline")}
        ${renderModelCard(transformer, "transformer")}
      </div>
    </div>

    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("table", 15)} Metricas detalladas — conjunto de prueba
      </div>
      ${renderMetricsTable(baseline, transformer)}
    </div>

    ${baseline.available && baseline.confusion_matrix ? `
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("grid", 15)} Matriz de confusion — ${baseline.model_display}
        <span class="dash-section-note">conjunto de prueba</span>
      </div>
      ${renderConfusionMatrix(baseline.confusion_matrix, baseline.clases)}
    </div>` : ""}

    ${transformer.available && transformer.confusion_matrix ? `
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("grid", 15)} Matriz de confusion — ${transformer.model_display}
      </div>
      ${renderConfusionMatrix(transformer.confusion_matrix, transformer.clases)}
    </div>` : ""}

    <div class="dash-section">
      <div class="dash-section-title">${icon("book-open", 15)} Referencia de metricas</div>
      ${renderMetricsLegend()}
    </div>
  </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [area] });
}

function renderModelCard(modelData, modelKey) {
  if (!modelData.available) {
    return `
    <div class="model-card model-card--pending">
      <div class="mcard-header">
        <span class="mcard-name">${modelData.model_display}</span>
        <span class="status-badge status-badge--pending">${icon("clock", 12)} Pendiente</span>
      </div>
      <div class="mcard-detail">${modelData.model_detail}</div>
      <div class="mcard-pending-msg">
        ${icon("loader", 20)}
        <p>Las metricas apareceran aqui cuando el modelo este listo.</p>
      </div>
    </div>`;
  }

  const t = modelData.metricas_test;
  const f1pct   = Math.round((t.f1_macro || 0) * 100);
  const f1color = (t.f1_macro || 0) >= 0.70 ? "#F59E0B"
                : (t.f1_macro || 0) >= 0.55 ? "#F59E0B"
                : "#EF4444";
  const trackColor = (t.f1_macro || 0) >= 0.70 ? "#FDE68A" : "#FCA5A5";

  return `
  <div class="model-card model-card--active">
    <div class="mcard-header">
      <span class="mcard-name">${modelData.model_display}</span>
      <span class="status-badge status-badge--ok">${icon("check-circle", 12)} Disponible</span>
    </div>
    <div class="mcard-detail">${modelData.model_detail}</div>

    <div class="mcard-star-metric">
      <div class="mstar-ring">
        <svg viewBox="0 0 44 44" class="mstar-svg">
          <circle cx="22" cy="22" r="18" fill="none" stroke="${trackColor}" stroke-width="4"/>
          <circle cx="22" cy="22" r="18" fill="none"
            stroke="${f1color}" stroke-width="4"
            stroke-dasharray="${Math.round(18 * 2 * Math.PI)}"
            stroke-dashoffset="${Math.round(18 * 2 * Math.PI * (1 - (t.f1_macro || 0)))}"
            stroke-linecap="round"
            transform="rotate(-90 22 22)"/>
        </svg>
        <div class="mstar-val">${(t.f1_macro || 0).toFixed(2)}</div>
      </div>
      <div class="mstar-label">F1-Macro<br><span>(conjunto de prueba)</span></div>
    </div>

    <div class="mcard-mini-grid">
      ${miniMetric("F1 Weighted",   t.f1_weighted,  0.80)}
      ${miniMetric("Bal. Accuracy", t.bal_accuracy, 0.70)}
      ${miniMetric("ROC-AUC",       t.roc_auc,      0.85)}
    </div>
  </div>`;
}

function miniMetric(label, val, target) {
  if (val == null) {
    return `
    <div class="mini-metric">
      <div class="mini-metric-val" style="color:var(--text-3);">—</div>
      <div class="mini-metric-label">${label}</div>
      <div class="mini-metric-target" style="color:var(--text-3);">≥ ${target}</div>
    </div>`;
  }
  const ok    = val >= target;
  const close = val >= target * 0.92;
  const cls   = ok ? "ok" : close ? "warn" : "";
  const color = ok ? "var(--green-dk)" : close ? "var(--amber-dk)" : "var(--red-dk)";

  return `
  <div class="mini-metric ${cls}">
    <div class="mini-metric-val" style="color:${color};">${val.toFixed(3)}</div>
    <div class="mini-metric-label">${label}</div>
    <div class="mini-metric-target" style="color:${color};">
      ${icon(ok ? "check" : "minus", 10)} ≥ ${target}
    </div>
  </div>`;
}

function renderMetricsTable(baseline, transformer) {
  const metrics = [
    { key: "f1_macro",    label: "F1-Macro",      target: 0.65, fmt: (v) => v.toFixed(4) },
    { key: "f1_weighted", label: "F1-Weighted",   target: 0.80, fmt: (v) => v.toFixed(4) },
    { key: "bal_accuracy",label: "Bal. Accuracy", target: 0.70, fmt: (v) => v.toFixed(4) },
    { key: "roc_auc",     label: "ROC-AUC",       target: 0.85, fmt: (v) => v.toFixed(4) },
  ];

  const bTest = baseline.available    ? baseline.metricas_test    : null;
  const tTest = transformer.available ? transformer.metricas_test : null;

  const rows = metrics.map(({ key, label, target, fmt }) => {
    const bVal = bTest ? bTest[key] : null;
    const tVal = tTest ? tTest[key] : null;

    const bCell = bVal != null
      ? `<span style="color:${bVal >= target ? "var(--green-dk)" : "var(--red-dk)"};">${fmt(bVal)}</span>`
      : `<span class="cell-na">—</span>`;

    const tCell = tVal != null
      ? `<span style="color:${tVal >= target ? "var(--green-dk)" : "var(--red-dk)"};">${fmt(tVal)}</span>`
      : `<span class="cell-na">pendiente</span>`;

    const bWins = bVal != null && tVal != null && bVal >= tVal;
    const tWins = tVal != null && bVal != null && tVal > bVal;

    return `
    <tr>
      <td><strong>${label}</strong></td>
      <td class="metric-target">≥ ${target}</td>
      <td class="${bWins ? "cell-winner" : ""}">${bCell}${bWins ? ` ${icon("award", 11)}` : ""}</td>
      <td class="${tWins ? "cell-winner" : ""}">${tCell}${tWins ? ` ${icon("award", 11)}` : ""}</td>
    </tr>`;
  }).join("");

  return `
  <div class="metrics-table-wrap">
    <table class="data-table metrics-compare-table">
      <thead>
        <tr>
          <th>Metrica</th><th>Objetivo</th>
          <th>${baseline.model_display}</th>
          <th>${transformer.model_display}</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="table-note">
      ${icon("info", 11)} Metricas sobre el conjunto de prueba (15% del dataset). Misma particion para ambos modelos.
    </div>
  </div>`;
}

function renderConfusionMatrix(matrix, classes) {
  if (!matrix || !classes.length) return `<div class="cell-na">No disponible</div>`;

  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const total     = rowTotals.reduce((a, b) => a + b, 0);
  let maxOff = 1;
  matrix.forEach((row, i) => row.forEach((cell, j) => {
    if (i !== j && cell > maxOff) maxOff = cell;
  }));

  const headerCells = classes.map((c) => `<th class="cm-header">${c}</th>`).join("");

  const bodyRows = matrix.map((row, i) => {
    const cells = row.map((cell, j) => {
      const isDiag = i === j;
      const pct    = rowTotals[i] > 0 ? (cell / rowTotals[i] * 100).toFixed(1) : "0.0";
      let bg = "transparent";
      if (isDiag) {
        const intensity = Math.min(0.85, 0.15 + (cell / rowTotals[i]) * 0.70);
        bg = `rgba(34,197,94,${intensity})`;
      } else if (cell > 0) {
        const intensity = Math.min(0.60, 0.05 + (cell / maxOff) * 0.55);
        bg = `rgba(239,68,68,${intensity})`;
      }
      return `
      <td class="cm-cell ${isDiag ? "cm-diag" : ""}" style="background:${bg};">
        <div class="cm-count">${cell.toLocaleString("es")}</div>
        <div class="cm-pct">${pct}%</div>
      </td>`;
    }).join("");

    return `
    <tr>
      <th class="cm-row-label">${classes[i]}</th>
      ${cells}
      <td class="cm-row-total">${rowTotals[i].toLocaleString("es")}</td>
    </tr>`;
  }).join("");

  return `
  <div class="cm-wrap">
    <div class="cm-labels">
      <div class="cm-axis-label cm-axis-y">Real</div>
      <div class="cm-table-outer">
        <div class="cm-axis-label cm-axis-x">Predicho</div>
        <table class="confusion-matrix">
          <thead>
            <tr><th></th>${headerCells}<th class="cm-header cm-total-header">Total</th></tr>
          </thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    </div>
    <div class="cm-legend">
      <span class="cm-leg-item"><span class="cm-leg-swatch" style="background:rgba(34,197,94,0.65);"></span> Correcto</span>
      <span class="cm-leg-item"><span class="cm-leg-swatch" style="background:rgba(239,68,68,0.45);"></span> Error</span>
    </div>
    <div class="table-note">${icon("info", 11)} Total muestras: ${total.toLocaleString("es")}</div>
  </div>`;
}

function renderMetricsLegend() {
  const rows = [
    ["F1-Macro",      "Media del F1 de cada clase, sin ponderar por frecuencia. Penaliza ignorar clases minoritarias.", "≥ 0.65"],
    ["F1-Weighted",   "Media del F1 ponderada por tamano de clase. Optimista con datasets desbalanceados.",             "≥ 0.80"],
    ["Bal. Accuracy", "Accuracy calculada por clase y promediada. No favorece a la clase mayoritaria.",                 "≥ 0.70"],
    ["ROC-AUC",       "Area bajo la curva ROC (macro). Capacidad del modelo de separar clases a distintos umbrales.",  "≥ 0.85"],
  ];

  return `
  <table class="legend-table">
    <thead><tr><th>Metrica</th><th>Que mide?</th><th>Objetivo</th></tr></thead>
    <tbody>
      ${rows.map(([m, d, t]) => `
      <tr>
        <td><strong>${m}</strong></td>
        <td>${d}</td>
        <td><span class="target">${t}</span></td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}