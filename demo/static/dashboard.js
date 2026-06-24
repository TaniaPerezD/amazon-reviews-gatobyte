/* ═══════════════════════════════════════════════════
   GATOBYTE · Dashboard de Métricas — dashboard.js
   Muestra métricas del baseline y transformer lado a lado.
   Datos mock hasta que los endpoints estén listos.
   ═══════════════════════════════════════════════════ */

"use strict";

// ── MOCK — reemplazar con fetch real ────────────────
// Estructura igual a metadata_modelo_final.json del notebook
const MOCK_METRICS = {
  baseline: {
    available: true,
    model_display: "Baseline",
    model_detail:  "LightGBM + TF-IDF (10k features)",
    metricas_val: {
      f1_macro:    0.7124,
      f1_weighted: 0.8731,
      bal_accuracy: 0.7201,
      roc_auc:     0.9320,
    },
    metricas_test: {
      f1_macro:    0.7089,
      f1_weighted: 0.8698,
      bal_accuracy: 0.7167,
      roc_auc:     0.9298,
    },
    clases: ["Negative", "Neutral", "Positive"],
    // confusion matrix: filas=real, cols=predicho, orden=clases arriba
    confusion_matrix: [
      [7823, 621,  456 ],   // real: Negative
      [1102, 4301, 2197],   // real: Neutral
      [812,  1543, 41963],  // real: Positive
    ],
  },
  transformer: {
    available: false,   // <- cambiar a true cuando esté listo
    model_display: "Transformer",
    model_detail:  "DistilBERT (inference only)",
    metricas_val: null,
    metricas_test: null,
    clases: [],
    confusion_matrix: null,
  },
};

// Cambia a false cuando /api/metrics esté listo
const DASH_USE_MOCK = false;

// ── INIT ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // El tab de dashboard carga cuando se activa (lazy load en app.js setupTabs)
  // también se puede llamar directamente:
  // loadDashboard();
});

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
      // ── ENDPOINTS REALES ──
      // GET /api/metrics/baseline  → mismo shape que MOCK_METRICS.baseline
      // GET /api/metrics/transformer → mismo shape que MOCK_METRICS.transformer
      const res = await fetch("/api/metrics");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Normalizar claves — el JSON usa bal_accuracy, el dashboard también, ok
      metrics = {
        baseline: {
          ...MOCK_METRICS.baseline,   // defaults para confusion_matrix y model_display
          ...data.baseline,
        },
        transformer: {
          ...MOCK_METRICS.transformer,
          ...data.transformer,
        },
      };
    }

    renderDashboard(metrics);
  } catch (err) {
    showError(area, "No se pudieron cargar las métricas.", err.message);
  }
}

// ── RENDER PRINCIPAL ────────────────────────────────
function renderDashboard(metrics) {
  const area = $("dashboardContent");
  if (!area) return;

  const { baseline, transformer } = metrics;

  area.innerHTML = `
  <div class="dash-layout">

    <!-- ── COMPARATIVA DE MODELOS ── -->
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

    <!-- ── TABLA DETALLADA ── -->
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("table", 15)} Métricas detalladas — conjunto de prueba
      </div>
      ${renderMetricsTable(baseline, transformer)}
    </div>

    <!-- ── MATRIZ DE CONFUSIÓN (solo baseline si transformer no disponible) ── -->
    ${baseline.available ? `
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("grid", 15)} Matriz de confusión — ${baseline.model_display}
        <span class="dash-section-note">conjunto de prueba</span>
      </div>
      ${renderConfusionMatrix(baseline.confusion_matrix, baseline.clases, "baseline")}
    </div>` : ""}

    ${transformer.available && transformer.confusion_matrix ? `
    <div class="dash-section">
      <div class="dash-section-title">
        ${icon("grid", 15)} Matriz de confusión — ${transformer.model_display}
      </div>
      ${renderConfusionMatrix(transformer.confusion_matrix, transformer.clases, "transformer")}
    </div>` : ""}

    <!-- ── REFERENCIA DE MÉTRICAS ── -->
    <div class="dash-section">
      <div class="dash-section-title">${icon("book-open", 15)} Referencia de métricas</div>
      ${renderMetricsLegend()}
    </div>

  </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [area] });
}

// ── CARD DE MODELO ───────────────────────────────────
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
        <p>Este modelo aún está en desarrollo.<br>Las métricas aparecerán aquí cuando esté listo.</p>
      </div>
    </div>`;
  }

  const t = modelData.metricas_test;
  const isBaseline = modelKey === "baseline";

  // Métrica estrella = F1 Macro (la más importante del proyecto)
  const f1pct = Math.round(t.f1_macro * 100);
  const f1color = t.f1_macro >= 0.70 ? "var(--green)"
                : t.f1_macro >= 0.55 ? "var(--amber)"
                : "var(--red)";

  return `
  <div class="model-card model-card--active">
    <div class="mcard-header">
      <span class="mcard-name">${modelData.model_display}</span>
      <span class="status-badge status-badge--ok">${icon("check-circle", 12)} Disponible</span>
    </div>
    <div class="mcard-detail">${modelData.model_detail}</div>

    <!-- Métrica estrella -->
    <div class="mcard-star-metric">
      <div class="mstar-ring" style="--pct:${f1pct};--color:${f1color};">
        <svg viewBox="0 0 44 44" class="mstar-svg">
          <circle cx="22" cy="22" r="18" fill="none" stroke="var(--border)" stroke-width="4"/>
          <circle cx="22" cy="22" r="18" fill="none"
            stroke="${f1color}" stroke-width="4"
            stroke-dasharray="${Math.round(18 * 2 * Math.PI)}"
            stroke-dashoffset="${Math.round(18 * 2 * Math.PI * (1 - t.f1_macro))}"
            stroke-linecap="round"
            transform="rotate(-90 22 22)"/>
        </svg>
        <div class="mstar-val">${t.f1_macro.toFixed(2)}</div>
      </div>
      <div class="mstar-label">F1-Macro<br><span>(conjunto de prueba)</span></div>
    </div>

    <!-- Mini grid de métricas secundarias -->
    <div class="mcard-mini-grid">
      ${miniMetric("F1 Weighted",  t.f1_weighted,  0.80, "weight")}
      ${miniMetric("Bal. Accuracy",t.bal_accuracy, 0.70, "scale")}
      ${miniMetric("ROC-AUC",      t.roc_auc,      0.85, "activity")}
    </div>
  </div>`;
}

function miniMetric(label, val, target, iconName) {
  // Si val es null o undefined, mostrar pendiente
  if (val == null) {
    return `
    <div class="mini-metric">
      <div class="mini-metric-val" style="color:var(--text-3);">—</div>
      <div class="mini-metric-label">${label}</div>
      <div class="mini-metric-target" style="color:var(--text-3);">≥ ${target}</div>
    </div>`;
  }
  const ok = val >= target;
  const color = ok ? "var(--green-dk)" : val >= target * 0.9 ? "var(--amber-dk)" : "var(--red-dk)";
  return `
  <div class="mini-metric">
    <div class="mini-metric-val" style="color:${color};">${val.toFixed(3)}</div>
    <div class="mini-metric-label">${label}</div>
    <div class="mini-metric-target" style="color:${ok ? "var(--green-dk)" : "var(--text-3)"};">
      ${icon(ok ? "check" : "minus", 10)} ≥ ${target}
    </div>
  </div>`;
}

// ── TABLA COMPARATIVA ───────────────────────────────
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

    // Winner highlight
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
          <th>Métrica</th>
          <th>Objetivo</th>
          <th>${baseline.model_display}</th>
          <th>${transformer.model_display}</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="table-note">
      ${icon("info", 11)} Métricas sobre el conjunto de prueba (15% del dataset). 
      Misma partición para ambos modelos.
    </div>
  </div>`;
}

// ── MATRIZ DE CONFUSIÓN ──────────────────────────────
function renderConfusionMatrix(matrix, classes, modelKey) {
  if (!matrix || !classes.length) return `<div class="cell-na">No disponible</div>`;

  // Totales por fila para calcular porcentajes
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);

  // Valor máximo fuera de diagonal para escala de color
  let maxOff = 1;
  matrix.forEach((row, i) =>
    row.forEach((cell, j) => {
      if (i !== j && cell > maxOff) maxOff = cell;
    })
  );

  const headerCells = classes.map((c) =>
    `<th class="cm-header">${c}</th>`
  ).join("");

  const bodyRows = matrix.map((row, i) => {
    const cells = row.map((cell, j) => {
      const isDiag = i === j;
      const pct = rowTotals[i] > 0 ? (cell / rowTotals[i] * 100).toFixed(1) : "0.0";

      let bg = "transparent";
      if (isDiag) {
        const intensity = Math.min(0.85, 0.15 + (cell / rowTotals[i]) * 0.70);
        bg = `rgba(34,197,94,${intensity})`;   // verde para diagonal
      } else if (cell > 0) {
        const intensity = Math.min(0.60, 0.05 + (cell / maxOff) * 0.55);
        bg = `rgba(239,68,68,${intensity})`;   // rojo para errores
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
      <div class="cm-axis-label cm-axis-y">Real →</div>
      <div class="cm-table-outer">
        <div class="cm-axis-label cm-axis-x">← Predicho</div>
        <table class="confusion-matrix">
          <thead>
            <tr>
              <th></th>
              ${headerCells}
              <th class="cm-header cm-total-header">Total</th>
            </tr>
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

// ── LEYENDA DE MÉTRICAS ──────────────────────────────
function renderMetricsLegend() {
  const rows = [
    ["F1-Macro",      "Media del F1 de cada clase, sin ponderar por frecuencia. Penaliza ignorar clases minoritarias (Neutral, Negativo).", "≥ 0.65"],
    ["F1-Weighted",   "Media del F1 ponderada por tamaño de clase. Optimista con datasets desbalanceados.", "≥ 0.80"],
    ["Bal. Accuracy", "Accuracy calculada por clase y promediada. No favorece a la clase mayoritaria.", "≥ 0.70"],
    ["ROC-AUC",       "Área bajo la curva ROC (macro). Capacidad del modelo de separar clases a distintos umbrales.", "≥ 0.85"],
  ];

  return `
  <table class="legend-table">
    <thead>
      <tr><th>Métrica</th><th>¿Qué mide?</th><th>Objetivo</th></tr>
    </thead>
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