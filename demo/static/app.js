/* ═══════════════════════════════════════════════════
   GATOBYTE · Review Search — app.js
   Vanilla JS, zero dependencies
   ═══════════════════════════════════════════════════ */

"use strict";

// ── STATE ──────────────────────────────────────────
const state = {
  info: null,
  lastResults: null,
  topK: 5,
  minScore: 0.0,
};

// ── DOM REFS ────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

// ── ICON HELPER ─────────────────────────────────────
function icon(name, size = 14) {
  return `<i data-lucide="${name}" style="width:${size}px;height:${size}px;vertical-align:-2px;flex-shrink:0;"></i>`;
}

// ── INIT ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupSearch();
  setupSliders();
  setupChips();
  await loadInfo();
  if (window.lucide) lucide.createIcons();
});

// ── TABS ────────────────────────────────────────────
function setupTabs() {
  const btns  = $$(".tab-btn");
  const panels = $$(".tab-panel");

  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      btns.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      $(`tab-${target}`).classList.add("active");

      // Lazy-load tab content
      if (target === "performance")  loadEval();
      if (target === "explorer")     loadFilters().then(() => loadChunks());
      if (target === "status")       loadStatus();
    });
  });
}

// ── SEARCH ──────────────────────────────────────────
function setupSearch() {
  const btn   = $("searchBtn");
  const input = $("queryInput");

  btn.addEventListener("click", doSearch);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });
}

async function doSearch() {
  const q = $("queryInput").value.trim();
  if (!q) return;

  const btn = $("searchBtn");
  btn.classList.add("loading");
  btn.innerHTML = `<span class="spinner"></span> Buscando…`;

  showLoading($("resultsArea"));

  try {
    const url = `/api/search?q=${encodeURIComponent(q)}&top_k=${state.topK}&min_score=${state.minScore}`;
    const res  = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.lastResults = data;
    renderResults(data);
  } catch (err) {
    showError($("resultsArea"), "Error al conectar con la API.", err.message);
  } finally {
    btn.classList.remove("loading");
    btn.innerHTML = `${icon("search", 16)} Buscar reseñas`;
    if (window.lucide) lucide.createIcons({ nodes: [btn] });
  }
}

function setupChips() {
  $$(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.query;
      $("queryInput").value = q;
      doSearch();
    });
  });
}

// ── SLIDERS ─────────────────────────────────────────
function setupSliders() {
  const topKSlider      = $("topKSlider");
  const topKVal         = $("topKVal");
  const minScoreSlider  = $("minScoreSlider");
  const minScoreVal     = $("minScoreVal");

  topKSlider.addEventListener("input", () => {
    state.topK = parseInt(topKSlider.value);
    topKVal.textContent = state.topK;
  });

  minScoreSlider.addEventListener("input", () => {
    state.minScore = parseFloat(minScoreSlider.value);
    minScoreVal.textContent = state.minScore.toFixed(2);
  });
}

// ── LOAD INFO ────────────────────────────────────────
async function loadInfo() {
  try {
    const res  = await fetch("/api/info");
    const info = await res.json();
    state.info = info;

    $("statChunks").textContent = info.num_chunks.toLocaleString("es");
    $("statSample").textContent = info.sample_size.toLocaleString("es");
    $("statDim").textContent    = info.embedding_dim;

    // Set default top_k from config
    if (info.default_top_k) {
      state.topK = info.default_top_k;
      $("topKSlider").value = info.default_top_k;
      $("topKVal").textContent = info.default_top_k;
    }
  } catch (err) {
    console.warn("No se pudo cargar /api/info", err);
  }
}

// ── RENDER RESULTS ────────────────────────────────────
function renderResults(data) {
  const area = $("resultsArea");

  if (!data.results || data.results.length === 0) {
    area.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">${icon("search", 44)}</div>
        <h3>Sin resultados</h3>
        <p>No encontramos reseñas para esta consulta.
           Intenta con otras palabras o baja el score mínimo.</p>
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [area] });
    return;
  }

  const { results, meta } = data;

  // ── SUMMARY ───────────────────────────────────────
  const avgRating = meta.avg_rating;
  const distHTML  = buildDistBars(results);
  const sentHTML  = buildSentChips(results);
  const starsHTML = buildStars(avgRating, 20);

  const summaryHTML = `
  <div class="summary-panel">
    <div class="sum-rating-block">
      <div class="sum-big-num">${avgRating != null ? avgRating.toFixed(1) : "—"}</div>
      <div class="sum-stars">${starsHTML}</div>
      <div class="sum-caption">${results.length} reseñas recuperadas</div>
    </div>
    <div class="sum-dist">
      <div class="sum-section-title">Distribución de calificaciones</div>
      ${distHTML}
    </div>
    <div class="sum-sent">
      <div class="sum-section-title">Sentimiento</div>
      ${sentHTML}
    </div>
  </div>`;

  // ── CARDS ─────────────────────────────────────────
  const cardsHTML = results.map((r, i) => buildCard(r, i + 1)).join("");

  // ── HEADER ────────────────────────────────────────
  const headerHTML = `
  <div class="results-header">
    <div class="results-title">Reseñas más relevantes</div>
    <div class="meta-pills">
      <span class="mpill">${icon("zap", 11)} ${meta.latency_ms} ms</span>
      <span class="mpill">${icon("crosshair", 11)} Top-${meta.top_k}</span>
      <span class="mpill">${icon("trending-up", 11)} Relevancia ${meta.avg_score.toFixed(2)}</span>
    </div>
  </div>`;

  // ── SIDEBAR ───────────────────────────────────────
  const info = state.info || {};
  const sidebarHTML = `
  <aside class="results-sidebar">
    <div class="scard">
      <div class="scard-title">${icon("radio", 12)} Búsqueda actual</div>
      <div class="scard-row"><span>Resultados</span><strong>${results.length}</strong></div>
      <div class="scard-row"><span>Relevancia media</span><strong>${meta.avg_score.toFixed(3)}</strong></div>
      <div class="scard-row"><span>Latencia</span><strong>${meta.latency_ms} ms</strong></div>
      <div class="scard-row"><span>Top-K</span><strong>${meta.top_k}</strong></div>
    </div>
    <div class="scard">
      <div class="scard-title">${icon("database", 12)} Índice FAISS</div>
      <div class="scard-row"><span>Fragmentos</span><strong>${(info.num_chunks || 0).toLocaleString("es")}</strong></div>
      <div class="scard-row"><span>Muestra</span><strong>${(info.sample_size || 0).toLocaleString("es")}</strong></div>
      <div class="scard-row"><span>Dimensión</span><strong>${info.embedding_dim || "—"}</strong></div>
    </div>
    <div class="scard scard-tip">
      <div class="scard-title">${icon("lightbulb", 12)} Tips de búsqueda</div>
      <ul class="tip-list">
        <li>Describe el <b>problema</b> con tus propias palabras</li>
        <li>Puedes escribir en <b>español o inglés</b></li>
        <li>Frases cortas y directas funcionan mejor</li>
        <li>Baja el <b>score mínimo</b> si hay pocos resultados</li>
      </ul>
    </div>
  </aside>`;

  // ── RAG CONTEXT ───────────────────────────────────
  const ragContext = results.map((r, i) =>
    `[Fuente ${i+1}] ASIN=${r.parent_asin} | chunk=${r.chunk_number} | sentiment=${r.sentiment}\n${r.chunk_text}`
  ).join("\n\n");

  const ragHTML = `
  <details class="rag-section">
    <summary class="rag-toggle">${icon("file-text", 14)} Contexto RAG generado</summary>
    <div class="rag-body">
      <p>Texto listo para enviar a un LLM junto con las fuentes recuperadas.</p>
      <textarea class="rag-textarea" rows="10" readonly>${escapeHtml(ragContext)}</textarea>
    </div>
  </details>`;

  area.innerHTML = `
  <div class="results-layout">
    <div>
      ${summaryHTML}
      ${headerHTML}
      <div class="rcards-list">${cardsHTML}</div>
      ${ragHTML}
    </div>
    ${sidebarHTML}
  </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [area] });
}

// ── BUILD CARD ────────────────────────────────────────
function buildCard(r, rank) {
  const stars     = buildStars(r.rating, 14);
  const ratingNum = r.rating != null ? parseFloat(r.rating).toFixed(1) : "—";
  const title     = escapeHtml(r.title || `Producto ${r.parent_asin}`);
  const category  = escapeHtml(r.main_category || "Electrónica");
  const asin      = escapeHtml(r.parent_asin);
  const text      = escapeHtml(truncate(r.chunk_text, 420));
  const votes     = r.helpful_vote || 0;
  const sbar      = buildScoreBar(r.score);
  const { cls, label } = sentimentInfo(r.sentiment);

  return `
  <article class="rcard">
    <div class="rcard-rank">${rank}</div>
    <div class="rcard-body">
      <div class="rcard-top">
        <div style="min-width:0;flex:1;">
          <div class="rcard-stars-row">
            <div class="rcard-stars">${stars}</div>
            <span class="rcard-rating-num">${ratingNum}</span>
          </div>
          <div class="rcard-title">${title}</div>
          <div class="rcard-pills">
            <span class="rpill">${icon("hash", 11)} ${asin}</span>
            <span class="rpill">${icon("folder", 11)} ${category}</span>
            <span class="rpill">Chunk #${r.chunk_number}</span>
          </div>
        </div>
        <div class="rcard-helpful">
          ${icon("thumbs-up", 13)}<span>${votes} útiles</span>
        </div>
      </div>
      <blockquote class="rcard-quote">${text}</blockquote>
      <div class="rcard-footer">
        <span class="badge badge-${cls}">${label}</span>
        <div class="score-wrap">
          <span class="score-caption">Relevancia</span>
          ${sbar}
        </div>
      </div>
    </div>
  </article>`;
}

// ── BUILD DIST BARS ───────────────────────────────────
function buildDistBars(results) {
  const counts = { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 };
  results.forEach((r) => {
    if (r.rating != null) {
      const n = Math.max(1, Math.min(5, Math.round(parseFloat(r.rating))));
      counts[n]++;
    }
  });
  const total = Math.max(1, Object.values(counts).reduce((a, b) => a + b, 0));

  return [5, 4, 3, 2, 1].map((star) => {
    const count = counts[star];
    const pct   = Math.max(1, Math.round((count / total) * 100));
    const color = star >= 4 ? "#22C55E" : star === 3 ? "#F59E0B" : "#EF4444";
    const label = "★".repeat(star);
    return `
    <div class="bar-row">
      <span class="bar-star">${label}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%;background:${color};"></div>
      </div>
      <span class="bar-count">${count}</span>
    </div>`;
  }).join("");
}

// ── BUILD SENTIMENT CHIPS ────────────────────────────
function buildSentChips(results) {
  const counts = { positive: 0, negative: 0, mixed: 0, neutral: 0 };
  results.forEach((r) => { counts[sentClass(r.sentiment)]++; });
  const total = Math.max(1, results.length);

  const items = [
    { key: "positive", lucide: "smile",        label: "Positivo", cls: "sent-positive" },
    { key: "negative", lucide: "frown",        label: "Negativo", cls: "sent-negative" },
    { key: "mixed",    lucide: "meh",          label: "Mixto",    cls: "sent-mixed"    },
    { key: "neutral",  lucide: "minus-circle", label: "Neutral",  cls: "sent-neutral"  },
  ];

  return items.map(({ key, lucide: iconName, label, cls }) => {
    const count = counts[key];
    const pct   = Math.round((count / total) * 100);
    return `
    <div class="sent-chip ${cls}">
      ${icon(iconName, 15)}
      <span class="s-label">${label}</span>
      <span class="s-count">${count}</span>
      <span class="s-pct">(${pct}%)</span>
    </div>`;
  }).join("");
}

// ── BUILD STARS ───────────────────────────────────────
function buildStars(rating, size = 16) {
  if (rating == null) return `<span style="color:#CBD5E1;font-size:${size}px;">☆☆☆☆☆</span>`;
  const n = Math.max(0, Math.min(5, Math.round(parseFloat(rating))));
  return `<span style="font-size:${size}px;letter-spacing:2px;">` +
    `<span style="color:#F59E0B;">${"★".repeat(n)}</span>` +
    `<span class="empty" style="color:#CBD5E1;">${"☆".repeat(5 - n)}</span>` +
    `</span>`;
}

// ── BUILD SCORE BAR ───────────────────────────────────
function buildScoreBar(score) {
  const pct   = Math.round(score * 100);
  const color = score >= 0.7 ? "#22C55E" : score >= 0.45 ? "#F59E0B" : "#EF4444";
  return `
  <div class="score-wrap">
    <div class="score-track">
      <div class="score-fill" style="width:${pct}%;background:${color};"></div>
    </div>
    <span class="score-num">${score.toFixed(2)}</span>
  </div>`;
}

// ── SENTIMENT HELPERS ─────────────────────────────────
function sentClass(sentiment) {
  const s = (sentiment || "").toLowerCase();
  if (s.includes("pos")) return "positive";
  if (s.includes("neg")) return "negative";
  if (s.includes("mix")) return "mixed";
  return "neutral";
}

function sentimentInfo(sentiment) {
  const cls = sentClass(sentiment);
  const iconMap = {
    positive: "smile",
    negative: "frown",
    mixed:    "meh",
    neutral:  "minus-circle",
  };
  const textMap = {
    positive: "Positivo",
    negative: "Negativo",
    mixed:    "Mixto",
    neutral:  "Neutral",
  };
  const label = `${icon(iconMap[cls] || "circle", 12)} ${textMap[cls] || "Neutral"}`;
  return { cls, label };
}

// ── EVAL ─────────────────────────────────────────────
async function loadEval() {
  const area = $("evalContent");
  showLoading(area);

  try {
    const res  = await fetch("/api/eval");
    const data = await res.json();

    if (!data.rows || data.rows.length === 0) {
      area.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">${icon("bar-chart-2", 44)}</div>
          <h3>Sin datos de evaluación</h3>
          <p>${data.message || "Ejecuta python src/02_evaluate_retrieval.py"}</p>
        </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [area] });
      return;
    }

    const rows = data.rows.map((r) => `
    <tr>
      <td>${escapeHtml(r.query)}</td>
      <td>${metricBadge(r.precision_at_k, 0.6)}</td>
      <td>${metricBadge(r.mrr, 0.6)}</td>
      <td>${metricBadge(r.cosine_sim, 0.35)}</td>
      <td>${latencyBadge(r.latency_ms)}</td>
    </tr>`).join("");

    area.innerHTML = `
    <div class="eval-table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Query</th>
            <th>Precision@K</th>
            <th>MRR</th>
            <th>Cosine Sim</th>
            <th>Latencia (ms)</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [area] });
  } catch (err) {
    showError(area, "No se pudo cargar el reporte.", err.message);
  }
}

function metricBadge(val, threshold) {
  const cls = val >= threshold ? "metric-good" : val >= threshold * 0.7 ? "metric-warn" : "metric-bad";
  return `<span class="metric-val ${cls}">${val.toFixed(3)}</span>`;
}

function latencyBadge(ms) {
  const cls = ms < 200 ? "metric-good" : ms < 500 ? "metric-warn" : "metric-bad";
  return `<span class="metric-val ${cls}">${ms.toFixed(0)} ms</span>`;
}

// ── EXPLORER ──────────────────────────────────────────
async function loadFilters() {
  try {
    const res  = await fetch("/api/filters");
    const data = await res.json();

    const sentSel = $("sentimentFilter");
    const catSel  = $("categoryFilter");

    data.sentiments.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s;
      sentSel.appendChild(opt);
    });

    data.categories.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c; opt.textContent = c;
      catSel.appendChild(opt);
    });
  } catch (err) {
    console.warn("No se pudieron cargar los filtros", err);
  }
}

async function loadChunks() {
  const area      = $("chunksContent");
  const sentiment = $("sentimentFilter").value;
  const category  = $("categoryFilter").value;
  const n         = $("nChunks").value;

  showLoading(area);

  try {
    const params = new URLSearchParams({ sentiment, category, n });
    const res    = await fetch(`/api/chunks?${params}`);
    const data   = await res.json();

    if (!data.rows || data.rows.length === 0) {
      area.innerHTML = `<div class="empty-state">
        <div class="empty-icon">${icon("database", 44)}</div>
        <h3>Sin chunks</h3>
        <p>No hay fragmentos con estos filtros.</p>
      </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [area] });
      return;
    }

    const rows = data.rows.map((r) => {
      const { cls, label } = sentimentInfo(r.sentiment);
      return `
      <tr>
        <td>${r.chunk}</td>
        <td><code style="font-size:11px;background:#F1F5F9;padding:2px 6px;border-radius:4px;">${escapeHtml(r.asin)}</code></td>
        <td><span class="badge badge-${cls}" style="font-size:11px;">${label}</span></td>
        <td>${escapeHtml(r.category)}</td>
        <td>${r.rating != null ? buildStars(r.rating, 12) : "—"}</td>
        <td style="font-family:'Lora',serif;font-size:13px;color:#475569;max-width:400px;">${escapeHtml(r.text)}…</td>
      </tr>`;
    }).join("");

    area.innerHTML = `
    <div class="eval-table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Chunk</th><th>ASIN</th><th>Sentimiento</th>
            <th>Categoría</th><th>Rating</th><th>Texto</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div style="font-size:12px;color:#94A3B8;margin-top:8px;text-align:right;">
      Mostrando ${data.rows.length} de ${data.total} chunks
    </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [area] });
  } catch (err) {
    showError(area, "No se pudieron cargar los chunks.", err.message);
  }
}

$("refreshChunks") && $("refreshChunks").addEventListener("click", loadChunks);

// ── STATUS ────────────────────────────────────────────
async function loadStatus() {
  const area = $("statusContent");
  showLoading(area);

  try {
    const res  = await fetch("/api/info");
    const info = await res.json();

    const isOk      = info.status === "OK";
    const statusCls = isOk ? "ok" : "warn";
    const statusTxt = isOk
      ? `${icon("check-circle", 18)} OK`
      : `${icon("alert-triangle", 18)} ACTUALIZAR`;

    area.innerHTML = `
    <div class="status-grid">
      <div class="status-card">
        <div class="status-label">Versión activa</div>
        <div class="status-value">${escapeHtml(info.version)}</div>
      </div>
      <div class="status-card">
        <div class="status-label">Edad del índice</div>
        <div class="status-value">${info.age_days} días</div>
      </div>
      <div class="status-card ${statusCls}">
        <div class="status-label">Estado</div>
        <div class="status-value">${statusTxt}</div>
      </div>
      <div class="status-card">
        <div class="status-label">Retención</div>
        <div class="status-value">${info.keep_n_versions} versiones</div>
      </div>
    </div>

    <div class="pipeline-card">
      <h3>${icon("git-merge", 14)} Flujo MLOps</h3>
      <div class="pipeline-steps">
        <div class="pstep">
          <div class="pstep-num">01</div>
          <div>
            <div class="pstep-name">build_index.py</div>
            <div class="pstep-desc">Data Prep · Embeddings · FAISS</div>
          </div>
        </div>
        <div class="pstep-arrow">→</div>
        <div class="pstep">
          <div class="pstep-num">02</div>
          <div>
            <div class="pstep-name">evaluate_retrieval.py</div>
            <div class="pstep-desc">Precision@K · MRR · Reportes</div>
          </div>
        </div>
        <div class="pstep-arrow">→</div>
        <div class="pstep">
          <div class="pstep-num">03</div>
          <div>
            <div class="pstep-name">update_policy.py</div>
            <div class="pstep-desc">Triggers · KEEP / REBUILD</div>
          </div>
        </div>
        <div class="pstep-arrow">→</div>
        <div class="pstep active">
          <div class="pstep-num">04</div>
          <div>
            <div class="pstep-name">main.py</div>
            <div class="pstep-desc">FastAPI · Demo · Deployment</div>
          </div>
        </div>
      </div>
    </div>

    <div class="model-info">
      <h3>Modelo e índice</h3>
      <div class="model-row">
        <span>Modelo de embeddings</span>
        <code>${escapeHtml(info.model_name)}</code>
      </div>
      <div class="model-row">
        <span>Versión del índice</span>
        <code>${escapeHtml(info.version)}</code>
      </div>
      <div class="model-row">
        <span>Chunk overlap</span>
        <code>${info.chunk_overlap}</code>
      </div>
      <div class="model-row">
        <span>Dimensión embedding</span>
        <code>${info.embedding_dim}</code>
      </div>
      <div class="model-row">
        <span>Fragmentos indexados</span>
        <code>${info.num_chunks.toLocaleString("es")}</code>
      </div>
    </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [area] });
  } catch (err) {
    showError(area, "No se pudo cargar el estado.", err.message);
  }
}

// ── HELPERS ───────────────────────────────────────────
function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function truncate(str, limit = 420) {
  str = String(str || "");
  return str.length <= limit ? str : str.slice(0, limit).trimEnd() + "…";
}

function showLoading(container) {
  container.innerHTML = `<div class="loading-state"><span class="spinner"></span> Cargando…</div>`;
}

function showError(container, msg, detail = "") {
  container.innerHTML = `
  <div class="error-state">
    <div class="error-icon">${icon("alert-circle", 44)}</div>
    <h3>Algo salió mal</h3>
    <p>${escapeHtml(msg)}</p>
    ${detail ? `<p style="margin-top:8px;font-size:12px;color:#94A3B8;">${escapeHtml(detail)}</p>` : ""}
  </div>`;
  if (window.lucide) lucide.createIcons({ nodes: [container] });
}
