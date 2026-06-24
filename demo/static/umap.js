/* ═══════════════════════════════════════════════════
   GATOBYTE · Visualización UMAP — umap.js
   Depende de helpers de app.js: $, icon, escapeHtml,
   showLoading, showError, sentimentInfo, buildStars
   ═══════════════════════════════════════════════════ */

"use strict";

// ── CONSTANTES DE COLOR ─────────────────────────────
const UMAP_COLORS = {
  // Por sentimiento
  sentiment: {
    positive: { fill: "#22C55E", stroke: "#15803D", label: "Positivo" },
    negative: { fill: "#EF4444", stroke: "#B91C1C", label: "Negativo" },
    neutral:  { fill: "#94A3B8", stroke: "#475569", label: "Neutral"  },
  },
  // Por rating (1-5)
  rating: {
    1: { fill: "#EF4444", stroke: "#B91C1C", label: "Rating 1 ★"     },
    2: { fill: "#F97316", stroke: "#C2410C", label: "Rating 2 ★★"    },
    3: { fill: "#EAB308", stroke: "#A16207", label: "Rating 3 ★★★"   },
    4: { fill: "#22C55E", stroke: "#15803D", label: "Rating 4 ★★★★"  },
    5: { fill: "#16A34A", stroke: "#14532D", label: "Rating 5 ★★★★★" },
  },
  // Por categoría (top 8 + otras)
  category: [
    "#7C3AED", "#2563EB", "#0891B2", "#059669",
    "#D97706", "#DC2626", "#DB2777", "#65A30D",
  ],
};

// ── ESTADO ──────────────────────────────────────────
const umapState = {
  points:     [],       // [{x, y, sentiment, rating, main_category}]
  colorBy:    "sentiment",
  selected:   null,     // índice del punto clickeado
  canvas:     null,
  ctx:        null,
  transform:  { scale: 1, tx: 0, ty: 0 },
  categories: [],       // lista única de categorías
  catColorMap: {},      // categoria → color hex
};

const USE_MOCK_UMAP = false; // ← cambiar a false cuando /api/umap esté listo

// ── INIT ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Se llama desde setupTabs en app.js cuando se activa el tab "umap"
});

async function loadUmap() {
  const area = $("umapContent");
  if (!area) return;
  if (umapState.points.length > 0) { renderUmapShell(); return; } // ya cargado
  showLoading(area);

  try {
    let raw;
    if (USE_MOCK_UMAP) {
      raw = generateMockPoints(300);
    } else {
      const res = await fetch("/api/embeddings/umap");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      raw = await res.json(); // [{umap1, umap2, sentiment, rating, main_category}]
    }

    umapState.points = raw.map((d) => ({
      x:            parseFloat(d.umap1 ?? d.UMAP1),
      y:            parseFloat(d.umap2 ?? d.UMAP2),
      sentiment:    (d.sentiment || "neutral").toLowerCase(),
      rating:       d.rating != null ? Math.round(parseFloat(d.rating)) : null,
      main_category: d.main_category || "Otros",
    }));

    // Categorías únicas → mapa de color
    const cats = [...new Set(umapState.points.map((p) => p.main_category))].sort();
    umapState.categories = cats;
    cats.forEach((c, i) => {
      umapState.catColorMap[c] = UMAP_COLORS.category[i % UMAP_COLORS.category.length];
    });

    renderUmapShell();
  } catch (err) {
    showError(area, "No se pudieron cargar los embeddings UMAP.", err.message);
  }
}

// ── SHELL HTML ──────────────────────────────────────
function renderUmapShell() {
  const area = $("umapContent");
  if (!area) return;

  const pts = umapState.points;

  // ── KPIs
  const total    = pts.length;
  const nPos     = pts.filter((p) => p.sentiment === "positive").length;
  const nNeg     = pts.filter((p) => p.sentiment === "negative").length;
  const nNeu     = pts.filter((p) => p.sentiment === "neutral").length;
  const avgRat   = pts.filter((p) => p.rating).reduce((s, p) => s + p.rating, 0) /
                   pts.filter((p) => p.rating).length || 0;
  const nCats    = umapState.categories.length;

  area.innerHTML = `

  <!-- KPIs -->
  <div class="umap-kpis">
    <div class="umap-kpi">
      <div class="umap-kpi-val">${total.toLocaleString("es")}</div>
      <div class="umap-kpi-label">${icon("layers", 12)} Embeddings</div>
    </div>
    <div class="umap-kpi umap-kpi--pos">
      <div class="umap-kpi-val">${nPos.toLocaleString("es")}</div>
      <div class="umap-kpi-label">${icon("smile", 12)} Positivos</div>
    </div>
    <div class="umap-kpi umap-kpi--neg">
      <div class="umap-kpi-val">${nNeg.toLocaleString("es")}</div>
      <div class="umap-kpi-label">${icon("frown", 12)} Negativos</div>
    </div>
    <div class="umap-kpi umap-kpi--neu">
      <div class="umap-kpi-val">${nNeu.toLocaleString("es")}</div>
      <div class="umap-kpi-label">${icon("minus-circle", 12)} Neutrales</div>
    </div>
    <div class="umap-kpi">
      <div class="umap-kpi-val">${avgRat.toFixed(1)} ★</div>
      <div class="umap-kpi-label">${icon("star", 12)} Rating medio</div>
    </div>
    <div class="umap-kpi">
      <div class="umap-kpi-val">${nCats}</div>
      <div class="umap-kpi-label">${icon("tag", 12)} Categorías</div>
    </div>
  </div>

  <!-- Segmented control -->
  <div class="umap-controls">
    <div class="umap-seg-wrap">
      <span class="umap-seg-label">Colorear por</span>
      <div class="umap-seg" role="group">
        <button class="umap-seg-btn active" data-colorby="sentiment">Sentimiento</button>
        <button class="umap-seg-btn"        data-colorby="rating">Rating</button>
        <button class="umap-seg-btn"        data-colorby="category">Categoría</button>
      </div>
    </div>
    <div class="umap-hint">${icon("mouse-pointer-click", 12)} Haz clic en un punto para ver detalles</div>
  </div>

  <!-- Scatter + panel lateral -->
  <div class="umap-main">
    <div class="umap-canvas-wrap">
      <canvas id="umapCanvas"></canvas>
    </div>
    <div class="umap-side" id="umapSide">
      ${renderSidePlaceholder()}
    </div>
  </div>

  <!-- Leyenda -->
  <div class="umap-legend" id="umapLegend"></div>
  `;

  if (window.lucide) lucide.createIcons({ nodes: [area] });

  setupSegControl();
  initCanvas();
  renderLegend();
}

// ── SEGMENTED CONTROL ───────────────────────────────
function setupSegControl() {
  const btns = document.querySelectorAll(".umap-seg-btn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      umapState.colorBy  = btn.dataset.colorby;
      umapState.selected = null;
      $("umapSide").innerHTML = renderSidePlaceholder();
      if (window.lucide) lucide.createIcons({ nodes: [$("umapSide")] });
      drawScatter();
      renderLegend();
    });
  });
}

// ── CANVAS ──────────────────────────────────────────
function initCanvas() {
  const wrap   = document.querySelector(".umap-canvas-wrap");
  const canvas = $("umapCanvas");
  if (!canvas || !wrap) return;

  umapState.canvas = canvas;
  umapState.ctx    = canvas.getContext("2d");

  const resize = () => {
    canvas.width  = wrap.clientWidth;
    canvas.height = Math.min(420, wrap.clientWidth * 0.65);
    computeTransform();
    drawScatter();
  };

  resize();
  window.addEventListener("resize", resize);
  canvas.addEventListener("click", onCanvasClick);
}

function computeTransform() {
  const pts = umapState.points;
  if (!pts.length) return;

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const W = umapState.canvas.width;
  const H = umapState.canvas.height;
  const pad = 28;

  const scaleX = (W - pad * 2) / (maxX - minX || 1);
  const scaleY = (H - pad * 2) / (maxY - minY || 1);
  const scale  = Math.min(scaleX, scaleY);

  const dataW = (maxX - minX) * scale;
  const dataH = (maxY - minY) * scale;

  umapState.transform = {
    scale,
    tx: pad + (W - pad * 2 - dataW) / 2 - minX * scale,
    ty: pad + (H - pad * 2 - dataH) / 2 - minY * scale,
    minX, maxX, minY, maxY,
  };
}

function toCanvas(x, y) {
  const { scale, tx, ty } = umapState.transform;
  return { cx: x * scale + tx, cy: y * scale + ty };
}

function drawScatter() {
  const { canvas, ctx, points, colorBy, selected } = umapState;
  if (!canvas || !ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Dibujar puntos no seleccionados primero
  points.forEach((p, i) => {
    if (i === selected) return;
    const { fill } = getPointColor(p, colorBy);
    const { cx, cy } = toCanvas(p.x, p.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = fill + "99"; // semi-transparente
    ctx.fill();
  });

  // Punto seleccionado encima
  if (selected !== null && points[selected]) {
    const p = points[selected];
    const { fill, stroke } = getPointColor(p, colorBy);
    const { cx, cy } = toCanvas(p.x, p.y);

    // Halo ámbar
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.fillStyle = "#F59E0B33";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.strokeStyle = "#F59E0B";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Punto
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function getPointColor(p, colorBy) {
  if (colorBy === "sentiment") {
    return UMAP_COLORS.sentiment[p.sentiment] || UMAP_COLORS.sentiment.neutral;
  }
  if (colorBy === "rating") {
    const key = p.rating != null ? Math.max(1, Math.min(5, p.rating)) : 3;
    return UMAP_COLORS.rating[key];
  }
  if (colorBy === "category") {
    const hex = umapState.catColorMap[p.main_category] || "#94A3B8";
    return { fill: hex, stroke: hex };
  }
  return { fill: "#94A3B8", stroke: "#475569" };
}

// ── CLICK EN CANVAS ─────────────────────────────────
function onCanvasClick(e) {
  const rect   = umapState.canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  let closest = null;
  let minDist = Infinity;

  umapState.points.forEach((p, i) => {
    const { cx, cy } = toCanvas(p.x, p.y);
    const d = Math.hypot(mouseX - cx, mouseY - cy);
    if (d < minDist) { minDist = d; closest = i; }
  });

  if (closest !== null && minDist < 14) {
    umapState.selected = closest;
    drawScatter();
    renderSideDetail(umapState.points[closest]);
  } else {
    umapState.selected = null;
    drawScatter();
    $("umapSide").innerHTML = renderSidePlaceholder();
    if (window.lucide) lucide.createIcons({ nodes: [$("umapSide")] });
  }
}

// ── PANEL LATERAL ───────────────────────────────────
function renderSidePlaceholder() {
  return `
  <div class="umap-side-placeholder">
    ${icon("mouse-pointer-click", 28)}
    <p>Haz clic en un punto del scatter para ver los detalles del embedding</p>
  </div>`;
}

function renderSideDetail(p) {
  const side = $("umapSide");
  if (!side) return;

  const { cls } = sentimentInfo(p.sentiment);
  const colorMap = {
    positive: { bg: "var(--green-lt)",  text: "var(--green-dk)",  border: "var(--green)"  },
    negative: { bg: "var(--red-lt)",    text: "var(--red-dk)",    border: "var(--red)"    },
    neutral:  { bg: "var(--amber-lt)",  text: "var(--amber-dk)",  border: "var(--amber)"  },
  };
  const col = colorMap[cls] || colorMap.neutral;

  const labelMap = { positive: "Positivo", negative: "Negativo", neutral: "Neutral" };

  // Color de categoría
  const catColor = umapState.catColorMap[p.main_category] || "#94A3B8";

  side.innerHTML = `
  <div class="umap-detail">
    <div class="umap-detail-title">${icon("info", 13)} Detalle del punto</div>

    <!-- Sentimiento -->
    <div class="umap-detail-badge" style="background:${col.bg};border:1px solid ${col.border};color:${col.text};">
      ${icon(cls === "positive" ? "smile" : cls === "negative" ? "frown" : "minus-circle", 14)}
      ${labelMap[cls] || "Neutral"}
    </div>

    <!-- Filas de datos -->
    <div class="umap-detail-rows">
      <div class="umap-detail-row">
        <span class="umap-detail-key">${icon("star", 11)} Rating</span>
        <span class="umap-detail-val">
          ${p.rating != null ? buildStars(p.rating, 13) : "—"}
        </span>
      </div>
      <div class="umap-detail-row">
        <span class="umap-detail-key">${icon("tag", 11)} Categoría</span>
        <span class="umap-detail-val" style="color:${catColor};font-weight:500;">
          ${escapeHtml(p.main_category)}
        </span>
      </div>
      <div class="umap-detail-row">
        <span class="umap-detail-key">${icon("map-pin", 11)} UMAP 1</span>
        <span class="umap-detail-val umap-detail-mono">${p.x.toFixed(4)}</span>
      </div>
      <div class="umap-detail-row">
        <span class="umap-detail-key">${icon("map-pin", 11)} UMAP 2</span>
        <span class="umap-detail-val umap-detail-mono">${p.y.toFixed(4)}</span>
      </div>
    </div>

    <div class="umap-detail-note">
      ${icon("info", 11)} Embedding generado con all-MiniLM-L6-v2 (384d → 2d vía UMAP)
    </div>
  </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [side] });
}

// ── LEYENDA ──────────────────────────────────────────
function renderLegend() {
  const leg = $("umapLegend");
  if (!leg) return;

  const { colorBy } = umapState;
  let items = [];

  if (colorBy === "sentiment") {
    items = Object.entries(UMAP_COLORS.sentiment).map(([k, v]) => ({
      color: v.fill,
      label: v.label,
      count: umapState.points.filter((p) => p.sentiment === k).length,
    }));
  } else if (colorBy === "rating") {
    items = Object.entries(UMAP_COLORS.rating).map(([k, v]) => ({
      color: v.fill,
      label: v.label,
      count: umapState.points.filter((p) => p.rating === parseInt(k)).length,
    }));
  } else {
    items = umapState.categories.slice(0, 8).map((cat) => ({
      color: umapState.catColorMap[cat],
      label: cat,
      count: umapState.points.filter((p) => p.main_category === cat).length,
    }));
    const otherCount = umapState.points.filter(
      (p) => !umapState.categories.slice(0, 8).includes(p.main_category)
    ).length;
    if (otherCount > 0) items.push({ color: "#CBD5E1", label: "Otras", count: otherCount });
  }

  leg.innerHTML = items.map((item) => `
    <div class="umap-leg-item">
      <span class="umap-leg-dot" style="background:${item.color};"></span>
      <span class="umap-leg-label">${escapeHtml(item.label)}</span>
      <span class="umap-leg-count">${item.count.toLocaleString("es")}</span>
    </div>
  `).join("");
}

// ── MOCK DATA ────────────────────────────────────────
function generateMockPoints(n) {
  const sentiments  = ["positive", "negative", "neutral"];
  const ratings     = [1, 2, 3, 4, 5];
  const categories  = ["All Electronics", "Computers", "Cell Phones", "Camera & Photo",
                       "Audio", "Amazon Devices", "Wearables", "Tablets"];
  const pts = [];

  // Cluster positivo (arriba derecha)
  for (let i = 0; i < n * 0.5; i++) {
    pts.push({
      umap1: 2 + (Math.random() - 0.5) * 4,
      umap2: 3 + (Math.random() - 0.5) * 3,
      sentiment: "positive",
      rating: Math.random() < 0.8 ? (Math.random() < 0.6 ? 5 : 4) : 3,
      main_category: categories[Math.floor(Math.random() * categories.length)],
    });
  }
  // Cluster negativo (izquierda)
  for (let i = 0; i < n * 0.3; i++) {
    pts.push({
      umap1: -2 + (Math.random() - 0.5) * 3,
      umap2: 1 + (Math.random() - 0.5) * 2.5,
      sentiment: "negative",
      rating: Math.random() < 0.8 ? (Math.random() < 0.6 ? 1 : 2) : 3,
      main_category: categories[Math.floor(Math.random() * categories.length)],
    });
  }
  // Cluster neutral (zona media)
  for (let i = 0; i < n * 0.2; i++) {
    pts.push({
      umap1: 0.5 + (Math.random() - 0.5) * 3,
      umap2: 2 + (Math.random() - 0.5) * 3,
      sentiment: "neutral",
      rating: Math.random() < 0.7 ? 3 : (Math.random() < 0.5 ? 4 : 2),
      main_category: categories[Math.floor(Math.random() * categories.length)],
    });
  }
  return pts;
}

window.loadUmap = loadUmap;