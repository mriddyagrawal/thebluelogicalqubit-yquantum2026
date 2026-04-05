// ── Interactive elements — all data-driven from results.json ──
// DATA is declared in index.html (shared)

async function loadData() {
  const resp = await fetch('data/results.json');
  DATA = await resp.json();
  populateSlides();
}

// ── Copy bitstring on click ──
function initBitstringCopy() {
  document.querySelectorAll('.bitstring').forEach(el => {
    el.style.cursor = 'pointer';
    el.title = 'Click to copy';
    el.addEventListener('click', function (e) {
      e.stopPropagation();
      navigator.clipboard.writeText(this.textContent.trim());
      const orig = this.textContent;
      this.textContent = 'Copied!';
      this.style.background = '#c8e6c9';
      setTimeout(() => { this.textContent = orig; this.style.background = ''; }, 1000);
    });
  });
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Main populate ──
function populateSlides() {
  if (!DATA) return;

  buildOverviewTable();

  // P5
  const p5 = DATA.p5;
  setText('p5-qubits', p5.num_qubits);
  setText('p5-orig-gates', p5.original.total_gates.toLocaleString());
  setText('p5-orig-u3', p5.original.gates.u3 || p5.original.gates.u || 0);
  setText('p5-orig-cz', p5.original.gates.cz || 0);
  setText('p5-orig-depth', p5.original.depth);
  setText('p5-final-gates', p5.final.total_gates.toLocaleString());
  setText('p5-final-depth', p5.final.depth);
  setText('p5-method', p5.simulation.method);
  setText('p5-shots', p5.simulation.shots.toLocaleString());
  setText('p5-bond', p5.simulation.bond_dimension);
  setText('p5-bitstring', p5.peak_bitstring);
  buildGateBreakdown('p5-gate-breakdown', p5.original.gates, p5.final.gates);

  // P6
  const p6 = DATA.p6;
  setText('p6-qubits', p6.num_qubits);
  setText('p6-orig-gates', p6.original.total_gates.toLocaleString());
  setText('p6-orig-cz', p6.original_cz.toLocaleString());
  setText('p6-final-gates', p6.final.total_gates.toLocaleString());
  setText('p6-final-cz', p6.final_cz);
  setText('p6-final-u', p6.final_u);
  setText('p6-reduction', p6.reduction_pct + '%');
  setText('p6-bitstring', p6.peak_bitstring);
  setText('p6-orig-depth', p6.original.depth);
  setText('p6-final-depth', p6.final.depth);
  buildGateAnimator('p6-gate-animator', p6);
  buildDonutChart('p6-donut-chart', p6);

  // P7
  const p7 = DATA.p7;
  setText('p7-qubits', p7.num_qubits);
  setText('p7-orig-gates', p7.original.total_gates.toLocaleString());
  setText('p7-components', p7.num_components);
  setText('p7-comp0-q', p7.components[0].num_qubits);
  setText('p7-comp1-q', p7.components[1].num_qubits);
  setText('p7-bitstring', p7.peak_bitstring);
  setText('p7-method', p7.simulation.method);
  setText('p7-shots', p7.simulation.shots_per_component.toLocaleString());
  buildInteractiveGraph('p7-graph-canvas', p7.graph, p7.components);

  initBitstringCopy();
}

// ══════════════════════════════════════════════════════════
// Overview table with real bitstrings (slide 3)
// ══════════════════════════════════════════════════════════
function buildOverviewTable() {
  const tbody = document.getElementById('overview-table-body');
  if (!tbody || !DATA.problems) return;

  const focusProblems = ['P5', 'P6', 'P7', 'P8', 'P9'];
  tbody.innerHTML = '';

  for (const p of DATA.problems) {
    const isFocus = focusProblems.includes(p.id);
    const tr = document.createElement('tr');
    if (isFocus) tr.classList.add('row-highlight');

    const bsDisplay = p.bitstring
      ? `<span class="bitstring" style="font-size:0.75em; padding:4px 8px;">${p.bitstring}</span>`
      : '<span style="color:#9e9e9e;">submitted</span>';

    tr.innerHTML = `
      <td><strong>${p.id}</strong></td>
      <td>${p.name}</td>
      <td>${p.qubits}</td>
      <td>${p.gates.toLocaleString()}</td>
      <td>${p.method}</td>
      <td>${bsDisplay}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ══════════════════════════════════════════════════════════
// P5: Interactive gate breakdown table
// ══════════════════════════════════════════════════════════
function buildGateBreakdown(containerId, origGates, finalGates) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const allKeys = [...new Set([...Object.keys(origGates), ...Object.keys(finalGates)])];
  let html = '<table class="interactive" style="width:100%;font-size:0.85em;">';
  html += '<tr><th>Gate</th><th>Original</th><th>Optimized</th><th>Change</th></tr>';
  for (const k of allKeys) {
    const o = origGates[k] || 0;
    const f = finalGates[k] || 0;
    const diff = f - o;
    const cls = diff < 0 ? 'color:#43a047' : diff > 0 ? 'color:#e53935' : '';
    const sign = diff > 0 ? '+' : '';
    html += `<tr><td><strong>${k}</strong></td><td>${o}</td><td>${f}</td><td style="${cls};font-weight:600">${sign}${diff}</td></tr>`;
  }
  html += '</table>';
  el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
// P6: Animated gate reduction slider
// ══════════════════════════════════════════════════════════
function buildGateAnimator(containerId, p6) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const origTotal = p6.original.total_gates;
  const finalTotal = p6.final.total_gates;
  const origCZ = p6.original_cz;
  const finalCZ = p6.final_cz;

  el.innerHTML = `
    <div class="interactive" style="padding:14px; background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="font-weight:600; color:#283593;">Optimization Progress</span>
        <span id="p6-slider-val" style="font-weight:700; color:#1a237e;">0%</span>
      </div>
      <input type="range" id="p6-slider" min="0" max="100" value="0"
        style="width:100%; cursor:pointer; accent-color:#1e88e5;">
      <div style="display:flex; justify-content:space-around; margin-top:10px; text-align:center;">
        <div>
          <div id="p6-anim-total" style="font-size:1.6em; font-weight:700; color:#1a237e;">${origTotal}</div>
          <div style="font-size:0.8em; color:#78909c;">Total Gates</div>
        </div>
        <div>
          <div id="p6-anim-cz" style="font-size:1.6em; font-weight:700; color:#e53935;">${origCZ}</div>
          <div style="font-size:0.8em; color:#78909c;">CZ Gates</div>
        </div>
        <div>
          <div id="p6-anim-depth" style="font-size:1.6em; font-weight:700; color:#43a047;">${p6.original.depth}</div>
          <div style="font-size:0.8em; color:#78909c;">Depth</div>
        </div>
      </div>
    </div>
  `;

  const slider = document.getElementById('p6-slider');
  slider.addEventListener('input', function () {
    const t = this.value / 100;
    const total = Math.round(origTotal + (finalTotal - origTotal) * t);
    const cz = Math.round(origCZ + (finalCZ - origCZ) * t);
    const depth = Math.round(p6.original.depth + (p6.final.depth - p6.original.depth) * t);
    document.getElementById('p6-anim-total').textContent = total.toLocaleString();
    document.getElementById('p6-anim-cz').textContent = cz.toLocaleString();
    document.getElementById('p6-anim-depth').textContent = depth;
    document.getElementById('p6-slider-val').textContent = Math.round(t * p6.reduction_pct) + '% reduced';
    // also update donut
    updateDonut(t, p6);
  });
}

// ══════════════════════════════════════════════════════════
// P6: Donut chart — gate composition before/after
// ══════════════════════════════════════════════════════════
let donutCanvas, donutCtx;
function buildDonutChart(containerId, p6) {
  const el = document.getElementById(containerId);
  if (!el) return;

  el.innerHTML = `
    <div class="interactive" style="background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); padding:12px; text-align:center;">
      <div style="font-weight:600; color:#283593; margin-bottom:6px;">Gate Composition</div>
      <canvas id="p6-donut" width="280" height="180"></canvas>
      <div id="p6-donut-legend" style="font-size:0.8em; color:#546e7a; margin-top:4px;"></div>
    </div>
  `;

  donutCanvas = document.getElementById('p6-donut');
  donutCtx = donutCanvas.getContext('2d');
  updateDonut(0, p6);
}

function updateDonut(t, p6) {
  if (!donutCtx) return;
  const ctx = donutCtx;
  const W = donutCanvas.width, H = donutCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const origCZ = p6.original_cz;
  const origU = p6.original.gates.u || 0;
  const origM = p6.original.gates.measure || 0;
  const finalCZ = p6.final_cz;
  const finalU = p6.final_u;
  const finalM = p6.final.gates.measure || 0;

  const cz = Math.round(origCZ + (finalCZ - origCZ) * t);
  const u = Math.round(origU + (finalU - origU) * t);
  const m = Math.round(origM + (finalM - origM) * t);
  const total = cz + u + m;

  const slices = [
    { label: 'CZ', val: cz, color: '#EF5350' },
    { label: 'U', val: u, color: '#42A5F5' },
    { label: 'Measure', val: m, color: '#66BB6A' },
  ];

  // Draw donut
  const cx = W / 2, cy = H / 2, r = 70, inner = 40;
  let angle = -Math.PI / 2;
  for (const s of slices) {
    if (s.val === 0) continue;
    const sweep = (s.val / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx + inner * Math.cos(angle), cy + inner * Math.sin(angle));
    ctx.arc(cx, cy, r, angle, angle + sweep);
    ctx.arc(cx, cy, inner, angle + sweep, angle, true);
    ctx.closePath();
    ctx.fillStyle = s.color;
    ctx.fill();
    angle += sweep;
  }

  // Center text
  ctx.fillStyle = '#1a237e';
  ctx.font = 'bold 18px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(total.toLocaleString(), cx, cy - 6);
  ctx.font = '11px sans-serif';
  ctx.fillStyle = '#78909c';
  ctx.fillText('total gates', cx, cy + 12);

  // Legend
  const legend = document.getElementById('p6-donut-legend');
  if (legend) {
    legend.innerHTML = slices.map(s =>
      `<span style="display:inline-block;margin:0 8px;"><span style="display:inline-block;width:10px;height:10px;background:${s.color};border-radius:2px;margin-right:3px;vertical-align:middle;"></span>${s.label}: ${s.val}</span>`
    ).join('');
  }
}

// ══════════════════════════════════════════════════════════
// P7: Interactive graph — hover + click to select component
// ══════════════════════════════════════════════════════════
function buildInteractiveGraph(_canvasId, graphData, components) {
  const wrapper = document.getElementById('p7-graph-wrapper');
  if (!wrapper) return;

  // Build everything inside the wrapper
  wrapper.innerHTML = `
    <div id="p7-graph-info" style="position:relative; z-index:11; margin-bottom:6px; padding:6px 10px; background:white; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.06); font-size:0.8em; color:#37474f;">
      <span style="color:#9e9e9e;">Click a node to explore</span>
    </div>
    <div id="p7-graph-buttons" style="position:relative; z-index:11; margin-bottom:6px; display:flex; gap:6px;">
      <button class="comp-btn" data-comp="-1" style="flex:1; padding:4px 6px; border:2px solid #78909c; background:white; color:#455a64; border-radius:5px; font-weight:600; cursor:pointer; font-size:0.78em;">All</button>
      <button class="comp-btn" data-comp="0" style="flex:1; padding:4px 6px; border:2px solid #4FC3F7; background:white; color:#0288d1; border-radius:5px; font-weight:600; cursor:pointer; font-size:0.78em;">Comp 0 (${components[0].num_qubits}q)</button>
      <button class="comp-btn" data-comp="1" style="flex:1; padding:4px 6px; border:2px solid #FF8A65; background:white; color:#e64a19; border-radius:5px; font-weight:600; cursor:pointer; font-size:0.78em;">Comp 1 (${components[1].num_qubits}q)</button>
    </div>
    <canvas id="p7-canvas" width="420" height="280" style="position:relative; z-index:11; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); background:white; cursor:crosshair; display:block; width:100%;"></canvas>
  `;

  const canvas = document.getElementById('p7-canvas');
  const btnRow = document.getElementById('p7-graph-buttons');

  const ctx = canvas.getContext('2d');
  const nodes = graphData.nodes;
  const edges = graphData.edges;

  // Normalize positions
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const n of nodes) {
    minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
    minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
  }
  const pad = 35;
  const scaleX = (canvas.width - pad * 2) / (maxX - minX || 1);
  const scaleY = (canvas.height - pad * 2) / (maxY - minY || 1);

  const posMap = {};
  for (const n of nodes) {
    posMap[n.id] = {
      x: pad + (n.x - minX) * scaleX,
      y: pad + (n.y - minY) * scaleY,
      comp: n.component
    };
  }

  let highlightComp = -1;
  let hoveredNode = -1;
  let selectedNode = -1;

  function drawGraph() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const colors = ['#4FC3F7', '#FF8A65'];
    const dimColors = ['rgba(79,195,247,0.15)', 'rgba(255,138,101,0.15)'];

    // Edges
    for (const [u, v] of edges) {
      const a = posMap[u], b = posMap[v];
      const dimA = highlightComp >= 0 && a.comp !== highlightComp;
      const dimB = highlightComp >= 0 && b.comp !== highlightComp;
      const isSelected = (selectedNode === u || selectedNode === v);

      if (dimA || dimB) {
        ctx.strokeStyle = 'rgba(200,200,200,0.15)';
        ctx.lineWidth = 0.5;
      } else if (isSelected) {
        ctx.strokeStyle = 'rgba(26,35,126,0.5)';
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = 'rgba(150,150,150,0.3)';
        ctx.lineWidth = 0.8;
      }
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // Nodes
    for (const n of nodes) {
      const p = posMap[n.id];
      const dim = highlightComp >= 0 && p.comp !== highlightComp;
      const isHovered = hoveredNode === n.id;
      const isSelected = selectedNode === n.id;
      const r = isHovered ? 11 : isSelected ? 10 : 7;

      // Glow for hovered/selected
      if (isHovered || isSelected) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
        ctx.fillStyle = dim ? 'rgba(200,200,200,0.15)' : (p.comp === 0 ? 'rgba(79,195,247,0.25)' : 'rgba(255,138,101,0.25)');
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = dim ? dimColors[p.comp] : colors[p.comp];
      ctx.fill();

      if (isHovered || isSelected) {
        ctx.strokeStyle = '#1a237e';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label
      if (!dim || isHovered) {
        ctx.fillStyle = dim ? 'rgba(150,150,150,0.5)' : '#333';
        ctx.font = (isHovered || isSelected) ? 'bold 10px sans-serif' : '8px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('q' + n.id, p.x, p.y - r - 4);
      }
    }

    // Tooltip
    if (hoveredNode >= 0) {
      const n = nodes.find(n => n.id === hoveredNode);
      const p = posMap[n.id];
      const compIdx = n.component;
      const neighbors = edges.filter(([u, v]) => u === n.id || v === n.id).length;
      const label = `q${n.id} | Component ${compIdx} (${components[compIdx].num_qubits}q) | ${neighbors} connections`;

      ctx.font = '11px sans-serif';
      const tw = ctx.measureText(label).width + 20;
      const tx = Math.max(5, Math.min(p.x - tw / 2, canvas.width - tw - 5));
      const ty = p.y > canvas.height / 2 ? p.y - 30 : p.y + 18;

      ctx.fillStyle = 'rgba(26,35,126,0.92)';
      ctx.beginPath();
      ctx.roundRect(tx, ty, tw, 24, 4);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.textAlign = 'left';
      ctx.fillText(label, tx + 10, ty + 16);
    }
  }

  // Update info panel
  function updateInfo() {
    const infoEl = document.getElementById('p7-graph-info');
    if (!infoEl) return;

    if (selectedNode >= 0) {
      const n = nodes.find(n => n.id === selectedNode);
      const neighbors = edges.filter(([u, v]) => u === n.id || v === n.id)
        .map(([u, v]) => u === n.id ? v : u).sort((a, b) => a - b);
      infoEl.innerHTML = `<strong>Qubit ${n.id}</strong> &mdash; Component ${n.component} (${components[n.component].num_qubits} qubits)<br>
        Connected to: ${neighbors.map(q => '<strong>q' + q + '</strong>').join(', ')} (${neighbors.length} neighbors)`;
    } else if (highlightComp >= 0) {
      const comp = components[highlightComp];
      infoEl.innerHTML = `<strong>Component ${highlightComp}</strong> &mdash; ${comp.num_qubits} qubits<br>
        Qubits: ${comp.qubits.map(q => '<strong>q' + q + '</strong>').join(', ')}`;
    } else {
      infoEl.innerHTML = `<strong>${nodes.length} qubits</strong>, <strong>${edges.length} CZ connections</strong> &mdash; split into 2 independent groups. Click a node for details.`;
    }
  }

  canvas.addEventListener('mousemove', function (e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    hoveredNode = -1;
    for (const n of nodes) {
      const p = posMap[n.id];
      if (Math.hypot(p.x - mx, p.y - my) < 14) { hoveredNode = n.id; break; }
    }
    drawGraph();
  });

  canvas.addEventListener('mouseleave', function () {
    hoveredNode = -1;
    drawGraph();
  });

  canvas.addEventListener('click', function (e) {
    e.stopPropagation();
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    selectedNode = -1;
    for (const n of nodes) {
      const p = posMap[n.id];
      if (Math.hypot(p.x - mx, p.y - my) < 14) {
        selectedNode = n.id;
        highlightComp = n.component;
        break;
      }
    }
    drawGraph();
    updateInfo();
    // Update button active states
    if (btnRow) btnRow.querySelectorAll('.comp-btn').forEach(b => {
      b.style.background = parseInt(b.dataset.comp) === highlightComp ? '#e3f2fd' : 'white';
    });
  });

  if (btnRow) btnRow.querySelectorAll('.comp-btn').forEach(btn => {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      highlightComp = parseInt(this.dataset.comp);
      selectedNode = -1;
      drawGraph();
      updateInfo();
      btnRow.querySelectorAll('.comp-btn').forEach(b => {
        b.style.background = parseInt(b.dataset.comp) === highlightComp ? '#e3f2fd' : 'white';
      });
    });
  });

  drawGraph();
  updateInfo();
}

// ══════════════════════════════════════════════════════════
// P10: Animated graph reduction — gate removal visualization
// ══════════════════════════════════════════════════════════
function initP10Animation() {
  var canvas = document.getElementById('p10-canvas');
  if (!canvas) return;

  // Use fixed canvas resolution — CSS scales it to fit the container
  canvas.width = 560;
  canvas.height = 400;

  var ctx = canvas.getContext('2d');
  var W = 560, H = 400;
  var N = 28;
  var label = document.getElementById('p10-stage-label');

  var nodes = [];
  var cx = W / 2, cy = H / 2, R = 140;
  for (var i = 0; i < N; i++) {
    var angle = (2 * Math.PI * i / N) - Math.PI / 2;
    nodes.push({ x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) });
  }

  var allEdges = [];
  for (var i = 0; i < N; i++) {
    for (var j = i + 1; j < N; j++) {
      var hash = (i * 31 + j * 17 + 7) % 100;
      if (hash < 55) {
        var cat = (i * 13 + j * 23) % 100;
        var type;
        if (cat < 25) type = 'rz';
        else if (cat < 45) type = 'ry';
        else if (cat < 60) type = 'cancel';
        else type = 'survive';
        allEdges.push({ i: i, j: j, type: type, alpha: 1, removing: false });
      }
    }
  }

  var totalEdges = allEdges.length;
  var stage = 0;

  function edgeColor(e) {
    if (e.removing) {
      if (e.type === 'rz') return 'rgba(229,57,53,' + e.alpha + ')';
      if (e.type === 'ry') return 'rgba(251,140,0,' + e.alpha + ')';
      if (e.type === 'cancel') return 'rgba(123,31,162,' + e.alpha + ')';
    }
    if (stage >= 4 && e.type === 'survive') return 'rgba(46,125,50,0.6)';
    return 'rgba(100,120,160,' + (0.15 * e.alpha) + ')';
  }

  function countVisible() {
    var c = 0;
    for (var k = 0; k < allEdges.length; k++) { if (allEdges[k].alpha > 0.05) c++; }
    return c;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (var k = 0; k < allEdges.length; k++) {
      var e = allEdges[k];
      if (e.alpha < 0.02) continue;
      ctx.beginPath();
      ctx.moveTo(nodes[e.i].x, nodes[e.i].y);
      ctx.lineTo(nodes[e.j].x, nodes[e.j].y);
      ctx.strokeStyle = edgeColor(e);
      ctx.lineWidth = e.removing ? 2.5 : 1;
      ctx.stroke();
    }
    var nColor = stage >= 4 ? '#2e7d32' : '#283593';
    for (var k = 0; k < N; k++) {
      ctx.beginPath();
      ctx.arc(nodes[k].x, nodes[k].y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = nColor;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.fillStyle = '#78909c';
    ctx.font = '13px Segoe UI, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Edges: ' + countVisible() + ' / ' + totalEdges, 14, H - 14);
  }

  function animateRemoval(type, callback) {
    var targets = [];
    for (var k = 0; k < allEdges.length; k++) {
      if (allEdges[k].type === type && allEdges[k].alpha > 0.5) targets.push(allEdges[k]);
    }
    for (var k = 0; k < targets.length; k++) targets[k].removing = true;
    draw();
    var step = 0, fadeSteps = 40;
    function fade() {
      step++;
      for (var k = 0; k < targets.length; k++) {
        targets[k].alpha = Math.max(0, 1 - step / fadeSteps);
      }
      draw();
      if (step < fadeSteps) requestAnimationFrame(fade);
      else {
        for (var k = 0; k < targets.length; k++) targets[k].removing = false;
        draw();
        if (callback) callback();
      }
    }
    requestAnimationFrame(fade);
  }

  function setStep(n, s) {
    var el = document.getElementById('p10-step-' + n);
    if (!el) return;
    if (s === 'active') { el.classList.add('active'); el.classList.remove('done'); el.style.opacity = '1'; }
    else if (s === 'done') { el.classList.remove('active'); el.classList.add('done'); el.style.opacity = '0.85'; }
  }

  function runAnimation() {
    stage = 0;
    for (var k = 0; k < allEdges.length; k++) { allEdges[k].alpha = 1; allEdges[k].removing = false; }
    for (var s = 1; s <= 4; s++) {
      var el = document.getElementById('p10-step-' + s);
      if (el) { el.classList.remove('active', 'done'); el.style.opacity = '0.35'; }
    }
    var rb = document.getElementById('p10-result-box');
    if (rb) rb.style.opacity = '0';
    if (label) label.textContent = 'Original Circuit \u2014 56 qubits, 3836 gates';
    draw();

    setTimeout(function() {
      stage = 1;
      if (label) label.textContent = 'Step 1: Removing all Rz gates (phase \u2260 probability)';
      setStep(1, 'active');
      animateRemoval('rz', function() {
        setStep(1, 'done');
        setTimeout(function() {
          stage = 2;
          if (label) label.textContent = 'Step 2: Removing small Ry(\u03B8) gates (\u03B8 < 0.5)';
          setStep(2, 'active');
          animateRemoval('ry', function() {
            setStep(2, 'done');
            setTimeout(function() {
              stage = 3;
              if (label) label.textContent = 'Step 3: CZ\u00B2 = I cancellation \u2014 pairs vanish';
              setStep(3, 'active');
              animateRemoval('cancel', function() {
                setStep(3, 'done');
                setTimeout(function() {
                  stage = 4;
                  if (label) label.textContent = 'Simplified graph \u2014 MPS simulation finds the peak';
                  setStep(4, 'active');
                  draw();
                  var rb2 = document.getElementById('p10-result-box');
                  if (rb2) rb2.style.opacity = '1';
                  var glowStep = 0;
                  function glow() {
                    glowStep++;
                    var pulse = 0.4 + 0.3 * Math.sin(glowStep * 0.08);
                    for (var k = 0; k < allEdges.length; k++) {
                      if (allEdges[k].type === 'survive') allEdges[k].alpha = pulse + 0.3;
                    }
                    draw();
                    if (glowStep < 120) requestAnimationFrame(glow);
                    else {
                      for (var k = 0; k < allEdges.length; k++) {
                        if (allEdges[k].type === 'survive') allEdges[k].alpha = 0.7;
                      }
                      draw();
                      setStep(4, 'done');
                    }
                  }
                  requestAnimationFrame(glow);
                }, 800);
              });
            }, 1000);
          });
        }, 1000);
      });
    }, 1500);
  }

  window.restartP10Anim = runAnimation;

  // Auto-play when slide becomes visible
  var slide8 = document.getElementById('slide-8');
  if (slide8) {
    var observer = new MutationObserver(function() {
      if (slide8.classList.contains('active')) runAnimation();
    });
    observer.observe(slide8, { attributes: true, attributeFilter: ['class'] });
    if (slide8.classList.contains('active')) runAnimation();
  }

  // Initial draw so canvas is not blank
  draw();
}

// loadData() is called from index.html after all slides are inserted into the DOM
