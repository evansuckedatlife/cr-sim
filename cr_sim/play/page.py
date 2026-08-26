"""The play page, as one string.

Kept as a Python string rather than a static file so the server stays a single
importable module with no asset paths to resolve, no packaging data to declare,
and nothing that can go missing when the package is moved. It is the same
reasoning the replay viewer uses: one file you can open, with everything in it.
"""

from __future__ import annotations

__all__ = ["PAGE"]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cr-sim</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --line: #30363d; --text: #e6edf3;
    --dim: #8b949e; --blue: #3b82f6; --red: #ef4444; --gold: #eab308;
    --elixir: #c026d3; --ok: #22c55e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; justify-content: center; padding: 16px; gap: 16px;
  }
  .stage { display: flex; flex-direction: column; gap: 10px; align-items: center; }
  canvas { background: #1b3a1f; border-radius: 8px; display: block; cursor: crosshair; }
  .bar { display: flex; align-items: center; gap: 10px; width: 100%; }
  .spacer { flex: 1; }
  button {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 12px; font: inherit; cursor: pointer;
  }
  button:hover { border-color: var(--dim); }
  button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
  .hand { display: flex; gap: 8px; }
  .card {
    width: 84px; border: 2px solid var(--line); border-radius: 8px; background: var(--panel);
    padding: 6px 4px 4px; text-align: center; cursor: pointer; user-select: none;
    transition: border-color .1s, transform .1s;
  }
  .card.sel { border-color: var(--gold); transform: translateY(-4px); }
  .card.poor { opacity: .45; cursor: not-allowed; }
  .card img { width: 52px; height: 52px; object-fit: contain; display: block; margin: 0 auto; }
  .card .nm { font-size: 10px; color: var(--dim); white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
  .card .co { font-size: 12px; font-weight: 700; color: var(--elixir); }
  .elixir { height: 16px; background: #2a1030; border-radius: 8px; overflow: hidden;
            flex: 1; position: relative; border: 1px solid var(--line); }
  .elixir > i { display: block; height: 100%; background: var(--elixir); width: 0%;
                transition: width .1s linear; }
  .elixir > span { position: absolute; inset: 0; text-align: center; font-size: 11px;
                   font-weight: 700; line-height: 14px; }
  aside { width: 300px; background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 12px; height: fit-content; }
  aside h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
             letter-spacing: .06em; color: var(--dim); }
  .deck { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-bottom: 10px; }
  .slot { border: 1px solid var(--line); border-radius: 6px; padding: 4px 2px;
          text-align: center; font-size: 10px; cursor: pointer; background: #0d1117; }
  .slot.sel { border-color: var(--gold); }
  .slot img { width: 34px; height: 34px; object-fit: contain; display: block; margin: 0 auto; }
  .pool { max-height: 260px; overflow-y: auto; border: 1px solid var(--line);
          border-radius: 6px; }
  .pool div { padding: 3px 8px; cursor: pointer; font-size: 12px;
              display: flex; justify-content: space-between; }
  .pool div:hover { background: #21262d; }
  .pool .c { color: var(--elixir); font-weight: 700; }
  input[type=search], input[type=number] {
    width: 100%; background: #0d1117; border: 1px solid var(--line); color: var(--text);
    border-radius: 6px; padding: 5px 8px; font: inherit; margin-bottom: 6px;
  }
  .msg { min-height: 20px; font-size: 12px; color: var(--gold); }
  .score { display: flex; gap: 14px; align-items: center; font-variant-numeric: tabular-nums; }
  .score b { font-size: 18px; }
  .you b { color: var(--blue); } .them b { color: var(--red); }
  .over { position: fixed; inset: 0; background: rgba(13,17,23,.86); display: none;
          align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
  .over.on { display: flex; }
  .over h1 { margin: 0; font-size: 40px; }
  .hint { color: var(--dim); font-size: 12px; }
</style>
</head>
<body>
<div class="stage">
  <div class="bar">
    <div class="score you">you <b id="youCrowns">0</b></div>
    <div class="score them">ai <b id="themCrowns">0</b></div>
    <div class="spacer"></div>
    <span class="hint" id="clock">0:00</span>
    <span class="hint" id="opp"></span>
    <button id="pause">pause</button>
    <button id="speed">1x</button>
  </div>
  <canvas id="board" width="540" height="960"></canvas>
  <div class="bar">
    <div class="hand" id="hand"></div>
    <div style="flex:1">
      <div class="elixir"><i id="elixirFill"></i><span id="elixirText">0</span></div>
      <div class="msg" id="msg"></div>
    </div>
  </div>
</div>

<aside>
  <h2>your deck</h2>
  <div class="deck" id="deck"></div>
  <h2>replace with</h2>
  <input type="search" id="search" placeholder="search 122 cards">
  <div class="pool" id="pool"></div>
  <h2 style="margin-top:12px">match</h2>
  <input type="number" id="seed" value="0" title="seed">
  <button class="primary" id="restart" style="width:100%">new match</button>
  <p class="hint" style="margin:8px 0 0">
    Click a card, then click your half of the arena. Spells reach anywhere.
  </p>
</aside>

<div class="over" id="over">
  <h1 id="overTitle">-</h1>
  <div class="hint" id="overSub"></div>
  <button class="primary" id="overAgain">play again</button>
</div>

<script>
const $ = (id) => document.getElementById(id);
const board = $('board'), ctx = board.getContext('2d');
let SETUP = null, STATE = null, ICONS = {}, selected = null, deck = [], slotIndex = null;
let speeds = [1, 2, 4, 0.5], speedAt = 0;

// The board is drawn from the shipped tilemap, not from a picture of one, so
// what you click is the same geometry the engine tests against.
let TILE = 30;   // pixels per tile, set once the arena size is known

async function api(path, body) {
  const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  return r.json();
}

function loadIcons(map) {
  for (const [name, src] of Object.entries(map)) {
    const im = new Image(); im.src = src; ICONS[name] = im;
  }
}

// ---------------------------------------------------------------- drawing

function drawTerrain() {
  const {terrain, halfWidth, halfHeight} = SETUP;
  const h = TILE / 2;
  for (let cy = 0; cy < halfHeight; cy++) {
    for (let cx = 0; cx < halfWidth; cx++) {
      const v = terrain[cy][cx];
      // Blue owns the low rows, so the board is drawn with y flipped: your
      // side is nearest you, which is how the game presents it.
      const py = (halfHeight - 1 - cy) * h;
      if (v === 2) ctx.fillStyle = '#1e4f6b';
      else if (v === 1) ctx.fillStyle = '#14261a';
      else ctx.fillStyle = (cx + cy) % 2 ? '#1f4324' : '#1b3a1f';
      ctx.fillRect(cx * h, py, h + 0.5, h + 0.5);
    }
  }
}

function worldToScreen(x, y) {
  return [x * TILE, (SETUP.heightTiles - y) * TILE];
}

function screenToWorld(px, py) {
  return [px / TILE, SETUP.heightTiles - py / TILE];
}

function drawTower(t, mine) {
  const [px, py] = worldToScreen(t.x, t.y);
  const size = t.n && t.n.indexOf('King') >= 0 ? TILE * 2.6 : TILE * 2.2;
  ctx.fillStyle = mine ? '#1d4ed8' : '#b91c1c';
  ctx.fillRect(px - size / 2, py - size / 2, size, size);
  ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 2;
  ctx.strokeRect(px - size / 2, py - size / 2, size, size);
  const frac = Math.max(0, t.hp / t.max);
  ctx.fillStyle = '#000a'; ctx.fillRect(px - size / 2, py - size / 2 - 8, size, 5);
  ctx.fillStyle = mine ? '#60a5fa' : '#f87171';
  ctx.fillRect(px - size / 2, py - size / 2 - 8, size * frac, 5);
  ctx.fillStyle = '#fff'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  ctx.fillText(t.hp, px, py + 4);
}

function drawEntity(e) {
  const [px, py] = worldToScreen(e.x, e.y);
  const mine = e.t === STATE.you.team;
  const r = TILE * 0.62;
  if (e.d) ctx.globalAlpha = 0.45;          // still deploying
  ctx.beginPath(); ctx.arc(px, py, r, 0, 6.284);
  ctx.fillStyle = mine ? 'rgba(59,130,246,.35)' : 'rgba(239,68,68,.35)';
  ctx.fill();
  ctx.strokeStyle = mine ? '#3b82f6' : '#ef4444'; ctx.lineWidth = 2; ctx.stroke();

  const img = ICONS[e.n];
  if (img && img.complete && img.naturalWidth) {
    const s = r * 1.9;
    ctx.drawImage(img, px - s / 2, py - s / 2, s, s);
  } else {
    ctx.fillStyle = '#fff'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText((e.n || '?').slice(0, 6), px, py + 3);
  }
  if (e.f) {                                 // a shadow marks air units
    ctx.beginPath(); ctx.ellipse(px, py + r * 0.9, r * 0.5, r * 0.2, 0, 0, 6.284);
    ctx.fillStyle = 'rgba(0,0,0,.35)'; ctx.fill();
  }
  const frac = Math.max(0, e.hp / e.max);
  if (frac < 1) {
    ctx.fillStyle = '#000a'; ctx.fillRect(px - r, py - r - 6, r * 2, 4);
    ctx.fillStyle = mine ? '#60a5fa' : '#f87171';
    ctx.fillRect(px - r, py - r - 6, r * 2 * frac, 4);
  }
  ctx.globalAlpha = 1;
}

let hover = null;
function drawPlacement() {
  if (!selected || !hover) return;
  const [px, py] = hover;
  ctx.beginPath(); ctx.arc(px, py, TILE * 0.8, 0, 6.284);
  ctx.strokeStyle = 'rgba(234,179,8,.9)'; ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
  ctx.stroke(); ctx.setLineDash([]);
}

function render() {
  if (!SETUP) return;
  drawTerrain();
  if (STATE) {
    for (const t of STATE.them.towers) drawTower(t, false);
    for (const t of STATE.you.towers) drawTower(t, true);
    const sorted = STATE.entities.slice().sort((a, b) => a.y - b.y);
    for (const e of sorted) if (e.k !== 2) drawEntity(e);
  }
  drawPlacement();
}

// ------------------------------------------------------------------- ui

function renderHand() {
  const el = $('hand'); el.innerHTML = '';
  const elixir = STATE ? STATE.you.elixir : 0;
  (STATE ? STATE.you.hand : []).forEach((c) => {
    const d = document.createElement('div');
    d.className = 'card' + (selected === c.name ? ' sel' : '')
                + (elixir < c.cost ? ' poor' : '');
    const img = SETUP.icons[c.name];
    d.innerHTML = (img ? `<img src="${img}" alt="">` : '<div style="height:52px"></div>')
                + `<div class="nm">${c.name}</div><div class="co">${c.cost}</div>`;
    d.onclick = () => { selected = (selected === c.name) ? null : c.name; renderHand(); };
    el.appendChild(d);
  });
}

function renderDeck() {
  const el = $('deck'); el.innerHTML = '';
  deck.forEach((name, i) => {
    const d = document.createElement('div');
    d.className = 'slot' + (slotIndex === i ? ' sel' : '');
    const img = SETUP.icons[name];
    d.innerHTML = (img ? `<img src="${img}" alt="">` : '')
                + `<div>${name.slice(0, 9)}</div>`;
    d.onclick = () => { slotIndex = (slotIndex === i ? null : i); renderDeck(); };
    el.appendChild(d);
  });
}

function renderPool() {
  const q = $('search').value.toLowerCase();
  const el = $('pool'); el.innerHTML = '';
  SETUP.cards.filter((c) => c.name.toLowerCase().includes(q)).slice(0, 140).forEach((c) => {
    const d = document.createElement('div');
    d.innerHTML = `<span>${c.name}</span><span class="c">${c.cost}</span>`;
    d.onclick = () => {
      if (slotIndex === null) { message('pick a deck slot to replace first'); return; }
      if (deck.includes(c.name)) { message(`${c.name} is already in the deck`); return; }
      deck[slotIndex] = c.name; slotIndex = null; renderDeck();
      message(`${c.name} added -- start a new match to use it`);
    };
    el.appendChild(d);
  });
}

function message(text) { $('msg').textContent = text; }

function renderStatus() {
  if (!STATE) return;
  $('youCrowns').textContent = STATE.you.crowns;
  $('themCrowns').textContent = STATE.them.crowns;
  const s = Math.floor(STATE.seconds);
  $('clock').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  $('opp').textContent = STATE.opponent || '';
  const e = STATE.you.elixir;
  $('elixirFill').style.width = (e / 10 * 100) + '%';
  $('elixirText').textContent = e.toFixed(1);
  $('pause').textContent = STATE.paused ? 'resume' : 'pause';
  if (STATE.result) {
    $('over').classList.add('on');
    $('overTitle').textContent = {win: 'You win', loss: 'You lose', draw: 'Draw'}[STATE.result.outcome];
    $('overSub').textContent = `${STATE.result.you} - ${STATE.result.them} (${STATE.result.reason})`;
  }
}

// ---------------------------------------------------------------- events

board.addEventListener('mousemove', (ev) => {
  const r = board.getBoundingClientRect();
  hover = [(ev.clientX - r.left) * board.width / r.width,
           (ev.clientY - r.top) * board.height / r.height];
});
board.addEventListener('mouseleave', () => { hover = null; });

board.addEventListener('click', async (ev) => {
  if (!selected) { message('pick a card first'); return; }
  const r = board.getBoundingClientRect();
  const px = (ev.clientX - r.left) * board.width / r.width;
  const py = (ev.clientY - r.top) * board.height / r.height;
  const [x, y] = screenToWorld(px, py);
  const res = await api('/api/play', {card: selected, x, y});
  if (res.ok) { selected = null; message(''); renderHand(); }
  else message(res.reason || 'could not place that');
});

$('pause').onclick = async () => {
  const res = await api('/api/control', {paused: !(STATE && STATE.paused)});
  if (STATE) STATE.paused = res.paused;
};
$('speed').onclick = async () => {
  speedAt = (speedAt + 1) % speeds.length;
  await api('/api/control', {speed: speeds[speedAt]});
  $('speed').textContent = speeds[speedAt] + 'x';
};
$('search').oninput = renderPool;
$('restart').onclick = newMatch;
$('overAgain').onclick = newMatch;

async function newMatch() {
  const res = await api('/api/new', {human: deck, ai: deck, seed: Number($('seed').value) || 0});
  if (!res.ok) { message(res.reason); return; }
  $('over').classList.remove('on');
  selected = null; message('');
}

// ----------------------------------------------------------------- loop

async function poll() {
  try {
    STATE = await api('/api/state');
    renderStatus();
    renderHand();
  } catch (err) { /* a dropped poll costs one frame, not the match */ }
  setTimeout(poll, 100);
}

function frame() { render(); requestAnimationFrame(frame); }

(async function start() {
  SETUP = await api('/api/setup');
  deck = SETUP.deck.slice();
  // Fit the canvas to the real arena rather than assuming 18x32.
  TILE = Math.min(Math.floor(560 / SETUP.widthTiles), Math.floor(940 / SETUP.heightTiles));
  board.width = Math.round(SETUP.widthTiles * TILE);
  board.height = Math.round(SETUP.heightTiles * TILE);
  loadIcons(SETUP.icons);
  renderDeck(); renderPool();
  poll(); frame();
})();
</script>
</body>
</html>
"""
