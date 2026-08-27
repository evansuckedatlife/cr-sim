"""The play page, as one string.

Kept as a Python string rather than a static file so the server stays a single
importable module with no asset paths to resolve, no packaging data to declare,
and nothing that can go missing when the package is moved.

The layout follows the game's own battle screen: the arena filling the frame, a
tree border, lanes running to the bridges, towers wearing a crown badge with
their hitpoints, and a card tray with a next-card slot and a segmented elixir
bar. That is not decoration. The point of the page is to make the simulator
legible at a glance, and a board laid out like the board it models is read far
faster than an abstract one.

Everything drawn comes from the engine. The grass, river and bridges are the
shipped tilemap rather than a background image, so what you click is the
geometry the simulation routes on.
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
    --ink: #f7f3e8; --shadow: rgba(0,0,0,.55);
    --tray: #2f6fd0; --tray-dark: #21529c; --tray-slot: #3d82e8;
    --elixir: #d81fd8; --elixir-dark: #4a1145;
    --blue: #3b7ddd; --red: #e0483c; --gold: #ffcf3f;
    --panel: rgba(12,18,28,.94); --line: #33415a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0b1016; color: var(--ink);
    font: 14px/1.4 "Segoe UI", ui-sans-serif, system-ui, sans-serif;
    display: flex; justify-content: center; align-items: flex-start;
    gap: 14px; padding: 12px; user-select: none;
  }

  .frame {
    position: relative; border-radius: 14px; overflow: hidden;
    background: #4a8a34; box-shadow: 0 10px 40px rgba(0,0,0,.6);
  }
  canvas { display: block; cursor: pointer; }
  .hud { position: absolute; inset: 0; pointer-events: none; }
  .hud > * { pointer-events: auto; }

  .who {
    position: absolute; top: 8px; left: 8px; display: flex; gap: 6px;
    align-items: center; text-shadow: 0 2px 3px var(--shadow);
  }
  .who .badge {
    width: 30px; height: 34px; border-radius: 5px 5px 12px 12px;
    background: linear-gradient(#c8443a, #8d2620); border: 2px solid #f0d9a0;
  }
  .who .nm { font-weight: 800; font-size: 13px; color: #ff8ce0; }
  .who .sub { font-size: 10px; color: #e8e0cd; }

  .clock {
    position: absolute; top: 8px; right: 8px; text-align: center;
    background: rgba(10,14,20,.72); border: 2px solid #6b7a90;
    border-radius: 6px; padding: 3px 9px; text-shadow: 0 2px 3px var(--shadow);
  }
  .clock .lbl { font-size: 9px; color: #b9c4d4; }
  .clock .val { font-size: 17px; font-weight: 800; font-variant-numeric: tabular-nums; }
  .clock.urgent .val { color: #ff6b5e; }

  .crowns { position: absolute; right: 6px; top: 34%; display: grid; gap: 44px; }
  .tally {
    width: 26px; border-radius: 5px; text-align: center; padding: 2px 0 3px;
    border: 2px solid rgba(0,0,0,.35); font-weight: 800; font-size: 11px;
  }
  .tally.them { background: linear-gradient(#e0483c, #a5281f); }
  .tally.you  { background: linear-gradient(#3b7ddd, #24549c); }

  .tray {
    position: absolute; left: 0; right: 0; bottom: 0;
    background: linear-gradient(var(--tray), var(--tray-dark));
    border-top: 3px solid #1b3f7a; padding: 6px 8px 22px;
    display: flex; gap: 7px; align-items: flex-end;
  }
  .next { width: 46px; flex: none; text-align: center; }
  .next .lbl { font-size: 9px; color: #cfe2ff; margin-bottom: 2px; }
  .next .slot {
    height: 54px; border-radius: 5px; background: var(--tray-slot);
    border: 2px solid #6ea8f5; display: grid; place-items: center; overflow: hidden;
  }
  .next img { width: 100%; height: 100%; object-fit: contain; }

  .hand { display: flex; gap: 7px; flex: 1; }
  .card {
    flex: 1; position: relative; border-radius: 6px; overflow: hidden;
    background: var(--tray-slot); border: 2px solid #6ea8f5; cursor: grab;
    transition: transform .08s ease, box-shadow .08s ease;
  }
  .card .art { display: block; width: 100%; height: 60px; object-fit: contain; }
  .card .blank { height: 60px; display: grid; place-items: center;
                 font-size: 9px; color: #cfe2ff; text-align: center; padding: 0 2px; }
  .card .cost {
    position: absolute; top: 2px; left: 2px; min-width: 17px; height: 17px;
    border-radius: 50%; background: var(--elixir); border: 2px solid #ff9df5;
    font-size: 10px; font-weight: 800; display: grid; place-items: center;
    text-shadow: 0 1px 2px var(--shadow);
  }
  .card.sel { transform: translateY(-8px); box-shadow: 0 0 0 3px var(--gold); }
  .card.poor { filter: grayscale(.7) brightness(.6); cursor: not-allowed; }
  .card.evo { box-shadow: 0 0 0 2px var(--gold), 0 0 12px rgba(255,207,63,.7); }
  .card .evotag {
    position: absolute; bottom: 0; left: 0; right: 0; font-size: 8px;
    font-weight: 800; letter-spacing: .6px; background: rgba(255,207,63,.9);
    color: #3a2b00; text-align: center;
  }

  .elixirbar {
    position: absolute; left: 8px; right: 8px; bottom: 3px; height: 15px;
    background: var(--elixir-dark); border-radius: 4px; overflow: hidden;
    display: flex; gap: 2px; padding: 2px; border: 2px solid #7a2470;
  }
  .elixirbar i { flex: 1; background: #3d1038; border-radius: 2px; }
  .elixirbar i.on { background: linear-gradient(var(--elixir), #a013a0); }
  .elixirCount {
    position: absolute; left: 12px; bottom: 1px; z-index: 2; font-size: 12px;
    font-weight: 800; color: #fff; text-shadow: 0 1px 3px #000;
  }

  aside {
    width: 268px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px; max-height: 94vh; overflow-y: auto;
  }
  aside h2 { margin: 14px 0 7px; font-size: 11px; text-transform: uppercase;
             letter-spacing: .08em; color: #8fa3bd; }
  aside h2:first-child { margin-top: 0; }
  .deck { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
  .slot2 {
    border: 1px solid var(--line); border-radius: 6px; padding: 3px 1px;
    text-align: center; font-size: 9px; cursor: pointer; background: #131c28;
  }
  .slot2.sel { border-color: var(--gold); }
  .slot2.evoslot { box-shadow: inset 0 0 0 1px var(--gold); }
  .slot2 img { width: 32px; height: 32px; object-fit: contain; display: block; margin: 0 auto; }
  .pool { max-height: 200px; overflow-y: auto; border: 1px solid var(--line);
          border-radius: 6px; margin-top: 5px; }
  .pool div { padding: 3px 8px; cursor: pointer; font-size: 12px;
              display: flex; justify-content: space-between; }
  .pool div:hover { background: #1c2836; }
  .pool .c { color: #ff7ae8; font-weight: 700; }
  input, button {
    width: 100%; background: #131c28; border: 1px solid var(--line); color: var(--ink);
    border-radius: 6px; padding: 6px 8px; font: inherit; margin-top: 5px;
  }
  button { cursor: pointer; font-weight: 600; }
  button.primary { background: var(--tray); border-color: #4b8ae8; }
  .row { display: flex; gap: 5px; }
  .msg { min-height: 18px; font-size: 11px; color: var(--gold); margin-top: 5px; }
  .hint { color: #7c8ea6; font-size: 11px; margin: 5px 0 0; }

  .over {
    position: absolute; inset: 0; background: rgba(6,10,16,.88); display: none;
    align-items: center; justify-content: center; flex-direction: column; gap: 10px;
    z-index: 5;
  }
  .over.on { display: flex; }
  .over h1 { margin: 0; font-size: 34px; text-shadow: 0 3px 8px #000; }
  .over button { width: auto; padding: 8px 20px; }
</style>
</head>
<body>

<div class="frame" id="frame">
  <canvas id="board"></canvas>
  <div class="hud">
    <div class="who">
      <div class="badge"></div>
      <div>
        <div class="nm">cr-sim</div>
        <div class="sub" id="oppKind">random</div>
      </div>
    </div>
    <div class="clock" id="clockBox">
      <div class="lbl">Time left:</div>
      <div class="val" id="clock">3:00</div>
    </div>
    <div class="crowns">
      <div class="tally them" id="themCrowns">0</div>
      <div class="tally you" id="youCrowns">0</div>
    </div>
    <div class="tray">
      <div class="next">
        <div class="lbl">Next:</div>
        <div class="slot" id="nextSlot"></div>
      </div>
      <div class="hand" id="hand"></div>
      <div class="elixirCount" id="elixirText">0</div>
      <div class="elixirbar" id="elixirBar"></div>
    </div>
  </div>
  <div class="over" id="over">
    <h1 id="overTitle">-</h1>
    <div class="hint" id="overSub"></div>
    <button class="primary" id="overAgain">play again</button>
  </div>
</div>

<aside>
  <h2>your deck</h2>
  <div class="deck" id="deck"></div>
  <p class="hint">Click a slot, then a card below to replace it.</p>
  <h2>card pool</h2>
  <input type="search" id="search" placeholder="search">
  <div class="pool" id="pool"></div>
  <h2>evolution slots</h2>
  <div class="deck" id="evos"></div>
  <p class="hint">Up to two. Pick a deck slot, then press + here.</p>
  <h2>match</h2>
  <input type="number" id="seed" value="0" title="seed">
  <button class="primary" id="restart">new match</button>
  <div class="row"><button id="pause">pause</button><button id="speed">1x</button></div>
  <div class="msg" id="msg"></div>
</aside>

<script>
const $ = (id) => document.getElementById(id);
const board = $('board'), ctx = board.getContext('2d');
let SETUP = null, STATE = null, ICONS = {}, selected = null, dragging = false;
let deck = [], evos = [], slotIndex = null, hover = null;
const SPEEDS = [1, 2, 4, 0.5]; let speedAt = 0;
let TILE = 20; const TRAY = 92;

async function api(path, body) {
  const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify(body)} : {};
  return (await fetch(path, opts)).json();
}

function worldToScreen(x, y) { return [x * TILE, (SETUP.heightTiles - y) * TILE]; }
function screenToWorld(px, py) { return [px / TILE, SETUP.heightTiles - py / TILE]; }

// ------------------------------------------------------------------ arena

// Drawn from the tilemap the engine routes on, so the bridges and the riverbank
// are where the simulation thinks they are rather than where art put them.
function drawArena() {
  const {terrain, halfWidth, halfHeight} = SETUP;
  const h = TILE / 2;
  for (let cy = 0; cy < halfHeight; cy++) {
    for (let cx = 0; cx < halfWidth; cx++) {
      const v = terrain[cy][cx];
      const py = (halfHeight - 1 - cy) * h;
      if (v === 2) ctx.fillStyle = '#2f8fd0';
      else if (v === 1) ctx.fillStyle = '#2c5a1f';
      else ctx.fillStyle = ((cx >> 1) + (cy >> 1)) % 2 ? '#5fa93c' : '#589f37';
      ctx.fillRect(cx * h, py, h + 0.6, h + 0.6);
    }
  }
  drawLanes(); drawBridges(); drawBorder();
}

// The tan paths. Cosmetic, but they are how a player reads where troops walk,
// and they follow the lane centres the router actually uses.
function drawLanes() {
  ctx.fillStyle = 'rgba(206,176,120,.8)';
  const top = worldToScreen(0, SETUP.heightTiles - 2.6)[1];
  const bot = worldToScreen(0, 2.6)[1];
  for (const lane of [3.5, 14.5]) {
    ctx.fillRect(worldToScreen(lane - 1.05, 0)[0], top, TILE * 2.1, bot - top);
  }
  for (const row of [6.4, SETUP.heightTiles - 6.4]) {
    const [ax, ay] = worldToScreen(2.45, row + 1.05);
    ctx.fillRect(ax, ay, TILE * 13.1, TILE * 2.1);
  }
}

function drawBridges() {
  const [rt, rb] = SETUP.riverBand;
  for (const lane of [3.5, 14.5]) {
    const [bx, by] = worldToScreen(lane - 1.0, rb + 0.4);
    const w = TILE * 2.0, hgt = (rb - rt + 0.8) * TILE;
    ctx.fillStyle = '#b98a4e'; ctx.fillRect(bx, by, w, hgt);
    ctx.strokeStyle = 'rgba(88,58,24,.7)'; ctx.lineWidth = 1.4;
    for (let i = 0; i <= 5; i++) {
      const y = by + hgt * i / 5;
      ctx.beginPath(); ctx.moveTo(bx, y); ctx.lineTo(bx + w, y); ctx.stroke();
    }
    ctx.strokeRect(bx, by, w, hgt);
  }
}

// A ring of dark foliage, the way the game frames the board.
function drawBorder() {
  const t = TILE * 0.8, h = board.height - TRAY;
  ctx.fillStyle = '#245018';
  ctx.fillRect(0, 0, board.width, t);
  ctx.fillRect(0, h - t, board.width, t);
  ctx.fillRect(0, 0, t, h);
  ctx.fillRect(board.width - t, 0, t, h);
  ctx.fillStyle = 'rgba(22,64,15,.6)';
  for (let i = 0; i < 30; i++) {
    const r = t * (0.4 + (i % 3) * 0.13);
    ctx.beginPath(); ctx.arc((i * 67) % board.width, t * 0.5, r, 0, 6.284); ctx.fill();
    ctx.beginPath(); ctx.arc((i * 51 + 25) % board.width, h - t * 0.5, r, 0, 6.284); ctx.fill();
  }
}

// ----------------------------------------------------------------- pieces

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// The crown plate above a tower, carrying its hitpoints.
function badge(cx, cy, value, mine) {
  const w = Math.max(36, String(value).length * 8 + 22), h = 15;
  ctx.fillStyle = 'rgba(10,14,20,.8)';
  roundRect(cx - w / 2, cy - h / 2, w, h, 4); ctx.fill();
  ctx.strokeStyle = mine ? '#5b95e8' : '#e8695e'; ctx.lineWidth = 1.5;
  roundRect(cx - w / 2, cy - h / 2, w, h, 4); ctx.stroke();
  const bx = cx - w / 2 + 4;
  ctx.fillStyle = '#ffcf3f';
  ctx.beginPath();
  ctx.moveTo(bx, cy + 3); ctx.lineTo(bx, cy - 3); ctx.lineTo(bx + 3, cy);
  ctx.lineTo(bx + 6, cy - 4); ctx.lineTo(bx + 9, cy); ctx.lineTo(bx + 12, cy - 3);
  ctx.lineTo(bx + 12, cy + 3); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = '700 11px "Segoe UI", sans-serif';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(value, cx + w / 2 - 5, cy + 1);
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
}

function drawTower(t, mine) {
  const [px, py] = worldToScreen(t.x, t.y);
  const king = (t.n || '').indexOf('King') >= 0;
  const w = TILE * (king ? 3.1 : 2.7), h = w * 0.86;
  const x = px - w / 2, y = py - h / 2;
  ctx.fillStyle = 'rgba(0,0,0,.28)'; ctx.fillRect(x + 3, y + 5, w, h);
  ctx.fillStyle = '#b9b3a4'; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = mine ? '#3b7ddd' : '#d8443a';
  ctx.fillRect(x, y, w, h * 0.42);
  ctx.strokeStyle = 'rgba(60,50,40,.8)'; ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = '#8e887b';
  for (let i = 0; i < 4; i++) ctx.fillRect(x + (w / 4) * i + 2, y + h - 6, w / 6, 5);
  if (t.hp > 0) badge(px, y - TILE * 0.5, t.hp, mine);
}

function drawEntity(e) {
  const [px, py] = worldToScreen(e.x, e.y);
  const mine = e.t === STATE.you.team;
  const r = TILE * 0.6;
  if (e.d) ctx.globalAlpha = 0.5;
  ctx.fillStyle = 'rgba(0,0,0,.3)';
  ctx.beginPath(); ctx.ellipse(px, py + r * 0.7, r * 0.8, r * 0.32, 0, 0, 6.284); ctx.fill();
  ctx.beginPath(); ctx.arc(px, py, r, 0, 6.284);
  ctx.fillStyle = mine ? 'rgba(59,125,221,.55)' : 'rgba(224,72,60,.55)'; ctx.fill();
  ctx.strokeStyle = mine ? '#8fc0ff' : '#ff9d92'; ctx.lineWidth = 2; ctx.stroke();

  const img = ICONS[e.n];
  if (img && img.complete && img.naturalWidth) {
    const s = r * 2.0; ctx.drawImage(img, px - s / 2, py - s / 2, s, s);
  } else {
    ctx.fillStyle = '#fff'; ctx.font = '700 8px "Segoe UI", sans-serif';
    ctx.textAlign = 'center'; ctx.fillText((e.n || '?').slice(0, 7), px, py + 3);
    ctx.textAlign = 'left';
  }
  const frac = Math.max(0, e.hp / e.max);
  if (frac < 1) {
    ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillRect(px - r, py - r - 5, r * 2, 3.5);
    ctx.fillStyle = mine ? '#59b6ff' : '#ff6b5e';
    ctx.fillRect(px - r, py - r - 5, r * 2 * frac, 3.5);
  }
  ctx.globalAlpha = 1;
}

// ------------------------------------------------------------- placement

function selectedCard() {
  if (!selected || !STATE) return null;
  return STATE.you.hand.find((c) => c.name === selected) || null;
}

// Mirrors the server's rules closely enough to colour the marker. The server
// still decides on release; this only avoids promising a tile it will refuse.
//
// The human side is always Team.BLUE (see SessionConfig.human_team), so
// "past the river" always means "into STATE.them's territory". Destroying
// one of their Princess Towers pushes that line forward, in that tower's
// lane only, up to the row it stood on -- see Arena.can_deploy, which is the
// rule this mirrors. STATE.them.towers already carries hp/x/y for exactly
// this reason.
function placementLegal(tx, ty) {
  const card = selectedCard();
  if (!card || !SETUP) return false;
  if (tx < 0 || ty < 0 || tx >= SETUP.widthTiles || ty >= SETUP.heightTiles) return false;
  const cell = (SETUP.terrain[Math.floor(ty * 2)] || [])[Math.floor(tx * 2)];
  if (cell === undefined || cell === 1) return false;
  if (cell === 2 && !card.water) return false;
  if (card.anywhere) return true;
  let limit = SETUP.riverBand[0];
  if (STATE) {
    for (const t of STATE.them.towers) {
      if (t.hp > 0 || (t.n || '').indexOf('King') >= 0) continue;
      const sameLane = (tx < SETUP.widthTiles / 2) === (t.x < SETUP.widthTiles / 2);
      if (sameLane) limit = Math.max(limit, t.y);
    }
  }
  return ty <= limit;
}

function snapped() {
  if (!hover) return null;
  const [tx, ty] = screenToWorld(hover[0], hover[1]);
  return [Math.floor(tx) + 0.5, Math.floor(ty) + 0.5];
}

function drawPlacement() {
  if (!selected || !hover) return;
  const tile = snapped(); if (!tile) return;
  const ok = placementLegal(tile[0], tile[1]);
  const [px, py] = worldToScreen(tile[0], tile[1]);
  ctx.save();
  ctx.strokeStyle = ok ? 'rgba(120,255,140,.95)' : 'rgba(255,90,80,.95)';
  ctx.fillStyle = ok ? 'rgba(120,255,140,.2)' : 'rgba(255,90,80,.18)';
  ctx.lineWidth = 2;
  ctx.fillRect(px - TILE / 2, py - TILE / 2, TILE, TILE);
  ctx.strokeRect(px - TILE / 2, py - TILE / 2, TILE, TILE);
  const card = selectedCard(), img = card && ICONS[card.name];
  if (img && img.complete && img.naturalWidth) {
    ctx.globalAlpha = 0.6;
    ctx.drawImage(img, px - TILE * 0.75, py - TILE * 0.95, TILE * 1.5, TILE * 1.5);
  }
  ctx.restore();
}

function render() {
  if (!SETUP) return;
  drawArena();
  if (STATE) {
    for (const t of STATE.them.towers) drawTower(t, false);
    for (const t of STATE.you.towers) drawTower(t, true);
    for (const e of STATE.entities.slice().sort((a, b) => b.y - a.y)) {
      if (e.k !== 2) drawEntity(e);
    }
  }
  drawPlacement();
}

// ------------------------------------------------------------------- hud

let handSignature = '';
function handState() {
  if (!STATE) return '';
  return STATE.you.hand.map((c) => c.name + (c.evo ? '*' : '')).join('|')
       + '/' + (selected || '') + '/' + (STATE.you.next ? STATE.you.next.name : '');
}

function refreshAffordability() {
  if (!STATE) return;
  document.querySelectorAll('#hand .card').forEach((el) => {
    el.classList.toggle('poor', STATE.you.elixir < Number(el.dataset.cost));
  });
}

// Rebuilt only when its contents change. Rebuilding on every poll -- ten times
// a second -- meant a press landing between down and up hit an element that had
// already been replaced, and cards frequently would not select at all.
function renderHand(force) {
  const sig = handState();
  if (!force && sig === handSignature) { refreshAffordability(); return; }
  handSignature = sig;

  const el = $('hand'); el.innerHTML = '';
  (STATE ? STATE.you.hand : []).forEach((c) => {
    const d = document.createElement('div');
    d.className = 'card' + (selected === c.name ? ' sel' : '') + (c.evo ? ' evo' : '');
    d.dataset.cost = c.cost;
    const src = SETUP.icons[c.name];
    d.innerHTML = (src ? `<img class="art" src="${src}" alt="">`
                       : `<div class="blank">${c.name}</div>`)
                + `<div class="cost">${c.cost}</div>`
                + (c.evo ? '<div class="evotag">EVO</div>' : '');
    d.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      selected = (selected === c.name) ? null : c.name;
      dragging = selected !== null;
      renderHand(true);
    });
    el.appendChild(d);
  });
  refreshAffordability();

  const nx = STATE && STATE.you.next;
  $('nextSlot').innerHTML =
    nx && SETUP.icons[nx.name] ? `<img src="${SETUP.icons[nx.name]}" alt="">` : '';
}

function renderElixir() {
  const bar = $('elixirBar');
  if (bar.children.length !== 10) {
    bar.innerHTML = '';
    for (let i = 0; i < 10; i++) bar.appendChild(document.createElement('i'));
  }
  const e = STATE ? STATE.you.elixir : 0;
  [...bar.children].forEach((seg, i) => seg.classList.toggle('on', i < Math.floor(e)));
  $('elixirText').textContent = Math.floor(e);
}

function renderStatus() {
  if (!STATE) return;
  $('youCrowns').textContent = STATE.you.crowns;
  $('themCrowns').textContent = STATE.them.crowns;
  const left = Math.max(0, 180 - Math.floor(STATE.seconds));
  $('clock').textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`;
  $('clockBox').classList.toggle('urgent', left <= 30);
  $('oppKind').textContent = STATE.opponent || 'random';
  $('pause').textContent = STATE.paused ? 'resume' : 'pause';
  renderElixir();
  if (STATE.result) {
    $('over').classList.add('on');
    $('overTitle').textContent =
      {win: 'Victory', loss: 'Defeat', draw: 'Draw'}[STATE.result.outcome];
    $('overSub').textContent =
      `${STATE.result.you} - ${STATE.result.them} (${STATE.result.reason})`;
  }
}

// ------------------------------------------------------------------ deck

function renderDeck() {
  const el = $('deck'); el.innerHTML = '';
  deck.forEach((name, i) => {
    const d = document.createElement('div');
    d.className = 'slot2' + (slotIndex === i ? ' sel' : '')
                + (evos.includes(name) ? ' evoslot' : '');
    const src = SETUP.icons[name];
    d.innerHTML = (src ? `<img src="${src}" alt="">` : '') + `<div>${name.slice(0, 9)}</div>`;
    d.onclick = () => { slotIndex = (slotIndex === i ? null : i); renderDeck(); };
    el.appendChild(d);
  });

  const ev = $('evos'); ev.innerHTML = '';
  evos.forEach((name) => {
    const d = document.createElement('div');
    d.className = 'slot2 evoslot';
    const src = SETUP.icons[name];
    d.innerHTML = (src ? `<img src="${src}" alt="">` : '') + `<div>${name.slice(0, 9)}</div>`;
    d.onclick = () => { evos = evos.filter((n) => n !== name); renderDeck(); };
    ev.appendChild(d);
  });
  if (evos.length < 2) {
    const add = document.createElement('div');
    add.className = 'slot2'; add.innerHTML = '<div style="padding:12px 0">+ slot</div>';
    add.onclick = () => {
      if (slotIndex === null) { message('pick a deck slot first'); return; }
      const name = deck[slotIndex];
      const card = SETUP.cards.find((c) => c.name === name);
      if (!card || !card.hasEvo) { message(`${name} has no evolution`); return; }
      if (!evos.includes(name)) evos.push(name);
      slotIndex = null; renderDeck();
      message(`${name} slotted -- start a new match to use it`);
    };
    ev.appendChild(add);
  }
}

function renderPool() {
  const q = $('search').value.toLowerCase();
  const el = $('pool'); el.innerHTML = '';
  SETUP.cards.filter((c) => c.name.toLowerCase().includes(q)).slice(0, 140).forEach((c) => {
    const d = document.createElement('div');
    d.innerHTML = `<span>${c.name}${c.hasEvo ? ' ✦' : ''}</span>`
                + `<span class="c">${c.cost}</span>`;
    d.onclick = () => {
      if (slotIndex === null) { message('pick a deck slot to replace'); return; }
      if (deck.includes(c.name)) { message(`${c.name} is already in the deck`); return; }
      evos = evos.filter((n) => n !== deck[slotIndex]);
      deck[slotIndex] = c.name; slotIndex = null; renderDeck();
      message(`${c.name} added -- start a new match to use it`);
    };
    el.appendChild(d);
  });
}

function message(t) { $('msg').textContent = t; }

// ---------------------------------------------------------------- events

function trackPointer(ev) {
  const r = board.getBoundingClientRect();
  hover = [(ev.clientX - r.left) * board.width / r.width,
           (ev.clientY - r.top) * board.height / r.height];
}
// On the window, because a drag that began on a card is already off the board
// by the time it starts moving.
window.addEventListener('pointermove', trackPointer);

async function place() {
  const tile = snapped(); if (!selected || !tile) return;
  const res = await api('/api/play', {card: selected, x: tile[0], y: tile[1]});
  if (res.ok) { selected = null; dragging = false; message(''); renderHand(true); }
  else message(res.reason || 'cannot place there');
}

board.addEventListener('pointerup', (ev) => { trackPointer(ev); if (selected) place(); });
board.addEventListener('click', (ev) => {
  if (selected && !dragging) { trackPointer(ev); place(); }
});
window.addEventListener('pointerup', () => { dragging = false; });

$('pause').onclick = async () => {
  const r = await api('/api/control', {paused: !(STATE && STATE.paused)});
  if (STATE) STATE.paused = r.paused;
};
$('speed').onclick = async () => {
  speedAt = (speedAt + 1) % SPEEDS.length;
  await api('/api/control', {speed: SPEEDS[speedAt]});
  $('speed').textContent = SPEEDS[speedAt] + 'x';
};
$('search').oninput = renderPool;
$('restart').onclick = newMatch;
$('overAgain').onclick = newMatch;

async function newMatch() {
  const res = await api('/api/new', {
    human: deck, ai: deck, seed: Number($('seed').value) || 0, humanEvolutions: evos,
  });
  if (!res.ok) { message(res.reason); return; }
  $('over').classList.remove('on');
  selected = null; handSignature = ''; message('');
}

// ----------------------------------------------------------------- loop

async function poll() {
  try { STATE = await api('/api/state'); renderStatus(); renderHand(false); }
  catch (err) { /* a dropped poll costs a frame, not the match */ }
  setTimeout(poll, 100);
}
function frame() { render(); requestAnimationFrame(frame); }

(async function start() {
  SETUP = await api('/api/setup');
  deck = SETUP.deck.slice();
  evos = (SETUP.evolutions || []).slice();

  // Sized to the viewport keeping the board's real proportions: the frame is
  // the arena plus the tray, never a stretched arena.
  const room = Math.max(420, window.innerHeight - 34) - TRAY;
  TILE = Math.max(12, Math.min(Math.floor(room / SETUP.heightTiles), 30));
  board.width = Math.round(SETUP.widthTiles * TILE);
  board.height = Math.round(SETUP.heightTiles * TILE) + TRAY;
  $('frame').style.width = board.width + 'px';

  for (const [name, src] of Object.entries(SETUP.icons)) {
    const im = new Image(); im.src = src; ICONS[name] = im;
  }
  renderDeck(); renderPool(); poll(); frame();
})();
</script>
</body>
</html>
"""
