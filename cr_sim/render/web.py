"""Self-contained HTML replay viewer.

Numbers in a test suite tell you the engine is self-consistent. They do not tell
you it looks like Clash Royale. Watching a push walk down a lane does, and it
catches whole classes of mistake -- a unit taking the wrong bridge, a tower
placed half a tile off, a lane assignment that quietly mirrors -- that no
assertion was written to look for.

The output is one HTML file with the replay embedded, no server and no external
assets, so it opens straight off disk. Frames are cosmetic and never enter the
state hash, so recording them cannot change what happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..engine.arena import Arena, Tile

__all__ = ["render_replay", "build_icon_map", "ICON_DIR"]

ICON_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "icons"


def build_icon_map(registry, icon_dir: Path | None = None) -> dict[str, str]:
    """Map each spawnable character name to its card's artwork file.

    The frames record *character* names (``Goblin_Stab``) while the artwork is
    keyed by *card* (``goblins.png``), so the mapping is built by walking each
    card's summons. A card whose art is missing from the pack simply has no
    entry and falls back to a plain circle.
    """
    icon_dir = icon_dir or ICON_DIR
    if not icon_dir.is_dir():
        return {}
    available = {p.name: p for p in icon_dir.glob("*.png")}
    mapping: dict[str, str] = {}
    for card in registry.standard():
        row = card.raw or {}
        filename = row.get("HighresImageFilename")
        if not filename and row.get("IconFile"):
            filename = f"image/chr/{row['IconFile']}.png"
        if not filename:
            continue
        path = available.get(Path(filename).name)
        if path is None:
            continue
        for character, _count in card.summons():
            mapping.setdefault(character, str(path))
        mapping.setdefault(card.name, str(path))
    return mapping


def _data_uri(path: Path) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>cr-sim replay</title>
<style>
  :root {{ color-scheme: dark; --bg:#12141c; --fg:#e8eaf2; --dim:#8b90a4; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  .wrap {{ display:flex; gap:20px; padding:20px; align-items:flex-start;
           flex-wrap:wrap; }}
  canvas {{ background:#1b2030; border-radius:8px; touch-action:none; }}
  .panel {{ min-width:250px; }}
  h1 {{ font-size:15px; margin:0 0 12px; letter-spacing:.04em; }}
  .row {{ display:flex; justify-content:space-between; gap:12px;
          padding:3px 0; border-bottom:1px solid #262b3d; }}
  .row span:last-child {{ color:var(--dim); }}
  input[type=range] {{ width:100%; }}
  button {{ background:#262b3d; color:var(--fg); border:0; border-radius:5px;
            padding:6px 12px; cursor:pointer; font:inherit; }}
  button:hover {{ background:#333a52; }}
  .bar {{ height:7px; background:#262b3d; border-radius:4px; overflow:hidden; margin-top:3px; }}
  .bar > i {{ display:block; height:100%; }}
  .hand {{ display:flex; gap:6px; margin-top:8px; align-items:flex-end; }}
  .card {{ width:52px; text-align:center; }}
  .card .art {{ width:52px; height:52px; border-radius:7px; background:#20263a;
                border:1px solid #333a52; object-fit:cover; display:block; }}
  .card.dim .art {{ opacity:.32; }}
  .card.next {{ opacity:.75; }}
  .card .nm {{ font-size:9px; color:var(--dim); margin-top:2px;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .card .co {{ font-size:10px; font-weight:700; }}
  .sep {{ width:1px; align-self:stretch; background:#333a52; margin:0 3px; }}
  .legend {{ margin-top:14px; color:var(--dim); font-size:12px; }}
  .legend b {{ color:var(--fg); font-weight:600; }}
</style>
<div class="wrap">
  <canvas id="c" width="{cw}" height="{ch}"></canvas>
  <div class="panel">
    <h1>cr-sim replay</h1>
    <div class="row"><span>tick</span><span id="tick">0</span></div>
    <div class="row"><span>time</span><span id="time">0.0s</span></div>
    <div class="row"><span>entities</span><span id="count">0</span></div>
    <div class="row"><span>blue elixir</span><span id="be">0</span></div>
    <div class="bar"><i id="beb" style="background:#4c8dff;width:0%"></i></div>
    <div class="hand" id="bh"></div>
    <div class="row" style="margin-top:10px"><span>red elixir</span><span id="re">0</span></div>
    <div class="bar"><i id="reb" style="background:#ff5c6c;width:0%"></i></div>
    <div class="hand" id="rh"></div>
    <p><input id="s" type="range" min="0" max="0" value="0"></p>
    <p><button id="play">play</button>
       <button id="slow">0.5x</button>
       <button id="fast">4x</button>
       <button id="hb">hitboxes</button></p>
    <div class="legend">
      <b>blue</b> defends the top, <b>red</b> the bottom.<br>
      art is enlarged for legibility; the thin ring is the real hitbox.<br>
      dashed ring = still deploying; dimmed card = unaffordable.<br>
      {meta}
    </div>
  </div>
</div>
<script>
const ARENA = {arena};
const FRAMES = {frames};
const SCALE = {scale};
const TPS = {tps};
const REAL_TPS = {real_tps};
const ICON_SRC = {icons};
const COSTS = {costs};
const ART_SCALE = {art_scale};
let SHOW_HITBOX = true;
const ICONS = {{}};
const HW = ARENA.half_width, HH = ARENA.half_height;
const c = document.getElementById('c'), g = c.getContext('2d');
const $ = id => document.getElementById(id);
const slider = $('s'); slider.max = Math.max(0, FRAMES.length - 1);
const SUB = 18000, HALF = SUB / 2;
const px = v => (v / SUB) * SCALE * 2;   // subtiles -> canvas px

function terrain() {{
  for (let y = 0; y < HH; y++) for (let x = 0; x < HW; x++) {{
    const f = ARENA.cells[y * HW + x];
    let col = '#222840';
    if (f & 32) col = '#1d3f6b';           // water
    else if (f & 16) col = '#171a26';      // blocked
    else if (f & 3) col = '#2c3450';       // lane road
    g.fillStyle = col;
    g.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
  }}
  // midline
  g.strokeStyle = 'rgba(255,255,255,.13)'; g.beginPath();
  g.moveTo(0, HH * SCALE / 2); g.lineTo(HW * SCALE, HH * SCALE / 2); g.stroke();
}}

function draw(i) {{
  const f = FRAMES[i]; if (!f) return;
  g.clearRect(0, 0, c.width, c.height);
  terrain();
  for (const e of f.e) {{
    const [id, team, kind, x, y, hp, mhp, deploying, name, cr] = e;
    const cx = px(x), cy = px(y);
    const tower = kind === 2;
    // Two radii. The hitbox is the unit's true CollisionRadius; the art is
    // drawn larger so the card is actually recognisable at this zoom. The
    // hitbox is still outlined, so nothing about the real footprint is lost.
    const hit = Math.max(3, px(cr || 0));
    const r = Math.max(tower ? SCALE * 1.4 : 11, hit * ART_SCALE);
    const img = ICONS[name];
    g.globalAlpha = deploying ? 0.5 : 1;
    // Team ring first, so the art sits on a coloured disc and ownership stays
    // readable even for art that is mostly transparent.
    g.beginPath(); g.arc(cx, cy, r, 0, 7);
    g.fillStyle = team === 0 ? '#4c8dff' : '#ff5c6c';
    g.fill();
    if (img && img.complete && img.naturalWidth) {{
      g.save();
      g.beginPath(); g.arc(cx, cy, r - 1, 0, 7); g.clip();
      const d = (r - 1) * 2;
      g.drawImage(img, cx - r + 1, cy - r + 1, d, d);
      g.restore();
      g.lineWidth = 2; g.strokeStyle = team === 0 ? '#9dc2ff' : '#ffb0b8';
      g.beginPath(); g.arc(cx, cy, r, 0, 7); g.stroke();
    }}
    if (SHOW_HITBOX && hit < r - 2) {{
      g.lineWidth = 1; g.strokeStyle = 'rgba(255,255,255,.30)';
      g.beginPath(); g.arc(cx, cy, hit, 0, 7); g.stroke();
    }}
    g.globalAlpha = 1;
    if (deploying) {{
      g.setLineDash([3, 3]); g.strokeStyle = '#fff';
      g.beginPath(); g.arc(cx, cy, r + 3, 0, 7); g.stroke(); g.setLineDash([]);
    }}
    if (mhp > 0 && hp < mhp) {{
      const w = r * 2;
      g.fillStyle = '#000a'; g.fillRect(cx - r, cy - r - 6, w, 3);
      g.fillStyle = '#5fe08a'; g.fillRect(cx - r, cy - r - 6, w * hp / mhp, 3);
    }}
  }}
  $('tick').textContent = f.t;
  $('time').textContent = (f.t / REAL_TPS).toFixed(1) + 's';
  $('count').textContent = f.e.length;
  $('be').textContent = f.x[0].toFixed(2);
  $('re').textContent = f.x[1].toFixed(2);
  $('beb').style.width = (f.x[0] * 10) + '%';
  $('reb').style.width = (f.x[1] * 10) + '%';
  // Hands are delta-encoded: scan back for the last frame that carried them.
  let h = null, n = null;
  for (let j = i; j >= 0 && !h; j--) {{ if (FRAMES[j].h) {{ h = FRAMES[j].h; n = FRAMES[j].n; }} }}
  if (h) {{
    hand($('bh'), h[0], n && n[0], f.x[0]);
    hand($('rh'), h[1], n && n[1], f.x[1]);
  }}
}}

function hand(host, cards, next, elixir) {{
  const want = (cards || []).concat(next ? ['|', next] : []);
  if (host.dataset.key === want.join(',') + ':' + Math.floor(elixir)) return;
  host.dataset.key = want.join(',') + ':' + Math.floor(elixir);
  host.innerHTML = '';
  for (const name of want) {{
    if (name === '|') {{
      const s = document.createElement('div'); s.className = 'sep'; host.appendChild(s);
      continue;
    }}
    const cost = COSTS[name] || 0;
    const el = document.createElement('div');
    el.className = 'card' + (cost > elixir ? ' dim' : '') + (name === next ? ' next' : '');
    el.title = name;
    const art = document.createElement(ICON_SRC[name] ? 'img' : 'div');
    art.className = 'art';
    if (ICON_SRC[name]) art.src = ICON_SRC[name];
    el.appendChild(art);
    const co = document.createElement('div');
    co.className = 'co';
    co.style.color = cost > elixir ? '#6d7391' : '#c9a3ff';
    co.textContent = cost ? cost + 'e' : '';
    el.appendChild(co);
    const nm = document.createElement('div'); nm.className = 'nm'; nm.textContent = name;
    el.appendChild(nm);
    host.appendChild(el);
  }}
}}

let playing = false, speed = 1, acc = 0, last = 0;
function loop(ts) {{
  if (playing) {{
    if (last) acc += (ts - last) / 1000 * TPS * speed;
    last = ts;
    while (acc >= 1) {{
      acc -= 1;
      let n = +slider.value + 1;
      if (n > +slider.max) {{ n = 0; }}
      slider.value = n;
    }}
    draw(+slider.value);
  }} else last = 0;
  requestAnimationFrame(loop);
}}
slider.oninput = () => draw(+slider.value);
$('play').onclick = e => {{ playing = !playing; e.target.textContent = playing ? 'pause' : 'play'; }};
$('hb').onclick = () => {{ SHOW_HITBOX = !SHOW_HITBOX; draw(+slider.value); }};
$('slow').onclick = () => speed = 0.5;
$('fast').onclick = () => speed = 4;
let pending = 0;
for (const k in ICON_SRC) {{
  const im = new Image(); pending++;
  im.onload = im.onerror = () => {{ if (--pending === 0) draw(+slider.value); }};
  im.src = ICON_SRC[k]; ICONS[k] = im;
}}
draw(0);
if (pending === 0) draw(0);
requestAnimationFrame(loop);
</script>
"""


def render_replay(
    arena: Arena,
    frames: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    ticks_per_second: float = 60,
    scale: int = 15,
    art_scale: float = 2.2,
    meta: str = "",
    real_tps: int | None = None,
    icons: Mapping[str, str] | None = None,
    costs: Mapping[str, int] | None = None,
) -> Path:
    """Write a standalone HTML replay of ``frames``.

    ``icons`` maps a character name to a PNG path. Only names that actually
    appear in ``frames`` are embedded, so a replay carries the handful of icons
    it needs rather than the whole card set.
    """
    arena_payload = {
        "half_width": arena.half_width,
        "half_height": arena.half_height,
        "cells": list(arena.cells),
    }
    # Names needing art: every unit that appears on the board, plus every card
    # that appears in either hand, since the panel draws those too.
    present = {e[8] for frame in frames for e in frame.get("e", ())}
    for frame in frames:
        for hand in frame.get("h", ()) or ():
            present.update(hand)
        for nxt in frame.get("n", ()) or ():
            if nxt:
                present.add(nxt)
    embedded = {
        name: _data_uri(Path(path))
        for name, path in (icons or {}).items()
        if name in present and Path(path).is_file()
    }
    page = _PAGE.format(
        icons=json.dumps(embedded, separators=(",", ":")),
        costs=json.dumps(
            {k: v for k, v in (costs or {}).items() if k in present}, separators=(",", ":")
        ),
        real_tps=real_tps or ticks_per_second,
        arena=json.dumps(arena_payload, separators=(",", ":")),
        frames=json.dumps(list(frames), separators=(",", ":")),
        scale=scale,
        art_scale=art_scale,
        tps=ticks_per_second,
        cw=arena.half_width * scale,
        ch=arena.half_height * scale,
        meta=meta,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def render_ascii(arena: Arena, entities, *, blue: str = "b", red: str = "r") -> str:
    """A quick terminal snapshot -- useful when a browser is overkill."""
    grid = [list(row) for row in arena.render().splitlines()]
    for entity in entities:
        if getattr(entity, "dead", False):
            continue
        cx, cy = arena.cell_at(entity.x, entity.y)
        if 0 <= cy < len(grid) and 0 <= cx < len(grid[cy]):
            glyph = blue if int(entity.team) == 0 else red
            grid[cy][cx] = glyph.upper() if int(entity.kind) == 2 else glyph
    return "\n".join("".join(row) for row in grid)
