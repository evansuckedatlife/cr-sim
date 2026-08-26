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

__all__ = ["render_replay"]

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
    <div class="row" style="margin-top:8px"><span>red elixir</span><span id="re">0</span></div>
    <div class="bar"><i id="reb" style="background:#ff5c6c;width:0%"></i></div>
    <p><input id="s" type="range" min="0" max="0" value="0"></p>
    <p><button id="play">play</button>
       <button id="slow">0.5x</button>
       <button id="fast">4x</button></p>
    <div class="legend">
      <b>blue</b> defends the top, <b>red</b> the bottom.<br>
      dashed ring = still deploying.<br>
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
    const [id, team, kind, x, y, hp, mhp, deploying, name] = e;
    const cx = px(x), cy = px(y);
    const tower = kind === 2;
    const r = tower ? SCALE * 1.5 : Math.max(4, SCALE * 0.55);
    g.beginPath(); g.arc(cx, cy, r, 0, 7);
    g.fillStyle = team === 0 ? '#4c8dff' : '#ff5c6c';
    g.globalAlpha = deploying ? 0.45 : 1; g.fill(); g.globalAlpha = 1;
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
$('slow').onclick = () => speed = 0.5;
$('fast').onclick = () => speed = 4;
draw(0); requestAnimationFrame(loop);
</script>
"""


def render_replay(
    arena: Arena,
    frames: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    ticks_per_second: float = 60,
    scale: int = 13,
    meta: str = "",
    real_tps: int | None = None,
) -> Path:
    """Write a standalone HTML replay of ``frames``."""
    arena_payload = {
        "half_width": arena.half_width,
        "half_height": arena.half_height,
        "cells": list(arena.cells),
    }
    page = _PAGE.format(
        real_tps=real_tps or ticks_per_second,
        arena=json.dumps(arena_payload, separators=(",", ":")),
        frames=json.dumps(list(frames), separators=(",", ":")),
        scale=scale,
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
