"""``python -m cr_sim.play`` -- play a match against a policy in a browser.

A local HTTP server on the standard library, deliberately: this is a tool for
looking at the simulator, and a tool for looking at something should not bring
its own dependency tree. There is no build step and no framework; the page is
one file the server prints.

The split is the same one :mod:`cr_sim.play.session` argues for. The engine runs
here and the browser draws what it is told. The page holds no rules -- it does
not know what a bridge costs to cross or whether a Miner may be placed in the
enemy half. It asks, and the same ``play_card`` the training environment calls
answers.

Polling rather than websockets. A match is one player on one machine, the state
is a few kilobytes, and a poll loop is a dozen lines that cannot desynchronise;
a socket would be a second code path for reconnects, backpressure and ordering,
to serve one localhost client.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from ..engine.arena import Tile, load_arena
from ..engine.entity import Team
from ..render.web import build_icon_map
from .page import PAGE
from .session import PlaySession, SessionConfig, random_controller

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "data_cache" / "csv_logic"

DEFAULT_DECK = (
    "Knight", "Musketeer", "Cannon", "Skeletons",
    "IceSpirits", "Log", "Fireball", "Goblins",
)


def _resolve(path: Path | None) -> Path | None:
    """Find a checkpoint given relative to the project rather than the shell.

    ``runs/dense-reward/final.pt`` is how the path is written down and how the
    trainer prints it, and it only resolves from the project root. Started from
    anywhere else it misses, so the project root is tried as well before giving
    up -- the alternative is a correct-looking command that silently plays a
    random opponent.
    """
    if path is None or path.is_file():
        return path
    candidate = ROOT / path
    return candidate if candidate.is_file() else path


class PlayServer:
    """Holds the one live session and the data every page load needs."""

    def __init__(self, build: Path, checkpoint: Path | None = None) -> None:
        self.data = LogicData.load(build)
        self.levels = build_level_table(self.data)
        self.registry = build_card_registry(self.data)
        self.arena = load_arena(self.data)
        self.icons = build_icon_map(self.registry)
        self.checkpoint = checkpoint
        self.lock = threading.Lock()
        self.session = self._new_session(DEFAULT_DECK, DEFAULT_DECK, seed=0)

    # -- session -----------------------------------------------------------

    def _controller(self, seed: int):
        """The trained policy if one was given, otherwise random play.

        Falling back rather than refusing to start: the point of the page is to
        watch the simulator, and that is worth doing before any policy exists.
        A failed load says so on the page rather than in a traceback.
        """
        if self.checkpoint is None:
            return random_controller(seed), "random"
        try:
            from .policy import policy_controller

            return policy_controller(self.checkpoint, self, seed), f"policy:{self.checkpoint.name}"
        except Exception as exc:  # noqa: BLE001 - reported to the page
            return random_controller(seed), f"random (policy failed: {exc})"

    def _new_session(self, human: tuple[str, ...], ai: tuple[str, ...], seed: int) -> PlaySession:
        controller, label = self._controller(seed)
        session = PlaySession(
            data=self.data,
            levels=self.levels,
            registry=self.registry,
            config=SessionConfig(human_deck=tuple(human), ai_deck=tuple(ai), seed=seed),
            controller=controller,
        )
        session.opponent_label = label  # type: ignore[attr-defined]
        return session

    # -- api ---------------------------------------------------------------

    def handle(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == "/api/setup":
            return self.setup()
        if path == "/api/state":
            with self.lock:
                self.session.advance()
                snapshot = self.session.snapshot()
            snapshot["opponent"] = getattr(self.session, "opponent_label", "random")
            if self.session.controller_error:
                snapshot["opponent"] += f" (stopped: {self.session.controller_error})"
            return snapshot
        if path == "/api/play":
            with self.lock:
                return self.session.play(
                    str(body.get("card", "")),
                    float(body.get("x", 0.0)),
                    float(body.get("y", 0.0)),
                )
        if path == "/api/new":
            human = tuple(body.get("human") or DEFAULT_DECK)
            ai = tuple(body.get("ai") or DEFAULT_DECK)
            seed = int(body.get("seed", 0))
            invalid = [c for c in human + ai if self.registry.get(c) is None]
            if invalid:
                return {"ok": False, "reason": f"unknown cards: {', '.join(sorted(set(invalid)))}"}
            if len(set(human)) != 8 or len(set(ai)) != 8:
                return {"ok": False, "reason": "each deck needs eight different cards"}
            with self.lock:
                self.session = self._new_session(human, ai, seed)
            return {"ok": True}
        if path == "/api/control":
            with self.lock:
                if "paused" in body:
                    self.session.paused = bool(body["paused"])
                if "speed" in body:
                    self.session.speed = max(0.1, min(4.0, float(body["speed"])))
                return {"ok": True, "paused": self.session.paused, "speed": self.session.speed}
        return {"ok": False, "reason": f"no route {path}"}

    def setup(self) -> dict[str, Any]:
        """The static things a page needs once: terrain, icons, the card pool."""
        arena = self.arena
        terrain = []
        for cy in range(arena.half_height):
            row = []
            for cx in range(arena.half_width):
                bits = arena.cell(cx, cy)
                if bits & Tile.WATER:
                    row.append(2)
                elif bits & Tile.BLOCKED:
                    row.append(1)
                else:
                    row.append(0)
            terrain.append(row)

        cards = sorted(
            (
                {
                    "name": card.name,
                    "cost": card.mana_cost,
                    "kind": card.kind.value,
                    "rarity": card.rarity,
                }
                for card in self.registry.standard()
            ),
            key=lambda c: (c["cost"], c["name"]),
        )
        return {
            "terrain": terrain,
            "halfWidth": arena.half_width,
            "halfHeight": arena.half_height,
            "widthTiles": arena.half_width / 2,
            "heightTiles": arena.half_height / 2,
            "icons": self.icons,
            "cards": cards,
            "deck": list(DEFAULT_DECK),
            "riverBand": [round(v / 18000, 2) for v in arena.river_band()],
        }


class _Handler(BaseHTTPRequestHandler):
    server_version = "cr-sim"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - quiet by default
        pass

    def _send(self, payload: Any, content_type: str = "application/json") -> None:
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(self.server.play.handle(self.path, {}))  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        self._send(self.server.play.handle(self.path, body))  # type: ignore[attr-defined]


def serve(build: Path, port: int, checkpoint: Path | None, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.play = PlayServer(build, checkpoint)  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{port}/"
    print(f"cr-sim play -> {url}")
    print(f"  opponent: {getattr(server.play.session, 'opponent_label', 'random')}")
    print("  ctrl-c to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-play")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument(
        "--policy", type=Path, default=None,
        help="a trained checkpoint to play against; random play if omitted",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(args.build, args.port, _resolve(args.policy), open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
