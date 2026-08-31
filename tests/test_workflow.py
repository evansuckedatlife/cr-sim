"""The guards around how work here is launched, saved and registered.

Three failures this project can afford exactly once each, and all three were
one keystroke away when these were written:

*   ``register_job.py`` wrote both of a run's files unconditionally, so one
    mistyped ``--name`` replaced a training run's several hundred rows and its
    entire config with a placeholder, exit code 0 and a success message.
*   ``supervise.ps1`` -- the launcher you reach for after a bugcheck -- passed
    neither ``--tower-level`` nor ``--elixir-weight``, so every crash-resilient
    run silently took the defaults the handoff calls not optional and got the
    opposite of both.
*   Every checkpoint was written by ``torch.save`` straight onto its
    destination, which truncates first. ``--resume`` from that one file is the
    whole crash-resilience strategy, and it was being taken apart and put back
    together every three updates on a machine that bugchecks.

None of these is a page or an algorithm, which is why none of them had a test.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ------------------------------------------------------ writing over a run


def test_registering_over_a_training_run_keeps_its_rows(tmp_path):
    """A mistyped ``--name`` used to be unrecoverable, silently and instantly.

    ``runs/`` is gitignored, there is no backup, and a run's ``config.json``
    is the only record of the tower level, the opponent, the weights it
    started from and the anchor it was held to -- so the rows and the meaning
    of the rows both went in the same write. This asserts on the data that
    survives rather than on the message, and on both sides of the guard: a
    name that holds nothing is still writable, or the guard would just be a
    script nobody can use.
    """
    import register_job
    from cr_sim.train.watch import read_metrics

    run = tmp_path / "learn-lvl5-kl01"
    run.mkdir()
    rows = [{"updates": u, "steps": u * 2048, "eval_lift_sd": 0.4 * u}
            for u in (1, 2, 3)]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "tower_level": 5,
        "eval_opponent": "random",
        "init_from": "checkpoints/clone.pt",
        "kl_reference": "checkpoints/clone.pt",
    }), encoding="utf-8")

    code = register_job.main([
        "--name", "learn-lvl5-kl01", "--note", "meant to type a new name",
        "--status", "running", "--runs", str(tmp_path)])
    assert code != 0, "a run holding measurements was written over anyway"

    survived = read_metrics(run / "metrics.jsonl")
    assert [r["updates"] for r in survived] == [1, 2, 3], \
        f"the run's rows are gone: {survived}"
    assert [r["eval_lift_sd"] for r in survived] == [0.4, 0.8, pytest.approx(1.2)]

    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for key, value in (("tower_level", 5), ("eval_opponent", "random"),
                       ("init_from", "checkpoints/clone.pt"),
                       ("kl_reference", "checkpoints/clone.pt")):
        assert config.get(key) == value, \
            f"{key} was overwritten, and it is the only record of what the rows mean"

    # The other side of it. A guard that also refuses the ordinary case is a
    # guard the next session deletes.
    assert register_job.main([
        "--name", "bench-tick-loop", "--note", "profiling. Running.",
        "--status", "running", "--runs", str(tmp_path)]) == 0
    # ... including saying so when it finishes, which is the instruction
    # CLAUDE.md gives and the one every session skips. One placeholder row is
    # not a measurement, so re-registering it is not destroying anything.
    assert register_job.main([
        "--name", "bench-tick-loop", "--note", "profiling. 61k ticks/s.",
        "--status", "done", "--runs", str(tmp_path)]) == 0

    # And both named ways past the guard keep every row that was there.
    assert register_job.main([
        "--name", "learn-lvl5-kl01", "--note", "adding a reading", "--append",
        "--runs", str(tmp_path)]) == 0
    kept = read_metrics(run / "metrics.jsonl")
    assert [r["updates"] for r in kept] == [1, 2, 3], "--append dropped rows"
    assert json.loads((run / "config.json").read_text(
        encoding="utf-8"))["tower_level"] == 5, "--append dropped the config"

    assert register_job.main([
        "--name", "learn-lvl5-kl01", "--note", "starting over", "--replace",
        "--runs", str(tmp_path)]) == 0
    moved = sorted(p.name for p in run.glob("metrics.*.jsonl"))
    assert moved, "--replace truncated the file instead of moving it aside"
    aside = read_metrics(run / moved[0])
    assert [r["updates"] for r in aside] == [1, 2, 3], \
        "the file --replace moved aside is not the file that was there"


# ------------------------------------- the settings that are not optional


def _supervise_common() -> "dict[str, str]":
    """The ``$common`` argv list ``supervise.ps1`` launches the trainer with.

    Read out of the script rather than by running it, because running it
    starts a training job.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    block = source.split("$common = @(", 1)[1].split("\n)", 1)[0]
    # Comments first: a flag named only in a comment is not a flag that is
    # passed, and this test would otherwise pass over one that had been
    # commented out.
    body = "\n".join(re.sub(r"#.*$", "", line) for line in block.splitlines())
    tokens = re.findall(r'"([^"]*)"', body)
    return {tokens[i]: tokens[i + 1]
            for i in range(len(tokens) - 1)
            if tokens[i].startswith("--") and not tokens[i + 1].startswith("--")}


def test_a_fresh_run_defaults_to_the_settings_that_are_not_optional():
    """Both of these used to default to the value the handoff argues against.

    At tower level 11 a 120-second match ends with 92% of tower health
    untouched and 92% of matches drawn, so crowns -- the only real objective
    -- almost never fire; ``runs/learn-1m-factored-lvl11`` is 557,056 steps of
    what that buys. At ``--elixir-weight 0.3`` a pass earns +0.071 more reward
    than a placement, and the searching bot at 0.3 never played a card at all.

    Two independent sources are read and neither is written here: argparse's
    own resolved defaults, and the argv list the crash-resilient launcher
    builds. Moving either one alone goes red, which is the point -- the
    launcher passing nothing is how the defaults came to matter.
    """
    from cr_sim.train.run import build_parser

    args = build_parser().parse_args([])
    assert args.tower_level == 5, \
        f"a run launched with no --tower-level trains at {args.tower_level}"
    assert args.elixir_weight == 0.0, \
        f"a run launched with no --elixir-weight is charged {args.elixir_weight}"

    common = _supervise_common()
    assert "--tower-level" in common and "--elixir-weight" in common, \
        f"supervise.ps1 leaves these to whatever run.py defaults to: {sorted(common)}"
    assert int(common["--tower-level"]) == args.tower_level
    assert float(common["--elixir-weight"]) == args.elixir_weight


def test_the_launcher_will_not_pick_a_run_name_for_you():
    """``-Name`` defaulted to ``projected-v2``, an existing 20-row run.

    So ``powershell -File supervise.ps1`` with no arguments -- what you type
    after a bugcheck, from muscle memory -- found that directory's
    ``checkpoint.pt``, resumed somebody else's experiment from it and appended
    to its metrics. Refused rather than declared ``Mandatory``, because a
    mandatory parameter prompts and half the callers here are not people.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    block = source.split("param(", 1)[1].split(")", 1)[0]
    declaration = [line for line in block.splitlines() if "$Name" in line]
    assert declaration, "supervise.ps1 no longer takes a -Name at all"
    assert "=" not in declaration[0], \
        f"-Name still carries a default, which is some other run: {declaration[0].strip()}"
    assert re.search(r"if\s*\(\s*-not\s+\$Name\s*\)", source), \
        "nothing stops the script when -Name is missing, so $runDir is runs\\"


# ------------------------------------------- a checkpoint written atomically


def test_a_checkpoint_survives_a_crash_during_its_own_write(tmp_path,
                                                            monkeypatch):
    """The 20 MB window this machine bugchecks inside of.

    ``torch.save`` onto the destination truncates it first, so for the length
    of the serialisation ``checkpoint.pt`` is neither the old checkpoint nor
    the new one -- and ``--resume`` from that single file is the entire
    strategy for surviving a crash. ``supervise.ps1`` opened that window every
    three updates.

    The failure is simulated at the only place it can be: a ``torch.save``
    that writes some bytes and then dies, which is what a killed process
    leaves. What is asserted is that the *previous* checkpoint still loads and
    still says what it said.
    """
    import torch

    from cr_sim.train.run import save_checkpoint

    path = tmp_path / "checkpoint.pt"
    save_checkpoint({"steps": 557056, "updates": 272,
                     "state_dict": {"trunk.weight": torch.ones(3, 3)}}, path)
    before = torch.load(path, map_location="cpu", weights_only=False)
    assert before["steps"] == 557056

    def dies_half_way(payload, target, *args, **kwargs):
        with open(target, "wb") as handle:
            handle.write(b"PK\x03\x04")  # a real checkpoint starts this way
        raise RuntimeError("the machine went down mid-write")

    monkeypatch.setattr(torch, "save", dies_half_way)
    with pytest.raises(RuntimeError):
        save_checkpoint({"steps": 999999, "state_dict": {}}, path)
    monkeypatch.undo()

    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["steps"] == 557056, \
        "the crash rewrote the checkpoint it was supposed to be replacing"
    assert after["updates"] == 272
    assert torch.equal(after["state_dict"]["trunk.weight"], torch.ones(3, 3))

    # And nothing half-written is left lying about: 20 MB per crash on a disk
    # with 3.4 GB free is its own outage.
    assert not list(tmp_path.glob("*.tmp")), \
        f"a partial checkpoint was left behind: {list(tmp_path.glob('*.tmp'))}"
