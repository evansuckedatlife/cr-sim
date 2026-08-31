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
import shutil
import subprocess
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

    # Both halves of the move, not just the rows. Dropping the config half
    # leaves --replace truncating the tower_level, eval_opponent, init_from
    # and kl_reference this test's own docstring calls the only record of what
    # the rows mean -- and the metrics assertions above stay green.
    moved_config = sorted(run.glob("config.*.json"))
    assert moved_config, \
        "--replace truncated config.json instead of moving it aside, so the " \
        "only record of what the moved rows mean is gone"
    kept_config = json.loads(moved_config[0].read_text(encoding="utf-8"))
    for key, value in (("tower_level", 5), ("eval_opponent", "random"),
                       ("init_from", "checkpoints/clone.pt"),
                       ("kl_reference", "checkpoints/clone.pt")):
        assert kept_config.get(key) == value, \
            f"{key} did not survive --replace: {sorted(kept_config)}"


def test_a_job_entry_that_holds_real_measurements_is_protected_too(tmp_path):
    """The rows clause of the guard is the only thing protecting fourteen runs.

    `_why_protected` refuses on two counts: the metrics file holds rows, or
    the config was not written by this script. The existing fixture trips both
    at once, so deleting the rows clause outright leaves it green -- while
    fourteen directories under `runs/` carry `kind: "job"` in their config,
    written by `run_ladder.py` or by this script itself, and hold real
    measurements: `audit-ladder-greedy` 18 rows, `agent-verify-metric` 34,
    `agent-ladder-v1` 12, `audit-ladder-sampled` 12, `bench-network` 5.

    For those, the rows clause is the whole guard. This is that shape.
    """
    import register_job
    from cr_sim.train.watch import read_metrics

    run = tmp_path / "audit-ladder-greedy"
    run.mkdir()
    rows = [{"updates": u, "steps": 0, "episodes": 12, "ladder_elo": 40.0 * u,
             "eval_opponent": "random"} for u in (1, 2, 3, 4)]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # Written by this script, so the config clause has nothing to say about it.
    (run / "config.json").write_text(json.dumps({
        "kind": "job", "note": "[done] ladder, greedy arms",
        "tower_level": 11, "eval_opponent": "random", "reward": "simple",
    }), encoding="utf-8")

    code = register_job.main([
        "--name", "audit-ladder-greedy", "--status", "running",
        "--note", "meant to type audit-ladder-sampled", "--runs", str(tmp_path)])
    assert code != 0, \
        "a job entry carrying real measurements was written over, and its " \
        "config says kind=job so nothing else was going to stop it"

    survived = read_metrics(run / "metrics.jsonl")
    assert [r["updates"] for r in survived] == [1, 2, 3, 4], \
        f"the entry's rows are gone: {survived}"
    assert [r["ladder_elo"] for r in survived] == [40.0, 80.0, 120.0, 160.0]
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["tower_level"] == 11 and config["reward"] == "simple", \
        f"the config was reduced to a placeholder: {sorted(config)}"


def test_appending_never_destroys_a_config_it_could_not_parse(tmp_path):
    """The default path refuses this directory *because* the config is unreadable.

    And its error message names `--append` as the safe route, whose documented
    contract is "keeps every key in the existing config except the note". It
    did the opposite: the parse failure was swallowed to `raw = None`, so
    nothing was carried forward and the write landed on top of perfectly
    readable text holding a tower_level, an eval_opponent and a total_steps
    behind one trailing comma. The old bytes were not moved aside; they were
    gone.
    """
    import register_job
    from cr_sim.train.watch import read_metrics

    run = tmp_path / "learn-lvl5-kl01"
    run.mkdir()
    rows = [{"updates": u, "eval_lift_sd": 0.1 * u} for u in range(1, 301)]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    damaged = ('{\n "tower_level": 11,\n "eval_opponent": "random",\n'
               ' "total_steps": 1000000,\n}\n')
    (run / "config.json").write_text(damaged, encoding="utf-8")

    # The default path refuses, and names --append.
    assert register_job.main([
        "--name", "learn-lvl5-kl01", "--note", "n", "--runs", str(tmp_path)]) != 0

    assert register_job.main([
        "--name", "learn-lvl5-kl01", "--note", "adding a reading", "--append",
        "--runs", str(tmp_path)]) == 0
    assert len(read_metrics(run / "metrics.jsonl")) == 300, "--append lost rows"

    aside = sorted(run.glob("config.*.json"))
    assert aside, \
        "the unreadable config was written over, so what those 300 rows were " \
        "measured under is gone -- and the route that did it is the one the " \
        "refusal above recommends"
    assert aside[0].read_text(encoding="utf-8") == damaged, \
        "the bytes that were moved aside are not the bytes that were there"


def test_updating_a_note_does_not_make_a_dead_run_look_alive(tmp_path):
    """The page derives "is this run live" from metrics.jsonl's mtime.

    And `register_job` rewrote that file even when it added no rows -- so
    doing the one thing CLAUDE.md tells every session to do when a job
    finishes, updating the note to say so, redrew a run that had not written a
    row in three days with a green pip, an accent progress bar and a
    seventeen-hour countdown to a finish that had already happened.
    """
    import hashlib
    import os
    import time as clock

    import register_job
    from cr_sim.train.watch import render_multi

    run = tmp_path / "learn-1m-flat"
    run.mkdir()
    rows = [{"updates": u, "steps": u * 2048, "eval_lift_sd": 0.3,
             "eval_win": 0.4, "control_win": 0.26, "eval_episodes": 40,
             "eval_opponent": "random"} for u in range(1, 25)]
    metrics = run / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows),
                       encoding="utf-8")
    (run / "config.json").write_text(
        json.dumps({"kind": "job", "note": "running"}), encoding="utf-8")

    dead = clock.time() - 74.8 * 3600
    os.utime(metrics, (dead, dead))
    before = (os.path.getmtime(metrics), hashlib.md5(metrics.read_bytes()).hexdigest())

    assert register_job.main([
        "--name", "learn-1m-flat", "--status", "done",
        "--note", "Dead. Superseded by learn-lvl5-kl01.",
        "--append", "--runs", str(tmp_path)]) == 0

    after = (os.path.getmtime(metrics), hashlib.md5(metrics.read_bytes()).hexdigest())
    assert before[1] == after[1], "the note update changed the rows"
    assert before[0] == after[0], \
        "the note update moved metrics.jsonl's mtime, which is the only thing " \
        "the page reads to decide a run is live"

    # And the note did land, which is what makes the above non-vacuous.
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["note"] == "[done] Dead. Superseded by learn-lvl5-kl01."

    # Read back the way the page reads it: 74.8 hours silent is not live.
    silence = clock.time() - os.path.getmtime(metrics)
    assert silence > 3600, \
        f"the run reads as {silence / 3600:.1f} hours silent, so it is drawn live"


# ------------------------------------- the settings that are not optional


def _strip_comments(block: str) -> str:
    """A fragment of PowerShell with its line comments gone.

    A flag named only in a comment is not a flag that is passed, and reading
    the block with them in it passes over one that has been commented out.
    """
    return "\n".join(re.sub(r"#.*$", "", line) for line in block.splitlines())


def _supervise_common() -> "dict[str, str]":
    """The ``$common`` argv list ``supervise.ps1`` launches the trainer with.

    Read out of the script rather than by running it, because running it
    starts a training job.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    block = source.split("$common = @(", 1)[1].split("\n)", 1)[0]
    tokens = re.findall(r'"([^"]*)"', _strip_comments(block))
    return {tokens[i]: tokens[i + 1]
            for i in range(len(tokens) - 1)
            if tokens[i].startswith("--") and not tokens[i + 1].startswith("--")}


def _supervise_overrides() -> "set[str]":
    """Every ``--flag`` the launcher appends after ``$common``.

    ``$common`` is the array that gets built; ``$argv`` is the array that gets
    passed, and reading only the first says nothing about the second. argparse
    takes the last occurrence of a repeated flag, so one token appended here
    silently overrides whatever ``$common`` declares -- and a test that reads
    only ``$common`` reports the two sources agreeing while every supervised
    run trains somewhere else.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    lines = [l for l in _strip_comments(source).splitlines()
             if re.match(r"\s*\$argv\s*=", l)]
    assert lines, "supervise.ps1 no longer builds an $argv to launch with"
    return {token for line in lines
            for token in re.findall(r'"(--[^"]*)"', line)}


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

    # And nothing appended after `$common` quietly overrides it. `$common` is
    # what gets declared; `$argv` is what gets passed, and argparse takes the
    # last occurrence -- so `$common + @("--tower-level","11")` trains every
    # supervised run at 11 while the two sources above go on agreeing on 5.
    clashes = _supervise_overrides() & set(common)
    assert not clashes, \
        f"the launcher appends {sorted(clashes)} after $common, and argparse " \
        "takes the last one -- so the settings read above are not the " \
        "settings the run gets"


def _tower_help(path: Path) -> "tuple[int, str]":
    """A module's ``--tower-level`` default and the help text beside it."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--tower-level"
                   for a in node.args):
            continue
        found = {k.arg: k.value for k in node.keywords}
        assert "default" in found, ast.dump(node)
        text = found.get("help")
        return (ast.literal_eval(found["default"]),
                ast.literal_eval(text) if text is not None else "")
    raise AssertionError(f"{path} no longer takes a --tower-level")


def test_a_scripts_help_does_not_describe_a_default_another_module_stopped_having():
    """`--tower-level`'s help said it agreed with a module it now differs from.

    `evaluate_vs_expert.py` defaults to 11 and said so "agreeing with
    cr_sim.train.evaluate's CLI and with cr_sim.train.run -- this defaulted to
    5 while both of those defaulted to 11". `cr_sim.train.run` defaults to 5
    now, deliberately, so the sentence printed by `--help` became false about
    the codebase it describes. The repaired test in test_measurement.py records
    the change and the user-facing help was not moved with it.

    Read out of the modules rather than written down, so the next move of
    either default lands here.
    """
    mine, text = _tower_help(ROOT / "scripts" / "evaluate_vs_expert.py")

    for module, path in (("cr_sim.train.evaluate",
                          ROOT / "cr_sim" / "train" / "evaluate.py"),
                         ("cr_sim.train.run",
                          ROOT / "cr_sim" / "train" / "run.py")):
        if module not in text:
            continue
        theirs, _ = _tower_help(path)
        at = text.index(module)
        window = text[max(0, at - 90):at + 90]
        if theirs != mine:
            assert "agree" not in window, \
                f"the help presents {module} as agreeing with this script, " \
                f"which defaults to {mine} while {module} defaults to {theirs}"
            assert f"defaults to {theirs}" in window, \
                f"the help says {module} differs but never says what it is: " \
                f"{window!r} -- it defaults to {theirs}"
        else:
            assert "differ" not in window, \
                f"the help presents {module} as differing, and both default " \
                f"to {mine}"


def _started_run(directory: Path, *, tower_level=11, elixir_weight=0.3,
                 rows=(1, 2, 3, 4), checkpoint=True) -> None:
    """A run directory as `cr_sim.train.run` leaves one, without training."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps({"updates": u, "steps": u * 32,
                            "eval_lift_sd": 0.1 * u}) + "\n" for u in rows),
        encoding="utf-8")
    config = {"total_steps": 128, "seed": 0, "reward": "projected",
              "match_seconds": 120, "frame_skip": 30}
    if tower_level is not None:
        config["tower_level"] = tower_level
    if elixir_weight is not None:
        config["elixir_weight"] = elixir_weight
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if checkpoint:
        torch.save({"steps": 128, "updates": 4, "state_dict": {}},
                   directory / "checkpoint.pt")


def _launch(out: Path, name: str, *extra) -> "tuple[int, str]":
    """`cr_sim.train.run.main`, stopped at the card data rather than trained.

    A guard that only fires after twenty minutes of rollouts is not a guard, so
    both of these run before anything is loaded -- which is exactly what lets
    this point `--build` at a directory that does not exist and read the
    resulting `FileNotFoundError` as "execution got past the guards".
    """
    from cr_sim.train.run import main

    argv = ["--out", str(out), "--name", name, "--envs", "1", "--horizon", "32",
            "--eval-every", "0", "--save-every", "1", "--workers", "0",
            "--steps", "32", "--build", str(out.parent / "no-such-build"),
            *extra]
    try:
        return int(main(argv) or 0), ""
    except SystemExit as refusal:
        return 1, str(refusal)
    except FileNotFoundError as reached:
        return 0, f"reached the trainer: {reached}"


def test_a_fresh_run_will_not_start_on_top_of_an_existing_one(tmp_path):
    """The guard `register_job.py` got, on the thing that produces the data.

    `register_job.py` writes no training data and is guarded; `run.py` owns
    every row, every config and every checkpoint under a run's name and was
    not. Re-running a name without `--resume` opened `metrics.jsonl` with mode
    "w", rewrote `config.json` and overwrote `checkpoint.pt`, `best.pt` and
    `final.pt` -- exit code 0 and a "done in 0.7 min" line. Measured before
    this existed: a run holding updates 1-4 out to step 128 came back holding
    update 1 alone, a config claiming `total_steps` 32 and `seed` 7, and a
    checkpoint at 32 steps.

    `runs/` is gitignored with no backup, which is the exact argument the
    commit message makes for guarding `register_job.py`.
    """
    import torch

    runs = tmp_path / "runs"
    run = runs / "resume-probe"
    _started_run(run)
    before = (run / "metrics.jsonl").read_bytes()
    before_config = (run / "config.json").read_bytes()

    code, said = _launch(runs, "resume-probe", "--seed", "7")
    assert code != 0, "a fresh run started on top of a run holding four updates"
    assert "--resume" in said and "--name" in said, \
        f"the refusal names neither way past it: {said!r}"

    assert (run / "metrics.jsonl").read_bytes() == before, "the rows were truncated"
    assert (run / "config.json").read_bytes() == before_config, \
        "config.json was rewritten, and it is the only record of what the rows mean"
    assert torch.load(run / "checkpoint.pt", map_location="cpu",
                      weights_only=False)["steps"] == 128

    # Both other sides of it. A name that holds nothing is still writable, or
    # the guard is a script nobody can use...
    _code, said = _launch(runs, "brand-new")
    assert "is not an empty slot" not in said, \
        f"an unused name was refused as though it held a run: {said!r}"

    # ... and so is a name holding only the placeholder row register_job.py
    # writes for a job that has measured nothing yet.
    placeholder = runs / "bench-tick-loop"
    placeholder.mkdir()
    (placeholder / "metrics.jsonl").write_text(
        json.dumps({"updates": 1, "steps": 0, "episodes": 0}) + "\n",
        encoding="utf-8")
    _code, said = _launch(runs, "bench-tick-loop")
    assert "is not an empty slot" not in said, \
        f"a job entry that has measured nothing was refused: {said!r}"

    # ... and --replace moves what is there aside rather than truncating it.
    _code, said = _launch(runs, "resume-probe", "--replace")
    assert "is not an empty slot" not in said, f"--replace was refused: {said!r}"
    moved = sorted(p.name for p in run.glob("metrics.*.jsonl"))
    assert moved, "--replace truncated the metrics instead of moving them aside"
    assert (run / moved[0]).read_bytes() == before, \
        "the file moved aside is not the file that was there"
    assert sorted(p.name for p in run.glob("checkpoint.*.pt")), \
        "--replace left the old checkpoint to be overwritten"


def test_resuming_a_run_cannot_relabel_the_rows_it_already_holds(tmp_path):
    """`config.json` is rewritten from the CLI on every start, resume included.

    So resuming a run trained at `--tower-level 11` with today's launcher,
    which passes 5, trained the remainder in a different arena *and* rewrote
    the run's own config to claim it had always been 5. The earlier rows record
    no level of their own -- no metrics row on disk carries one -- so that
    config was the only thing that knew what they meant. Seven runs here hold a
    resumable checkpoint and a config that predates these being recorded at
    all, and `supervise.ps1 -Name projected-v2`, the command its own header
    documents, was one of them.

    `_ladder_ratings` already refuses a rating table fitted at another tower
    level, with the same argument. This is that refusal for a resume.
    """
    runs = tmp_path / "runs"
    run = runs / "resume-probe"
    _started_run(run, tower_level=11, elixir_weight=0.3)
    before = (run / "config.json").read_bytes()

    code, said = _launch(runs, "resume-probe", "--resume",
                         "--tower-level", "5", "--elixir-weight", "0.0")
    assert code != 0, \
        "the run was resumed into a different arena and its config rewritten " \
        "to say it had always been that one"
    assert "tower_level" in said and "11" in said and "5" in said, \
        f"the refusal does not say which two arenas disagree: {said!r}"
    assert (run / "config.json").read_bytes() == before, \
        "the config was rewritten by the very command that was refused"

    # Agreeing is not refused, or --resume would be unusable.
    _code, said = _launch(runs, "resume-probe", "--resume",
                          "--tower-level", "11", "--elixir-weight", "0.3")
    assert "records tower_level" not in said, \
        f"a resume that agrees with the recorded arena was refused: {said!r}"

    # And the record that gets written, asked of the function that decides it.
    # Every arena key, both directions, including the case seven real runs on
    # disk are in: a config that predates the key being recorded at all, where
    # stamping today's value puts a number where the file's silence is the
    # only honest answer.
    from cr_sim.train.run import _ARENA_KEYS, _check_resume_arena

    recorded = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert set(_ARENA_KEYS) <= set(recorded), \
        f"the fixture no longer records every arena key: {sorted(recorded)}"

    for key in _ARENA_KEYS:
        agreeing = {k: recorded[k] for k in _ARENA_KEYS}
        kept = _check_resume_arena(run, dict(agreeing, resumed=True))
        assert kept[key] == recorded[key], \
            f"{key} was dropped from a config that records it"

        disagreeing = dict(agreeing)
        disagreeing[key] = "something-else"
        with pytest.raises(SystemExit) as refused:
            _check_resume_arena(run, disagreeing)
        assert key in str(refused.value), \
            f"a resume disagreeing about {key} was allowed through"

    silent = runs / "projected-v2"
    _started_run(silent, tower_level=None, elixir_weight=None)
    written = _check_resume_arena(
        silent, {"tower_level": 5, "elixir_weight": 0.0, "reward": "projected",
                 "match_seconds": 120, "frame_skip": 30, "seed": 0})
    for absent in ("tower_level", "elixir_weight"):
        assert absent not in written, \
            f"a resume would stamp {absent}={written[absent]!r} onto a config " \
            "describing rows measured before that key was recorded"
    assert written["seed"] == 0, "an ordinary key stopped being refreshed"
    # The keys it does record are still checked against.
    assert written["reward"] == "projected" and written["frame_skip"] == 30


powershell = pytest.mark.skipif(
    shutil.which("powershell") is None, reason="needs Windows PowerShell")


def _supervise_header() -> str:
    """Everything in ``supervise.ps1`` above the first line that opens the log.

    The parameter block, the guard, and the two paths derived from ``-Name``.
    Running that much is safe -- it starts nothing -- and running it is the
    only way to tell a guard from a comment.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    head, sep, _ = source.partition("$log = Join-Path")
    assert sep, "supervise.ps1 no longer opens a log where this test splits it"
    return head


@powershell
def test_the_launcher_will_not_pick_a_run_name_for_you(tmp_path):
    """``-Name`` defaulted to ``projected-v2``, an existing 20-row run.

    So ``powershell -File supervise.ps1`` with no arguments -- what you type
    after a bugcheck, from muscle memory -- found that directory's
    ``checkpoint.pt``, resumed somebody else's experiment from it and appended
    to its metrics. Refused rather than declared ``Mandatory``, because a
    mandatory parameter prompts and half the callers here are not people.

    Run, not read. Asserting that the param line carries no ``=`` and that an
    ``if (-not $Name)`` exists is a description of the guard's shape, and both
    assertions stay true with ``exit 2`` replaced by ``$Name = "projected-v2"``
    -- which is the original defect, verbatim, under a green test.
    """
    probe = tmp_path / "header.ps1"
    probe.write_text(
        _supervise_header()
        + '\nWrite-Output "name=[$Name]"\nWrite-Output "runDir=$runDir"\n',
        encoding="utf-8")

    def call(*extra):
        return subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(probe), *extra],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)

    missing = call()
    assert missing.returncode != 0, \
        "the launcher accepted no -Name at all, so $runDir is some other run"
    assert "name=[" not in missing.stdout, \
        f"the guard let the script carry on and pick a name: {missing.stdout!r}"
    assert "runDir=" not in missing.stdout, \
        f"a run directory was resolved with no -Name given: {missing.stdout!r}"
    # Whitespace collapsed: PowerShell wraps Write-Error at the console width,
    # so the message arrives with newlines through the middle of it.
    said = " ".join((missing.stderr + missing.stdout).split())
    assert "-Name" in said and "required" in said, \
        f"the refusal says nothing about what is missing: {said[:200]!r}"

    # The other side of it: a guard that also refuses the ordinary case is a
    # guard the next session deletes.
    given = call("-Name", "a-new-run")
    assert given.returncode == 0, given.stderr
    assert "name=[a-new-run]" in given.stdout
    assert given.stdout.rstrip().endswith("runs\\a-new-run")


@powershell
def test_the_launcher_can_tell_a_finished_run_from_a_crashed_one(tmp_path):
    """``$p.ExitCode`` is ``$null`` for every outcome without ``$p.Handle``.

    ``Start-Process -PassThru`` in Windows PowerShell 5.1 hands back a Process
    whose native handle is never opened, so ``WaitForExit()`` has nothing to
    cache the exit code from. ``if ($code -eq 0) { break }`` then never fires,
    and a run that finished cleanly is relaunched ``MaxRestarts`` more times --
    fifty by default -- each attempt resuming an already-finished checkpoint
    and logging ``exited , restarting in 15s`` with the code missing.

    The launcher's own three lines are spliced verbatim into a harness that
    supplies ``$root``, ``$Name`` and ``$argv``, so what runs here is the
    source and not a copy of it.
    """
    source = (ROOT / "supervise.ps1").read_text(encoding="utf-8")
    start = source.index("    $p = Start-Process")
    end = source.index("    $code = $p.ExitCode") + len("    $code = $p.ExitCode")
    fragment = source[start:end]
    assert "Start-Process" in fragment and "WaitForExit" in fragment, fragment

    for wanted in (0, 3):
        # A script file rather than `python -c`: PowerShell re-quotes an
        # ArgumentList and splits the inline program on its first space.
        child = tmp_path / f"child{wanted}.py"
        child.write_text(f"import sys\nsys.exit({wanted})\n", encoding="utf-8")

        probe = tmp_path / f"exit{wanted}.ps1"
        probe.write_text(
            f'$root = "{tmp_path}"\n'
            f'$Name = "probe{wanted}"\n'
            f'$argv = @("{child}")\n'
            + fragment
            + '\nWrite-Output "code=[$code]"\n',
            encoding="utf-8")
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180)
        assert f"code=[{wanted}]" in done.stdout, \
            f"a child that exited {wanted} was reported as " \
            f"{done.stdout.strip()!r}, so the launcher cannot tell a clean " \
            f"finish from a crash and restarts either one"


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


def _torch_save_sites(path: Path) -> "list[tuple[str, int]]":
    """Every ``torch.save`` call in a module, as (enclosing function, line)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner, out = {}, []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                owner.setdefault(inner, node.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = (f"{func.value.id}.{func.attr}"
                 if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                 else getattr(func, "id", ""))
        if named in ("torch.save", "save"):
            out.append((owner.get(node, "<module>"), node.lineno))
    return out


def _checkpoint_destinations(path: Path) -> "list[tuple[str, int]]":
    """Every call that is handed a path ending in ``.pt``, as (callee, line)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        writes = any(
            isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div)
            and isinstance(arg.right, ast.Constant)
            and str(arg.right.value).endswith(".pt")
            for arg in node.args)
        if not writes:
            continue
        func = node.func
        named = (func.attr if isinstance(func, ast.Attribute)
                 else getattr(func, "id", "?"))
        out.append((named, node.lineno))
    return out


def test_no_checkpoint_is_ever_written_straight_onto_its_destination():
    """The helper being atomic proves nothing about what actually uses it.

    ``test_a_checkpoint_survives_a_crash_during_its_own_write`` drives
    ``save_checkpoint`` directly, so three separate reversions of real call
    sites to a bare ``torch.save`` all survived the whole suite -- including
    the ``checkpoint.pt`` site that test's own docstring names, and an extra
    ``torch.save`` inserted into the ``--save-every`` branch, which reopens the
    truncate-first window every three updates because that is what
    ``supervise.ps1`` sets.

    So this asks the modules themselves, rather than the helper: no
    ``torch.save`` outside ``save_checkpoint``, and every ``.pt`` a trainer
    writes goes through it. Cheap enough to be honest -- a real run to a
    ``final.pt`` is half a minute.
    """
    writers = {ROOT / "cr_sim" / "train" / "run.py": ["save_checkpoint"],
               ROOT / "scripts" / "clone_policy.py": []}

    for path, allowed in writers.items():
        stray = [(where, line) for where, line in _torch_save_sites(path)
                 if where not in allowed]
        assert not stray, \
            f"{path.name} calls torch.save outside {allowed or 'anything'}: " \
            f"{stray} -- torch.save truncates its destination first, and " \
            "--resume from that one file is the whole crash-resilience strategy"

        destinations = _checkpoint_destinations(path)
        assert destinations, f"{path.name} writes no checkpoint at all any more"
        wrong = [(callee, line) for callee, line in destinations
                 if callee != "save_checkpoint"]
        assert not wrong, \
            f"{path.name} hands a .pt path to something other than " \
            f"save_checkpoint: {wrong}"

    # Non-vacuous: the three destinations the crash-resilience story is about
    # are all there and all go through the helper.
    lines = (ROOT / "cr_sim" / "train" / "run.py").read_text(encoding="utf-8")
    for name in ("checkpoint.pt", "best.pt", "final.pt"):
        assert f'"{name}"' in lines, f"run.py no longer writes {name}"
