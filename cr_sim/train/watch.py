"""``python -m cr_sim.train.watch`` -- see how a run is going, while it goes.

A training run writes one JSON line per update and prints a scrolling wall of
numbers. Neither answers the question you actually have, which is whether the
thing is getting better -- that is a shape over time, and a column of figures
is the worst way to show a shape.

So this reads the metrics file and draws it, refreshing while the run
continues. It is deliberately a separate process: it must not be able to slow
the run down, crash it, or hold a lock on the file it is reading.

**The chart that matters is the lift, not the return.** The trainer's own
return is measured while the policy is exploring, averaged over a sliding
window, on whatever seeds the rollout drew -- and against a paired-seed control
it has run about eighteen points optimistic. So the lift against that control,
in units of the control's own standard deviation, is drawn first and largest,
with a zero line, because "indistinguishable from random" is the null result
this project keeps landing on and it should be impossible to miss.

Everything is plotted against *steps* rather than updates, so runs with
different batch sizes can be compared honestly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "overnight-selfplay"


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Load the metrics file, tolerating a half-written final line.

    The run appends while this reads, so the last line can be incomplete. That
    is normal rather than an error, and dropping it is the whole handling
    required.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows



def _next_evaluation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How long until this run evaluates again.

    Both numbers are inferred from the run's own history rather than read
    from its config, so this works for a run whose config was lost and for
    one whose cadence was changed by a resume.

    The cadence comes from the spacing of past evaluations; the pace from how
    long recent updates actually took, which is what a countdown should
    follow rather than an average over a run that has changed speed.
    """
    updates = [r["updates"] for r in rows if "eval_lift_sd" in r]
    if len(updates) < 2 or len(rows) < 2:
        return {"eval_every": None, "next_eval_seconds": None}
    gaps = [b - a for a, b in zip(updates, updates[1:]) if b > a]
    if not gaps:
        return {"eval_every": None, "next_eval_seconds": None}
    cadence = min(gaps)

    # Paced on the recent stretch. A run that was throttled partway through
    # would otherwise be timed by an average it no longer runs at.
    window = rows[-min(len(rows), 12):]
    span = (window[-1].get("elapsed_seconds"), window[0].get("elapsed_seconds"))
    if None in span or window[-1]["updates"] <= window[0]["updates"]:
        return {"eval_every": cadence, "next_eval_seconds": None}
    per_update = (span[0] - span[1]) / (window[-1]["updates"] - window[0]["updates"])
    if per_update <= 0:
        return {"eval_every": cadence, "next_eval_seconds": None}

    done = rows[-1]["updates"]
    remaining = cadence - (done - updates[-1]) % cadence
    return {"eval_every": cadence,
            "next_eval_seconds": max(0.0, remaining * per_update)}


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The handful of facts worth putting at the top of the page."""
    if not rows:
        return {"updates": 0}
    last = rows[-1]
    evals = [r for r in rows if "eval_lift_sd" in r]
    best = max(evals, key=lambda r: r["eval_lift_sd"]) if evals else None
    return {
        "updates": last.get("updates", 0),
        "steps": last.get("steps", 0),
        "episodes": last.get("episodes", 0),
        "steps_per_second": last.get("steps_per_second", 0.0),
        "entropy": last.get("entropy", 0.0),
        "evaluations": len(evals),
        "latest_lift": evals[-1]["eval_lift_sd"] if evals else None,
        # `rotating_probe` writes the sampled arm to `eval_lift_sd` and the
        # argmax beside it. Where both are on the row, both are reported: one
        # of them alone is not the checkpoint.
        "latest_lift_greedy": (evals[-1].get("eval_lift_sd_greedy")
                               if evals else None),
        "modes_recorded": bool(evals and "eval_lift_sd_greedy" in evals[-1]),
        "latest_win": evals[-1].get("eval_win") if evals else None,
        "control_win": evals[-1].get("control_win") if evals else None,
        "best_lift": best["eval_lift_sd"] if best else None,
        "best_at_steps": best.get("steps") if best else None,
        "total_steps": last.get("total_steps"),
        "elapsed_seconds": last.get("elapsed_seconds"),
        **_next_evaluation(rows),
        "ancestor_win": (
            [r["ancestor_win"] for r in rows if "ancestor_win" in r] or [None]
        )[-1],
        "ancestor_loss": (
            [r["ancestor_loss"] for r in rows if "ancestor_loss" in r] or [None]
        )[-1],
        "ancestor_age": (
            [r["ancestor_age"] for r in rows if "ancestor_age" in r] or [None]
        )[-1],
        # Hours left at the current rate. Wrong the moment the rate changes,
        # which is why it is labelled as an estimate rather than a countdown.
        "eta_hours": (
            (last.get("total_steps", 0) - last.get("steps", 0))
            / max(1e-9, last.get("steps_per_second", 0.0)) / 3600
            if last.get("total_steps") else None
        ),
    }




def _started_at(run: Path) -> float:
    """When a run began, as a timestamp.

    Taken from config.json, which is written once before the first update, so
    it dates the start rather than the most recent write. Falls back to the
    metrics file for runs that predate the config being written at all.
    """
    for name in ("config.json", "metrics.jsonl"):
        path = run / name
        if path.is_file():
            stat = path.stat()
            # st_ctime is creation time on Windows and metadata-change time
            # elsewhere; either dates the start closely enough to order by,
            # and st_mtime would sort by last write, which is a different
            # question.
            return min(stat.st_ctime, stat.st_mtime)
    return 0.0


def _label_for(run: Path) -> str:
    """A name that stays unique once runs nest and worktrees join in.

    ``run.name`` alone was fine while every run sat directly in ``runs/``.
    A sweep writes ``runs/<sweep>/<variant>/metrics.jsonl``, and every
    variant of every sweep is then called the same thing as some other
    sweep's -- so the page silently showed one and dropped the rest.
    """
    parts = run.resolve().parts
    if "runs" not in parts:
        return run.name
    index = len(parts) - 1 - parts[::-1].index("runs")
    label = "/".join(parts[index + 1:])
    if "worktrees" in parts:
        tag = parts[parts.index("worktrees") + 1].replace("agent-", "")[:7]
        return f"{tag}:{label}"
    return label


def _note_of(run: Path) -> str:
    """One line of prose from config.json saying what this entry is.

    ``config.json`` has always been written and never read except for its
    timestamp. Everything that is not a training run -- the search expert, a
    cloned policy, a benchmark, a head-to-head between checkpoints -- looks
    on the index like a run that produced two flat points, and what it
    actually measured lived only in whatever conversation produced it.

    A config that is valid JSON but not an object -- ``null``, a list, a bare
    string -- has no ``.get``, and a file that is not UTF-8 raises a
    ``UnicodeDecodeError`` rather than an ``OSError``. Either one used to
    propagate out of ``once()``, and the refresh loop catches only
    ``KeyboardInterrupt``, so one unreadable config killed the watcher and
    froze every served page on whatever it had last written.
    """
    raw = _read_json(run / "config.json")
    return str(raw.get("note", "")) if isinstance(raw, dict) else ""


def _plottable(value: Any) -> bool:
    """Whether a value survives the trip through JSON into a browser.

    ``json.dumps`` writes a bare ``NaN`` token and ``json.loads`` reads it
    back, so a non-finite number is invisible from Python and fatal in the
    page: ``JSON.parse`` rejects the whole document, ``poll()``'s ``r.json()``
    rejects with it, the rejection lands in its own empty ``catch``, and the
    view silently freezes on whatever it loaded with. One run's critic wrote
    ``NaN`` for explained variance and froze the served page for a day.

    A row missing a key is already how this file says "no reading here", so
    dropping the point needs no handling anywhere downstream.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, int)


def _json_safe(value: Any) -> Any:
    """The same structure with every non-finite number replaced by null.

    `json.dumps` writes bare `NaN` and `Infinity` tokens, which are not JSON
    and which `JSON.parse` refuses. Null is a value the page already renders
    as `--` everywhere a number can be missing.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _series_of(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def pair(key):
        return [(r.get("steps", 0), r[key])
                for r in rows if key in r and _plottable(r[key])]

    return {
        "steps": [r.get("steps", 0) for r in rows],
        "lift": pair("eval_lift_sd"),
        # The argmax arm, where the probe recorded one. `rotating_probe`
        # writes both on a single row and `eval_lift_sd` is the *sampled*
        # one, so plotting that alone draws the distribution's trajectory and
        # silently omits the greedy one -- the reading a fine-tune was
        # written off on when only the other half was looked at.
        "lift_greedy": pair("eval_lift_sd_greedy"),
        "win": pair("eval_win"),
        "control": pair("control_win"),
        "entropy": pair("entropy"),
        "value_loss": pair("value_loss"),
        "rollout_win": pair("win_rate"),
        # Added once the critic turned out to be the bottleneck. Value loss
        # cannot be read without knowing the spread of what it is fitting;
        # explained variance is that comparison done properly, and is what
        # decides whether PPO has a usable signal at all.
        "explained_variance": pair("explained_variance"),
        "ret_std": pair("ret_std"),
        "noop": pair("noop_fraction"),
        # The ladder: how the policy fares against the oldest version of
        # itself still in the pool. Only self-play runs record it.
        "ancestor_win": pair("ancestor_win"),
        "ancestor_loss": pair("ancestor_loss"),
        # The eval rows joined back together. Lift and control are separate
        # pair-lists above and cannot be zipped downstream, so a reader who
        # wants a lift together with the control rate that scales it -- which
        # is the only honest way to show one -- has nowhere to get it.
        # One entry per measurement rather than per row, so a row holding
        # both a sampled and a greedy arm appears as the two readings it is.
        "evals": [{"steps": r["steps"], "lift": r["lift"], "win": r.get("win"),
                   "control": r.get("control"), "arm": r.get("arm"),
                   "mode": r.get("mode"), "opponent": r.get("opponent")}
                  for r in _readings_of(rows)],
    }


# ------------------------------------------------------------------ all time
#
# A cross-run view has exactly one way to go badly wrong on this project, and
# it has already gone wrong once: a lift is a number against an opponent, and
# the opponents differ. The in-run probe used to face an agent that never
# plays a card while the paired verdicts faced a random one; both wrote to
# ``eval_lift_sd``, and the two were compared. The control wins 92% of the
# first kind of match and 26% of the second, so those are not the same scale
# and a column that sorts both is a confident lie.
#
# ``eval_opponent`` settles it, and is read wherever it is written -- which
# is now mandatory, since ``selfplay.check_lift_is_named`` refuses a row that
# carries a lift without it, but is still absent from most of the rows
# already on disk. Where it is missing, the only cross-run ordering drawn
# here is one where a single job measured every arm against one opponent on
# one seed set, over the same number of battles each, and said so in its own
# note. Everything else is grouped by what the reading records about its
# opponent -- the name where there is one, otherwise the control's own win
# rate, a property of the measurement rather than a score -- and never ranked
# across groups. Nothing on this page is prose about the data: every count it
# states is computed from the same rows it draws.

#: The two evaluation setups whose control has actually been measured here.
#: A control that never plays a card wins 92.5% of its matches; a random one
#: wins 26%. A reading sitting on neither is on a scale nobody has
#: identified, and says so rather than being rounded to whichever is closer.
_ANCHORS = ((0.925, "idle"), (0.26, "random"))
_ANCHOR_TOLERANCE = 0.005

#: What one in-run evaluation costs, for the estimate in the battle ledger
#: that is deliberately not added into the total.
_PROBE_EPISODES = 40

#: Job rows whose ``episodes`` really are battles that were played. Jobs are
#: excluded from every counter by default, because they reuse standard key
#: names for unrelated quantities -- a batch size in ``steps``, a speedup
#: ratio in ``steps_per_second``, a count of spreadsheet comparisons in
#: ``episodes``. This list is rendered on the page beside the number it
#: contributes, so the exception is as visible as the rule.
_BATTLE_ROWS = ("sim vs arithmetic winner",)

#: Two readings this close are the same reading, for the exhibit that shows
#: one number meaning two different things.
_COINCIDENCE = 0.01

#: Sentence boundaries in a job note. Notes are prose with hard line breaks
#: and em-dash asides, so the split is deliberately coarse: it only has to be
#: fine enough that a denial and a claim do not land in the same fragment.
_NOTE_SENTENCE = re.compile(r"(?<=[.!?:])\s+|\n+")

#: A claim that is being denied rather than made. Matched case-insensitively,
#: so a note shouting NOT and one muttering "not" are both refused.
_NOTE_NEGATION = re.compile(
    r"\b(?:not|never|no|none|nothing|cannot|isn't|aren't|weren't|wasn't|"
    r"didn't|don't|doesn't|without)\b", re.IGNORECASE)


def _anchor_for(control: Any) -> "str | None":
    """Which measured control this reading's control rate matches, if any."""
    if not isinstance(control, (int, float)) or isinstance(control, bool):
        return None
    for value, label in _ANCHORS:
        if abs(control - value) <= _ANCHOR_TOLERANCE:
            return label
    return None


def _scale_of(control: Any, stated: bool = False,
              opponent: Any = None) -> dict[str, Any]:
    """The provenance chip that travels with a lift, in the same object.

    Never a bare number. A screenshot of a figure with no scale beside it is
    how an idle-scale reading gets quoted as a random-scale one.

    ``opponent`` is the name the row or verdict record wrote down for itself.
    Where it exists it wins outright: ``selfplay.check_lift_is_named`` refuses
    a metrics row that carries ``eval_lift_sd`` without ``eval_opponent``, so
    every new row names its opponent, and inferring a different one from the
    control's win rate would put a confidently wrong name on a number that
    names itself. Where the two disagree the disagreement is carried rather
    than resolved: a third opponent whose control happens to win at the idle
    rate is neither idle nor a rounding error.
    """
    anchor = _anchor_for(control)
    named = str(opponent).strip() if opponent not in (None, "") else ""
    return {"control": control,
            "anchor": anchor,
            "named": named or None,
            "opponent": named or anchor,
            "source": "recorded" if named else ("inferred" if anchor else None),
            "stated": bool(stated and (named or anchor)),
            "conflict": bool(named and anchor and named != anchor)}


def _same_scale(one: dict[str, Any], two: dict[str, Any]) -> "bool | None":
    """Whether two readings were read against the same thing.

    ``None`` where there is not enough evidence to say, which is a third
    answer and not a soft yes: two readings that both record nothing are not
    thereby comparable. Names beat control rates, because a name is recorded
    and a rate is inferred.
    """
    if not one or not two:
        return None
    if one.get("named") and two.get("named"):
        return one["named"] == two["named"]
    a, b = one.get("control"), two.get("control")
    if _plottable(a) and _plottable(b):
        return abs(a - b) <= _ANCHOR_TOLERANCE
    if one.get("named") or two.get("named"):
        # One names an opponent and the other has only a rate, or nothing.
        other = two if one.get("named") else one
        anchor = other.get("anchor")
        if anchor:
            return (one.get("named") or two.get("named")) == anchor
    return None


def _mode_of(arm: Any) -> "tuple[str, str | None]":
    """Split a free-text arm label into the weights it names and how they played.

    Greedy and sampled play are different numbers for the same weights and
    must never be collapsed: a fine-tuning run was written off as worthless
    because its greedy arm was flat -- the argmax had not moved -- while
    sampled had gone from +0.718 to +1.239 with the intervals cleanly apart.

    ``arm`` is free text with no validator, and ``register_job.py`` is the one
    writer of a metrics file that never calls ``check_lift_is_named``. So
    anything unrecognised comes back as an unknown mode. It must never
    default to greedy, which is the assumption that hid that run.
    """
    text = str(arm or "").strip()
    head, sep, tail = text.rpartition(",")
    mode = tail.strip().lower()
    if sep and mode in ("greedy", "sampled"):
        return head.strip(), mode
    return text, None


def _readings_of(rows: Sequence[dict[str, Any]],
                 default_weight: str = "") -> list[dict[str, Any]]:
    """Every lift a run's rows actually contain, one object per measurement.

    A row is not a reading. ``evaluate.rotating_probe`` writes two arms on one
    row: ``eval_lift_sd`` is the *sampled* one -- its own docstring says so,
    because that is what a policy still being trained actually plays -- and
    ``eval_lift_sd_greedy`` is the argmax beside it. Selecting rows on
    ``"eval_lift_sd" in row`` reads the sampled number, calls it the run's
    lift, shows no mode, and drops the greedy arm off the page entirely. That
    is the same collapse in the other direction from the one that wrote off a
    working fine-tune: greedy and sampled are two numbers and both were
    measured.

    ``eval_episodes`` is the recorded size of the evaluation. ``episodes`` on
    a training row is a cumulative counter of a different quantity, so only
    the former is read as a battle count here.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if "eval_lift_sd" not in row or not _plottable(row["eval_lift_sd"]):
            continue
        weight, mode = _mode_of(row.get("arm"))
        paired = _plottable(row.get("eval_lift_sd_greedy"))
        if paired and mode is None:
            # The writer's contract, not a guess: this row holds both arms.
            mode = "sampled"
        base = {
            "lift": row["eval_lift_sd"], "win": row.get("eval_win"),
            "control": row.get("control_win"),
            "arm": row.get("arm"), "mode": mode,
            "weight": weight or default_weight,
            "opponent": row.get("eval_opponent"),
            "steps": row.get("steps", 0),
            "episodes": row.get("episodes"),
            "eval_episodes": (row.get("eval_episodes")
                              if _plottable(row.get("eval_episodes")) else None),
        }
        out.append(base)
        if paired:
            out.append(dict(base, lift=row["eval_lift_sd_greedy"],
                            win=row.get("eval_win_greedy"), mode="greedy",
                            arm=((base["weight"] + ", greedy").strip(", ")
                                 or "greedy")))
    return out


def _segment_total(rows: Sequence[dict[str, Any]], key: str) -> float:
    """A cumulative counter totalled across resumes.

    These counters restart when a run is resumed into a fresh process, so the
    last row is not the total and neither is the largest value. Every fall
    starts a new segment, and the segments add.

    Two defensible rules disagree here by a few hundred episodes, which is
    why this one is printed on the page beside every number it produces
    rather than left for a reader to assume.
    """
    total = 0.0
    seen = None
    segment = 0.0
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(value):
            continue
        if seen is not None and value < seen:
            total += segment
            segment = 0.0
        segment = max(segment, value)
        seen = value
    return total + segment


def _repeats(previous: dict[str, Any], row: dict[str, Any]) -> bool:
    """Whether ``row`` is the same update written twice rather than a resume.

    Both look identical on the update number, and telling them apart is worth
    getting right: keying a dict on ``updates`` across a whole file collapses
    a resume's replayed numbers onto their pre-resume namesakes and deletes
    the earlier segment outright -- 816 real training battles and eleven
    minutes of compute, from the two runs on this machine that resumed.

    A repeat writes the same counters again. A resume restarts them, so a
    counter that goes backwards marks a new segment and both rows are kept.
    """
    updates = row.get("updates")
    if updates is None or previous.get("updates") != updates:
        return False
    for key in ("steps", "episodes", "elapsed_seconds", "total_steps"):
        before, after = previous.get(key), row.get(key)
        if _plottable(before) and _plottable(after) and after < before:
            return False
    return True


def _verdict_shape(raw: Any) -> str:
    """What kind of verdict object this is, from the fields it actually has.

    Four have been written and three of them disagree about where the lift
    lives. More are being written right now, so a shape that is not
    recognised says so on the page instead of being guessed at. Guessing is
    what makes the flat mirror in ``runs/cloned/verdict.json`` dangerous: it
    is a byte-identical copy of the greedy sub-object, so reading it hands
    you greedy without telling you, and hides a sampled reading 2.3x lower.
    """
    if isinstance(raw, list):
        if raw and all(isinstance(r, dict) and "lift" in r and "mode" in r
                       for r in raw):
            return "arms"
        return "unrecognised"
    if isinstance(raw, dict):
        both = [k for k in ("greedy", "sampled")
                if isinstance(raw.get(k), dict) and "lift" in raw[k]]
        if len(both) == 2:
            return "paired"
        if "lift" in raw:
            return "flat"
    return "unrecognised"


def _states_shared_conditions(note: str) -> bool:
    """Whether a job's note claims its arms are actually comparable.

    Read as a flag and never mined for numbers. The one job that qualifies
    says its arms met "the SAME random opponent on the SAME 150 paired
    seeds" -- the author's own emphasis, twice. The sweep that sits at the
    same control rate and also labels its arms makes no such claim, and its
    note carries a caveat saying only the within-sweep ordering is sound.
    Equal control rates do not license ranking; a stated method does.

    A note that stops matching costs the page its ladder and produces the
    empty state, which is the safe direction to fail in.

    Counting the token is not enough on its own. "Arms did NOT face the SAME
    opponent and were NOT played on the SAME seeds" says SAME twice, carries
    no caveat, and means the exact opposite -- and a page that read it as a
    licence printed a record, a sorted ladder and a "(stated)" chip on top of
    a note denying every one of them. So the claim has to be affirmative, and
    it has to name both things it is claiming: one opponent and one seed set.
    """
    if "CAVEAT" in note:
        return False
    claims = [part for part in _NOTE_SENTENCE.split(note) if "SAME" in part]
    if len(claims) < 1 or sum(part.count("SAME") for part in claims) < 2:
        return False
    if any(_NOTE_NEGATION.search(part) for part in claims):
        return False
    joined = " ".join(claims).lower()
    return "opponent" in joined and "seed" in joined


def _ci_of(raw: dict[str, Any]) -> "list[float] | None":
    """An interval, only where both ends are fields of the object.

    No lift row anywhere on disk carries one. Intervals exist in the verdict
    files and in note prose, and prose is never parsed for a number here --
    the note is printed whole instead, which loses nothing.
    """
    low, high = raw.get("ci_low"), raw.get("ci_high")
    if _plottable(low) and _plottable(high):
        return [low, high]
    return None


def _verdict_of(raw: Any) -> dict[str, Any]:
    """One verdict file, normalised, or a note saying it could not be read."""
    shape = _verdict_shape(raw)
    out: dict[str, Any] = {"shape": shape}
    if shape == "flat":
        out.update(lift=raw.get("lift"), ci=_ci_of(raw), win=raw.get("win"),
                   loss=raw.get("loss"), draw=raw.get("draw"),
                   episodes=raw.get("episodes"),
                   control_win=raw.get("control_win"),
                   control_draw=raw.get("control_draw"),
                   # Which weights this verdict measured. Dropping it is how
                   # one run's peak got paired with another checkpoint's
                   # replay: runs/poc-vs-random's verdict is final.pt, while
                   # the peak it was printed beside came from best.pt, and
                   # the file's own note says best.pt replays at -0.033.
                   checkpoint=str(raw.get("checkpoint", "") or ""),
                   opponent=raw.get("eval_opponent"),
                   note=str(raw.get("note", "") or ""))
    elif shape == "paired":
        for mode in ("greedy", "sampled"):
            sub = raw[mode]
            out[mode] = {"lift": sub.get("lift"), "ci": _ci_of(sub),
                         "win": sub.get("win"), "loss": sub.get("loss"),
                         "opponent": sub.get("eval_opponent")
                         or raw.get("eval_opponent")}
        out.update(episodes=raw.get("episodes"),
                   checkpoint=str(raw.get("checkpoint", "") or ""),
                   note=str(raw.get("note", "") or ""))
    elif shape == "arms":
        out["records"] = [{
            # "_diag" names a scratch directory, not an experiment. It is the
            # unmodified clone every other arm here is a variation on.
            "name": ("baseline clone" if r.get("name") == "_diag"
                     else str(r.get("name", ""))),
            "checkpoint": str(r.get("checkpoint", "") or ""),
            "mode": r.get("mode"), "lift": r.get("lift"), "ci": _ci_of(r),
            "win": r.get("win"), "loss": r.get("loss"),
            "episodes": r.get("episodes"),
            # The only field on this machine that puts a lift and the
            # opponent that produced it on the same object.
            "opponent": r.get("eval_opponent"),
        } for r in raw]
    return out


def _weight_key(label: Any) -> str:
    """The part of an arm name identifying the weights rather than the run.

    ``w0.5 hard`` in a sweep's metrics and ``w0.5`` in its verdict are the
    same checkpoint measured twice.
    """
    parts = str(label or "").split()
    return parts[0].lower() if parts else ""


#: The order the play modes are laid out in. Fixed, so the sections do not
#: reshuffle when one of them happens to hold the largest number.
_MODE_ORDER = ("greedy", "sampled", None)


def _block_of(entries: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """The one measurement block whose arms may be ranked against each other.

    Every condition has to hold at once: the arms come from a single job, the
    rows carry an ``arm`` label, every row shares one control rate, that rate
    matches a control that has actually been measured or the rows name their
    opponent and agree on it, every arm records the same number of battles,
    and the job's own note states the conditions. Nothing else on the page is
    sorted by value.

    The shared seed count is a gate rather than a decoration. Without it a job
    whose arms ran 150 and 90 battles still built a block, and the record card
    printed "0 paired seeds" directly above the words "one opponent, one seed
    set" -- fabricating the provenance claim this page exists to protect.

    Within the block the arms are partitioned by play mode and ranked only
    inside a partition. Greedy and sampled are two different numbers for the
    same weights, so one ordering over both is a ranking of how each
    checkpoint happened to be played.
    """
    qualifying = []
    for entry in entries:
        evals = entry["evals"]
        if not entry["job"] or len(evals) < 2:
            continue
        if not all(str(r.get("arm") or "").strip() for r in evals):
            continue
        controls = {r.get("control") for r in evals}
        if len(controls) != 1:
            continue
        control = controls.pop()
        named = {str(r.get("opponent")).strip() for r in evals
                 if r.get("opponent")}
        if len(named) > 1:
            continue
        opponent = named.pop() if named else None
        if opponent is None and _anchor_for(control) is None:
            continue
        seeds = {r.get("eval_episodes") if r.get("eval_episodes") is not None
                 else r.get("episodes") for r in evals}
        if len(seeds) != 1:
            continue
        count = seeds.pop()
        if not _plottable(count) or count <= 0:
            continue
        if not _states_shared_conditions(entry["note"]):
            continue
        qualifying.append((entry, control, opponent, int(count)))
    if not qualifying:
        return None
    entry, control, opponent, seeds = max(
        qualifying, key=lambda q: (len(q[0]["evals"]), q[0]["name"]))
    scale = _scale_of(control, stated=True, opponent=opponent)
    arms = []
    for row in entry["evals"]:
        arms.append({"arm": str(row.get("arm")), "weight": row["weight"],
                     "mode": row["mode"], "lift": row["lift"],
                     "win": row.get("win"), "episodes": seeds,
                     "scale": scale})
    groups = []
    for mode in _MODE_ORDER:
        found = sorted([a for a in arms if a["mode"] == mode],
                       key=lambda a: a["lift"], reverse=True)
        if found:
            groups.append({"mode": mode, "arms": found})
    # Flattened in section order, never in value order: any consumer that
    # takes arms[0] must get the top of a partition, not a cross-mode max.
    ordered = [a for g in groups for a in g["arms"]]
    return {
        "job": entry["name"],
        "note": entry["note"],
        "control": control,
        "scale": scale,
        "seeds": seeds,
        "arms": ordered,
        "groups": groups,
        # The control is drawn as the zero row, and is one of the arms the
        # job ran.
        "count": len(ordered) + 1,
    }


def _pairs_of(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arms that are the same weights played two ways, greedy beside sampled."""
    by_weight: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for arm in arms:
        if arm.get("mode") is None:
            continue
        key = arm["weight"]
        if key not in by_weight:
            by_weight[key] = {}
            order.append(key)
        by_weight[key][arm["mode"]] = arm
    pairs = []
    for key in order:
        found = by_weight[key]
        if "greedy" in found and "sampled" in found:
            pairs.append({"weight": key,
                          "greedy": found["greedy"], "sampled": found["sampled"],
                          "gap": found["greedy"]["lift"] - found["sampled"]["lift"]})
    return pairs


def _groups_of(entries: list[dict[str, Any]],
               verdicts: dict[str, Any]) -> list[dict[str, Any]]:
    """Every lift on disk, gathered by the control rate it was read against.

    Grouped, never ranked across groups, and ordered by the control rate
    itself -- descending, because that is a property of the measurement and
    an ordering on it cannot be misread as a ranking of the policies.
    """
    buckets: dict[Any, dict[str, Any]] = {}
    for entry in entries:
        for row in entry["evals"]:
            control = row.get("control")
            named = str(row.get("opponent") or "").strip() or None
            # A reading with no control rate at all is not on the same scale
            # as every other reading with no control rate. Where it names its
            # opponent that name is the bucket; where it names nothing there
            # is no scale, and the card it lands in is not ranked.
            if isinstance(control, (int, float)) and not isinstance(control, bool):
                # A row that names the opponent its own control rate points
                # at is the same measurement as one that only has the rate,
                # so they share a card. A row naming a third opponent at that
                # rate is not, and gets its own.
                anchor = _anchor_for(control)
                key = (0, round(control, 6),
                       "" if (not named or named == anchor) else named)
            elif named:
                key = (1, named, "")
            else:
                key = (2, "", "")
            bucket = buckets.setdefault(key, {
                "control": control if key[0] == 0 else None,
                "named": named, "rankable": key[0] != 2, "runs": {}})
            # One confirmation is enough to name the card: a rate agreeing
            # with a name is stronger evidence than the rate alone.
            bucket["named"] = bucket["named"] or named
            run = bucket["runs"].setdefault(entry["name"], {
                "name": entry["name"], "readings": [], "job": entry["job"],
                "ranking": False, "arms": [],
            })
            run["readings"].append(row)
    groups = []
    # Control-rate cards first, descending by rate; then cards held together
    # only by a recorded opponent name; then, last, the readings that recorded
    # neither. The rate is a property of the measurement, so ordering on it
    # cannot be misread as a ranking of the policies.
    def order(key):
        return (key[0], -key[1] if key[0] == 0 else 0.0,
                "" if key[0] == 0 else str(key[1]), key[2])

    for key in sorted(buckets, key=order):
        bucket = buckets[key]
        runs = []
        for name in sorted(bucket["runs"]):
            run = bucket["runs"][name]
            rows = run["readings"]
            best = max(rows, key=lambda r: r["lift"])
            # A job whose every row is a labelled arm is a table of separate
            # measurements, not a trajectory. It has no "current" value and
            # its last row is simply its worst arm.
            ranking = run["job"] and all(str(r.get("arm") or "").strip()
                                         for r in rows)
            entry = {
                "name": name, "job": run["job"], "ranking": ranking,
                "evals": len(rows),
                "best": best["lift"],
                "best_mode": best.get("mode"),
                "best_at_steps": best.get("steps", 0),
                "best_win": best.get("win"),
                # Whether "best of N" is even one policy's number. Where the
                # readings were played two different ways the largest of them
                # is the largest of two scales, so it is reported per mode.
                "modes": sorted({str(r.get("mode")) for r in rows}),
            }
            if ranking:
                entry["arms"] = [
                    {"arm": str(a.get("arm")), "lift": a["lift"],
                     "win": a.get("win"), "mode": a.get("mode")}
                    for mode in _MODE_ORDER
                    for a in sorted([r for r in rows if r.get("mode") == mode],
                                    key=lambda r: r["lift"], reverse=True)]
            # A verdict that re-measured the same policy and disagreed about
            # the control's own win rate makes this whole bucketing
            # provisional, so it is printed inside the card it undermines.
            verdict = verdicts.get(name) or {}
            other = verdict.get("control_win")
            if (verdict.get("shape") == "flat" and _plottable(other)
                    and _plottable(bucket["control"])
                    and abs(other - bucket["control"]) > _ANCHOR_TOLERANCE):
                entry["disputed"] = {
                    "in_run": bucket["control"], "verdict": other,
                    "verdict_draw": verdict.get("control_draw"),
                    "episodes": verdict.get("episodes"),
                }
            runs.append(entry)
        # Sorting exists only inside a card, and only where the card is one
        # scale. The card holding everything that recorded neither a control
        # rate nor an opponent is not one scale, so it is listed by name.
        if bucket["rankable"]:
            runs.sort(key=lambda r: r["best"], reverse=True)
        else:
            runs.sort(key=lambda r: r["name"])
        groups.append({
            "control": bucket["control"],
            "scale": _scale_of(bucket["control"], opponent=bucket["named"]),
            "rankable": bucket["rankable"],
            "rows": sum(r["evals"] for r in runs),
            "runs": runs,
        })
    return groups


def _exhibits_of(entries: list[dict[str, Any]], block: "dict[str, Any] | None",
                 verdicts: dict[str, Any]) -> dict[str, Any]:
    """Three short demonstrations of why the rest of the page is shaped as it is."""
    out: dict[str, Any] = {}

    # (a) The same figure, read against two different opponents. Found rather
    # than asserted: the closest pair of near-identical lifts that sit in
    # different scale groups.
    if block:
        best = None
        for arm in block["arms"]:
            for entry in entries:
                if entry["name"] == block["job"]:
                    continue
                for row in entry["evals"]:
                    scale = _scale_of(row.get("control"),
                                      opponent=row.get("opponent"))
                    if _same_scale(scale, block["scale"]) is not False:
                        continue
                    gap = abs(row["lift"] - arm["lift"])
                    if gap <= _COINCIDENCE and (best is None or gap < best[0]):
                        best = (gap, arm, entry, row, scale)
        if best:
            _, arm, entry, row, scale = best
            out["coincidence"] = {
                "ladder": {"arm": arm["arm"], "lift": arm["lift"],
                           "scale": arm["scale"], "job": block["job"]},
                "in_run": {"run": entry["name"], "lift": row["lift"],
                           "scale": scale, "steps": row.get("steps", 0)},
                "gap": best[0],
            }

    # (b) A peak chosen from many short readings, replayed at length.
    #
    # Both halves have to be the same thing measured twice, or the panel
    # attributes to selection whatever else changed. Two things else did
    # change on this machine: runs/poc-vs-random's verdict measures final.pt
    # while the peak beside it came from best.pt, and the verdict's control
    # wins 4% of its own matches where the in-run readings' control won 30%.
    # So the checkpoint and the scale are both carried, both compared, and
    # a mismatch is stated rather than folded into the number.
    chosen = None
    for entry in entries:
        verdict = verdicts.get(entry["name"]) or {}
        if verdict.get("shape") != "flat" or not verdict.get("note"):
            continue
        if not entry["evals"]:
            continue
        peak = max(entry["evals"], key=lambda r: r["lift"])
        if not _plottable(verdict.get("lift")) or peak["lift"] <= verdict["lift"]:
            continue
        if chosen is None or peak["lift"] > chosen["best_in_run"]:
            in_scale = _scale_of(peak.get("control"),
                                 opponent=peak.get("opponent"))
            verdict_scale = _scale_of(verdict.get("control_win"),
                                      opponent=verdict.get("opponent"))
            chosen = {"run": entry["name"], "best_in_run": peak["lift"],
                      "best_scale": in_scale, "best_mode": peak.get("mode"),
                      "readings": len(entry["evals"]),
                      "verdict_lift": verdict["lift"],
                      "verdict_scale": verdict_scale,
                      "verdict_ci": verdict.get("ci"),
                      "verdict_episodes": verdict.get("episodes"),
                      "verdict_checkpoint": verdict.get("checkpoint") or None,
                      "same_scale": _same_scale(in_scale, verdict_scale),
                      # No metrics row on this machine records which
                      # checkpoint produced it, so where the verdict names
                      # one there is no way to confirm it is the same weights
                      # -- and saying so is the whole difference between this
                      # panel and a wrong subtraction.
                      "same_checkpoint": None,
                      "note": verdict["note"]}
    if chosen:
        out["selection"] = chosen

    # (c) How much a reading moves when it is simply run again. Greedy play on
    # a fixed seed set is deterministic, so the same weights read twice give
    # the same digits; sampled play on the same seeds does not.
    #
    # Joined on the checkpoint path, never on the arm name: two sweeps here
    # both call an arm "w0.5". And only on a recorded mode -- an unknown one
    # bucketed under the string "None" and then described as greedy is three
    # collapses in one sentence, since only greedy play is the deterministic
    # thing this exhibit is about.
    seen: dict[str, list[dict[str, Any]]] = {}
    for source, verdict in sorted(verdicts.items()):
        found = []
        if verdict.get("shape") == "paired":
            for mode in ("greedy", "sampled"):
                found.append({"weight": source, "mode": mode,
                              "checkpoint": verdict.get("checkpoint") or source,
                              "opponent": verdict[mode].get("opponent"),
                              "lift": verdict[mode]["lift"]})
        elif verdict.get("shape") == "arms":
            for record in verdict.get("records", []):
                found.append({"weight": _weight_key(record["name"]),
                              "mode": record.get("mode"),
                              "checkpoint": record.get("checkpoint") or "",
                              "opponent": record.get("opponent"),
                              "lift": record.get("lift")})
        for item in found:
            if not _plottable(item["lift"]):
                continue
            if not item["checkpoint"]:
                continue
            seen.setdefault(str(item["mode"]), []).append(dict(item, source=source))
    identical, spread = None, None
    # The two recorded modes and nothing else. A mode of None bucketed under
    # the string "None" and then described as greedy is the assumption that
    # hid a working fine-tune: only greedy play on fixed seeds is the
    # deterministic thing this exhibit is about.
    for mode in ("greedy", "sampled"):
        items = seen.get(mode, [])
        by_value: dict[tuple, list[dict[str, Any]]] = {}
        for item in items:
            by_value.setdefault(
                (item["checkpoint"], str(item["opponent"]), item["lift"]),
                []).append(item)
        for (checkpoint, opponent, value), group in by_value.items():
            sources = {g["source"] for g in group}
            if len(sources) > 1 and identical is None:
                identical = {"mode": mode, "value": value,
                             "digits": repr(value), "checkpoint": checkpoint,
                             "opponent": (None if opponent == "None"
                                          else opponent),
                             "deterministic": mode == "greedy",
                             "sources": sorted(sources)}
        if identical and mode != identical["mode"]:
            values = sorted({i["lift"] for i in items
                             if i["checkpoint"] == identical["checkpoint"]
                             and i["source"] in identical["sources"]})
            if len(values) > 1:
                spread = {"mode": mode, "values": values,
                          "range": max(values) - min(values)}
    if identical:
        out["resolution"] = {"identical": identical, "spread": spread}
        # The two leaders of one section of the ladder. Taking arms[0] and
        # arms[1] off the flattened list would compare a greedy reading with
        # a sampled one and call the difference a tower level.
        section = next((g for g in (block or {}).get("groups", [])
                        if g["mode"] == identical["mode"] and len(g["arms"]) > 1),
                       None)
        if section:
            top, runner = section["arms"][0], section["arms"][1]
            out["resolution"]["top"] = {
                "mode": section["mode"],
                "arm": top["arm"], "lift": top["lift"],
                "runner_arm": runner["arm"], "runner_lift": runner["lift"],
                "gap": top["lift"] - runner["lift"],
            }
    return out


def _ever_of(entries: list[dict[str, Any]], verdicts: dict[str, Any],
             soak: Any, readings: list[dict[str, Any]],
             lift_files: int) -> dict[str, Any]:
    """The all-time counters, each carrying the rule that produced it."""
    models = [e for e in entries if not e["job"]]
    jobs = [e for e in entries if e["job"]]
    # A model directory that recorded no gradient steps did not train. Two of
    # them exist -- the behavioural clone and the one-ply search -- and their
    # `episodes` are the battles of the evaluation their own verdict file
    # already counts, so adding them to the training line counts 190 battles
    # twice. Their configs say as much: "No training, no gradients".
    trained = [e for e in models if _segment_total(e["rows"], "steps") > 0]
    untrained = [e for e in models if e not in trained]

    def total(group, key):
        return sum(_segment_total(e["rows"], key) for e in group)

    def per_row(group, key):
        """A plain sum, for rows where the key is not a running counter.

        A job writes one independent quantity per row -- 150 battles on each
        of seven arms is 1,050 battles, not the 150 a cumulative rule reads
        off the largest row.
        """
        return sum(row[key] for e in group for row in e["rows"]
                   if _plottable(row.get(key)))

    # Counted over models only, because the figure it annotates is a sum over
    # models only. A job that logged elapsed time would otherwise raise the
    # denominator without moving the total it qualifies.
    reporting = sum(1 for e in models
                    if _segment_total(e["rows"], "elapsed_seconds") > 0)

    counted = [
        {"what": "training episodes",
         "n": int(round(total(trained, "episodes"))),
         "rule": "every battle a training run played, resumes added up; jobs and the runs that recorded no gradient steps left out, because those are evaluations their verdict files already count"},
    ]
    if isinstance(soak, dict) and _plottable(soak.get("matches")):
        counted.append({
            "what": "engine soak matches", "n": int(soak["matches"]),
            "rule": "runs/soak-spells, which has a summary but no metrics file, so the run itself is invisible to this page"})
    verdict_battles = 0
    for verdict in verdicts.values():
        if verdict.get("shape") == "flat" and _plottable(verdict.get("episodes")):
            verdict_battles += int(verdict["episodes"])
        elif verdict.get("shape") == "paired" and _plottable(verdict.get("episodes")):
            verdict_battles += 2 * int(verdict["episodes"])
        elif verdict.get("shape") == "arms":
            verdict_battles += sum(int(r["episodes"]) for r in verdict["records"]
                                   if _plottable(r.get("episodes")))
    if verdict_battles:
        counted.append({"what": "paired verdict battles", "n": verdict_battles,
                        "rule": "every arm of every verdict file, greedy and sampled counted separately"})
    duels = []
    excluded_rows = []
    unnamed: dict[str, int] = {}
    for entry in jobs:
        for row in entry["rows"]:
            what = str(row.get("what", "") or "")
            episodes = row.get("episodes")
            if not _plottable(episodes) or not episodes:
                continue
            if what in _BATTLE_ROWS:
                duels.append({"job": entry["name"], "what": what,
                              "n": int(episodes)})
            elif what:
                excluded_rows.append({"job": entry["name"], "what": what,
                                      "n": int(episodes)})
            else:
                # Rows with no `what` used to be invisible here, which hid
                # the two largest real-battle contributors -- including the
                # job the record itself is drawn from.
                unnamed[entry["name"]] = unnamed.get(entry["name"], 0) + int(episodes)
    for duel in duels:
        counted.append({"what": duel["what"], "n": duel["n"],
                        "rule": "a job row that records battles actually played, named one by one rather than counted in by kind"})
    excluded_rows += [{"job": name, "what": "rows with no label of their own",
                       "n": n} for name, n in sorted(unnamed.items())]
    excluded_rows.sort(key=lambda r: r["n"], reverse=True)
    duel_total = sum(d["n"] for d in duels)

    # The evaluation battles behind the in-run readings. Estimated at the
    # probe default only where the reading did not record its own size:
    # `evaluate.rotating_probe` writes `eval_episodes`, and 14 rows on this
    # machine record 150 while the rule text used to claim every one of them
    # was the 40-episode default.
    estimated, recorded_size = 0, 0
    for reading in readings:
        size = reading.get("eval_episodes")
        if _plottable(size) and size > 0:
            estimated += int(size)
            recorded_size += 1
        else:
            estimated += _PROBE_EPISODES
    estimated += lift_files * _PROBE_EPISODES

    return {
        "steps": {"models": int(round(total(models, "steps"))),
                  "jobs": int(round(total(jobs, "steps")))},
        "episodes": {"models": int(round(total(trained, "episodes"))),
                     "untrained": int(round(total(untrained, "episodes"))),
                     "jobs": int(per_row(jobs, "episodes"))},
        "updates": {"models": int(round(total(models, "updates"))),
                    "jobs": int(round(total(jobs, "updates")))},
        "seconds": int(round(total(models, "elapsed_seconds"))),
        "reporting_elapsed": reporting,
        "runs": len(entries),
        "models": len(models),
        "trained": len(trained),
        "untrained": sorted(e["name"] for e in untrained),
        "jobs": len(jobs),
        "lift_rows": len(readings),
        "battles": {
            "counted": counted,
            "total": sum(c["n"] for c in counted),
            # Not added in. It is part estimate at the evaluation default,
            # and an estimate does not belong in a total that says "exactly".
            "estimated": {
                "what": "in-run probe battles",
                "n": estimated,
                "recorded": recorded_size,
                "rule": (str(recorded_size) + " of " + str(len(readings))
                         + " readings record their own eval_episodes and are counted at it; the rest are estimated at the "
                         + str(_PROBE_EPISODES) + "-episode default, plus one control run per run that evaluated"),
            },
            "excluded": {
                "what": "job episodes",
                "n": int(per_row(jobs, "episodes")) - duel_total,
                "items": excluded_rows[:4],
                "rule": "jobs reuse the episode key for things that are not battles, and each row is its own quantity rather than a running counter, so the rows add; the duels above are counted in by name and are not part of this figure",
            },
        },
        "soak": soak if isinstance(soak, dict) else None,
    }


def _record_of(block: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """The best comparable reading in each way of playing, never one of them.

    Always a pair, and never a maximum across both. ``clone_policy.py`` writes
    ``max(greedy, sampled)`` and discards the loser, so the single number in a
    metrics row is whichever flattered the checkpoint -- and a record slot
    that took the largest arm of a block regardless of mode performed exactly
    that operation on the page that condemns it, crowning one checkpoint's
    sampled reading over another's greedy one.

    So there is one record per play mode. Each is the best arm within its own
    section, shown beside the same weights played the other way where that was
    measured, at the same size, and neither can be folded away.
    """
    if not block or not block["groups"]:
        return None
    modes = []
    for group in block["groups"]:
        top = group["arms"][0]
        twin = next((a for a in block["arms"]
                     if a["weight"] == top["weight"] and a["mode"] != top["mode"]
                     and a["mode"] is not None and top["mode"] is not None),
                    None)
        modes.append({"mode": group["mode"], "top": top, "twin": twin,
                      "arms": len(group["arms"])})
    return {
        "job": block["job"], "arms": block["count"], "seeds": block["seeds"],
        "control": block["control"], "scale": block["scale"],
        "modes": modes,
    }


def _demoted_of(entries: list[dict[str, Any]], block: "dict[str, Any] | None",
                verdicts: dict[str, Any]) -> "dict[str, Any] | None":
    """The largest lift anywhere, and every reason it is not the record.

    It appears here so it can never appear anywhere else. The biggest number
    on this machine belongs to a one-ply search that consults the engine at
    decision time, over a fifth of the battles the comparable arms ran, at a
    control rate matching nothing that has been measured.

    A maximum taken across every reading on disk is a maximum across scales,
    so the first thing the panel has to say is which scale this one is on and
    whether that is the record's. Printed directly beneath a random-scale
    record with no such line, an idle-scale number reconstructs the 92%-vs-26%
    collapse this whole view exists to prevent, in its two headline panels.
    """
    job = block["job"] if block else None
    best = None
    for entry in entries:
        if entry["name"] == job:
            continue
        for row in entry["evals"]:
            if best is None or row["lift"] > best[1]["lift"]:
                best = (entry, row)
    if best is None:
        return None
    entry, row = best
    verdict = verdicts.get(entry["name"]) or {}
    control = row.get("control")
    scale = _scale_of(control, opponent=row.get("opponent"))
    record_scale = block["scale"] if block else None
    return {
        "name": entry["name"], "lift": row["lift"],
        "win": row.get("win"), "mode": row.get("mode"), "scale": scale,
        "note": entry["note"],
        # Only where an interval is a field of an object. Never from prose.
        "ci": verdict.get("ci") if verdict.get("shape") == "flat" else None,
        "episodes": (verdict.get("episodes") if verdict.get("shape") == "flat"
                     else row.get("eval_episodes") or row.get("episodes")),
        "verdict_note": (verdict.get("note", "")
                         if verdict.get("shape") == "flat" else ""),
        "compare_episodes": block["seeds"] if block else None,
        "anchored": _anchor_for(control) is not None,
        "named": bool(scale["named"]),
        "control_recorded": _plottable(control),
        # The comparison the two headline panels invite, answered before it
        # can be made: True, False, or "there is not enough recorded to say".
        "record_scale": record_scale,
        "record_control": block["control"] if block else None,
        "same_scale": _same_scale(scale, record_scale) if record_scale else None,
    }


def _run_of_checkpoint(path, names):
    """Which run on the page a verdict's checkpoint path belongs to.

    ``runs/passweight/w0.1/cloned.pt`` belongs to ``passweight/w0.1`` and to
    nothing else. Matched against the labels actually on the page, longest
    first, so a sweep variant wins over the sweep it sits in, and a path that
    matches no run on the page returns None rather than the nearest guess.
    """
    text = str(path or "").replace(chr(92), "/").strip()
    if not text:
        return None
    parts = [part for part in text.split("/") if part and part != ".."]
    if "runs" in parts:
        parts = parts[parts.index("runs") + 1:]
    stem = "/".join(parts[:-1]) if parts else ""
    if not stem:
        return None
    for name in sorted(names, key=len, reverse=True):
        tail = name.split(":", 1)[-1]
        if stem == tail or stem.endswith("/" + tail):
            return name
    return None


def _modes_of(entries: list[dict[str, Any]], block: "dict[str, Any] | None",
              verdicts: dict[str, Any], lift_rows: int,
              with_mode: int) -> dict[str, Any]:
    """Every checkpoint where both ways of playing the same weights survive."""
    pairs = []
    if block:
        for pair in _pairs_of(block["arms"]):
            pairs.append({
                "source": block["job"], "weight": pair["weight"],
                "greedy": {"lift": pair["greedy"]["lift"],
                           "win": pair["greedy"]["win"], "ci": None},
                "sampled": {"lift": pair["sampled"]["lift"],
                            "win": pair["sampled"]["win"], "ci": None},
                "gap": pair["gap"], "recorded": False,
                "opponent": block["scale"]["opponent"],
                "opponent_source": block["scale"]["source"],
                "opponent_mismatch": False,
            })
    names = [e["name"] for e in entries]
    recorded = []
    for source, verdict in sorted(verdicts.items()):
        shape = verdict.get("shape")
        if shape == "paired":
            # A paired verdict is the plainest case of both halves surviving,
            # and excluding it is why the half-transcription guard never fired
            # for runs/cloned -- the very file whose flat mirror hands a
            # reader greedy while hiding a sampled reading 2.3x lower.
            greedy, sampled = verdict["greedy"], verdict["sampled"]
            if not (_plottable(greedy.get("lift"))
                    and _plottable(sampled.get("lift"))):
                continue
            left, right = greedy.get("opponent"), sampled.get("opponent")
            mismatch = bool(left and right and left != right)
            recorded.append({
                "source": source, "weight": source,
                "checkpoint": verdict.get("checkpoint") or "",
                "run": (source if source in names
                        else _run_of_checkpoint(verdict.get("checkpoint"), names)),
                "greedy": {"lift": greedy["lift"], "win": greedy.get("win"),
                           "ci": greedy.get("ci")},
                "sampled": {"lift": sampled["lift"], "win": sampled.get("win"),
                            "ci": sampled.get("ci")},
                "gap": None if mismatch else greedy["lift"] - sampled["lift"],
                "opponent": None if mismatch else (left or right),
                "opponent_source": "recorded",
                "opponent_greedy": left, "opponent_sampled": right,
                "opponent_mismatch": mismatch,
                "recorded": bool(left or right),
                "flips": (greedy["lift"] < 0) != (sampled["lift"] < 0),
                "straddles_zero": bool(
                    sampled.get("ci") and sampled["ci"][0] < 0 < sampled["ci"][1]),
            })
            continue
        if shape != "arms":
            continue
        by_weight: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for record in verdict["records"]:
            key = record["name"]
            if key not in by_weight:
                by_weight[key] = {}
                order.append(key)
            if record.get("mode") in ("greedy", "sampled"):
                by_weight[key][record["mode"]] = record
        for key in order:
            found = by_weight[key]
            if "greedy" not in found or "sampled" not in found:
                continue
            greedy, sampled = found["greedy"], found["sampled"]
            if not (_plottable(greedy.get("lift")) and _plottable(sampled.get("lift"))):
                continue
            # A checkpoint whose two modes fall either side of zero. Reading
            # one of them alone is not a partial answer, it is the wrong sign.
            flips = (greedy["lift"] < 0) != (sampled["lift"] < 0)
            straddles = bool(sampled.get("ci")
                             and sampled["ci"][0] < 0 < sampled["ci"][1])
            # Two opponents on one row is not a greedy-vs-sampled gap. The
            # subtraction is withheld rather than printed in a Gap column
            # under a heading saying the only difference is how it played.
            left, right = greedy.get("opponent"), sampled.get("opponent")
            mismatch = bool(left and right and left != right)
            checkpoint = greedy.get("checkpoint") or sampled.get("checkpoint")
            recorded.append({
                "source": source, "weight": key,
                "checkpoint": checkpoint,
                "run": _run_of_checkpoint(checkpoint, names),
                "greedy": {"lift": greedy["lift"], "win": greedy.get("win"),
                           "ci": greedy.get("ci")},
                "sampled": {"lift": sampled["lift"], "win": sampled.get("win"),
                            "ci": sampled.get("ci")},
                "gap": None if mismatch else greedy["lift"] - sampled["lift"],
                "opponent": None if mismatch else (left or right),
                "opponent_source": "recorded",
                "opponent_greedy": left, "opponent_sampled": right,
                "opponent_mismatch": mismatch,
                "recorded": bool(left or right), "flips": flips,
                "straddles_zero": straddles,
            })
    # Where a metrics file transcribed only one mode of a checkpoint whose
    # other mode was measured, the per-run dashboard is reporting a number
    # that reverses when read the other way. Worth saying -- but only where
    # the checkpoint can actually be identified. Two different sweeps on this
    # machine both call an arm "w0.5", so a join on the name alone would
    # attribute one sweep's reading to the other, which is the same class of
    # mistake this whole view exists to prevent. The join is therefore on the
    # run the verdict's own checkpoint path names, and a verdict whose
    # checkpoint names no run on this page joins to nothing at all.
    owners: dict[str, set] = {}
    for item in recorded:
        owners.setdefault(_weight_key(item["weight"]), set()).add(
            item["checkpoint"] or item["source"])
    by_run: dict[str, list] = {}
    for entry in entries:
        for row in entry["evals"]:
            by_run.setdefault(entry["name"], []).append(row)
    for item in recorded:
        key = _weight_key(item["weight"])
        if len(owners.get(key, ())) > 1:
            # Same name, different checkpoints. Say so instead of guessing.
            item["ambiguous"] = sorted(owners[key])
            continue
        run = item["run"]
        if not run:
            continue
        seen = []
        for row in by_run.get(run, []):
            mode = row.get("mode")
            if mode and _weight_key(row["weight"]) not in (key, ""):
                continue
            if not mode:
                # No mode on the row, but an exact match against one of the
                # two measured lifts identifies which half was written down.
                # runs/cloned's rows are bit-identical to its greedy arm.
                for candidate in ("greedy", "sampled"):
                    if row["lift"] == item[candidate]["lift"]:
                        mode = candidate
                        break
            if mode:
                seen.append((run, mode, row["lift"]))
        modes = {mode for _run, mode, _lift in seen}
        if len(modes) == 1 and seen:
            only = sorted(modes)[0]
            item["half_transcribed"] = {
                "mode": only,
                # The number each run is actually showing, read off that
                # run's own rows rather than off the verdict, and quoted for
                # every run named rather than for the first one only.
                "runs": [{"run": name, "lift": lift} for name, lift
                         in sorted({(n, v) for n, _m, v in seen})],
                "hidden": item["sampled" if only == "greedy" else "greedy"]["lift"],
            }
    return {
        "pairs": pairs, "recorded": recorded,
        "with_mode": with_mode, "lift_rows": lift_rows,
    }


def _all_time(runs, notes, kinds, extras) -> dict[str, Any]:
    """Everything the all-time view shows, computed from the same rows.

    Nothing in here may vary with the clock, the path or the machine. The
    page re-renders when the payload fingerprint moves, so a term that
    drifted on its own would throw away scroll position, an open glossary and
    any chart mid-scrub every fifteen seconds.

    Nothing in here is a sentence about today's data either. Every count the
    page states in prose is computed here and read there, because a hardcoded
    "it takes five values, two of which match a control that was measured" is
    true until the next evaluation finishes and false for ever afterwards --
    printed, as it was, directly above the six group cards contradicting it.
    """
    notes = notes or {}
    kinds = kinds or {}
    extras = extras or {}
    soak = (extras.get("soak") if isinstance(extras.get("soak"), dict) else None)
    verdicts = {name: _verdict_of(raw)
                for name, raw in sorted((extras.get("verdicts") or {}).items())}

    # Keyed by label, exactly as `render_multi` keys `body["runs"]`. Two runs
    # that produce one label collapse into one tab there, so counting the
    # census off a list of triples printed "29 runs" above 28 tabs, and
    # passing one run path twice on the command line doubled every total it
    # contributed. The last wins, which is what the runs dict does.
    by_name: dict[str, dict[str, Any]] = {}
    for name, rows, _live in runs:
        rows = list(rows)
        readings = _readings_of(rows, default_weight=name)
        by_name[name] = {
            "name": name,
            "rows": rows,
            "evals": readings,
            "note": str(notes.get(name, "") or ""),
            "job": kinds.get(name) == "job",
        }
    entries = list(by_name.values())
    collisions: list[str] = []
    seen_names: set[str] = set()
    for name, _rows, _live in runs:
        if name in seen_names:
            collisions.append(name)
        seen_names.add(name)

    readings = [r for e in entries for r in e["evals"]]
    lift_rows = len(readings)
    lift_files = sum(1 for e in entries if e["evals"])
    # Two runs wrote the same evaluation twice, under updates 1 and 2, before
    # the double-write was fixed. The dedupe keys on `updates`, so both
    # survive it and the row count overstates the measurements.
    distinct = len({(e["name"], r["lift"], r.get("win"), r.get("control"),
                     r.get("mode"))
                    for e in entries for r in e["evals"]})
    with_mode = sum(1 for r in readings if r.get("mode") is not None)

    block = _block_of(entries)
    groups = _groups_of(entries, verdicts)
    return {
        "census": len(entries),
        "collisions": sorted(set(collisions)),
        "lift_rows": lift_rows,
        # Readings whose control rate matches neither control that has been
        # measured and which do not name their opponent either. Around half
        # of them, which is the single most important fact on the page.
        "unidentified": sum(1 for r in readings
                            if _scale_of(r.get("control"),
                                         opponent=r.get("opponent"))["opponent"]
                            is None),
        # How many name the opponent on their own row. The page says in two
        # places that not one lift on disk does; that has to be counted
        # rather than asserted, because `selfplay.check_lift_is_named` now
        # refuses to write a row that does not.
        "named_rows": sum(1 for r in readings if r.get("opponent")),
        "control_values": len(groups),
        "anchored_groups": sum(1 for g in groups
                               if g["scale"]["opponent"] is not None),
        "distinct_evals": distinct,
        "block": block,
        "record": _record_of(block),
        "demoted": _demoted_of(entries, block, verdicts),
        "groups": groups,
        "modes": _modes_of(entries, block, verdicts, lift_rows, with_mode),
        "exhibits": _exhibits_of(entries, block, verdicts),
        "ever": _ever_of(entries, verdicts, soak, readings, lift_files),
        "unreadable": sorted(name for name, v in verdicts.items()
                             if v["shape"] == "unrecognised"),
        # Run directories that exist and produced no row, and verdict files
        # that were skipped as duplicates. A census is asked exactly about
        # the run that started and wrote nothing, so dropping it silently is
        # the one answer it must not give.
        "skipped": sorted(str(s) for s in (extras.get("skipped") or [])),
        "duplicate_verdicts": sorted(
            str(d) for d in (extras.get("duplicate_verdicts") or [])),
    }


def render_multi(
    runs: "Sequence[tuple[str, Sequence[dict[str, Any]], bool]]",
    back: str | None = None,
    notes: "dict[str, str] | None" = None,
    kinds: "dict[str, str] | None" = None,
    extras: "dict[str, Any] | None" = None,
) -> "tuple[str, dict[str, Any]]":
    """One page holding several runs, as ``(name, rows, live)`` triples.

    Runs are compared far more often than they are watched alone -- the
    question is almost always "is this one doing better than that one", and
    answering it by opening two files and scrolling both is how a difference
    gets missed. Tabs switch between them and a split view puts two
    side by side against shared axes.

    ``live`` marks a run whose metrics file was written recently, so a
    finished run is not presented as though it were still moving.
    """
    body = {
        "runs": {
            name: {
                "series": _series_of(rows),
                "summary": summarise(rows),
                "live": bool(live),
                # What this entry *is*. A training run is legible from its
                # curves; a benchmark, a head-to-head or a piece of work an
                # agent is doing is not, and an index entry whose meaning
                # lives only in a chat log is an index entry nobody can read.
                "note": (notes or {}).get(name, ""),
            }
            for name, rows, live in runs
        },
        # Newest first. The question is nearly always what the run started
        # most recently is doing, and burying it under a fortnight of finished
        # runs makes that a scroll. De-duplicated, because `runs` above is
        # keyed by label: a label arriving twice is one tab, and listing it
        # twice paints two tabs for it and counts it twice in the footer.
        "order": list(dict.fromkeys(name for name, _, _ in reversed(list(runs)))),
        "back": back,
    }
    # The cross-run view. Attached here rather than after the hash, because
    # anything added below ships in the HTML and is invisible to `version`,
    # so the page would never re-render for it -- stale aggregates sitting
    # beside fresh per-run curves on the same screen.
    body["alltime"] = _all_time(runs, notes, kinds, extras)
    # A fingerprint of the data, so the page can re-render only when something
    # actually moved. Reloading on a timer throws away scroll position, an
    # open glossary and any chart mid-scrub, several times an hour, to redraw
    # numbers that had not changed.
    # When this was built, so a countdown keeps ticking between polls rather
    # than freezing at whatever it said when the page last loaded.
    # Applied to the whole payload, not only the series. `summarise` reads
    # its figures straight off the last row, so a non-finite lift or entropy
    # reaches the page around `_plottable` and breaks `JSON.parse` on the
    # served file exactly as a NaN in a series does. Done before the
    # fingerprint, so the hash covers what actually ships.
    body = _json_safe(body)
    body["generated_at"] = time.time()
    body["version"] = hashlib.sha1(
        json.dumps({k: v for k, v in body.items() if k != "generated_at"},
                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    return _PAGE.replace("__DATA__", json.dumps(body)), body


def render(rows: Sequence[dict[str, Any]], title: str,
           back: str | None = None) -> str:
    """The page for a single run. One file, no dependencies, no build step."""
    return render_multi([(title, rows, True)], back=back)[0]


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>cr-sim training</title>
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="cr-sim">
<meta name="theme-color" content="#0D1218">
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Crect width='180' height='180' rx='40' fill='%230D1218'/%3E%3Cpath d='M40 128V52h18l22 34 22-34h18v76h-18V82l-22 33-22-33v46z' fill='%2358B4D0'/%3E%3C/svg%3E">
<link rel="manifest" href="data:application/json,%7B%22name%22%3A%22cr-sim%20training%22%2C%22short_name%22%3A%22cr-sim%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%230D1218%22%2C%22theme_color%22%3A%220D1218%22%2C%22start_url%22%3A%22.%22%7D">
<!-- No webfont link. This page is read over the LAN from a phone that is
     often on a wifi with no route out, and three render-blocking requests to
     a font host it cannot reach buy nothing: every font-family here already
     carries a real fallback stack, and the manifest and icon are data: URIs
     precisely so the file has no dependencies at all. -->
<style>
:root{
  --ground:#EFF2F5;--panel:#FFFFFF;--panel2:#F6F8FA;
  --ink:#121A24;--soft:#3C4855;--muted:#626F7E;
  --rule:#D8DFE6;--hair:#E7ECF1;
  --accent:#14657F;--accentw:#E3EFF4;
  --good:#26704F;--goodw:#E2F0E9;
  --warn:#8A6516;--warnw:#F7EFDD;
  --crit:#A2352C;--critw:#F7E7E5;
  --grey:#8695A4;
  --shadow:0 1px 2px rgba(18,26,36,.06),0 8px 24px -12px rgba(18,26,36,.18);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0D1218;--panel:#141C25;--panel2:#1A232D;
  --ink:#E6ECF2;--soft:#C2CCD7;--muted:#8695A4;
  --rule:#26313D;--hair:#1E2833;
  --accent:#58B4D0;--accentw:#12303C;
  --good:#5FB68C;--goodw:#163529;
  --warn:#DCAB57;--warnw:#362B15;
  --crit:#E58177;--critw:#3A201D;
  --grey:#7C8B9A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#0D1218;--panel:#141C25;--panel2:#1A232D;
  --ink:#E6ECF2;--soft:#C2CCD7;--muted:#8695A4;
  --rule:#26313D;--hair:#1E2833;
  --accent:#58B4D0;--accentw:#12303C;
  --good:#5FB68C;--goodw:#163529;
  --warn:#DCAB57;--warnw:#362B15;
  --crit:#E58177;--critw:#3A201D;
  --grey:#7C8B9A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
  padding:env(safe-area-inset-top) 20px 60px;
  font-family:"Source Sans 3",ui-sans-serif,system-ui,-apple-system,sans-serif;
  font-size:16.5px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto}
h1,h2,h3,.lbl,.v,.kpi-n{font-family:Archivo,ui-sans-serif,system-ui,sans-serif}
.mono,.v,.kpi-n,td.n,.lbl-s{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.lbl{font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}

header{padding:26px 0 14px;display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap}
h1{font-size:clamp(22px,4vw,30px);font-weight:700;letter-spacing:-.02em;margin:0;flex:1}
.stamp{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 3px var(--goodw)}
.dot.stale{background:var(--muted);box-shadow:0 0 0 3px var(--hair)}
.bell{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  color:var(--soft);font-family:Archivo,sans-serif;font-size:11px;font-weight:600;
  letter-spacing:.05em;padding:5px 10px;cursor:pointer}
.bell[data-on="1"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}

.tabbar{display:flex;align-items:flex-end;gap:10px;border-bottom:1px solid var(--rule)}
.tabs{display:flex;gap:2px;flex:1;overflow-x:auto;scrollbar-width:none;padding-bottom:1px}
.tabs::-webkit-scrollbar{display:none}
/* Expanded: every run visible at once instead of one scrolling row. With two dozen runs the strip hid most of them behind a scroll gesture that gives no sign there is anything to scroll to. */
.tabs.expanded{flex-wrap:wrap;overflow:visible;max-height:none;row-gap:3px;padding-bottom:4px}
.tab{appearance:none;border:1px solid transparent;border-bottom:0;background:none;
  font-family:Archivo,sans-serif;font-size:13px;font-weight:600;color:var(--muted);
  padding:8px 12px;border-radius:3px 3px 0 0;cursor:pointer;display:flex;align-items:center;
  gap:7px;white-space:nowrap}
.tab[aria-selected="true"]{background:var(--panel);border-color:var(--rule);color:var(--ink);margin-bottom:-1px;padding-bottom:9px}
.tab .pip{width:6px;height:6px;border-radius:50%;background:var(--muted);flex:none}
.tab .pip.on{background:var(--good);box-shadow:0 0 0 2.5px var(--goodw)}
.tab .val{font-family:"JetBrains Mono",monospace;font-size:11.5px;font-variant-numeric:tabular-nums}
.split-toggle{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  font-family:Archivo,sans-serif;font-size:11px;font-weight:600;color:var(--soft);
  padding:6px 11px;cursor:pointer;margin-bottom:7px}
.split-toggle[aria-pressed="true"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}
@media (max-width:700px){.split-toggle{display:none}}

/* Its own class, deliberately not .split-toggle. That class is hidden below
   700px -- correctly, since the split panes collapse to one column at 900px
   anyway -- and this button is the only way into the all-time view. Sharing
   the class made the entire view unreachable on the 390px phone this page
   exists to be read on. It must stay visible at every width. */
.view-toggle{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  font-family:Archivo,sans-serif;font-size:11px;font-weight:600;color:var(--soft);
  padding:6px 11px;cursor:pointer;margin-bottom:7px;white-space:nowrap}
.view-toggle[aria-pressed="true"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}

.at{margin-top:4px}
.at .panel{background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  box-shadow:var(--shadow);padding:14px 16px;margin-top:10px}
.at .seen{font-size:12.5px;color:var(--muted);padding:9px 12px;background:var(--panel2);
  border-radius:3px;margin-top:14px}
.lift{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums;font-weight:700;
  display:inline-flex;align-items:baseline;gap:6px;flex-wrap:wrap}
.lift.unknown{color:var(--muted)}
.lift.big{font-size:30px;letter-spacing:-.02em}
.wl{font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.at .rec{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--good);
  border-radius:3px;box-shadow:var(--shadow);padding:16px 18px;margin-top:10px}
.at .rec.demote{border-left-color:var(--warn)}
.at .twin{margin-top:12px;padding-top:12px;border-top:1px solid var(--hair)}
.at .twin .lift.big{font-size:22px}
.at .prov{margin-top:8px;font-size:12.5px;color:var(--muted)}
.at ul.why{margin:8px 0 0;padding-left:18px;font-size:13px;color:var(--soft)}
.at ul.why li{margin:3px 0}
.at .caption{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.45}
.at .scroll{overflow-x:auto}
.ladder{margin-top:10px}
.lrow{padding:7px 0;border-top:1px solid var(--hair)}
.lrow:first-child{border-top:0}
.lrow .name{font-size:13px;display:flex;gap:7px;align-items:baseline;flex-wrap:wrap}
.lrow .track{position:relative;height:7px;background:var(--panel2);border-radius:2px;margin-top:5px}
.lrow .track i{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:2px;min-width:1px}
.lrow .track b{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.7}
/* Two arms that are the same weights played two ways, tied together so the
   pair cannot be read as two separate policies. */
.lrow.bracket{border-left:2px solid var(--accent);padding-left:9px}
.lrow.zero{border-top:1px solid var(--rule);margin-top:8px;padding-top:9px}
.ledger{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
.ledger th{text-align:left;font-family:Archivo,sans-serif;font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);font-weight:700;padding:4px 8px 4px 0}
.ledger td{padding:6px 8px 6px 0;border-top:1px solid var(--hair);vertical-align:top}
.ledger td.n{text-align:right;white-space:nowrap;font-family:"JetBrains Mono",monospace;
  font-variant-numeric:tabular-nums}
/* The greedy-vs-sampled table stacks on a phone. Four columns in a 301px
   scroll box parked the sampled figure at x=406 and the gap off the end
   entirely, so the default state of the page's own anti-greedy-bias table on
   the device it is built for showed the greedy number alone -- including on
   the rows flagged "sign flips", where the hidden half is the other sign.
   Swipeable is not shown. */
@media (max-width:700px){
  .ledger.modes thead{position:absolute;left:-9999px}
  .ledger.modes tr{display:block;border-top:1px solid var(--rule);padding:6px 0}
  .ledger.modes td{display:flex;justify-content:space-between;align-items:baseline;
    gap:12px;border-top:0;padding:2px 0}
  .ledger.modes td.n{text-align:right}
  .ledger.modes td.n::before{content:attr(data-h);font-family:Archivo,sans-serif;
    font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
    font-weight:700}
  .ledger.modes td .caption{text-align:left}
}
.tiles2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.tile2{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:11px 13px;
  box-shadow:var(--shadow);min-width:0}
.tile2 .v{font-family:"JetBrains Mono",monospace;font-size:18px;font-weight:700;display:block;
  margin:3px 0;letter-spacing:-.02em;overflow-wrap:anywhere}
.tile2 .rule{font-size:11px;color:var(--muted);line-height:1.35}
.at details.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  margin-top:8px;box-shadow:var(--shadow)}
.at details.card summary{padding:11px 13px;cursor:pointer;font-size:13.5px;display:flex;
  gap:8px;align-items:center;flex-wrap:wrap}
.at details.card .body{padding:0 13px 13px}
.at .panel.close{border-left:3px solid var(--crit)}


.panes{display:grid;grid-template-columns:1fr;gap:18px;margin-top:16px}
.panes.split{grid-template-columns:1fr 1fr}
@media (max-width:900px){.panes.split{grid-template-columns:1fr}}
.pane{min-width:0}
.note{font-size:13.5px;line-height:1.5;color:var(--muted);margin:0 0 12px;padding:10px 12px;
  border-left:2px solid var(--rule);background:rgba(255,255,255,.02);border-radius:0 6px 6px 0;white-space:pre-wrap}
.pane-head{display:flex;align-items:center;gap:9px;margin-bottom:10px}
.pane-head select{font-family:Archivo,sans-serif;font-size:13px;font-weight:600;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:5px 8px}

.readout{background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow);overflow:hidden}
.bar{height:3px;background:var(--hair)}
.bar i{display:block;height:100%;background:var(--accent)}
.readout-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr))}
.cell{padding:13px 15px;border-right:1px solid var(--hair);border-top:1px solid var(--hair)}
.cell:first-child{border-top:0}
@media (min-width:760px){.cell{border-top:0}}
.cell:last-child{border-right:0}
.kpi-n{font-size:20px;font-weight:600;letter-spacing:-.02em;display:block;margin-bottom:2px}
.kpi-n small{font-size:12px;font-weight:400;color:var(--muted)}

.hero{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--sev,var(--rule));
  border-radius:3px;padding:18px 20px;box-shadow:var(--shadow);margin-top:12px}
.hero.good{--sev:var(--good)}.hero.warn{--sev:var(--warn)}.hero.crit{--sev:var(--crit)}.hero.dim{--sev:var(--grey)}
.hero-top{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.hero .v{font-size:36px;font-weight:700;letter-spacing:-.03em;line-height:1}
.chip{font-family:Archivo,sans-serif;font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;padding:3px 7px;border-radius:2px;white-space:nowrap}
.chip.good{background:var(--goodw);color:var(--good)}
.chip.warn{background:var(--warnw);color:var(--warn)}
.chip.crit{background:var(--critw);color:var(--crit)}
.chip.dim{background:var(--hair);color:var(--muted)}
.hero .best{margin-top:8px;font-size:12.5px;color:var(--muted)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}
@media (max-width:700px){.tiles{grid-template-columns:1fr 1fr}}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:12px 14px;box-shadow:var(--shadow)}
.tile .v{font-size:21px;font-weight:700;letter-spacing:-.02em;margin:5px 0 0;display:block}
.good{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}.dim{color:var(--muted)}

.chart{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:14px 16px;
  box-shadow:var(--shadow);margin-top:10px;position:relative}
.chart h3{font-size:14px;font-weight:600;margin:0 0 8px;letter-spacing:-.01em;
  display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.chart h3 .read{font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:400;
  color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}
.chart svg{display:block;width:100%;height:auto;touch-action:pan-y}
.empty{padding:18px;text-align:center;color:var(--muted);font-size:13.5px;background:var(--panel2);border-radius:3px}
.axis{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;opacity:.5}
.grid-l{stroke:var(--hair);stroke-width:1}
.lbl-s{font:600 10px "JetBrains Mono",monospace;fill:var(--muted)}
.scrub{stroke:var(--accent);stroke-width:1;opacity:.7}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}
@media (max-width:700px){.grid2{grid-template-columns:1fr}}
h2{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin:26px 0 2px}

details.gloss{margin-top:26px;background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow)}
details.gloss summary{padding:12px 18px;cursor:pointer;font-family:Archivo,sans-serif;
  font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
details.gloss summary::-webkit-details-marker{color:var(--muted)}
details.gloss dl{margin:0;padding:0 18px 16px}
details.gloss dt{font-family:Archivo,sans-serif;font-weight:700;margin-top:14px;font-size:14px}
details.gloss dd{margin:3px 0 0;color:var(--soft);font-size:13.5px;max-width:78ch}
code{font-family:"JetBrains Mono",monospace;font-size:.87em;background:var(--accentw);color:var(--accent);padding:1px 4px;border-radius:2px}
footer{margin-top:28px;padding-top:12px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted)}
</style></head><body>
<div class="wrap">

<header>
  <h1 id="title">cr-sim</h1>
  <button class="bell" id="bell" data-on="0" title="Notify when a run posts a new evaluation">Alerts off</button>
  <div class="stamp"><span class="dot" id="pulse"></span><span id="stamp"></span></div>
</header>

<div class="tabbar">
  <div class="tabs" id="tabs" role="tablist"></div>
  <button class="view-toggle" id="viewall" aria-pressed="false" title="Everything measured so far, across every run">All time</button>
  <button class="split-toggle" id="expand" aria-pressed="false" title="Show every run at once">All runs</button>
  <button class="split-toggle" id="split" aria-pressed="false">Split</button>
</div>

<div class="panes" id="panes"></div>
<div class="at" id="alltime" hidden></div>

<details class="gloss">
  <summary>What the numbers mean</summary>
  <dl>
    <dt>Lift vs control</dt>
    <dd>Return against a random agent over the same fixed battles, in standard deviations of the control's own spread. 0 is no better than random. Under about 0.25 is inside the noise &mdash; a +0.375 reading on this project measured &minus;0.033 over 300 battles.</dd>
    <dt>Beats its past self</dt>
    <dd>Self-play only. Wins and losses against the oldest version still in the opponent pool. Fixed reference, so more sensitive than lift &mdash; but movement, not skill.</dd>
    <dt>Explained variance</dt>
    <dd>How much of the outcome the critic predicts. 0 is no better than guessing the average, which makes PPO's advantages noise. Sat at 0.00 for four runs.</dd>
    <dt>Entropy</dt>
    <dd>How undecided the policy is. Falling means committing &mdash; good only alongside rising lift.</dd>
    <dt>Value loss and return spread</dt>
    <dd>Squared error, and the spread of what it fits. Meaningless apart; explained variance is the same comparison done properly.</dd>
    <dt>Rollout win rate</dt>
    <dd>Measured while exploring, about eighteen points optimistic here. Do not steer by it.</dd>
    <dt>Pass rate</dt>
    <dd>How often it declined to play. Never punished, so a run can quietly collapse into it.</dd>
    <dt>Control win</dt>
    <dd>How often the control agent won its own half of the same battles. It stands in for a missing opponent: where a row does not carry <code>eval_opponent</code>, this is the only evidence in the file of what it faced. A control that wins 92.5% never played a card; one that wins 26% played randomly. A rate matching neither is left unnamed, and a name written on the row is never overruled by a rate.</dd>
    <dt>Scale group</dt>
    <dd>Every reading sharing one control win rate. Groups are ordered by that rate, which is a property of the measurement, so the order is not a ranking. Numbers are compared inside a group and never across two.</dd>
    <dt>Greedy vs sampled</dt>
    <dd>The same weights either always playing their best action or drawing from their distribution. Two different numbers for one checkpoint, and never averaged or collapsed: cloning wrote +1.623 greedy and +0.709 sampled for identical weights. Where only one is recorded, the other is unmeasured rather than equal.</dd>
    <dt>Paired seeds</dt>
    <dd>The policy and the control played the same battles from the same starts. Two lifts are comparable only if they used one opponent and one seed set, and only a job that says so in its own note is treated as having done that.</dd>
    <dt>Job vs model</dt>
    <dd>A model trained; a job recorded work someone did. Split on <code>kind=="job"</code> in config.json, which register_job writes and no trainer does. Never on the name &mdash; <code>bench-*</code> and <code>agent-*</code> are jobs while <code>probe-*</code> and <code>ab-*</code> are models. Jobs are out of every counter, because their rows park a batch size in steps and spreadsheet comparisons in episodes.</dd>
    <dt>Battles ever</dt>
    <dd>Counted from named sources only: training episodes, the engine soak, every arm of every verdict file, and job rows individually allowed in. The in-run probe figure is an estimate at the evaluation default and sits below the rule, outside the total.</dd>
    <dt>Recorded hours</dt>
    <dd>A floor and never a total. Fewer than half the runs write elapsed time at all, and five of the largest report zero, so the real figure is well above this one.</dd>
    <dt>Distinct evaluations</dt>
    <dd>Lift rows, less the ones that are the same measurement written twice. Two runs wrote one evaluation under both <code>updates</code> 1 and 2, and the de-duplication keys on <code>updates</code>, so both rows survive it and the row count overstates the evidence.</dd>
  </dl>
</details>

<footer id="foot"></footer>
</div>

<script>
var DATA = __DATA__;

function num(x,d){return (x===null||x===undefined||(typeof x==='number'&&isNaN(x)))?'--':Number(x).toFixed(d===undefined?2:d);}
function pct(x){return (x===null||x===undefined)?'--':(x*100).toFixed(0)+'%';}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function last(p){return (p&&p.length)?p[p.length-1][1]:null;}
/* `--` rather than 0 for a missing count. A null seed count rendered as "0
   paired seeds" directly under a headline claiming one seed set, which reads
   as a measured zero instead of an absent figure. */
function commas(n){return (n===null||n===undefined)?'--':Number(n).toLocaleString();}
function dur(s){if(!s&&s!==0)return '--';var h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?(h+'h '+m+'m'):(m+'m');}
/* Counts the "next eval" cells down once a second. Separate from rendering
   because the page deliberately re-renders only when the data changes, and a
   countdown that moves only then is wrong for the twenty minutes in between. */
function tickCountdowns(){
  var now=Date.now()/1000;
  Array.prototype.forEach.call(document.querySelectorAll('[data-role="next"]'),function(el){
    var at=parseFloat(el.dataset.at||'');
    if(!at||el.dataset.live!=='1'){el.textContent='--';return;}
    var left=at-now;
    if(left<=0){el.textContent='due';el.className='good';return;}
    el.className='';
    var m=Math.floor(left/60),sec=Math.floor(left%60);
    el.textContent=m>=60?((m/60).toFixed(1)+'h'):(m+'m '+(sec<10?'0':'')+sec+'s');
  });
}

function remember(k,v){try{localStorage.setItem(k,v);}catch(e){}}
function recall(k,f){try{var v=localStorage.getItem(k);return v===null?f:v;}catch(e){return f;}}

/* ---------------------------------------------------------------- charts */

var CHARTS = [];   // every drawn chart, so scrubbing can find them

function chart(id,title,series,opts){
  opts=opts||{};
  var all=[]; series.forEach(function(s){(s.points||[]).forEach(function(p){all.push(p);});});
  if(all.length<1){
    return '<div class="chart"><h3>'+title+'</h3><div class="empty">'
      +(opts.emptyText||'no evaluations yet')+'</div></div>';
  }
  if(all.length<2||series.every(function(s){return (s.points||[]).length<2;})){
    var only=series.filter(function(s){return (s.points||[]).length;}).map(function(s){
      var pt=s.points[s.points.length-1];
      return '<span style="color:'+s.color+'"><b>'+(opts.asPct?pct(pt[1]):num(pt[1],3))+'</b> '+s.name+'</span>';
    }).join('<span style="color:var(--muted)"> / </span>');
    return '<div class="chart"><h3>'+title+'</h3><div class="empty" style="font-size:14px">'+only
      +'<div style="margin-top:5px;font-size:12px;color:var(--muted)">one reading &mdash; a trend needs two</div></div></div>';
  }
  var w=640,h=170,padL=42,padR=Math.max(50,18+Math.max.apply(null,series.map(function(s){
    return (s.points||[]).length?s.name.length:0;}))*6.2),padT=10,padB=20;
  var xs=all.map(function(p){return p[0];}),ys=all.map(function(p){return p[1];});
  var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var lo=Math.min.apply(null,ys),hi=Math.max.apply(null,ys);
  if(opts.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  if(hi-lo<1e-9){hi+=0.5;lo-=0.5;}
  var pd=(hi-lo)*0.12;hi+=pd;lo-=pd;
  var X=function(v){return padL+(x1===x0?0.5:(v-x0)/(x1-x0))*(w-padL-padR);};
  var Y=function(v){return padT+(1-(v-lo)/(hi-lo))*(h-padT-padB);};
  var body='';
  [0.33,0.66].forEach(function(f){var y=padT+f*(h-padT-padB);
    body+='<line class="grid-l" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+y.toFixed(1)+'" y2="'+y.toFixed(1)+'"/>';});
  if(opts.zero) body+='<line class="zero" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+Y(0).toFixed(1)+'" y2="'+Y(0).toFixed(1)+'"/>';
  series.forEach(function(s){
    var pts=s.points||[]; if(pts.length<2) return;
    if(opts.fill){
      var base=Y(Math.max(lo,0));
      var a=pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ')
        +' L'+X(pts[pts.length-1][0]).toFixed(1)+' '+base.toFixed(1)
        +' L'+X(pts[0][0]).toFixed(1)+' '+base.toFixed(1)+' Z';
      body+='<path d="'+a+'" fill="'+s.color+'" opacity=".08"/>';
    }
    body+='<path d="'+pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ')
      +'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
    var lp=pts[pts.length-1];
    body+='<circle cx="'+X(lp[0]).toFixed(1)+'" cy="'+Y(lp[1]).toFixed(1)+'" r="3.2" fill="'+s.color+'"/>';
    body+='<text class="lbl-s" x="'+(w-padR+6)+'" y="'+(Y(lp[1])+3.4).toFixed(1)+'" fill="'+s.color+'">'+s.name+'</text>';
  });
  body+='<line class="axis" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+(h-padB)+'" y2="'+(h-padB)+'"/>';
  body+='<text class="lbl-s" x="3" y="'+(Y(hi)+8).toFixed(1)+'">'+hi.toFixed(2)+'</text>';
  body+='<text class="lbl-s" x="3" y="'+Y(lo).toFixed(1)+'">'+lo.toFixed(2)+'</text>';
  if(x1>x0){
    body+='<text class="lbl-s" x="'+padL+'" y="'+(h-5)+'">'+(x0/1000).toFixed(0)+'k</text>';
    body+='<text class="lbl-s" x="'+(w-padR)+'" y="'+(h-5)+'" text-anchor="end">'+(x1/1000).toFixed(0)+'k</text>';
  }else{
    body+='<text class="lbl-s" x="'+((padL+w-padR)/2)+'" y="'+(h-5)+'" text-anchor="middle">'+(x0/1000).toFixed(0)+'k</text>';
  }
  body+='<line class="scrub" id="'+id+'-line" x1="0" x2="0" y1="'+padT+'" y2="'+(h-padB)+'" style="display:none"/>';
  CHARTS.push({id:id,series:series,x0:x0,x1:x1,padL:padL,padR:padR,w:w,asPct:!!opts.asPct});
  return '<div class="chart"><h3>'+title+'<span class="read" id="'+id+'-read"></span></h3>'
    +'<svg id="'+id+'" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(title)+'">'+body+'</svg></div>';
}

/* Reading exact values by dragging along a chart. Endpoint labels tell you
   where a line finished; the interesting question is usually what it did in
   the middle, and squinting at a 170px-tall svg does not answer it. */
function wireScrub(){
  CHARTS.forEach(function(c){
    var svg=document.getElementById(c.id), read=document.getElementById(c.id+'-read'),
        line=document.getElementById(c.id+'-line');
    if(!svg||!read) return;
    function at(clientX){
      var box=svg.getBoundingClientRect();
      var vx=(clientX-box.left)/box.width*c.w;
      var t=Math.max(0,Math.min(1,(vx-c.padL)/(c.w-c.padL-c.padR)));
      var step=c.x0+t*(c.x1-c.x0);
      var parts=[];
      c.series.forEach(function(s){
        var pts=s.points||[]; if(!pts.length) return;
        var best=pts[0],bd=Infinity;
        pts.forEach(function(p){var d=Math.abs(p[0]-step); if(d<bd){bd=d;best=p;}});
        parts.push('<span style="color:'+s.color+'">'+(c.asPct?pct(best[1]):num(best[1],3))+'</span>');
        step=best[0];
      });
      read.innerHTML=commas(Math.round(step))+' &middot; '+parts.join(' / ');
      line.setAttribute('x1',c.padL+(step-c.x0)/((c.x1-c.x0)||1)*(c.w-c.padL-c.padR));
      line.setAttribute('x2',line.getAttribute('x1'));
      line.style.display='';
    }
    function clear(){read.innerHTML='';line.style.display='none';}
    svg.addEventListener('mousemove',function(e){at(e.clientX);});
    svg.addEventListener('mouseleave',clear);
    svg.addEventListener('touchstart',function(e){at(e.touches[0].clientX);},{passive:true});
    svg.addEventListener('touchmove',function(e){at(e.touches[0].clientX);},{passive:true});
    svg.addEventListener('touchend',clear);
  });
}

/* ------------------------------------------------------------------ panes */

function verdictFor(l){
  if(l===null||l===undefined) return ['dim','not measured','First evaluation lands at update 20.'];
  if(l>=0.5) return ['good','clearly better','Outside the control&rsquo;s noise.'];
  if(l>=0.25) return ['good','probably better','Outside the noise, but not by much.'];
  if(l>-0.25) return ['warn','inside the noise','A single positive reading here means nothing.'];
  return ['crit','worse than random','Committed to something actively bad.'];
}

function cell(v,k,sm){return '<div class="cell"><span class="kpi-n">'+v+(sm?' <small>'+sm+'</small>':'')
  +'</span><span class="lbl">'+k+'</span></div>';}

function paneMarkup(id){
  return '<div class="pane" data-pane="'+id+'">'
    +'<div class="pane-head"><span class="lbl">showing</span><select data-role="pick"></select></div>'
    +'<div data-role="note" class="note"></div>'
    +'<div class="readout"><div class="bar"><i data-role="bar" style="width:0%"></i></div>'
    +'<div class="readout-grid" data-role="general"></div></div>'
    +'<div data-role="hero"></div><div class="tiles" data-role="tiles"></div>'
    +'<h2>Is it learning</h2><div data-role="c1"></div>'
    +'<h2>Is the critic working</h2><div data-role="c2"></div></div>';
}

function fillPane(pane,name,slot){
  var run=DATA.runs[name]; if(!run) return;
  var S=run.series,s=run.summary,q=function(r){return pane.querySelector('[data-role="'+r+'"]');};
  var noteEl=q('note');
  if(noteEl){noteEl.innerHTML=run.note?esc(run.note):'';noteEl.style.display=run.note?'block':'none';}
  var done=s.total_steps?Math.min(1,(s.steps||0)/s.total_steps):0;
  q('bar').style.width=(done*100).toFixed(1)+'%';
  var remain=(s.total_steps&&s.steps_per_second)?(s.total_steps-s.steps)/s.steps_per_second:null;
  q('general').innerHTML=[
    cell(commas(s.steps),'steps',s.total_steps?('/ '+commas(s.total_steps)):''),
    cell(num(s.steps_per_second,1),'per sec'),
    cell(commas(s.episodes),'matches'),
    cell(dur(s.elapsed_seconds),'elapsed'),
    cell(remain===null?'--':dur(remain),'left'),
    cell(commas(s.evaluations),'evals'),
    cell('<span data-role="next">--</span>','next eval')
  ].join('');
  // Kept as a number the tick loop counts down, rather than re-rendered:
  // the page only re-renders when the data changes, which is every twenty
  // updates, and a countdown that only moves then is a clock that is wrong
  // for twenty minutes at a time.
  var nextEl=q('next');
  if(nextEl){
    nextEl.dataset.at=(s.next_eval_seconds===null||s.next_eval_seconds===undefined)
      ? '' : String((DATA.generated_at||0)+s.next_eval_seconds);
    nextEl.dataset.live=run.live?'1':'0';
  }

  var lift=s.latest_lift,r=verdictFor(lift);
  q('hero').innerHTML='<div class="hero '+r[0]+'"><div class="lbl">lift vs control'
    +(s.modes_recorded?' &middot; sampled':'')+'</div>'
    +'<div class="hero-top"><span class="v '+r[0]+'">'
    +((lift===null||lift===undefined)?'&mdash;':((lift>0?'+':'')+num(lift)+' sd'))
    +'</span><span class="chip '+r[0]+'">'+r[1]+'</span></div>'
    +((s.best_lift===null||s.best_lift===undefined)?'':
      '<div class="best">best '+(s.best_lift>0?'+':'')+num(s.best_lift)+' at '+commas(s.best_at_steps)+'</div>')
    +(s.modes_recorded?('<div class="best">the same weights greedy: '
      +((s.latest_lift_greedy>0?'+':'')+num(s.latest_lift_greedy))
      +' sd. Neither number is the checkpoint on its own.</div>'):'')
    +'</div>';

  var ev=last(S.explained_variance),ent=last(S.entropy),vl=last(S.value_loss),
      rs=last(S.ret_std),np=last(S.noop),rw=last(S.rollout_win);
  var evCls=ev===null?'dim':(ev>=0.3?'good':ev>=0.1?'warn':'crit');
  function tile(k,v,cls){return '<div class="tile"><div class="lbl">'+k+'</div><span class="v '+(cls||'')+'">'+v+'</span></div>';}
  q('tiles').innerHTML=[
    tile('expl var',ev===null?'--':((ev>0?'+':'')+num(ev,3)),evCls),
    tile('win v control',(s.latest_win===null||s.latest_win===undefined)?'--':
      (pct(s.latest_win)+' <span class="dim" style="font-size:14px">v '+pct(s.control_win)+'</span>')),
    tile('past self',(s.ancestor_win===null||s.ancestor_win===undefined)?'--':
      (pct(s.ancestor_win)+' <span class="dim" style="font-size:14px">/ '+pct(s.ancestor_loss)+'</span>'),
      (s.ancestor_win===null||s.ancestor_win===undefined)?'dim':(s.ancestor_win>s.ancestor_loss?'good':'warn')),
    tile('entropy',num(ent,3)),
    tile('critic err',num(vl,3)+' <span class="dim" style="font-size:14px">/ '+num(rs,2)+'</span>'),
    tile('pass',np===null?'--':pct(np),(np!==null&&np>0.5)?'crit':'')
  ].join('');

  var A='#2E86AB',B='#8A6516',C='#A2352C',D='#26704F',G='#8695A4',p=slot;
  q('c1').innerHTML=chart('ch'+p+'a','Lift vs control',
      [{name:S.lift_greedy.length?'sampled':'lift',color:A,points:S.lift}]
      .concat(S.lift_greedy.length?[{name:'greedy',color:B,points:S.lift_greedy}]:[]),
      {zero:true,fill:!S.lift_greedy.length})
    +'<div class="grid2">'
    +chart('ch'+p+'b','Beating its past',
      [{name:'wins',color:D,points:S.ancestor_win},{name:'losses',color:C,points:S.ancestor_loss}],
      {asPct:true,emptyText:'not self-play'})
    +chart('ch'+p+'c','Win rate',
      [{name:'agent',color:A,points:S.win},{name:'random',color:G,points:S.control}],{asPct:true})
    +'</div>';
  q('c2').innerHTML=chart('ch'+p+'d','Explained variance',
      [{name:'expl var',color:D,points:S.explained_variance}],{zero:true,fill:true,emptyText:'too few updates'})
    +'<div class="grid2">'
    +chart('ch'+p+'e','Critic error and spread',
      [{name:'loss',color:C,points:S.value_loss},{name:'spread',color:G,points:S.ret_std}],{emptyText:'too few updates'})
    +chart('ch'+p+'f','Entropy and pass rate',
      [{name:'entropy',color:B,points:S.entropy},{name:'pass',color:G,points:S.noop}],{emptyText:'too few updates'})
    +'</div>';
}

/* ------------------------------------------------------------------ all time

   One page-wide view of everything measured so far. It draws no charts on
   purpose: chart() plots x as `steps`, and every ladder row and every cloned
   checkpoint has steps=0, so a curve here would stack eight separate
   measurements on one vertical line and call it a trend. */

function sd(v){return (v===null||v===undefined)?'--':((v>=0?'+':'-')+Math.abs(Number(v)).toFixed(3));}
function mag(v){return (v===null||v===undefined)?'--':Math.abs(Number(v)).toFixed(3);}
function plural(n,one,many){return n===1?one:(many||(one+'s'));}

/* The scale a lift was read on, as a chip that travels in the same element as
   the number. A figure with no scale beside it is how an idle-scale reading
   gets screenshotted and quoted as a random-scale one.

   A name written on the row wins over one inferred from the control's win
   rate. `selfplay.check_lift_is_named` refuses to write a lift without its
   opponent, so new rows say who they faced -- and calling such a row "vs idle
   (inferred)" because its control happened to win at the idle rate puts a
   confidently wrong name on a number that names itself. */
function scaleChip(sc){
  if(!sc) return {cls:'dim',text:'scale not recorded'};
  if(sc.named) return {cls:'good',text:'vs '+sc.named+' (recorded)'};
  if(sc.opponent) return {cls:sc.stated?'good':'dim',
    text:'vs '+sc.opponent+(sc.stated?' (stated)':' (inferred)')};
  return {cls:'warn',text:'scale unidentified (control '+num(sc.control)+')'};
}

function liftNode(v,sc,extra){
  var c=scaleChip(sc);
  return '<span class="lift'+(c.cls==='warn'?' unknown':'')+(extra?' '+extra:'')+'">'
    +sd(v)+'<span class="chip '+c.cls+'">'+esc(c.text)+'</span>'
    +((sc&&sc.conflict)?('<span class="chip crit">control says '+esc(sc.anchor)+'</span>'):'')
    +'</span>';
}

/* Never defaults to greedy. `arm` is free text with no validator, and reading
   an unlabelled number as greedy is how a working fine-tune was written off:
   its argmax had not moved while the distribution around it had. */
function modeChip(m){
  if(m==='greedy'||m==='sampled') return '<span class="chip dim">'+m+'</span>';
  return '<span class="chip warn">mode unknown</span>';
}

function winLoss(w,l){
  if(w===null||w===undefined) return '';
  return '<span class="wl">'+pct(w)+' w'+((l===null||l===undefined)?'':(' / '+pct(l)+' l'))+'</span>';
}

/* What has moved since this browser last opened the view. Compared on
   evaluation counts and the record itself, never on row counts: a job that
   re-registers rewrites its file from scratch, so rows can go down without
   anything new having been measured. Kept in this browser, never in the
   payload -- a per-viewer fact in the payload would change the fingerprint
   and redraw the page for everybody. */
var seenLine=null;
function recordSignature(A){
  if(!A.record) return null;
  return A.record.modes.map(function(m){
    return (m.mode||'unknown')+' '+sd(m.top.lift);}).join(', ');
}
function sinceLastLook(A){
  /* Worked out once per page load and then held. Recomputing it on every
     poll would answer "since the last fifteen seconds", which is a question
     nobody has, and would wipe the answer to the one they did ask the moment
     any run wrote a row. */
  if(seenLine!==null) return seenLine;
  var now={version:2,perRun:{},record:recordSignature(A)};
  DATA.order.forEach(function(n){now.perRun[n]=DATA.runs[n].summary.evaluations||0;});
  var prev=null;
  try{prev=JSON.parse(recall('crsim-seen','')||'null');}catch(e){prev=null;}
  remember('crsim-seen',JSON.stringify(now));
  if(!prev||prev.version!==2){seenLine='First look on this device. Nothing to compare against yet.';return seenLine;}
  var fresh=[],moved=[];
  Object.keys(now.perRun).forEach(function(n){
    if(!(n in prev.perRun)) fresh.push(n);
    else if(now.perRun[n]>prev.perRun[n]) moved.push(n+' +'+(now.perRun[n]-prev.perRun[n]));
  });
  var bits=[];
  if(fresh.length) bits.push(fresh.length+' new '+plural(fresh.length,'run')+': '+fresh.join(', '));
  if(moved.length) bits.push('new evaluations: '+moved.join(', '));
  if(prev.record!==now.record) bits.push('the record changed, '+(prev.record||'none')+' to '+(now.record||'none'));
  seenLine=bits.length?bits.join(' · '):'Nothing new since you last looked.';
  return seenLine;
}

function recordMarkup(A){
  var r=A.record;
  if(!r) return '<div class="panel"><b>No comparable ranking on disk.</b><div class="caption">'
    +'A record needs one job that measured several arms against one opponent on one seed set, '
    +'over the same number of battles each, and said so in its own note. '
    +'Nothing here does, so nothing is called best -- every reading is grouped by scale below instead.</div></div>';
  var out='';
  r.modes.forEach(function(m){
    out+='<div class="rec">';
    out+='<div class="lbl">Best of the '+(m.mode?(esc(m.mode)+' arms'):'arms whose play mode was never recorded')
      +' &mdash; '+m.arms+' of them</div>';
    out+='<div>'+liftNode(m.top.lift,m.top.scale,'big')+'</div>';
    out+='<div class="prov"><b>'+esc(m.top.weight)+'</b> '+modeChip(m.top.mode)+' '
      +winLoss(m.top.win)+' &middot; '+commas(r.seeds)+' paired seeds &middot; control '+pct(r.control)+' w</div>';
    if(m.twin){
      out+='<div class="twin"><div class="lbl">The same weights, '+esc(m.twin.mode)+'</div>'
        +'<div>'+liftNode(m.twin.lift,m.twin.scale,'big')+'</div>'
        +'<div class="prov">'+winLoss(m.twin.win)+' &middot; the same checkpoint, played the other way. '
        +'Neither number is the checkpoint on its own.</div></div>';
    }else{
      out+='<div class="twin"><div class="prov">Only one way of playing these weights was recorded. '
        +'clone_policy writes max(greedy, sampled) and discards the other, so the missing half is not zero, it is unmeasured.</div></div>';
    }
    out+='</div>';
  });
  if(r.modes.length>1) out+='<div class="caption">Two records, one per way of playing, and deliberately not one. '
    +'Greedy and sampled are different numbers for the same weights, so a single best across both ranks how each '
    +'checkpoint happened to be played -- which is the operation this page exists to refuse.</div>';
  out+='<div class="caption">'+r.arms+' arms in '+esc(r.job)
    +'. The conditions are that job’s own note, quoted under the ladder below, and the intervals are in it.</div>';
  return out;
}

/* The two headline panels sit one above the other, so the second one has to
   say, before anything else, whether it is even on the first one's scale.
   Printed without that line, an idle-anchored number under a random-scale
   record rebuilds the 92%-vs-26% collapse this whole view exists to prevent. */
function demotedHeading(A){
  var d=A.demoted;
  if(d&&d.same_scale===false) return 'The highest number anywhere, and the different scale it was read on';
  if(d&&d.same_scale===null&&d.record_scale) return 'The highest number anywhere, and why it cannot be placed beside the record';
  return 'The highest number, and why it is not the record';
}

function demotedMarkup(A){
  var d=A.demoted;
  if(!d) return '';
  var why=[];
  if(d.record_scale){
    var mine=scaleChip(d.scale).text,theirs=scaleChip(d.record_scale).text;
    if(d.same_scale===false)
      why.push('<b>Not the record’s scale.</b> This was read against a control that won '+pct(d.scale.control)
        +' of its own matches ('+esc(mine)+'); the record was read against one that won '+pct(d.record_control)
        +' ('+esc(theirs)+'). Different opponents give the same policy wildly different numbers, so these two '
        +'cannot be compared at all, never mind ranked.');
    else if(d.same_scale===null)
      why.push('<b>Whether this is the record’s scale cannot be established</b> from what is on disk ('
        +esc(mine)+' against '+esc(theirs)+'), so the comparison the layout invites cannot be made either way.');
    else
      why.push('Read on the record’s own scale ('+esc(mine)+'), so it is comparable — and still not the record, for the rest of these.');
  }
  if(d.verdict_note) why.push('Its own verdict says: “'+esc(d.verdict_note)+'”');
  if(d.episodes&&d.compare_episodes&&d.episodes<d.compare_episodes)
    why.push(commas(d.episodes)+' episodes, against '+commas(d.compare_episodes)+' for every comparable number.');
  if(!d.control_recorded)
    why.push('No control win rate on the row at all'+(d.named?', so the name it records is the only evidence of what it faced.':', so there is no evidence in the file of what it faced.'));
  else if(!d.anchored&&!d.named)
    why.push('Control win '+num(d.scale.control)+' matches neither control that has been measured, and the row names no opponent, so what it faced is genuinely unknown.');
  else if(d.scale.conflict)
    why.push('The row says it faced '+esc(d.scale.named)+', while its control rate of '+num(d.scale.control)
      +' matches the '+esc(d.scale.anchor)+' control. The name is kept and the disagreement is not resolved here.');
  if(!d.named) why.push('No opponent recorded on the row'
    +(A.named_rows?(', unlike '+A.named_rows+' of '+A.lift_rows+' readings on disk.'):', like every other lift on disk.'));
  var out='<div class="rec demote">';
  out+='<div>'+liftNode(d.lift,d.scale,'big')+(d.ci?(' <span class="prov">['+sd(d.ci[0])+', '+sd(d.ci[1])+']</span>'):'')+'</div>';
  out+='<div class="prov"><b>'+esc(d.name)+'</b> '+modeChip(d.mode)+' '+winLoss(d.win)+'</div>';
  out+='<ul class="why">'+why.map(function(w){return '<li>'+w+'</li>';}).join('')+'</ul>';
  out+='<div class="caption">It is shown here, once, so it can never turn up in the record slot.</div>';
  out+='</div>';
  return out;
}

/* One sub-ladder per play mode, each scaled to its own arms. Bar lengths are
   not comparable across sections and the caption says so: greedy exceeds
   sampled for every paired checkpoint here, so a single sorted list is driven
   by how each policy was played more than by its weights. */
function ladderSection(g,b,weights){
  var lifts=g.arms.map(function(a){return a.lift;}).concat([0]);
  var lo=Math.min.apply(null,lifts),hi=Math.max.apply(null,lifts),span=(hi-lo)||1;
  var zero=(0-lo)/span*100;
  var out='<div class="lbl" style="margin-top:14px">'
    +(g.mode?(esc(g.mode)+' play'):'play mode never recorded')+' &mdash; '+g.arms.length+' '+plural(g.arms.length,'arm')+'</div>';
  out+='<div class="ladder">';
  g.arms.forEach(function(a){
    var v=(a.lift-lo)/span*100;
    var left=Math.min(zero,v),width=Math.abs(v-zero);
    out+='<div class="lrow'+(weights[a.weight]>1?' bracket':'')+'">'
      +'<div class="name">'+liftNode(a.lift,a.scale)+' '+modeChip(a.mode)+' <b>'+esc(a.weight)+'</b> '+winLoss(a.win)+'</div>'
      +'<div class="track"><i style="left:'+left+'%;width:'+width+'%"></i><b style="left:'+zero+'%"></b></div>'
      +'</div>';
  });
  out+='<div class="lrow zero"><div class="name"><span class="lift">0.000<span class="chip dim">the control itself</span></span> '
    +'<b>the control</b> '+winLoss(b.control)+'</div></div>';
  out+='</div>';
  return out;
}

function ladderMarkup(A){
  var b=A.block;
  if(!b) return '';
  var weights={};
  b.arms.forEach(function(a){weights[a.weight]=(weights[a.weight]||0)+1;});
  var c=scaleChip(b.scale);
  var out='<div class="panel"><div class="lbl">'+b.count+' arms &middot; '+commas(b.seeds)
    +' paired seeds each &middot; <span class="chip '+c.cls+'">'+esc(c.text)+'</span> control '+pct(b.control)
    +' w &middot; conditions as the note states them, quoted in full below</div>';
  out+='<div class="note">'+esc(b.note)+'</div>';
  b.groups.forEach(function(g){out+=ladderSection(g,b,weights);});
  if(b.groups.length>1) out+='<div class="caption">One section per way of playing, each scaled to its own arms: '
    +'bar lengths do not carry from one section to the next, and neither do the rankings. The same weights greedy '
    +'and sampled are two measurements, not two policies.</div>';
  var res=A.exhibits&&A.exhibits.resolution;
  if(res&&res.top&&res.top.gap!==null&&res.spread){
    out+='<div class="caption">The top two '+esc(String(res.top.mode))+' arms are '+mag(res.top.gap)
      +' apart, and two runs of identical weights already move '+mag(res.spread.range)+'. Treat the leaders as tied.</div>';
  }
  out+='<div class="caption">No whiskers and no interval column: no lift row on disk carries one. '
    +'The intervals are in the note above, printed as written.</div>';
  out+='</div>';
  return out;
}

function modesMarkup(A){
  var m=A.modes,out='<div class="panel">';
  out+='<div class="caption">Only '+m.with_mode+' of '+m.lift_rows+' readings record how the policy played at all. '
    +'clone_policy keeps max(greedy, sampled) and discards the other half, so the rest are one number '
    +'with no way back to the pair it came from.</div>';
  out+='<div class="scroll"><table class="ledger modes"><thead><tr><th>Checkpoint</th><th class="n">Greedy</th>'
    +'<th class="n">Sampled</th><th class="n">Gap</th></tr></thead><tbody>';
  var rows=m.pairs.concat(m.recorded);
  if(!rows.length) out+='<tr><td colspan="4">No checkpoint on disk was measured both ways.</td></tr>';
  rows.forEach(function(p){
    out+='<tr><td>'+esc(p.weight)
      +(p.opponent?(' <span class="chip good">vs '+esc(p.opponent)+' ('+esc(p.opponent_source||'recorded')+')</span>'):'')
      +(p.opponent_mismatch?' <span class="chip crit">two different opponents</span>':'')
      +(p.flips?' <span class="chip crit">sign flips</span>':'')
      +'<div class="caption">'+esc(p.source)+'</div></td>'
      +'<td class="n" data-h="Greedy">'+sd(p.greedy.lift)+'</td>'
      +'<td class="n" data-h="Sampled">'+sd(p.sampled.lift)
      +(p.straddles_zero?'<div class="caption">interval straddles zero</div>':'')+'</td>'
      +'<td class="n" data-h="Gap">'+(p.gap===null||p.gap===undefined?'&mdash;':sd(p.gap))+'</td></tr>';
    if(p.opponent_mismatch) out+='<tr><td colspan="4" class="caption">Greedy was measured against '
      +esc(String(p.opponent_greedy))+' and sampled against '+esc(String(p.opponent_sampled))
      +'. The difference between them is not a play-mode gap, so none is shown.</td></tr>';
    if(p.ambiguous) out+='<tr><td colspan="4" class="caption">Two different checkpoints are recorded under this name ('
      +esc(p.ambiguous.join(', '))+'), so no metrics row can be attributed to either of them.</td></tr>';
    if(p.half_transcribed) out+='<tr><td colspan="4" class="caption">Only the '+esc(p.half_transcribed.mode)
      +' half of this checkpoint was written into metrics, so '
      +p.half_transcribed.runs.map(function(r){return esc(r.run)+' reports '+sd(r.lift);}).join(', ')
      +' for a checkpoint that reads '+sd(p.half_transcribed.hidden)+' the other way.</td></tr>';
  });
  out+='</tbody></table></div>';
  out+='<div class="caption">A row carries an opponent chip only where the file it came from put the lift, the play '
    +'mode and the opponent on one object. Every future writer should copy that shape.</div>';
  out+='</div>';
  return out;
}

function exhibitsMarkup(A){
  var x=A.exhibits||{},out='';
  if(x.coincidence){
    var c=x.coincidence;
    out+='<div class="panel"><div class="lbl">The same number, twice, meaning different things</div>'
      +'<div>'+liftNode(c.ladder.lift,c.ladder.scale)+' <span class="caption">'+esc(c.ladder.arm)+', in '+esc(c.ladder.job)+'</span></div>'
      +'<div>'+liftNode(c.in_run.lift,c.in_run.scale)+' <span class="caption">'+esc(c.in_run.run)+', in-run at '+commas(c.in_run.steps)+' steps</span></div>'
      +'<div class="caption">'+mag(c.gap)+' apart, different opponents and different measurement paths. '
      +'This is why sorting every lift produces a confident lie -- and why those two numbers live on '
      +'opposite sides of a labelled boundary on this page.</div></div>';
  }
  if(x.selection){
    var s=x.selection;
    out+='<div class="panel"><div class="lbl">Best of N is a selection, not a result</div>'
      +'<div>'+liftNode(s.best_in_run,s.best_scale)+' <span class="caption">'+esc(s.run)+', best of '+s.readings+' in-run readings</span></div>'
      +'<div>'+liftNode(s.verdict_lift,s.verdict_scale)+(s.verdict_ci?(' <span class="prov">['+sd(s.verdict_ci[0])+', '+sd(s.verdict_ci[1])+']</span>'):'')
      +' <span class="caption">replayed over '+commas(s.verdict_episodes)+' episodes'
      +(s.verdict_checkpoint?(', on '+esc(s.verdict_checkpoint)):'')+'</span></div>';
    if(s.same_scale===false)
      out+='<div class="caption warn"><b>These two were not read against the same control</b> ('
        +num(s.best_scale.control)+' in-run, '+num(s.verdict_scale.control)
        +' in the verdict), so the drop between them is not the selection effect on its own. '
        +'Whatever else changed changed with it.</div>';
    else if(s.same_scale===null)
      out+='<div class="caption warn">Neither reading records enough about its opponent to establish that they '
        +'are on one scale, so the difference between them is not attributable to selection alone.</div>';
    if(s.verdict_checkpoint)
      out+='<div class="caption">The verdict measured <b>'+esc(s.verdict_checkpoint)+'</b>. No metrics row on this '
        +'machine records which checkpoint produced it, so there is no way to confirm the peak above came from the '
        +'same weights — read the note, which names the checkpoint it replayed and the figure it got.</div>';
    out+='<div class="note">'+esc(s.note)+'</div></div>';
  }
  if(x.resolution){
    var r=x.resolution,i=r.identical;
    out+='<div class="panel"><div class="lbl">How much a reading moves when you simply run it again</div>'
      +'<div class="caption">The same checkpoint (<b>'+esc(i.checkpoint)+'</b>) played '+esc(i.mode)
      +(i.opponent?(' against '+esc(i.opponent)):'')+', on a fixed seed set, measured in '
      +esc(i.sources.join(' and '))+': <span class="mono">'+esc(i.digits)+'</span> both times, identical to every digit. '
      +(i.deterministic
        ? 'Greedy play on fixed seeds is deterministic, so that is not agreement between measurements, it is the same measurement.'
        : 'Sampled play is not deterministic, so digit-for-digit agreement means the same evaluation was written down twice rather than run twice.')
      +'</div>';
    if(r.spread) out+='<div class="caption">The same checkpoint played '+esc(r.spread.mode)+' in those same files: '
      +r.spread.values.map(sd).join(' and ')+', a spread of '+mag(r.spread.range)+'.</div>';
    if(r.top) out+='<div class="caption">And the leading '+esc(String(r.top.mode))+' arm of the ladder above reads '+sd(r.top.lift)
      +' -- a different tower level, not batch noise on one measurement. Do not read the '
      +mag(r.top.gap)+' between '+esc(r.top.arm)+' and '+esc(r.top.runner_arm)+' as a difference.</div>';
    out+='</div>';
  }
  return out;
}

function groupsMarkup(A){
  var unnamed=A.control_values-A.anchored_groups;
  var out='<div class="panel"><b>Grouped by what the reading was measured against, and never ranked across groups.</b>'
    +'<div class="caption">Where a row names its opponent that name is the group. Where it does not, the control’s '
    +'own win rate is the only evidence of what it faced, and on this payload the readings fall into '
    +A.control_values+' '+plural(A.control_values,'group')+': '+A.anchored_groups+' where the opponent can be put a name to and '
    +unnamed+' where '+plural(unnamed,'it cannot','they cannot')+'. '
    +A.unidentified+' of '+A.lift_rows+' readings sit on a scale nobody has identified, and they are not rounded to '
    +'whichever known scale is nearer. Sorting exists only inside a card.</div></div>';
  A.groups.forEach(function(g){
    var c=scaleChip(g.scale);
    out+='<details class="card"><summary><span class="chip '+c.cls+'">'+esc(c.text)+'</span>'
      +'<span class="caption">'+g.rows+' readings across '+g.runs.length+' '+plural(g.runs.length,'run')
      +(g.rankable?'':' &mdash; not ordered')+'</span></summary><div class="body">';
    if(!g.rankable) out+='<div class="caption">These readings record neither a control rate nor an opponent. '
      +'They are not one scale and nothing here says they are, so they are listed by name and not by size.</div>';
    g.runs.forEach(function(r){
      if(r.ranking){
        out+='<div class="lbl" style="margin-top:10px">'+esc(r.name)+' &mdash; a table of arms, not a trajectory</div>';
        if(DATA.runs[r.name]&&DATA.runs[r.name].note)
          out+='<div class="note">'+esc(DATA.runs[r.name].note)+'</div>';
        out+='<div class="scroll"><table class="ledger"><tbody>';
        r.arms.forEach(function(a){
          out+='<tr><td>'+esc(a.arm)+' '+modeChip(a.mode)+'</td><td class="n">'+sd(a.lift)+'</td>'
            +'<td class="n">'+((a.win===null||a.win===undefined)?'':pct(a.win)+' w')+'</td></tr>';
        });
        out+='</tbody></table></div>';
        out+='<div class="caption">Separate measurements, grouped by play mode and ordered only inside one. '
          +'This job has no current value and no best step; its last row is simply its worst arm.</div>';
      }else{
        out+='<div class="lrow"><div class="name">'+liftNode(r.best,g.scale)+' '+modeChip(r.best_mode)+' <b>'+esc(r.name)+'</b> '
          +'<span class="caption">best of '+r.evals+' '+plural(r.evals,'reading')+' (optimistic)'
          +(r.best_at_steps?(' at '+commas(r.best_at_steps)+' steps'):'')
          +(r.modes.length>1?(', over '+r.modes.length+' different ways of playing'):'')+'</span></div></div>';
      }
      if(r.disputed) out+='<div class="caption">Provisional: '+esc(r.name)+' reads control '+num(r.disputed.in_run)
        +' in-run and '+num(r.disputed.verdict)+' in its own '+commas(r.disputed.episodes)
        +'-episode verdict'+(r.disputed.verdict_draw?(', at '+pct(r.disputed.verdict_draw)+' draws'):'')
        +'. The quantity this grouping rests on is not stable across measurement paths.</div>';
    });
    out+='</div></details>';
  });
  return out;
}

function everMarkup(A){
  var e=A.ever,b=e.battles;
  var twice=A.lift_rows-A.distinct_evals;
  var tiles=[
    ['Steps',commas(e.steps.models),'Training runs only. Jobs park a batch size here, so their '+commas(e.steps.jobs)+' is left out.'],
    ['Episodes',commas(e.episodes.models),'Battles played while training, resumes added segment by segment rather than taking the largest row. '
      +(e.untrained.length?('Another '+commas(e.episodes.untrained)+' belong to '+e.untrained.length+' '
        +plural(e.untrained.length,'run')+' that recorded no gradient steps ('+e.untrained.join(', ')
        +'); those are evaluations, counted through their verdict files instead.'):'Every model run here recorded gradient steps.')],
    ['Updates',commas(e.updates.models),'Gradient updates. Counted off the same de-duplicated rows the run tabs show.'],
    ['Distinct evaluations',commas(A.distinct_evals),
      commas(A.lift_rows)+' lift rows'+(twice?(', '+twice+' of which '+plural(twice,'is','are')+' a measurement already counted under another update number.'):', none of them a repeat of another.')],
    ['Runs',commas(e.models)+' models + '+commas(e.jobs)+' jobs','Split on config kind=="job", which register_job writes and no trainer does. Not on the name.'],
    ['Hours',
      'at least '+dur(e.seconds),
      'A floor, not a total: only '+e.reporting_elapsed+' of '+e.models+' model runs record elapsed time at all, and the figure sums those.']
  ];
  var out='<div class="tiles2">'+tiles.map(function(t){
    return '<div class="tile2"><div class="lbl">'+t[0]+'</div><span class="v">'+t[1]+'</span>'
      +'<div class="rule">'+esc(t[2])+'</div></div>';}).join('')+'</div>';

  out+='<div class="panel"><div class="lbl">Battles ever, as line items</div><div class="scroll"><table class="ledger"><tbody>';
  b.counted.forEach(function(c){
    out+='<tr><td>'+esc(c.what)+'<div class="caption">'+esc(c.rule)+'</div></td><td class="n">'+commas(c.n)+'</td></tr>';
  });
  out+='<tr><td><b>Counted exactly</b></td><td class="n"><b>'+commas(b.total)+'</b></td></tr>';
  out+='</tbody></table></div>';
  out+='<div class="scroll"><table class="ledger"><thead><tr><th>Below the rule, not added in</th><th class="n"></th></tr></thead><tbody>';
  out+='<tr><td>'+esc(b.estimated.what)+'<div class="caption">'+esc(b.estimated.rule)+'</div></td>'
    +'<td class="n">~'+commas(b.estimated.n)+'</td></tr>';
  out+='<tr><td>Excluded: '+esc(b.excluded.what)+'<div class="caption">'+esc(b.excluded.rule)+'</div>'
    +b.excluded.items.map(function(i){return '<div class="caption">'+commas(i.n)+' &mdash; '+esc(i.what)+', in '+esc(i.job)+'</div>';}).join('')
    +'</td><td class="n">'+commas(b.excluded.n)+'</td></tr>';
  out+='</tbody></table></div></div>';

  if(e.soak){
    var s=e.soak,reasons=(s.reasons||[]).map(function(r){return commas(r[1])+' '+esc(r[0]);}).join(', ');
    out+='<div class="panel"><div class="lbl">Engine health, from the run this page cannot otherwise see</div>'
      +'<div class="caption">'+esc(s.run)+': '+commas(s.matches)+' matches, '
      +((s.anomalies||[]).length)+' anomalies, '+reasons+', mean '+num(s.mean_ticks,2)+' ticks. '
      +'It has a summary and no metrics file, so it is invisible to the run list.</div></div>';
  }
  return out;
}

function closingMarkup(A){
  var out='<div class="panel close"><b>What this page cannot tell you.</b><div class="caption">'
    +(A.named_rows
      ? (A.named_rows+' of '+A.lift_rows+' readings name the opponent they were measured against; the other '
         +(A.lift_rows-A.named_rows)+' have their scale read off the control’s own win rate, and ')
      : 'Not one lift on disk names the opponent it was measured against, so every scale here is read off the control’s own win rate and ')
    +A.unidentified+' of '+A.lift_rows+' readings match nothing that was measured. '
    +'Greedy and sampled play are collapsed at the source outside the '+A.modes.with_mode+' rows that carry a play mode, '
    +'and the half that was discarded is not zero, it is unmeasured. A peak picked from many short readings is a selection '
    +'and not a skill, which is why every best on this page says so. Recorded time is a floor: most runs never wrote it down. '
    +'And the one ranking here is one job directory deep &mdash; if that job is re-registered, the ladder goes away rather than '
    +'quietly becoming a sort of whatever is left.</div>';
  if(A.skipped&&A.skipped.length) out+='<div class="caption">Not in the '+A.census+': '+A.skipped.length+' run '
    +plural(A.skipped.length,'directory','directories')+' with a config and no rows at all &mdash; '+esc(A.skipped.join(', '))
    +'. A run that started and produced nothing is exactly what a census is asked about, so it is named here rather than dropped.</div>';
  if(A.duplicate_verdicts&&A.duplicate_verdicts.length) out+='<div class="caption">'+A.duplicate_verdicts.length
    +' verdict '+plural(A.duplicate_verdicts.length,'file')+' reachable through a second root held the same content as one already read, '
    +'and '+plural(A.duplicate_verdicts.length,'was','were')+' counted once rather than twice: '+esc(A.duplicate_verdicts.join(', '))+'.</div>';
  if(A.collisions&&A.collisions.length) out+='<div class="caption">'+A.collisions.length+' run '
    +plural(A.collisions.length,'label')+' arrived more than once ('+esc(A.collisions.join(', '))
    +'). Each is shown and counted once; the last one read wins.</div>';
  if(A.unreadable&&A.unreadable.length) out+='<div class="caption">'+A.unreadable.length+' verdict '
    +plural(A.unreadable.length,'file')+' had a shape this page does not recognise and '
    +plural(A.unreadable.length,'was','were')+' not read for a number: '+esc(A.unreadable.join(', '))+'.</div>';
  return out+'</div>';
}

function allTimeMarkup(A){
  if(!A) return '<div class="panel">No aggregate in this payload.</div>';
  var out='<div class="seen">'+esc(sinceLastLook(A))+'</div>';
  out+='<h2>The record</h2>'+recordMarkup(A);
  out+='<h2>'+demotedHeading(A)+'</h2>'+demotedMarkup(A);
  out+='<h2>The comparable ladder</h2>'+(A.block?ladderMarkup(A)
    :'<div class="panel">No comparable ranking on disk.<div class="caption">Nothing is sorted in its place.</div></div>');
  out+='<h2>Greedy vs sampled, wherever both survive</h2>'+modesMarkup(A);
  /* Counted, not named. An exhibit is dropped where the data stops
     supporting it -- a selected peak whose replay turns out to be a
     different checkpoint, two files that cannot be shown to hold the same
     weights -- and a heading that says "three" over two of them is the same
     hardcoded claim about one afternoon's payload as the rest of them. */
  var shown=['coincidence','selection','resolution'].filter(function(k){return (A.exhibits||{})[k];}).length;
  out+='<h2>'+(shown?(shown+' '+plural(shown,'exhibit')):'No exhibit the data still supports')+'</h2>'+exhibitsMarkup(A);
  out+='<h2>Everything else, grouped by scale</h2>'+groupsMarkup(A);
  out+='<h2>Ever, in numbers</h2>'+everMarkup(A);
  out+=closingMarkup(A);
  return out;
}

/* ------------------------------------------------------------ notifications */

var seenEvals={};

function alertsOn(){return recall('crsim-alerts','0')==='1';}

function noticeNewEvaluations(){
  if(!alertsOn()||typeof Notification==='undefined'||Notification.permission!=='granted') return;
  DATA.order.forEach(function(n){
    var s=DATA.runs[n].summary,count=s.evaluations||0;
    if(seenEvals[n]===undefined){seenEvals[n]=count;return;}
    if(count>seenEvals[n]){
      seenEvals[n]=count;
      var lift=s.latest_lift;
      try{
        new Notification(n,{body:'lift '+((lift>0?'+':'')+num(lift))+' sd at '
          +commas(s.steps)+' steps',tag:n,renotify:false});
      }catch(e){}
    }else{seenEvals[n]=count;}
  });
}

/* ------------------------------------------------------------------- shell */

var split=false,picked=['',''];
/* Which of the two views is showing. The tab rail stays rendered in both, so
   tapping any run is the way back -- there is no second URL and no second
   file, because a second page would have to be written, served and kept in
   step with this one for a view that is derived from exactly the same rows. */
var view=(recall('crsim-view','runs')==='alltime')?'alltime':'runs';

function paintTabs(){
  document.getElementById('tabs').innerHTML=DATA.order.map(function(n){
    var r=DATA.runs[n],l=r.summary.latest_lift;
    var val=(l===null||l===undefined)?'':'<span class="val">'+(l>0?'+':'')+Number(l).toFixed(2)+'</span>';
    var on=view==='runs'&&picked.slice(0,split?2:1).indexOf(n)>=0;
    return '<button class="tab" role="tab" data-run="'+n+'" aria-selected="'+on+'">'
      +'<span class="pip'+(r.live?' on':'')+'"></span>'+n+val+'</button>';
  }).join('');
  Array.prototype.forEach.call(document.querySelectorAll('.tab'),function(t){
    t.addEventListener('click',function(){
      picked[0]=t.dataset.run;remember('crsim-pane0',picked[0]);
      view='runs';remember('crsim-view','runs');draw();});
  });
  var vt=document.getElementById('viewall');
  vt.setAttribute('aria-pressed',String(view==='alltime'));
  var ex=document.getElementById('expand');
  ex.textContent=expanded?'Collapse':'All '+DATA.order.length;
  ex.setAttribute('aria-pressed',String(expanded));
  document.getElementById('tabs').classList.toggle('expanded',expanded);
}

/* Repainted on every poll, so the expanded state has to survive a repaint
   and a reload -- collapsing the strip under someone mid-scan is how the
   page loses the run they were looking for. */
var expanded=recall('crsim-expanded','')==='1';

function wireExpand(){
  document.getElementById('expand').addEventListener('click',function(){
    expanded=!expanded;
    remember('crsim-expanded',expanded?'1':'');
    paintTabs();
  });
}

function draw(){
  var panesEl=document.getElementById('panes');
  var allEl=document.getElementById('alltime');
  CHARTS=[];
  /* Drawn from in here rather than beside it: this function clears CHARTS and
     overwrites #panes unconditionally, so a view rendered anywhere else would
     have its markup thrown away on the next poll. */
  if(view==='alltime'){
    document.getElementById('title').textContent='cr-sim, all time';
    panesEl.innerHTML='';panesEl.style.display='none';
    allEl.hidden=false;
    allEl.innerHTML=allTimeMarkup(DATA.alltime);
    paintTabs();
    var A=DATA.alltime;
    document.getElementById('foot').textContent=A
      ? (A.census+' runs, '+A.lift_rows+' lift readings, '+A.unidentified+' on an unidentified scale')
      : '';
    var anyLive=DATA.order.some(function(n){return DATA.runs[n].live;});
    document.getElementById('pulse').className='dot'+(anyLive?'':' stale');
    document.getElementById('stamp').textContent=new Date().toLocaleTimeString();
    return;
  }
  panesEl.style.display='';
  allEl.hidden=true;allEl.innerHTML='';
  document.getElementById('title').textContent=split?'cr-sim':picked[0];
  panesEl.className='panes'+(split?' split':'');
  panesEl.innerHTML=paneMarkup(0)+(split?paneMarkup(1):'');
  for(var i=0;i<(split?2:1);i++){
    (function(i){
      var pane=panesEl.querySelector('[data-pane="'+i+'"]');
      var pick=pane.querySelector('[data-role="pick"]');
      pick.innerHTML=DATA.order.map(function(n){
        return '<option value="'+n+'"'+(n===picked[i]?' selected':'')+'>'+n+'</option>';}).join('');
      pick.addEventListener('change',function(){
        picked[i]=pick.value;remember('crsim-pane'+i,pick.value);draw();});
      pane.querySelector('.pane-head').style.display=split?'flex':'none';
      fillPane(pane,picked[i],i);
    })(i);
  }
  paintTabs();
  wireScrub();
  var s=DATA.runs[picked[0]];
  document.getElementById('foot').textContent=s
    ? (s.summary.updates+' updates, '+s.summary.evaluations+' evaluations')
    : '';
  var pulse=document.getElementById('pulse');
  pulse.className='dot'+(s&&s.live?'':' stale');
  document.getElementById('stamp').textContent=new Date().toLocaleTimeString();
}

function apply(next){
  DATA=next;
  DATA.order.forEach(function(n){if(picked.indexOf(n)<0&&!picked[0])picked[0]=n;});
  if(!DATA.runs[picked[0]])picked[0]=DATA.order[0];
  if(!DATA.runs[picked[1]])picked[1]=DATA.order[Math.min(1,DATA.order.length-1)];
  draw();
  noticeNewEvaluations();
}

/* Polled, and re-rendered only when the payload actually changed. A timed
   reload throws away scroll position, an open glossary and any chart being
   scrubbed, several times an hour, to redraw numbers that had not moved. */
function poll(){
  fetch('data.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(next){
    if(next.version!==DATA.version) apply(next);
    else document.getElementById('stamp').textContent=new Date().toLocaleTimeString();
  }).catch(function(){}).then(function(){setTimeout(poll,15000);});
}

(function start(){
  picked=[recall('crsim-pane0',DATA.order[0]),
          recall('crsim-pane1',DATA.order[Math.min(1,DATA.order.length-1)])]
         .map(function(n){return DATA.runs[n]?n:DATA.order[0];});
  split=recall('crsim-split','0')==='1'&&DATA.order.length>1;
  var toggle=document.getElementById('split');
  toggle.setAttribute('aria-pressed',String(split));
  if(DATA.order.length<2) toggle.style.display='none';
  toggle.addEventListener('click',function(){
    split=!split;remember('crsim-split',split?'1':'0');
    toggle.setAttribute('aria-pressed',String(split));draw();});

  wireExpand();
  if(DATA.order.length<3) document.getElementById('expand').style.display='none';

  var viewBtn=document.getElementById('viewall');
  viewBtn.addEventListener('click',function(){
    view=(view==='alltime')?'runs':'alltime';
    remember('crsim-view',view);draw();});

  var bell=document.getElementById('bell');
  function paintBell(){
    var on=alertsOn()&&typeof Notification!=='undefined'&&Notification.permission==='granted';
    bell.dataset.on=on?'1':'0';
    bell.textContent=on?'Alerts on':'Alerts off';
  }
  bell.addEventListener('click',function(){
    if(typeof Notification==='undefined'){bell.textContent='Unsupported';return;}
    if(alertsOn()){remember('crsim-alerts','0');paintBell();return;}
    Notification.requestPermission().then(function(p){
      remember('crsim-alerts',p==='granted'?'1':'0');paintBell();});
  });
  paintBell();

  DATA.order.forEach(function(n){seenEvals[n]=DATA.runs[n].summary.evaluations||0;});
  draw();
  tickCountdowns();
  setInterval(tickCountdowns,1000);
  if(location.protocol==='http:'||location.protocol==='https:') setTimeout(poll,15000);
})();
</script>
</body></html>
"""





def _run_roots() -> list[Path]:
    """Every directory a run might be under.

    Agents work in git worktrees, each with its own runs/ directory, so the
    experiments actually moving right now are usually not in this checkout at
    all. A page showing only main's runs showed nothing live while four
    sweeps were running a directory away.
    """
    roots = [ROOT / "runs"]
    worktrees = ROOT / ".claude" / "worktrees"
    if worktrees.is_dir():
        roots += [w / "runs" for w in sorted(worktrees.iterdir())
                  if (w / "runs").is_dir()]
    return [r for r in roots if r.is_dir()]


def _kind_of(run: Path) -> "str | None":
    """Whether this directory is a training run or a piece of work someone did.

    ``scripts/register_job.py`` writes ``kind: "job"`` on every entry it
    registers and no trainer writes the key at all, so this is exact. Name
    prefixes are not: ``bench-*`` and ``agent-*`` are jobs, ``probe-*`` and
    ``ab-*`` are models, and two of the jobs carry lift rows that look
    exactly like a model's.

    Read through ``_read_json`` and type-checked, because this gates the
    job/model split the census, every counter and the ranking rest on, and a
    config that is valid JSON but not an object has no ``.get`` at all.
    """
    raw = _read_json(run / "config.json")
    return raw.get("kind") if isinstance(raw, dict) else None


def _read_json(path: Path) -> Any:
    """One JSON file, or None if it cannot be read.

    Two evaluation processes are writing verdict files right now, so a
    half-written one is normal rather than an error.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        # ValueError covers UnicodeDecodeError, which a file saved as UTF-16
        # raises and which is not an OSError.
        return None


def _extras_of(roots: Sequence[Path]) -> dict[str, Any]:
    """The measurements that live beside the runs rather than inside them.

    Verdict files hold the paired evaluations -- the intervals, and the only
    record anywhere of which opponent was faced -- and one of them sits in a
    directory with no metrics file at all, so ``discover`` cannot see it.
    ``runs/soak-spells`` is the same: ten thousand matches, a summary, no
    metrics file, and therefore invisible to this page until now.
    """
    verdicts: dict[str, Any] = {}
    duplicates: list[str] = []
    soak = None
    for root in roots:
        for path in sorted(root.rglob("*verdict*.json")):
            raw = _read_json(path)
            if raw is None:
                continue
            # Keyed by the run it belongs to, so it joins against the rows.
            # A verdict under some other name keeps that name instead.
            name = _label_for(path.parent) if path.stem == "verdict" else path.stem
            if name in verdicts:
                if raw == verdicts[name]:
                    # The same file reached through two roots -- a checkout
                    # and a worktree holding a copy. Counting it again adds a
                    # second set of battles for one evaluation, and the copy
                    # joins to no run, so it is dropped and reported instead.
                    duplicates.append(str(path))
                    continue
                name = _label_for(path.parent) + "/" + path.stem
            verdicts[name] = raw
        for path in sorted(root.rglob("summary.json")):
            raw = _read_json(path)
            if isinstance(raw, dict) and "matches" in raw and soak is None:
                soak = dict(raw, run=_label_for(path.parent))
    return {"verdicts": verdicts, "soak": soak,
            "duplicate_verdicts": duplicates}


#: A run whose metrics file has not been touched this recently is shown as
#: finished. Generous on purpose: an update at the slowest cadence measured
#: here takes a couple of minutes, and calling a live run dead is worse than
#: being slow to notice a finished one.
_LIVE_SECONDS = 300.0



def _local_addresses() -> list[str]:
    """Every address this machine can plausibly be reached on.

    Printed rather than guessed at, because the useful one is the wifi
    address and a machine typically has several -- loopback, a virtual
    adapter or two, and the real one.
    """
    import socket

    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       family=socket.AF_INET):
            address = info[4][0]
            if address not in found and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    return found


def serve(page: Path, port: int) -> None:
    """Serve ``page`` on every interface until interrupted.

    The file is read per request rather than held in memory: the refresh loop
    rewrites it every few seconds, and a cached copy would show a phone the
    state at the moment the server started.
    """
    import http.server
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the stdlib spells it this way
            wanted = page
            kind = "text/html; charset=utf-8"
            if self.path.rstrip("/").endswith("data.json"):
                wanted = page.with_suffix(".json")
                kind = "application/json"
            elif self.path not in ("/", "/index.html", "/progress.html"):
                self.send_error(404)
                return
            try:
                body = wanted.read_bytes()
            except OSError:
                self.send_error(503, "not written yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            # No caching: the whole point is that it changes.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console for training output
            pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("0.0.0.0", port), Handler) as httpd:
        print(f"serving {page.name} on port {port}", flush=True)
        for address in _local_addresses():
            print(f"  http://{address}:{port}", flush=True)
        print("  (same wifi; Windows may ask to allow it through the firewall)",
              flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-watch")
    parser.add_argument(
        "runs", nargs="*", type=Path, default=None,
        help="run directories to watch. Several are shown as tabs on one "
             "page, with a split view for comparing two at once -- the "
             "question is almost always whether this run is beating that "
             "one, and answering it across two files is how a difference "
             "gets missed. Defaults to every run under runs/.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the page (default: progress.html)")
    parser.add_argument("--open", action="store_true", help="open it in a browser")
    parser.add_argument("--once", action="store_true",
                        help="write once and exit rather than refreshing")
    parser.add_argument("--every", type=float, default=30.0)
    parser.add_argument(
        "--serve", type=int, default=0, metavar="PORT",
        help="also serve the page on this port, on every interface, so a "
             "phone on the same wifi can watch a run. 0 disables it.",
    )
    args = parser.parse_args(argv)

    def discover() -> list[Path]:
        """Which runs to show, re-asked on every refresh.

        Rescanned rather than listed once at startup: a run begun after the
        watcher was already going would otherwise never appear, and the page
        looks identical whether a run is missing or merely quiet. That is
        exactly how two diagnostic runs went unnoticed for several minutes.
        """
        if args.runs:
            # De-duplicated like the discovery branch below. The same path
            # given twice produced one tab and two entries: every model total
            # doubled while `body["runs"]` collapsed to one key, so the
            # census identity the page prints in its own footer broke.
            return list(dict.fromkeys(args.runs))
        found: list[Path] = []
        roots = _run_roots()
        for root in roots:
            if not root.is_dir():
                continue
            # rglob rather than iterdir: a sweep nests its variants one level
            # deeper than a plain run, and iterdir could not see them.
            found += [m.parent for m in root.rglob("metrics.jsonl")]
        found = list(dict.fromkeys(found))
        # Ordered by when each run started, not by name. Alphabetical put
        # today's run between two from last week, and the question being
        # asked of this page is almost always "what changed since the last
        # one" -- which only reads properly in the order they happened.
        return sorted(found, key=_started_at)

    runs = discover()
    if not runs:
        print("no runs to watch", file=sys.stderr)
        return 1

    out = args.out or (runs[0] / "progress.html" if len(runs) == 1
                       else ROOT / "progress.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    def once() -> int:
        now = time.time()
        collected = []
        notes: dict[str, str] = {}
        kinds: dict[str, str] = {}
        skipped: list[str] = []
        for run in discover():
            metrics = run / "metrics.jsonl"
            rows = read_metrics(metrics)
            label = _label_for(run)
            if not rows:
                # A directory with a config and no rows is a run that started
                # and produced nothing, which is exactly what a census is
                # asked about. It cannot be shown as a tab -- there is no
                # series -- so it is named instead of vanishing.
                if (run / "config.json").is_file():
                    skipped.append(label)
                continue
            # Deduplicated by update, but only where the repeat is adjacent:
            # the files written before the double-write fix are still on
            # disk, and counting each update twice is what once made a
            # stalled run look busy. A resume replays the same update numbers
            # with the counters reset, so keying a dict on `updates` across
            # the whole file overwrote every pre-resume row with its
            # post-resume namesake and deleted a whole segment -- 816 real
            # training battles, from the two runs that actually resumed,
            # under a tile claiming resumes are added segment by segment.
            deduped: list[dict[str, Any]] = []
            for row in rows:
                if deduped and _repeats(deduped[-1], row):
                    deduped[-1] = row
                else:
                    deduped.append(row)
            rows = deduped
            live = metrics.is_file() and (now - metrics.stat().st_mtime) < _LIVE_SECONDS
            collected.append((label, rows, live))
            note = _note_of(run)
            if note:
                notes[label] = note
            kind = _kind_of(run)
            if kind:
                kinds[label] = kind
        if not collected:
            return 0
        extras = _extras_of(_run_roots())
        extras["skipped"] = skipped
        html, body = render_multi(collected, notes=notes, kinds=kinds,
                                  extras=extras)
        out.write_text(html, encoding="utf-8")
        # Beside the page, because a served copy polls this rather than
        # reloading itself. Written second so a reader never sees data newer
        # than the page that shipped with it.
        out.with_suffix(".json").write_text(
            json.dumps(body), encoding="utf-8")
        return sum(len(rows) for _, rows, _ in collected)

    count = once()
    print(f"{len(discover())} run(s), {count} updates -> {out}", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    if args.once:
        return 0

    if args.serve:
        # In a thread, because the refresh loop below is the thing that keeps
        # the page current and the server only hands out whatever it last
        # wrote.
        import threading

        threading.Thread(target=serve, args=(out, args.serve),
                         daemon=True).start()

    # Polled rather than watched. The page refreshes on a timer anyway, so
    # reacting within a second of a write buys nothing, and a filesystem
    # watcher would be more machinery than the interval deserves.
    try:
        while True:
            time.sleep(max(1.0, args.every))
            try:
                once()
            except Exception as failure:  # noqa: BLE001 - see below
                # A watcher that dies takes every served page down with it,
                # frozen on whatever it last wrote and giving no sign it has
                # stopped -- the same symptom as the NaN freeze and just as
                # invisible. One unreadable file on disk is not worth that,
                # so the refresh is skipped and the reason is printed.
                print(f"refresh failed, keeping the last page: {failure!r}",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
