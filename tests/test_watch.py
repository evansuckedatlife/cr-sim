"""The training progress view.

Its job is to answer one question -- is this getting better -- from a file that
is being appended to while it reads. So the things worth testing are that it
survives a half-written line, that it distinguishes "no evaluations yet" from
"evaluated at zero", and that it puts the honest number in front rather than
the flattering one.

That last point is the reason this exists at all. The trainer's own return is
measured while the policy is exploring and has run about eighteen points
optimistic against a paired-seed control; a progress page that led with it
would show a run improving when it was not.

The page itself now renders client-side: the Python side ships a data blob and
a script, and the script draws the DOM. That makes most of the interesting
behaviour untestable by grepping the HTML string, because the same JS source
-- including every branch's literal text -- is present in the page regardless
of what the data says. The node-backed tests below exist for exactly that
reason: they run the page's own functions and check what they actually
produce, not what strings happen to appear in the template.
"""

from __future__ import annotations

import json
import time

import pytest

from cr_sim.train.watch import read_metrics, render, render_multi, summarise


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(update, steps, **extra):
    row = {
        "updates": update, "steps": steps, "episodes": steps // 20,
        "steps_per_second": 25.0, "entropy": 4.3, "value_loss": 1.0,
        "policy_loss": -0.01, "mean_return": 0.2, "win_rate": 0.3,
        "noop_fraction": 0.01,
    }
    row.update(extra)
    return row


# ------------------------------------------------------------------ reading


def test_a_half_written_final_line_is_dropped_not_fatal(tmp_path):
    """The run appends while this reads, so the last line can be incomplete.

    That is normal rather than an error, and a viewer that crashed on it would
    fail exactly when someone was checking on a long run.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(_row(1, 1000)) + "\n" + '{"updates": 2, "steps": 20',
        encoding="utf-8",
    )
    rows = read_metrics(path)
    assert len(rows) == 1 and rows[0]["updates"] == 1


def test_a_missing_file_reads_as_empty(tmp_path):
    """Asking about a run that has not written anything yet is ordinary."""
    assert read_metrics(tmp_path / "nope.jsonl") == []


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(_row(1, 100)) + "\n\n\n" + json.dumps(_row(2, 200)) + "\n",
        encoding="utf-8",
    )
    assert len(read_metrics(path)) == 2


# ---------------------------------------------------------------- summarising


def test_no_evaluations_is_distinct_from_evaluating_at_zero():
    """One means "not measured yet", the other means "no better than random",
    and showing a zero for the first would be a lie about a run that has not
    been judged."""
    none_yet = summarise([_row(1, 1000)])
    assert none_yet["latest_lift"] is None
    assert none_yet["evaluations"] == 0

    measured = summarise([_row(1, 1000, eval_lift_sd=0.0, eval_win=0.3, control_win=0.3)])
    assert measured["latest_lift"] == 0.0
    assert measured["evaluations"] == 1


def test_the_best_evaluation_is_kept_not_the_last(tmp_path):
    """A run that peaks and regresses should still report the peak, because
    that is the checkpoint that was saved."""
    rows = [
        _row(1, 1000, eval_lift_sd=0.1),
        _row(2, 2000, eval_lift_sd=0.8),
        _row(3, 3000, eval_lift_sd=-0.2),
    ]
    summary = summarise(rows)
    assert summary["best_lift"] == 0.8
    assert summary["best_at_steps"] == 2000
    assert summary["latest_lift"] == -0.2


def test_an_empty_run_summarises_without_raising():
    assert summarise([])["updates"] == 0


# ------------------------------------------------------------------ the page


def test_the_page_leads_with_the_lift_not_the_rollout_return():
    """The rollout return is the flattering number and the misleading one.

    The page draws client-side now, so the DOM order is fixed by the order
    the script's own `chart(...)` calls appear in the source -- the first
    "Is it learning" chart is the lift, and only after it does a win-rate
    chart appear, which is what this checks instead of the old server-
    rendered heading order.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.3)], "run")
    assert page.index("'Lift vs control'") < page.index("'Win rate'")
    assert "eighteen points optimistic" in page, "the caveat is not stated"


def test_the_page_embeds_the_series_it_draws():
    rows = [
        _row(1, 1000, eval_lift_sd=0.1, eval_win=0.3, control_win=0.3),
        _row(2, 2000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.3),
    ]
    page = render(rows, "run")
    payload = json.loads(page.split("var DATA = ", 1)[1].split(";\n", 1)[0])
    run = payload["runs"]["run"]
    assert run["series"]["lift"] == [[1000, 0.1], [2000, 0.4]]
    assert run["summary"]["best_lift"] == 0.4
    assert payload["order"] == ["run"]


def test_the_page_carries_its_own_logic_and_data():
    """No build step and no fetch on open -- it is opened from disk while a
    run runs, and once open it only ever reaches out for its own data.json.

    Web fonts are the one exception, and they are allowed only because they
    degrade silently: the page must name a real fallback stack so it still
    reads correctly on a machine with no network, which is the state this is
    most likely to be opened in.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.2)], "run")
    assert "<script" in page and "</html>" in page
    assert "var DATA = " in page, "the page doesn't carry its own data"

    external = [
        line for line in page.splitlines()
        if ("http://" in line or "https://" in line)
        and "fonts.googleapis.com" not in line
        and "fonts.gstatic.com" not in line
        # A data: URI is embedded, not fetched -- the apple-touch-icon is an
        # inline SVG whose xmlns is itself a URL, which is not a reach outside
        # the page even though it contains "http://".
        and 'href="data:' not in line
    ]
    assert not external, f"page reaches outside for {external}"
    assert "ui-sans-serif" in page and "ui-monospace" in page, "no fallback stack"


# ------------------------------------------------------ explaining the numbers


def test_every_metric_on_the_page_is_explained_somewhere_on_it():
    """A dashboard of unexplained numbers is a dashboard nobody can act on.

    Two of these were actively misread on this project: value loss was called
    mis-calibrated when it was only measured against a different reward scale,
    and a pair of positive lift readings were called a trend when six of them
    averaged to noise. The glossary exists so the page cannot be read that way
    again.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.3, eval_win=0.4, control_win=0.3)], "run")
    for term in ("Lift vs control", "Beats its past self", "Explained variance",
                 "Entropy", "Value loss and return spread", "Rollout win rate",
                 "Pass rate",
                 # Every quantity the all-time view introduces. Each entry
                 # states the rule it was computed under, not a definition --
                 # a definition of "battles ever" would not tell anyone that
                 # 8,765 of them are spreadsheet comparisons.
                 "Control win", "Scale group", "Greedy vs sampled",
                 "Paired seeds", "Job vs model", "Battles ever",
                 "Recorded hours", "Distinct evaluations",
                 # And every quantity the five all-time pictures introduce.
                 # Each of these is an axis or a bar length somewhere, which
                 # is exactly the kind of number a reader cannot look up.
                 "Greedy&ndash;sampled gap", "Matched pair", "Sweep family",
                 "Reading number", "Interval width"):
        assert term in page, f"{term!r} is shown but never explained"


def test_the_page_says_what_zero_means_for_the_two_numbers_that_need_it():
    """Both have a meaningful zero that is not obvious from the value alone."""
    page = render([_row(1, 1000, eval_lift_sd=0.0)], "run")
    assert "no better than random" in page
    assert "no better than guessing the average" in page


def test_the_page_carries_the_critic_series_it_now_leads_on():
    rows = [
        _row(1, 1000, explained_variance=-0.02, ret_std=0.5),
        _row(2, 2000, explained_variance=0.31, ret_std=0.6),
    ]
    payload = json.loads(
        render(rows, "run").split("var DATA = ", 1)[1].split(";\n", 1)[0])
    series = payload["runs"]["run"]["series"]
    assert series["explained_variance"] == [[1000, -0.02], [2000, 0.31]]
    assert series["ret_std"] == [[1000, 0.5], [2000, 0.6]]


# ---------------------------------------------------- charts with thin data
#
# The page renders in JavaScript, so asserting on the HTML string tests the
# template rather than the behaviour -- `no evaluations yet` appears in the
# script source whatever the data is, which is why the test that looked for it
# passed for a year without checking anything. These run the page's own
# functions under node and assert on what they actually return.

import re
import shutil
import subprocess

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="needs node to run the page's JS")


def _script_body(rows):
    """The page's function definitions, as JS source, with the DOM-touching
    start-up block stripped off. Shared by every node-backed test below."""
    page = render(rows, "run")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    return script.split("(function start()", 1)[0]


def _node(source):
    """Run `source` under node and return its stdout.

    Through a file rather than `node -e`: the page's script outgrew the
    Windows command-line limit the day the all-time view landed, and every
    node-backed test here died with "the filename or extension is too long"
    -- eight of the only genuinely non-vacuous tests in this file, all at
    once, for a reason that has nothing to do with what they check.
    """
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / "page.js"
        script.write_text(source, encoding="utf-8")
        # Decoded as UTF-8 and not as the console codepage: node writes its
        # stdout as UTF-8, and the page draws a literal middle dot between a
        # point's arm and the sweep it came out of. Read as cp1252 that
        # arrives as two mojibake characters and no test can assert on the
        # label at all.
        result = subprocess.run([node, str(script)], capture_output=True,
                                text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _call(rows, expression):
    """Evaluate `expression` against the page's own chart code."""
    body = _script_body(rows)
    return _node(body + chr(10) + "process.stdout.write(String("
                 + expression + "));")


def _call_with_storage(rows, expression):
    """Like `_call`, but with a fake `localStorage`.

    Plain `node -e` has no `localStorage` global, so without this,
    `remember`/`recall` silently no-op through their own catch branch and
    every persistence test would pass whether or not persistence actually
    worked.
    """
    body = _script_body(rows)
    stub = (
        "var __store={};"
        "var localStorage={setItem:function(k,v){__store[k]=String(v);},"
        "getItem:function(k){return Object.prototype.hasOwnProperty.call(__store,k)?__store[k]:null;}};"
    )
    return _node(body + stub + chr(10) + "process.stdout.write(String("
                 + expression + "));")


def _run_js(rows, harness):
    """Run arbitrary JS `harness` statements after the page's function
    definitions, for tests that need more than one expression -- a scripted
    sequence, or a fake DOM/fetch/localStorage the page's functions call
    into. Returns whatever the harness wrote to stdout."""
    return _node(_script_body(rows) + harness)


@needs_node
def test_one_evaluation_is_reported_rather_than_called_none():
    """A single reading cannot be drawn as a line, but it is not nothing.

    The page printed "no evaluations yet" over the top of a real evaluation
    for the first hour of a run -- the one thing a progress view must never
    do, which is deny a measurement it is holding.
    """
    rows = [_row(1, 15360, eval_lift_sd=0.118, eval_win=0.32, control_win=0.30)]
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{zero:true})")
    assert "no evaluations yet" not in out, "denied an evaluation it was given"
    assert "0.118" in out, "the single reading is not shown"
    assert "one reading" in out


@needs_node
def test_no_evaluations_still_reads_as_none():
    out = _call([_row(1, 1000)], "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{})")
    assert "no evaluations yet" in out


@needs_node
def test_two_readings_at_the_same_step_do_not_produce_a_degenerate_axis():
    """Identical endpoints must not be rendered as "15k -> 15k", which reads
    as a bug in the chart rather than a fact about the data. The page's fix is
    a single centred label instead of a start and end label that happen to
    match, so this checks the label appears exactly once."""
    rows = [_row(1, 15360, eval_lift_sd=0.1), _row(2, 15360, eval_lift_sd=0.2)]
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{})")
    assert out.count("15k</text>") == 1, "should be one centred label, not a degenerate range"


@needs_node
def test_the_legend_is_not_clipped_by_a_long_series_name():
    """Fixed right padding cut "rollout win" down to "rollout wc"."""
    rows = [_row(i, i * 1000, win_rate=0.2 + i / 100) for i in range(1, 5)]
    out = _call(rows, "chart('t','n',[{name:'rollout win',color:'#000',points:DATA.runs['run'].series.rollout_win}],{})")
    assert "rollout win</text>" in out, "series label truncated"
    # the label starts inside the viewBox and its text has room to the edge
    x = float(re.search(r'x="([0-9.]+)"[^>]*>rollout win</text>', out).group(1))
    assert x + len("rollout win") * 6.3 <= 640, "label runs past the right edge"


@needs_node
def test_the_headline_verdict_reads_a_noisy_lift_as_noise():
    """A lift inside the control's own bounce must not be presented as a win.

    The failure this prevents is real: an early +0.23 was reported as the
    first positive signal, and six evaluations later the mean was +0.04. The
    verdict text now lives in client-side JS, so it is part of the static
    template regardless of the data (every branch's wording is always in the
    page source) -- only calling `verdictFor` proves which branch a given
    lift actually lands in.
    """
    assert _call([_row(1, 1000)], "verdictFor(0.2)[1]") == "inside the noise"
    assert _call([_row(1, 1000)], "verdictFor(0.3)[1]") == "probably better"
    assert _call([_row(1, 1000)], "verdictFor(0.6)[1]") == "clearly better"
    assert _call([_row(1, 1000)], "verdictFor(-0.4)[1]") == "worse than random"
    assert _call([_row(1, 1000)], "verdictFor(null)[1]") == "not measured"


@needs_node
def test_a_chart_can_be_scrubbed_to_read_an_exact_value():
    """Endpoint labels say where a line finished; the interesting question is
    usually what it did in the middle, and squinting at a 170px svg does not
    answer it. `wireScrub` is what lets a drag along the chart read out the
    nearest point instead."""
    harness = r"""
var elements = {};
function makeEl(id){
  var el = {
    style: {display:''}, innerHTML: '',
    getBoundingClientRect: function(){ return {left:0, width:640}; },
    addEventListener: function(evt, fn){ this['on_'+evt] = fn; },
    setAttribute: function(k,v){ this['attr_'+k] = v; },
    getAttribute: function(k){ return this['attr_'+k]; },
  };
  elements[id] = el;
  return el;
}
var document = { getElementById: function(id){ return elements[id] || null; } };

chart('t','n',[{name:'lift',color:'#000',points:[[0,0.1],[1000,0.5]]}],{});
makeEl('t'); makeEl('t-read'); makeEl('t-line');
wireScrub();

elements['t'].on_mousemove({clientX:42});   // left edge -> the first point
var atStart = elements['t-read'].innerHTML;
var lineShownAtStart = elements['t-line'].style.display;

elements['t'].on_mousemove({clientX:590});  // right edge -> the last point
var atEnd = elements['t-read'].innerHTML;

elements['t'].on_mouseleave();
var afterLeave = elements['t-read'].innerHTML;
var lineHiddenAfterLeave = elements['t-line'].style.display;

process.stdout.write(JSON.stringify({
  atStart: atStart, lineShownAtStart: lineShownAtStart,
  atEnd: atEnd, afterLeave: afterLeave, lineHiddenAfterLeave: lineHiddenAfterLeave,
}));
"""
    out = json.loads(_run_js([_row(1, 1000)], harness))
    assert "0.100" in out["atStart"], "scrubbing to the first point misreads it"
    assert out["lineShownAtStart"] == "", "the scrub line should be visible while dragging"
    assert "0.500" in out["atEnd"], "scrubbing to the last point misreads it"
    assert out["afterLeave"] == "", "the readout should clear once the pointer leaves"
    assert out["lineHiddenAfterLeave"] == "none", "the scrub line should hide once the pointer leaves"


# ------------------------------------------------------------ several at once


def test_several_runs_are_carried_on_one_page():
    """Comparing runs is the common case, and doing it across two files is
    how a difference gets missed."""
    page, payload = render_multi([
        ("alpha", [_row(1, 1000, eval_lift_sd=0.4)], True),
        ("beta", [_row(1, 1000, eval_lift_sd=-0.1)], False),
    ])
    assert set(payload["order"]) == {"alpha", "beta"}
    assert payload["runs"]["alpha"]["summary"]["latest_lift"] == 0.4
    assert payload["runs"]["beta"]["summary"]["latest_lift"] == -0.1


def test_a_finished_run_is_not_presented_as_moving():
    """The pip beside a tab says whether that run is still writing. Showing a
    run that stopped hours ago as live is how a dead watcher went unnoticed
    for an hour on this project."""
    page, payload = render_multi([("done", [_row(1, 1000)], False)])
    assert payload["runs"]["done"]["live"] is False


def test_the_page_offers_a_split_view_only_when_there_is_something_to_compare():
    page, payload = render_multi([("only", [_row(1, 1000)], True)])
    assert "split-toggle" in page
    assert "order.length<2" in page, "the toggle is never hidden for one run"


def test_runs_are_ordered_by_when_they_started(tmp_path):
    """Chronological, not alphabetical.

    Sorting by name puts today's run between two from last week, and the
    question this page answers is almost always what changed since the
    previous one -- which only reads in the order they happened.
    """
    import os

    from cr_sim.train.watch import _started_at

    # Deliberately reverse-alphabetical, so a name sort would fail this.
    order = ["zulu", "mike", "alpha"]
    for i, name in enumerate(order):
        run = tmp_path / name
        run.mkdir()
        (run / "metrics.jsonl").write_text(
            json.dumps(_row(1, 1000)) + chr(10), encoding="utf-8")
        config = run / "config.json"
        config.write_text("{}", encoding="utf-8")
        # Backdated, so the real _started_at (which takes the earlier of
        # creation and modification) reads these as the start times.
        stamp = 1_000_000 + i * 3600
        os.utime(config, (stamp, stamp))

    found = sorted((d for d in tmp_path.iterdir() if d.is_dir()), key=_started_at)
    assert [d.name for d in found] == order
    assert found != sorted(tmp_path.iterdir()), "this would pass on a name sort"


def test_a_run_without_a_config_still_sorts(tmp_path):
    """Runs from before the config was written must not crash the ordering."""
    from cr_sim.train.watch import _started_at

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "metrics.jsonl").write_text(
        json.dumps(_row(1, 1000)) + chr(10), encoding="utf-8")
    assert _started_at(bare) > 0
    assert _started_at(tmp_path / "missing") == 0.0


def test_the_newest_run_leads_the_payload_order():
    """`render_multi` takes runs oldest-first (the same order `discover()`
    sorts them in) but presents them newest-first in the payload -- the
    question this page answers is almost always what the most recently
    started run is doing, and burying it under a fortnight of finished runs
    made that a scroll."""
    page, payload = render_multi([
        ("first", [_row(1, 1000)], False),
        ("second", [_row(1, 1000)], False),
        ("third", [_row(1, 1000)], True),
    ])
    assert payload["order"] == ["third", "second", "first"]


# ------------------------------------------------------------------ the ladder


def test_the_ladder_reaches_the_page():
    """Self-play measures itself against its own past, and that has to show.

    It is the most sensitive number available: the reference is fixed and
    roughly the right difficulty, where lift against a random control has a
    spread wide enough that a +0.375 reading on this project measured -0.033
    when replayed over 300 battles.
    """
    rows = [
        _row(1, 1000, ancestor_win=0.30, ancestor_loss=0.50, ancestor_age=1),
        _row(2, 2000, ancestor_win=0.62, ancestor_loss=0.21, ancestor_age=3),
    ]
    page, _ = render_multi([("sp", rows, True)])
    # Through JSON, as it actually ships on the page -- the Python-side
    # payload holds (steps, value) tuples, which is an implementation detail
    # the page itself never sees.
    payload = json.loads(page.split("var DATA = ", 1)[1].split(";\n", 1)[0])
    run = payload["runs"]["sp"]
    assert run["series"]["ancestor_win"] == [[1000, 0.30], [2000, 0.62]]
    assert run["summary"]["ancestor_win"] == 0.62
    assert run["summary"]["ancestor_age"] == 3


def test_a_run_without_a_ladder_says_so_rather_than_showing_zero():
    """Most runs were not self-play. Reporting 0% would read as 'it loses
    every one', which is a different claim from 'this was never measured'."""
    page, payload = render_multi([("plain", [_row(1, 1000)], True)])
    assert payload["runs"]["plain"]["summary"]["ancestor_win"] is None
    assert payload["runs"]["plain"]["series"]["ancestor_win"] == []


# --------------------------------------------------------- the payload version


def test_the_payload_fingerprint_changes_only_when_the_data_does():
    """The page polls data.json and only re-renders when `version` differs
    from what it already has, so the fingerprint must be a pure function of
    the data -- if it also moved with wall-clock time, every poll would look
    like new data and the whole point of comparing versions would be lost."""
    rows = [_row(1, 1000, eval_lift_sd=0.1)]
    _, first = render_multi([("run", rows, True)])
    time.sleep(0.01)
    _, second = render_multi([("run", rows, True)])
    assert first["version"] == second["version"], "the fingerprint drifted with no data change"

    _, third = render_multi([("run", rows + [_row(2, 2000, eval_lift_sd=0.4)], True)])
    assert third["version"] != first["version"], "new data produced the same fingerprint"


# ---------------------------------------------------- installed, polled, alerted


def test_there_is_no_meta_refresh_tag():
    """A `<meta http-equiv="refresh">` reloads the whole document -- scroll
    position, an open glossary, and a chart mid-scrub, gone every 15 seconds,
    to redraw numbers that usually had not moved. The page now polls
    data.json and re-renders in place instead, so no refresh tag should ship."""
    page = render([_row(1, 1000)], "run")
    assert "http-equiv" not in page.lower()
    assert "refresh" not in page.lower()


def test_the_page_is_installable_to_a_home_screen():
    """Meant to be left open on a phone next to the desk, not bookmarked --
    an icon of its own so checking a run doesn't mean digging back through
    open tabs."""
    page = render([_row(1, 1000)], "run")
    assert 'apple-mobile-web-app-capable' in page and 'content="yes"' in page
    assert 'rel="manifest"' in page
    assert 'rel="apple-touch-icon"' in page
    assert "standalone" in page, "the manifest should ask to run standalone, not in a browser chrome"


@needs_node
def test_alerts_are_off_by_default_and_persist_once_turned_on():
    """The opt-in choice is stored under a fixed localStorage key,
    `crsim-alerts`, so a reload does not silently re-ask and does not
    silently forget a yes."""
    assert _call_with_storage([_row(1, 1000)], "alertsOn()") == "false"
    assert _call_with_storage(
        [_row(1, 1000)], "(remember('crsim-alerts','1'), alertsOn())"
    ) == "true"


def test_permission_is_requested_only_on_a_click_not_on_page_load():
    """A page that asks for notification permission the instant it opens gets
    an instinctive "block", which burns the ask for good -- browsers do not
    let a site ask again after that. The request must be nested inside the
    bell's click handler, not run eagerly during start-up."""
    page = render([_row(1, 1000)], "run")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    assert script.count("Notification.requestPermission(") == 1

    marker = "bell.addEventListener('click',function(){"
    start = script.index(marker) + len(marker) - 1
    depth, end = 0, None
    for i in range(start, len(script)):
        if script[i] == "{":
            depth += 1
        elif script[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "unbalanced braces reading the click handler"
    request_at = script.index("Notification.requestPermission(")
    assert start < request_at < end, "the request runs outside the click handler"


@needs_node
def test_the_page_polls_for_new_data_and_only_reapplies_it_on_a_new_version():
    """Re-rendering on every poll -- even when nothing changed -- is the
    behaviour the version fingerprint exists to prevent, so this drives
    `poll()` itself (with fake `fetch`/`apply`/timers) rather than just
    checking that the word "poll" appears in the source."""
    harness = r"""
var realTimer = require('timers').setTimeout;
function setTimeout(fn, ms){}          // swallow poll's own re-scheduling
var fetchCalls = 0;
var nextPayload = {version:'same'};
function fetch(url){
  fetchCalls++;
  return Promise.resolve({ json: function(){ return Promise.resolve(nextPayload); } });
}
var appliedWith = null;
function apply(next){ appliedWith = next; }   // stand in for the real, DOM-heavy apply
var stampText = null;
var document = { getElementById: function(id){ return { set textContent(v){ stampText = v; } }; } };
var DATA = {version:'same'};

poll();
realTimer(function(){
  var afterSame = {fetchCalls: fetchCalls, appliedWith: appliedWith, stampText: stampText};
  nextPayload = {version:'different'};
  poll();
  realTimer(function(){
    process.stdout.write(JSON.stringify({
      afterSame: afterSame,
      appliedAfterDifferent: appliedWith,
    }));
  }, 30);
}, 30);
"""
    out = json.loads(_run_js([_row(1, 1000)], harness))
    assert out["afterSame"]["fetchCalls"] == 1, "should have polled once"
    assert out["afterSame"]["appliedWith"] is None, "an unchanged version must not be re-applied"
    assert out["afterSame"]["stampText"], "the timestamp should still refresh on an unchanged poll"
    assert out["appliedAfterDifferent"] == {"version": "different"}, \
        "a changed version must be applied"


# --------------------------------------------------------- time to next eval


def test_the_evaluation_cadence_is_inferred_from_the_run_itself():
    """Read from history rather than from config, so it still works for a run
    whose config was lost and for one whose cadence a resume changed."""
    rows = [_row(i, i * 100, elapsed_seconds=i * 30.0) for i in range(1, 25)]
    for i in (5, 10, 15, 20):
        rows[i - 1]["eval_lift_sd"] = 0.3
    summary = summarise(rows)
    assert summary["eval_every"] == 5


def test_the_countdown_uses_the_recent_pace_not_the_whole_run():
    """A run throttled partway through would otherwise be timed by an average
    it no longer runs at. This one halves its speed at the midpoint, and the
    estimate has to follow the slow half."""
    rows = []
    elapsed = 0.0
    for i in range(1, 25):
        elapsed += 10.0 if i <= 12 else 60.0
        rows.append(_row(i, i * 100, elapsed_seconds=elapsed))
    for i in (5, 10, 15, 20):
        rows[i - 1]["eval_lift_sd"] = 0.3
    seconds = summarise(rows)["next_eval_seconds"]
    # Next evaluation is at update 25, one update away, at the slow pace.
    assert 40 < seconds < 80, f"estimated {seconds:.0f}s, not the slow pace"


def test_a_run_with_too_little_history_admits_it():
    """One evaluation cannot imply a cadence, and guessing produces a
    confident countdown to a time that means nothing."""
    summary = summarise([_row(1, 100, eval_lift_sd=0.2, elapsed_seconds=10.0)])
    assert summary["eval_every"] is None
    assert summary["next_eval_seconds"] is None


def test_the_countdown_ticks_between_polls():
    """The page re-renders only when the data changes -- every twenty updates.
    A countdown that moved only then would be wrong for the twenty minutes in
    between, which is exactly the interval it exists to cover."""
    page = render([_row(1, 1000)], "run")
    assert "tickCountdowns" in page
    assert "setInterval(tickCountdowns,1000)" in page
    assert "generated_at" in page, "no wall-clock anchor to count down from"


def test_the_generation_time_does_not_change_the_fingerprint():
    """Otherwise every poll looks like new data and the page re-renders
    constantly, which is the behaviour polling replaced."""
    from cr_sim.train.watch import render_multi
    import time as _time

    _, first = render_multi([("a", [_row(1, 1000, eval_lift_sd=0.2)], True)])
    _time.sleep(0.01)
    _, later = render_multi([("a", [_row(1, 1000, eval_lift_sd=0.2)], True)])
    assert first["generated_at"] != later["generated_at"]
    assert first["version"] == later["version"]


def test_a_note_reaches_the_page_and_changes_the_fingerprint():
    """An entry has to be able to say what it is.

    The index carries benchmarks, head-to-head comparisons and long-running
    jobs beside the training runs, and those are not legible from their
    curves the way a training run is. Without this, what a number meant lived
    only in whatever conversation produced it.
    """
    rows = [_row(1, 1000, eval_lift_sd=0.2)]
    _, plain = render_multi([("a", rows, True)])
    page, noted = render_multi([("a", rows, True)],
                               notes={"a": "measured against a random opponent"})
    assert plain["runs"]["a"]["note"] == ""
    assert noted["runs"]["a"]["note"] == "measured against a random opponent"
    assert "measured against a random opponent" in page
    # A note is data like any other: changing it must invalidate the cached
    # page, or a reader keeps the old explanation next to the new numbers.
    assert plain["version"] != noted["version"]
    assert 'data-role="note"' in page, "nowhere on the page to put it"


def test_a_note_is_read_from_the_run_directory(tmp_path):
    """config.json has always been written and only ever read for its
    timestamp. The note lives there because that is the file a run already
    writes before its first update."""
    from cr_sim.train.watch import _note_of

    run = tmp_path / "someJob"
    run.mkdir()
    assert _note_of(run) == "", "a run without a config should not raise"

    (run / "config.json").write_text(json.dumps({"note": "what this is"}),
                                     encoding="utf-8")
    assert _note_of(run) == "what this is"

    (run / "config.json").write_text("{not json", encoding="utf-8")
    assert _note_of(run) == "", "a corrupt config must not take the page down"


def test_registering_a_job_makes_an_entry_the_watcher_can_read(tmp_path):
    """A job with nothing measured yet still has to appear.

    The watcher skips a directory whose metrics file is empty, and an entry
    that shows up only once it has an answer is exactly the entry you needed
    while waiting for one -- which is how three copies of the same script once
    ran unnoticed.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from register_job import register
    from cr_sim.train.watch import _note_of, read_metrics

    out = register("waiting", note="not finished yet", runs_dir=tmp_path,
                   status="running")
    assert read_metrics(out / "metrics.jsonl"), "no row means no entry"
    assert _note_of(out) == "[running] not finished yet"

    out = register("measured", note="done", rows=[{"eval_lift_sd": 1.5}],
                   runs_dir=tmp_path)
    rows = read_metrics(out / "metrics.jsonl")
    assert rows[0]["updates"] == 1, "unnumbered rows must still plot in order"
    assert summarise(rows)["latest_lift"] == 1.5


# ------------------------------------------------- the payload a browser reads


@needs_node
def test_the_served_payload_parses_in_a_browser_not_only_in_python(tmp_path):
    """`json.dumps` and `JSON.parse` disagree about NaN, and the page loses.

    A critic that diverges writes `NaN` for explained variance. Python emits it
    as a bare token and reads it straight back, so every test here passed and
    every figure looked right from this side -- while `poll()`'s `r.json()`
    rejected on the served file, the rejection landed in its own empty
    `catch`, and the page sat frozen on the data it had loaded with,
    re-arming its timer forever. It was stuck for a day before anyone opened
    a console. Parsed with node, because Python cannot see this bug.
    """
    rows = [
        _row(1, 1000, explained_variance=float("nan"), eval_lift_sd=0.4,
             eval_win=0.5, control_win=0.26),
        _row(2, 2000, explained_variance=0.31, eval_lift_sd=float("inf")),
    ]
    page = render(rows, "run")
    blob = page.split("var DATA = ", 1)[1].split(";\n", 1)[0]
    assert "NaN" not in blob, "a bare NaN token ships in the payload"

    path = tmp_path / "data.json"
    path.write_text(blob, encoding="utf-8")
    script = (
        "var d=JSON.parse(require('fs').readFileSync(process.argv[1],'utf8'));"
        "process.stdout.write(JSON.stringify(d.runs.run.series));"
    )
    result = subprocess.run([node, "-e", script, str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    series = json.loads(result.stdout)

    # The finite reading survives at its own step; the two unusable ones are
    # simply absent, which is already how this file says "no reading here".
    assert series["explained_variance"] == [[2000, 0.31]]
    assert series["lift"] == [[1000, 0.4]], "a non-finite lift was plotted"


# ------------------------------------------------------------ the all-time view
#
# The one thing this view must never do is put two numbers in one column when
# they were measured against different opponents. The control wins 92% of its
# matches against an agent that never plays a card and 26% against a random
# one, so the two scales are nothing like each other, and `eval_opponent` --
# which would say which is which -- is on none of the 93 lift rows on disk.


def _block(name="headtohead", control=0.26, note=None):
    """A job that measured several arms under one stated set of conditions."""
    rows = [
        _row(1, 0, episodes=150, arm="beta, greedy",
             eval_lift_sd=0.55, eval_win=0.61, control_win=control),
        _row(2, 0, episodes=150, arm="alpha, greedy",
             eval_lift_sd=0.40, eval_win=0.52, control_win=control),
        _row(3, 0, episodes=150, arm="beta, sampled",
             eval_lift_sd=0.22, eval_win=0.44, control_win=control),
    ]
    if note is None:
        note = ("All three arms against the SAME random opponent on the SAME "
                "150 paired seeds.")
    return name, rows, note


def test_only_arms_from_one_stated_measurement_are_ranked_against_each_other():
    """A lift is meaningless without the opponent that produced it.

    This project lost a session to exactly that. The in-run probe faced an
    opponent that never plays a card while the paired verdicts faced a random
    one, both wrote the number to `eval_lift_sd`, and the two were compared.
    The control wins 92% of the idle matches and 26% of the random ones, so
    an all-time board that sorts every lift it can find ranks one scale
    against the other and is confidently wrong.

    The decoy here holds the largest number on the page, so an implementation
    that sorts everything fails loudly instead of reordering quietly.
    """
    name, ladder, note = _block()
    idler = [_row(1, 1000, eval_lift_sd=0.90, eval_win=0.95, control_win=0.925)]

    _page, payload = render_multi(
        [("idler", idler, False), (name, ladder, False)],
        notes={name: note}, kinds={name: "job"})
    block = payload["alltime"]["block"]

    assert block is not None, "the one comparable measurement was not found"
    assert block["job"] == name
    # Partitioned by play mode, best-first inside a partition, and computed
    # rather than copied: the fixture is in none of these orders and beta
    # appears twice under two different ways of playing.
    assert [(g["mode"], [(a["arm"], a["lift"]) for a in g["arms"]])
            for g in block["groups"]] == [
        ("greedy", [("beta, greedy", 0.55), ("alpha, greedy", 0.40)]),
        ("sampled", [("beta, sampled", 0.22)])]
    assert block["seeds"] == 150

    # The decoy is not on the ladder at all -- asserted on the lifts, because
    # arm labels are built from the `arm` field and could never carry a run
    # name whatever the implementation did.
    assert 0.90 not in [a["lift"] for a in block["arms"]], \
        "an idle-scale lift was ranked against random-scale ones"
    # And it is not simply dropped -- the largest number on the machine is
    # shown, in the one place it cannot be mistaken for the record.
    assert payload["alltime"]["demoted"]["name"] == "idler"
    assert payload["alltime"]["demoted"]["lift"] == 0.90
    # ... which is a different scale from the record's, and the payload says
    # so rather than leaving the two headline numbers side by side.
    assert payload["alltime"]["demoted"]["same_scale"] is False
    record = payload["alltime"]["record"]
    assert [(m["mode"], m["top"]["lift"]) for m in record["modes"]] == [
        ("greedy", 0.55), ("sampled", 0.22)], \
        "the record was taken from the biggest number rather than the best comparable one in each mode"


def test_an_equal_control_rate_does_not_by_itself_license_a_ranking():
    """Two jobs sit at the same control rate and only one of them is comparable.

    The sweep labels its arms and reads 0.26 exactly like the head-to-head
    does, and its own note says only the within-sweep ordering is sound
    because it changed the training targets at the same time. Ranking the two
    jobs together would compare a hard-target arm against a soft-target one
    and call the difference pass weight.
    """
    good_name, good_rows, good_note = _block("headtohead")
    sweep_name, sweep_rows, _ = _block("sweep")
    sweep_note = ("150 paired battles each, random opponent.\n"
                  "CAVEAT: the sweep runs hard targets while the baseline "
                  "used soft, so only the within-sweep ordering is sound.")

    _page, payload = render_multi(
        [(good_name, good_rows, False), (sweep_name, sweep_rows, False)],
        notes={good_name: good_note, sweep_name: sweep_note},
        kinds={good_name: "job", sweep_name: "job"})

    assert payload["alltime"]["block"]["job"] == good_name, \
        "a caveated sweep was treated as a comparable measurement"
    # It is still shown, grouped with everything else at its control rate.
    at_26 = [g for g in payload["alltime"]["groups"] if g["control"] == 0.26][0]
    assert sorted(r["name"] for r in at_26["runs"]) == ["headtohead", "sweep"]


def test_with_no_comparable_measurement_the_page_says_so_rather_than_ranking_anyway():
    """The ladder is one job directory deep, and that job can vanish.

    `register_job.register` writes with `write_text`, which truncates, so
    re-registering a job to update its status wipes the rows unless the
    caller supplies them again. The failure has to be an empty state, not a
    quiet fallback to sorting whatever lifts are left -- that fallback is the
    original bug wearing the ladder's clothes.
    """
    idler = [_row(1, 1000, eval_lift_sd=0.90, eval_win=0.95, control_win=0.925)]
    random_scale = [_row(1, 2000, eval_lift_sd=0.31, eval_win=0.4, control_win=0.26)]

    _page, payload = render_multi([("idler", idler, False),
                                   ("cloned", random_scale, False)])
    alltime = payload["alltime"]

    assert alltime["block"] is None, "ranked arms that no job vouched for"
    assert alltime["record"] is None, "named a record with nothing to compare"
    # Every reading is still on the page, grouped by what it was read against.
    assert [g["control"] for g in alltime["groups"]] == [0.925, 0.26]
    assert alltime["lift_rows"] == 2


def test_a_run_is_grouped_by_its_control_rate_and_unmatched_rates_stay_unnamed():
    """Three of the five control rates on disk match nothing that was measured.

    47 of 93 readings sit on one of them. Rounding those to the nearer of the
    two known controls would put a number on an idle scale or a random one on
    no evidence at all, so they are labelled unidentified and dimmed instead.
    """
    _page, payload = render_multi([
        ("idle-ish", [_row(1, 1000, eval_lift_sd=0.5, control_win=0.925)], False),
        ("neither", [_row(1, 1000, eval_lift_sd=0.9, control_win=0.85)], False),
        ("random-ish", [_row(1, 1000, eval_lift_sd=0.3, control_win=0.26)], False),
    ])
    groups = payload["alltime"]["groups"]

    # Descending by control rate, which is a property of the measurement --
    # so the order cannot be read as a ranking of the policies.
    assert [g["control"] for g in groups] == [0.925, 0.85, 0.26]
    assert [g["scale"]["opponent"] for g in groups] == ["idle", None, "random"]
    # The largest lift on the page sits in the group nobody has identified,
    # and gets no name from being large.
    assert groups[1]["runs"][0]["best"] == 0.9
    assert groups[1]["scale"]["opponent"] is None, "an unmatched rate was guessed"


def test_greedy_and_sampled_readings_of_one_checkpoint_are_kept_apart():
    """The same weights give two different numbers and only one gets written.

    `clone_policy.py` keeps `max(greedy, sampled)` and discards the other, so
    a metrics row holds whichever flattered the checkpoint. A fine-tuning run
    was written off as worthless on that basis: its greedy arm was flat
    because the argmax had not moved, while sampled had gone from +0.718 to
    +1.239 with the intervals cleanly apart.
    """
    name, rows, note = _block()
    _page, payload = render_multi([(name, rows, False)],
                                  notes={name: note}, kinds={name: "job"})
    modes = payload["alltime"]["modes"]

    # beta was measured both ways; alpha only greedy, so it is not a pair.
    assert [(p["weight"], p["greedy"]["lift"], p["sampled"]["lift"])
            for p in modes["pairs"]] == [("beta", 0.55, 0.22)]
    # The gap is computed, and equals neither reading.
    assert modes["pairs"][0]["gap"] == pytest.approx(0.33)
    assert modes["with_mode"] == 3 and modes["lift_rows"] == 3

    record = payload["alltime"]["record"]
    # One record per way of playing, each beside the same weights played the
    # other way. A single record across both modes is `max(greedy, sampled)`,
    # which is the operation the page condemns clone_policy for.
    greedy, sampled = record["modes"]
    assert greedy["mode"] == "greedy" and greedy["top"]["arm"] == "beta, greedy"
    assert greedy["twin"]["mode"] == "sampled" and greedy["twin"]["lift"] == 0.22, \
        "the record is not a pair, so one of the two numbers can be quoted alone"
    assert sampled["mode"] == "sampled" and sampled["top"]["arm"] == "beta, sampled"
    assert sampled["twin"]["lift"] == 0.55


def test_an_unrecognised_arm_label_reads_as_unknown_rather_than_greedy():
    """`arm` is free text with no validator anywhere.

    `register_job.py` is the one writer of a metrics file that never calls
    `check_lift_is_named`, so an arm can say anything. Defaulting an
    unlabelled reading to greedy is the assumption that hid a working
    fine-tune for a day, and it must never be the fallback.
    """
    from cr_sim.train.watch import _mode_of

    assert _mode_of("cloned, greedy") == ("cloned", "greedy")
    assert _mode_of("selfplay-1m best, sampled") == ("selfplay-1m best", "sampled")
    assert _mode_of("checkpoint 12, argmax") == ("checkpoint 12, argmax", None)
    assert _mode_of("") == ("", None)
    assert _mode_of(None) == ("", None)


def test_every_run_on_the_page_is_counted_once_in_the_census():
    """Two runs that produce the same label overwrite each other silently.

    `_label_for`'s output is the key of the runs dict, so a collision drops a
    run with no error -- which is how the page showed 22 of 29 for a week.
    Counting the census off the same dict is the only way the number on the
    page and the tabs beside it can disagree loudly rather than quietly.
    """
    runs = [(f"run{i}", [_row(1, 1000 * i)], False) for i in range(1, 6)]
    _page, payload = render_multi(runs)
    assert payload["alltime"]["census"] == len(payload["runs"]) == 5

    # The collision this docstring is about, actually built. `body["runs"]`
    # is keyed by label, so the duplicate collapses into one tab there; a
    # census counted off the input list instead says 3 above two tabs, and
    # every total the duplicate touches is counted twice. Passing the same
    # run path twice on the command line produces exactly this.
    doubled = [("dup", [_row(1, 1000, episodes=40)], False),
               ("dup", [_row(1, 1000, episodes=40)], False),
               ("solo", [_row(1, 500, episodes=10)], False)]
    _page, payload = render_multi(doubled)
    assert payload["alltime"]["census"] == len(payload["runs"]) == 2
    assert payload["order"] == ["solo", "dup"], "one run, two tabs"
    assert payload["alltime"]["ever"]["episodes"]["models"] == 50, \
        "a label that arrived twice was counted twice in the totals"
    assert payload["alltime"]["collisions"] == ["dup"], \
        "the collision was absorbed without a word"


def test_the_same_evaluation_written_twice_counts_once():
    """Two runs wrote one evaluation into two rows, under updates 1 and 2.

    The dedupe in `once()` keys on `updates`, so both rows survive it and the
    lift-row count says 93 where 91 measurements were actually taken. A page
    headlining "93 evaluations" is overstating the evidence by two.
    """
    rows = [_row(1, 4096, eval_lift_sd=1.62, eval_win=0.83, control_win=0.26),
            _row(2, 4096, eval_lift_sd=1.62, eval_win=0.83, control_win=0.26),
            _row(3, 8192, eval_lift_sd=0.71, eval_win=0.55, control_win=0.26)]
    _page, payload = render_multi([("cloned", rows, False)])
    assert payload["alltime"]["lift_rows"] == 3
    assert payload["alltime"]["distinct_evals"] == 2


def test_a_resumed_run_is_totalled_by_its_segments_not_its_largest_row():
    """These counters restart when a run is resumed into a fresh process.

    Four runs on disk are non-monotone for that reason, and the rules
    disagree by hundreds of episodes and by ten minutes of compute. Neither
    the last row nor the biggest one is the total, so the numbers here are
    chosen to equal no single row -- a fixture whose answer happens to match
    one of the wrong rules proves nothing.
    """
    resumed = [
        _row(1, 1000, episodes=100, elapsed_seconds=60),
        _row(2, 2000, episodes=200, elapsed_seconds=120),
        # Resumed here: a fresh process, so both counters restart near zero
        # while steps carry on from the checkpoint.
        _row(3, 3000, episodes=50, elapsed_seconds=30),
        _row(4, 4000, episodes=120, elapsed_seconds=75),
    ]
    plain = [_row(1, 500, episodes=40, elapsed_seconds=10)]
    # A job: a batch size parked in `steps` and a count of spreadsheet
    # comparisons parked in `episodes`.
    job = [_row(1, 512, episodes=7), _row(2, 512, episodes=9000)]

    _page, payload = render_multi([("resumed", resumed, False),
                                   ("plain", plain, False),
                                   ("bench", job, False)],
                                  kinds={"bench": "job"})
    ever = payload["alltime"]["ever"]

    # 200 + 120 across the resume, plus 40. The largest row would give 240
    # and the last row 160, so neither wrong rule can pass this.
    assert ever["episodes"]["models"] == 360
    assert ever["seconds"] == 205
    # Rule-independent, because nothing here resets them.
    assert ever["steps"]["models"] == 4500
    assert ever["updates"]["models"] == 5

    assert ever["models"] == 2 and ever["jobs"] == 1
    # 7 + 9000. A job row is not a running counter -- it is one independent
    # quantity per row -- so the rows add, where the cumulative rule read the
    # largest and reported 150 for a job that played 1,050 battles.
    assert ever["episodes"]["jobs"] == 9007, "a job leaked into the model tally"
    assert ever["steps"]["models"] == 4500, "a batch size was counted as training steps"
    assert ever["runs"] == 3


def test_the_battle_ledger_names_its_sources_and_leaves_the_estimate_out():
    """One number for "battles ever" hides that most of them are not battles.

    A job stores 8,765 engine-versus-spreadsheet comparisons in `episodes`,
    which is a table lookup rather than a match, and the in-run probe figure
    is an estimate at the evaluation default rather than anything recorded.
    Adding either into a total that says "exactly" makes the whole ledger
    unusable, so they sit below the rule with their reasons.
    """
    trained = [_row(1, 1000, episodes=120)]
    job = [
        _row(1, 0, episodes=600, what="sim vs arithmetic winner"),
        _row(2, 0, episodes=8765, what="engine vs community sheet"),
    ]
    # No episodes of its own, so the training figure stays readable here.
    evaluated = [_row(1, 1000, episodes=0, eval_lift_sd=0.3, eval_win=0.4,
                      control_win=0.26)]
    extras = {
        "soak": {"matches": 10000, "mean_ticks": 1770.15},
        "verdicts": {"cloned": {"episodes": 150, "greedy": {"lift": 1.6},
                                "sampled": {"lift": 0.7}}},
    }

    _page, payload = render_multi([("trained", trained, False),
                                   ("gate", job, False),
                                   ("evaluated", evaluated, False)],
                                  kinds={"gate": "job"}, extras=extras)
    battles = payload["alltime"]["ever"]["battles"]
    counted = {c["what"]: c["n"] for c in battles["counted"]}

    assert counted["training episodes"] == 120
    assert counted["engine soak matches"] == 10000
    # Both modes of the paired verdict, because they are separate battles.
    assert counted["paired verdict battles"] == 300
    assert counted["sim vs arithmetic winner"] == 600
    assert battles["total"] == 120 + 10000 + 300 + 600 == 11020

    # One evaluation and one run that evaluated, at the 40-episode default.
    assert battles["estimated"]["n"] == 80
    # Compares the estimate against what the total is actually made of. The
    # old form compared 80 against 11,020 and could never fire.
    assert battles["estimated"]["what"] not in counted, "the estimate was added in"
    assert battles["total"] == sum(c["n"] for c in battles["counted"])
    assert battles["excluded"]["n"] == 8765
    assert battles["excluded"]["items"][0]["what"] == "engine vs community sheet"
    # Every line says how it was counted, or it does not belong on the page.
    lines = list(battles["counted"]) + [battles["estimated"], battles["excluded"]]
    assert all(str(line.get("rule", "")).strip() for line in lines),         "a figure shipped with no counting rule beside it"


def test_a_verdict_is_read_by_its_fields_and_an_unknown_shape_says_so():
    """Four verdict files exist and three of them disagree about the lift.

    `runs/cloned/verdict.json` carries a flat lift beside its greedy and
    sampled sub-objects, and that flat block is a byte-identical copy of the
    greedy one -- so reading it hands you greedy without telling you and
    hides a sampled reading 2.3x lower. Two more evaluation processes are
    writing verdicts right now, so a fifth shape is a matter of time and has
    to be reported rather than guessed at.
    """
    extras = {"verdicts": {
        "cloned": {"episodes": 150,
                   "greedy": {"lift": 1.623, "ci_low": 1.39, "ci_high": 1.86},
                   "sampled": {"lift": 0.709, "ci_low": 0.46, "ci_high": 0.96},
                   "lift": 1.623, "ci_low": 1.39, "ci_high": 1.86},
        "expert": {"episodes": 40, "lift": 2.716, "ci_low": 2.37, "ci_high": 3.06},
        "paired": [{"name": "_diag", "mode": "greedy", "lift": 1.623,
                    "episodes": 150, "eval_opponent": "random"},
                   {"name": "_diag", "mode": "sampled", "lift": 0.734,
                    "ci_low": -0.15, "ci_high": 0.35, "episodes": 150,
                    "eval_opponent": "random"}],
        "future": {"arms": [{"score": 3}]},
    }}
    _page, payload = render_multi([("cloned", [_row(1, 1000)], False)],
                                  extras=extras)
    alltime = payload["alltime"]

    assert alltime["unreadable"] == ["future"], \
        "a shape nobody has seen was read as though it were understood"

    recorded = alltime["modes"]["recorded"]
    assert [r["weight"] for r in recorded] == ["cloned", "baseline clone"], \
        "the paired verdict was skipped, so the flat mirror's hidden half stays hidden"
    # The paired file: both halves survive, so both are shown even though the
    # flat block beside them would have handed a reader greedy alone.
    paired = recorded[0]
    assert (paired["greedy"]["lift"], paired["sampled"]["lift"]) == (1.623, 0.709)
    assert paired["gap"] == pytest.approx(1.623 - 0.709)
    assert paired["opponent"] is None, "an opponent was invented for a file that names none"
    # Renamed for the reader: "_diag" is a scratch directory, not an arm.
    arms = recorded[1]
    assert arms["opponent"] == "random", \
        "the one file that records its opponent was not read for it"
    assert arms["gap"] == pytest.approx(1.623 - 0.734)
    assert arms["straddles_zero"] is True

    # 150 greedy + 150 sampled for the clone, 40 for the expert, 300 for the
    # arms file -- and nothing at all from the shape that was not recognised.
    counted = {c["what"]: c["n"] for c in alltime["ever"]["battles"]["counted"]}
    assert counted["paired verdict battles"] == 300 + 40 + 300


def test_no_interval_is_invented_for_a_row_that_does_not_carry_one():
    """No lift row on disk has ever carried a confidence interval.

    All 93 of them lack `eval_ci_low` and `eval_ci_high`; the intervals live
    in the verdict files and in note prose. Drawing a whisker on a ladder row
    means inventing one, and reading one out of prose with a regular
    expression means inventing one more carefully. The note is printed whole
    above the ladder instead, which loses nothing.
    """
    name, rows, note = _block()
    # The expert's verdict has to belong to a run on the page, or nothing
    # reaches it and the fixture proves nothing about intervals at all.
    expert = [_row(1, 0, episodes=40, eval_lift_sd=2.7, eval_win=1.0,
                   control_win=0.925)]
    _page, payload = render_multi([("expert", expert, False), (name, rows, False)],
                                  notes={name: note}, kinds={name: "job"},
                                  extras={"verdicts": {
                                      "expert": {"episodes": 40, "lift": 2.7,
                                                 "ci_low": 2.4, "ci_high": 3.1}}})
    alltime = payload["alltime"]

    for arm in alltime["block"]["arms"]:
        assert "ci" not in arm, "a ladder row grew an interval from nowhere"
    for pair in alltime["modes"]["pairs"]:
        assert pair["greedy"]["ci"] is None and pair["sampled"]["ci"] is None
    # Where an interval is a field of an object it is read, and only there.
    assert alltime["demoted"]["name"] == "expert"
    assert alltime["demoted"]["ci"] == [2.4, 3.1]
    # The note is the interval channel, and it ships verbatim.
    assert alltime["block"]["note"] == note


def test_the_all_time_aggregate_is_inside_the_payload_fingerprint():
    """A stale total beside a fresh curve on the same screen is worse than no total.

    The page re-renders only when `version` changes. Anything attached after
    the hash ships in the HTML and is invisible to it, so the aggregates
    would sit at whatever they were when the tab was opened while the run
    tabs beside them kept moving.
    """
    first = [_row(1, 1000, episodes=50)]
    second = [_row(1, 1000, episodes=50), _row(2, 2000, episodes=90)]

    _p, before = render_multi([("run", first, True)])
    _p, same = render_multi([("run", first, True)])
    _p, after = render_multi([("run", second, True)])

    assert before["version"] == same["version"], "the fingerprint drifted with no data change"
    assert before["alltime"]["ever"]["episodes"]["models"] == 50
    assert after["alltime"]["ever"]["episodes"]["models"] == 90
    assert before["version"] != after["version"], \
        "the aggregate moved and the page would never have redrawn"

    # The rows moving is not the test. Changing the rows moves `version`
    # through `body["runs"]` wherever the aggregate is attached, so an
    # aggregate computed after the hash passes on the fixture above. This
    # moves something only the aggregate can see: the soak summary, which has
    # no metrics file and therefore no run tab of its own.
    extras = {"soak": {"matches": 10000, "mean_ticks": 1770.15}}
    louder = {"soak": {"matches": 99999, "mean_ticks": 1770.15}}
    _p, quiet = render_multi([("run", first, True)], extras=extras)
    _p, loud = render_multi([("run", first, True)], extras=louder)

    assert (quiet["alltime"]["ever"]["battles"]["total"]
            != loud["alltime"]["ever"]["battles"]["total"])
    assert quiet["version"] != loud["version"], \
        "the ledger moved by 90,000 battles under an unchanged fingerprint"
    assert quiet["runs"] == loud["runs"], \
        "the fixture moved a run as well, so this proves nothing about the aggregate"


def _multi_body(runs, **kwargs):
    """The page's function definitions for a multi-run payload."""
    page = render_multi(runs, **kwargs)[0]
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    return script.split("(function start()", 1)[0]


def _call_multi(runs, expression, stored=None, **kwargs):
    """Evaluate `expression` against a multi-run page, with real localStorage.

    The stub goes in front of the page's own code rather than after it,
    because the view the page opens in is read out of storage as the script
    loads -- put the stub after and every restore test passes without the
    restore working.
    """
    stub = ("var __store=" + json.dumps(stored or {}) + ";"
            "var localStorage={setItem:function(k,v){__store[k]=String(v);},"
            "getItem:function(k){return Object.prototype.hasOwnProperty.call"
            "(__store,k)?__store[k]:null;}};")
    return _node(stub + _multi_body(runs, **kwargs) + chr(10)
                 + "process.stdout.write(String(" + expression + "));")


def test_the_way_into_the_all_time_view_is_never_hidden_by_a_media_query():
    """The page exists to be read on a phone, and this button is the only door.

    `--serve` prints the machine's addresses so a phone on the same wifi can
    watch a run, and the split and expand buttons are hidden below 700px on
    purpose -- the panes collapse to one column at 900px anyway, so those two
    control nothing there. Putting the all-time button in that same class
    made the entire view unreachable at 390px while looking perfectly fine on
    the laptop it was built on.
    """
    page = render([_row(1, 1000)], "run")
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]

    assert ".view-toggle{" in style, "the button has no styling of its own"
    assert 'class="view-toggle"' in page, "the button does not use it"

    # Every @media block, brace-matched, so a nested rule cannot hide one.
    hidden = []
    for start in [m.start() for m in re.finditer(r"@media", style)]:
        depth, i = 0, style.index("{", start)
        for j in range(i, len(style)):
            if style[j] == "{":
                depth += 1
            elif style[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        block = style[start:j + 1]
        if "display:none" in block.replace(" ", ""):
            hidden.append(block)

    assert hidden, "no media query hides anything, so this test proves nothing"
    # By class, by id, and by container -- matching the class name alone let
    # three plausible ways of making the button unreachable at 390px through:
    # hiding `#viewall`, hiding `.tabbar` around it, or `visibility:hidden`
    # rather than `display:none`.
    for start in [m.start() for m in re.finditer(r"@media", style)]:
        depth, i = 0, style.index("{", start)
        for j in range(i, len(style)):
            if style[j] == "{":
                depth += 1
            elif style[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        block = style[start:j + 1].replace(" ", "")
        if not ("display:none" in block or "visibility:hidden" in block
                or "opacity:0" in block):
            continue
        for target in (".view-toggle", "#viewall", ".tabbar"):
            assert target not in block, \
                f"{target} is hidden at some width, and the all-time view goes with it"


@needs_node
def test_the_ladder_draws_the_arms_it_ranked_and_not_the_one_it_refused_to():
    """The excluded reading is the largest number on the page.

    An implementation that sorted every lift it could find would put it at
    the top of the ladder, and this asserts on the markup rather than the
    payload so that a board computed correctly and drawn from the wrong list
    still fails. The decoy's control rate says it faced an opponent that
    never plays a card, where the ladder's arms faced a random one.
    """
    name, ladder, note = _block()
    idler = [_row(1, 1000, eval_lift_sd=0.90, eval_win=0.95, control_win=0.925)]
    runs = [("idler", idler, False), (name, ladder, False)]
    kw = {"notes": {name: note}, "kinds": {name: "job"}}

    drawn = _call_multi(runs, "ladderMarkup(DATA.alltime)", **kw)
    assert "+0.550" in drawn, "the ranking is computed but never shown"
    assert "+0.400" in drawn and "+0.220" in drawn
    assert "+0.900" not in drawn, \
        "an idle-scale lift was drawn onto a random-scale ladder"
    # The provenance claim the header makes is checkable against its source,
    # because the note that licensed the ranking is quoted whole beside it.
    assert note.replace('"', "&quot;") in drawn, \
        "the ladder asserts conditions without showing what stated them"

    # It is not suppressed either -- it is shown once, where it cannot be
    # mistaken for the best result.
    demoted = _call_multi(runs, "demotedMarkup(DATA.alltime)", **kw)
    assert "+0.900" in demoted and "idler" in demoted


@needs_node
def test_a_lift_and_the_scale_it_was_read_on_are_one_element():
    """A number screenshotted without its scale is how this went wrong before.

    An idle-scale reading and a random-scale one differ by a factor the page
    cannot recover, so the chip has to be inside the same element as the
    figure -- not beside it in a neighbouring cell that a crop can remove.
    Readings whose control matches nothing measured are dimmed and say so
    rather than being given the nearer of the two known names.
    """
    name, ladder, note = _block()
    unknown = [_row(1, 1000, eval_lift_sd=0.61, eval_win=0.7, control_win=0.85)]
    # A reading at the measured idle rate, so the naming machinery is live in
    # the same payload: "vs idle" is absent below because 0.85 was withheld a
    # name, not because nothing on this page could ever have earned one.
    idler = [_row(1, 1000, eval_lift_sd=0.44, eval_win=0.6, control_win=0.925)]
    runs = [("mystery", unknown, False), ("idler", idler, False),
            (name, ladder, False)]
    kw = {"notes": {name: note}, "kinds": {name: "job"}}

    drawn = _call_multi(runs, "ladderMarkup(DATA.alltime)", **kw)
    assert '+0.550<span class="chip good">vs random (stated)</span>' in drawn, \
        "the number and its scale are not in the same element"

    groups = _call_multi(runs, "groupsMarkup(DATA.alltime)", **kw)
    assert '+0.610<span class="chip warn">scale unidentified (control 0.85)</span>' in groups
    assert '+0.440<span class="chip dim">vs idle (inferred)</span>' in groups, \
        "a rate that does match a measured control was not named"
    assert "vs idle" not in groups.split("+0.610", 1)[0].rsplit("<details", 1)[-1], \
        "an unmatched control rate was given a name"


@needs_node
def test_with_nothing_comparable_the_view_says_so_instead_of_ranking():
    """Silence is the correct output, and a sorted list is not.

    The ladder is one job directory deep and `register_job` writes with
    `write_text`, so re-registering that job to update its status empties it.
    The page then has two readings on two different scales and no licence to
    order them, and must say that rather than fall back to whichever sort it
    can still manage.
    """
    runs = [("idler", [_row(1, 1000, eval_lift_sd=0.90, control_win=0.925)], False),
            ("cloned", [_row(1, 2000, eval_lift_sd=0.31, control_win=0.26)], False)]

    out = _call_multi(runs, "allTimeMarkup(DATA.alltime)")
    assert "No comparable ranking on disk" in out
    # The bigger number must not have been quietly crowned instead.
    assert out.index("No comparable ranking on disk") < out.index("+0.900")

    # And the ladder section itself has to be the empty state. Both checks
    # above are satisfiable from elsewhere on the page -- the phrase is also
    # emitted by the record panel, and the demoted strip is always drawn
    # before the ladder -- so a fallback that sorted every group's runs into
    # one cross-scale ranking passed them both while rendering exactly the
    # bug this test is named for.
    ladder = out.split("The comparable ladder</h2>", 1)[1].split("<h2>", 1)[0]
    assert "No comparable ranking on disk" in ladder
    assert "+0.900" not in ladder and "+0.310" not in ladder, \
        "an idle-scale reading was ranked against a random-scale one"


@needs_node
def test_the_all_time_view_is_the_one_the_page_opens_in_next_time():
    """Somebody reading the totals wants them again on the next look.

    The two existing toggles both persist for the same reason, and this one
    matters more: it is a whole view rather than a layout, and losing it on
    every poll would send a reader back to a single run's curves without
    their having asked to go.
    """
    runs = [("run", [_row(1, 1000)], False)]
    assert _call_multi(runs, "view") == "runs", "opens somewhere nobody chose"
    assert _call_multi(runs, "view", stored={"crsim-view": "alltime"}) == "alltime", \
        "the chosen view was not restored"
    # And an unrecognised stored value falls back rather than blanking the page.
    assert _call_multi(runs, "view", stored={"crsim-view": "sideways"}) == "runs"


@needs_node
def test_the_battle_ledger_on_the_page_adds_up_to_what_it_prints():
    """A total nobody can check is a total nobody should trust.

    Every source is printed as its own line with the rule that produced it,
    so the sum is verifiable by eye -- which matters because two of the
    largest blocks of battles ever run here are invisible to the run list,
    and one plausible-looking figure of 9,071 is not battles at all.
    """
    runs = [("trained", [_row(1, 1000, episodes=120)], False),
            ("gate", [_row(1, 0, episodes=600, what="sim vs arithmetic winner")], False)]
    kw = {"kinds": {"gate": "job"},
          "extras": {"soak": {"matches": 10000, "mean_ticks": 1770.15,
                              "reasons": [["tick limit", 9751]], "anomalies": []}}}

    out = _call_multi(runs, "everMarkup(DATA.alltime)", **kw)
    for figure in ("120", "10,000", "600", "10,720"):
        assert figure in out, f"{figure} is part of the total but never shown"

    # Each figure has to be its own printed line item, and the printed line
    # items have to add up to the printed total. Substring checks alone pass
    # when the ledger stops printing its rows: 10,000 also appears in the
    # soak panel, 600 in the excluded row, and the total in its own cell --
    # so the "verifiable by eye" property was never being tested.
    counted = out.split("Below the rule", 1)[0]
    items = re.findall(r"<td>([^<]*)<div class=\"caption\">[^<]*"
                       r"</div></td><td class=\"n\">([\d,]+)</td>", counted)
    named = {what: int(n.replace(",", "")) for what, n in items}
    assert named == {"training episodes": 120, "engine soak matches": 10000,
                     "sim vs arithmetic winner": 600}, \
        "the ledger printed a total with its own line items missing"
    assert sum(named.values()) == 10720
    assert ">10,720<" in out, "the total nothing sums to"
    # The soak run has no metrics file, so nothing else on the page knows it
    # happened -- which is exactly why it is counted here.
    assert "1770.15 ticks" in out


@needs_node
def test_draw_puts_the_all_time_view_where_the_per_run_panes_were():
    """The view has to be drawn from inside `draw`, and nothing else will do.

    `draw` clears CHARTS and overwrites the panes container unconditionally on
    every poll. A view rendered anywhere else has its markup thrown away
    fifteen seconds later with no error and no clue why, which is how the
    expand button's first attempt was lost. This runs the real function
    against a stub DOM instead of trusting that the branch exists.
    """
    name, ladder, note = _block()
    runs = [("run", [_row(1, 1000, episodes=40)], True), (name, ladder, False)]
    stub = """
var __store={'crsim-view':'alltime'};
var localStorage={setItem:function(k,v){__store[k]=String(v);},
  getItem:function(k){return Object.prototype.hasOwnProperty.call(__store,k)?__store[k]:null;}};
function El(){this.innerHTML='';this.textContent='';this.className='';this.hidden=null;
  this.style={};this.dataset={};this.attrs={};
  this.setAttribute=function(k,v){this.attrs[k]=v;};
  this.addEventListener=function(){};
  this.classList={toggle:function(){}};
  this.querySelector=function(){return null;};}
var NODES={};
['panes','alltime','title','tabs','expand','viewall','foot','pulse','stamp','split','bell']
  .forEach(function(id){NODES[id]=new El();});
var document={getElementById:function(id){return NODES[id];},
              querySelectorAll:function(){return [];}};
"""
    harness = """
draw();
process.stdout.write(JSON.stringify({
  hidden:NODES.alltime.hidden,
  panes:NODES.panes.innerHTML,
  panesDisplay:NODES.panes.style.display,
  title:NODES.title.textContent,
  pressed:NODES.viewall.attrs['aria-pressed'],
  foot:NODES.foot.textContent,
  markup:NODES.alltime.innerHTML.length,
  hasRecord:NODES.alltime.innerHTML.indexOf('+0.550')>=0,
  charts:CHARTS.length
}));
"""
    out = json.loads(_node(stub + _multi_body(runs, notes={name: note},
                                              kinds={name: "job"}) + harness))

    assert out["hidden"] is False, "the container was never revealed"
    assert out["markup"] > 0 and out["hasRecord"], "the view drew nothing"
    assert out["panes"] == "" and out["panesDisplay"] == "none", \
        "the per-run panes are still in the document underneath"
    assert out["pressed"] == "true", "the toggle does not show which view is on"
    assert "all time" in out["title"]
    # No chart registered: chart() plots x as steps, and every arm here has
    # steps=0, so a curve would stack the whole ranking on one vertical line.
    assert out["charts"] == 0, "a chart was registered on a view with no time axis"
    # The tab rail is still painted, because tapping a run is the way back.
    assert "2 runs" in out["foot"]


# ------------------------------------------- the all-time view: the pictures
#
# Five hand-rolled SVGs, and one thing they can all do wrong: put two numbers
# on one axis that were not read against the same thing. Each is tested for the
# refusal as well as for the drawing, and every assertion below is made against
# the numbers that actually reached the geometry -- circle positions, path
# coordinates, bar percentages -- rather than against a literal from the
# template. Every branch's text is in the page source whatever the data says,
# which is how a test here once passed for a year checking nothing.


def _tags(markup, name):
    """Every `<name ...>` in some markup, as a list of attribute dicts."""
    return [dict(re.findall(r'([a-zA-Z_:][\w:.-]*)="([^"]*)"', match.group(1)))
            for match in re.finditer("<" + name + r"\b([^>]*)>", markup)]


def _texts(markup):
    """The contents of every `<text>` drawn."""
    return re.findall(r"<text[^>]*>([^<]*)</text>", markup)


def _path_points(d):
    """The (x, y) pairs of an SVG path's `d`, as floats."""
    return [(float(x), float(y))
            for x, y in re.findall(r"[ML]([-\d.]+) ([-\d.]+)", d)]


def _percent(style, key):
    """A `left:` or `width:` out of a ladder bar's inline style, as a float."""
    return float(re.search(key + r":([-\d.]+)%", style).group(1))


def _record(name, family, arm, mode, lift, low=None, high=None,
            episodes=150, opponent="random", win=0.5):
    out = {"name": name,
           "checkpoint": "runs/" + family + "/" + arm + "/cloned.pt",
           "mode": mode, "lift": lift, "win": win, "episodes": episodes,
           "eval_opponent": opponent}
    if low is not None:
        out["ci_low"], out["ci_high"] = low, high
    return out


#: Three checkpoints measured both ways against one recorded opponent over one
#: battle count. `level` reads the same number both ways, so it must land
#: exactly on the diagonal; `collapsed` reads negative greedy and positive
#: sampled, which is the sign flip the whole picture exists for.
_BOTH_WAYS = [
    _record("healthy", "fam", "healthy", "greedy", 1.60, 1.40, 1.80),
    _record("healthy", "fam", "healthy", "sampled", 0.70, 0.50, 0.90),
    _record("level", "fam", "level", "greedy", 0.40, 0.20, 0.60),
    _record("level", "fam", "level", "sampled", 0.40, 0.20, 0.60),
    _record("collapsed", "fam", "collapsed", "greedy", -1.90, -2.10, -1.70),
    _record("collapsed", "fam", "collapsed", "sampled", 0.05, -0.15, 0.25),
]

#: The plain page these verdict-driven pictures are drawn over: one run, one
#: reading, nothing that could produce a block or a ladder of its own.
_QUIET = [("run", [_row(1, 1000, eval_lift_sd=0.1, eval_win=0.4,
                        control_win=0.26, eval_opponent="random")], False)]


def test_the_greedy_sampled_scatter_carries_one_point_per_recorded_checkpoint():
    """The modes table's own data, with the two arms as the two axes.

    Both fields the drawing needs are new: without `episodes` the picture
    cannot tell one population from another, and without `scale` it cannot put
    a provenance chip on itself.
    """
    _page, payload = render_multi(_QUIET, extras={"verdicts": {"v": _BOTH_WAYS}})
    recorded = payload["alltime"]["modes"]["recorded"]

    assert [(r["weight"], r["episodes"], r["scale"]["opponent"],
             r["scale"]["source"], r["flips"]) for r in recorded] == [
        ("healthy", 150, "random", "recorded", False),
        ("level", 150, "random", "recorded", False),
        ("collapsed", 150, "random", "recorded", True)]


@needs_node
def test_the_greedy_sampled_scatter_puts_both_arms_on_one_domain():
    """The 45 degree line is the whole point, and two domains would break it.

    A checkpoint that read the same number both ways must land exactly on the
    diagonal. If the axes were scaled independently -- greedy spans 3.5 here
    and sampled 0.65 -- it would sit far off it, and a reader would take the
    distance from the diagonal for a gap that was never measured.
    """
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"v": _BOTH_WAYS}})
    dots = sorted((float(t["cx"]), float(t["cy"]), t["fill"])
                  for t in _tags(out, "circle") if t.get("r") == "3.2")
    assert len(dots) == 3, "one point per checkpoint measured both ways"
    collapsed, level, healthy = dots        # greedy -1.90 < 0.40 < 1.60

    # padL is 44 and the plot box is square 328 by 328, so a point whose two
    # readings are equal satisfies cx - padL == (h - padB) - cy. That identity
    # holds only under one shared domain.
    assert abs((level[0] - 44) - (356 - level[1])) < 0.2, \
        "the two axes were given separate domains, so the diagonal is a lie"
    assert (healthy[0] - 44) - (356 - healthy[1]) > 5, \
        "a checkpoint whose greedy beats its sampled is not above the diagonal"
    assert (collapsed[0] - 44) - (356 - collapsed[1]) < -5

    # The sign flip is the one thing this picture is for, and it is coloured.
    assert collapsed[2] == "#A2352C" and healthy[2] == "#2E86AB"
    assert level[2] == "#2E86AB"

    # The chip is the gate's answer read back to the reader, and nothing was
    # checking it: every one of these records was measured against a recorded
    # random opponent over 150 battles, so naming any other opponent or any
    # other battle count here is the project's signature failure printed as a
    # label. The negative half is what makes this non-vacuous.
    assert "vs random (recorded)" in out and "idle" not in out
    assert "150 battles" in out and "40 battles" not in out

    # One marker per sweep, and the sweep on every label -- not only where two
    # arms happen to share a name. These three came out of one sweep, so they
    # share a marker and the caption says which.
    assert "<b>fam</b> (circle): healthy, level, collapsed" in out
    assert "healthy\u00b7fam" in out and "collapsed\u00b7fam" in out

    # The intervals are drawn from the recorded ones and from nothing else.
    # Asserted as a ratio against the distance between two points, so it holds
    # whatever the padding rule is: healthy's greedy interval is 0.40 wide and
    # its reading sits 3.50 from collapsed's.
    bars = [t for t in _tags(out, "line") if t.get("opacity") == ".45"]
    across = [t for t in bars if t["y1"] == t["y2"]]
    assert len(bars) == 6 and len(across) == 3, "the recorded intervals are not drawn"
    width = max(abs(float(t["x2"]) - float(t["x1"])) for t in across)
    assert abs(width / (healthy[0] - collapsed[0]) - 0.40 / 3.50) < 0.01


@needs_node
def test_the_greedy_sampled_scatter_counts_a_second_population_rather_than_merging_it():
    """150 battles and 300 battles are two precisions, not one scatter.

    The decoy holds the largest numbers on the page, so an implementation that
    plots every recorded pair fails on the axis label rather than quietly
    stretching the domain around readings that do not belong on it.
    """
    other = [_record("big", "other", "big", "greedy", 8.0, 7.5, 8.5, episodes=300),
             _record("big", "other", "big", "sampled", 7.0, 6.5, 7.5, episodes=300),
             _record("huge", "other", "huge", "greedy", 9.0, 8.5, 9.5, episodes=300),
             _record("huge", "other", "huge", "sampled", 7.5, 7.0, 8.0, episodes=300)]
    # And one checkpoint measured both ways that names no opponent at all --
    # the pre-fix shape of every verdict on this project. Counted in the
    # caption, exactly as the second population is, because the two halves of
    # "counted, never merged" are one contract.
    blind = [dict(r) for r in _BOTH_WAYS[:2]]
    for record in blind:
        record["name"] = record["checkpoint"] = "runs/fam/blind/cloned.pt"
        record.pop("eval_opponent")
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"a": _BOTH_WAYS, "b": other,
                                           "c": blind}})

    dots = [t for t in _tags(out, "circle") if t.get("r") == "3.2"]
    assert len(dots) == 3, "two battle counts were plotted on one axis"
    # The domain is the drawn population's own: -2.10 and 1.80, each padded by
    # 8% of the 3.90 between them. A merged domain would run to about 9.4.
    labels = _texts(out)
    assert "2.11" in labels and "-2.41" in labels, \
        "the axis stretched around a population that is not drawn"
    assert "Another 2 checkpoints faced random over 300 battles" in out
    assert "1 more checkpoint records no opponent at all and is not drawn." in out


@needs_node
def test_one_checkpoint_measured_both_ways_is_still_drawn():
    """A scatter of one point is one checkpoint at its two readings.

    That is honest in a way a line of one point is not, so unlike chart() this
    draws it -- with both zero lines and the diagonal, and a caption saying it
    is a gap rather than a pattern.
    """
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"v": _BOTH_WAYS[:2]}})

    assert len([t for t in _tags(out, "circle") if t.get("r") == "3.2"]) == 1
    assert len([t for t in _tags(out, "line") if t.get("class") == "zero"]) == 3, \
        "the diagonal and the two zero lines are what make one point readable"
    assert len([t for t in _tags(out, "line") if t.get("opacity") == ".45"]) == 2
    assert "not yet a pattern" in out
    assert "no verdict records both ways" not in out


@needs_node
def test_two_sweeps_on_the_scatter_are_two_markers_and_never_one():
    """The scatter's only guard is the opponent and the battle count.

    That is a control-based guard, and it cannot separate two checkpoints out
    of different sweeps that state the same configuration, faced the same
    recorded opponent over the same battles and read opposite signs three
    standard deviations apart -- their controls genuinely match. The sweep is
    the only thing on disk that records what was held fixed, so it is drawn
    into the mark and printed on every label, and a checkpoint out of no
    sweep is marked as belonging to none.
    """
    two = _BOTH_WAYS + [
        _record("v1", "obsablate", "v1", "greedy", -1.60, -1.86, -1.34),
        _record("v1", "obsablate", "v1", "sampled", 0.12, -0.10, 0.34)]
    solo = [dict(record) for record in _BOTH_WAYS[:2]]
    for record in solo:
        record["name"] = "baseline"
        record["checkpoint"] = "runs/_diag/cloned.pt"

    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"a": two, "b": solo}})
    circles = [t for t in _tags(out, "circle") if t.get("r") == "3.2"]
    filled = [t for t in circles if t.get("fill") != "none"]
    rings = [t for t in circles if t.get("fill") == "none"]

    assert len(filled) == 3, "the largest sweep keeps the filled circle"
    assert len(_tags(out, "rect")) == 1, "the second sweep shares the first's mark"
    assert len(rings) == 1, "a checkpoint out of no sweep is not marked as one"
    assert "<b>fam</b> (circle): healthy, level, collapsed" in out
    assert "<b>obsablate</b> (square): v1" in out
    assert "<b>no sweep</b> (ring): baseline" in out
    assert "comparable inside one sweep only" in out


@needs_node
def test_the_scatter_says_how_many_points_it_could_not_label():
    """Label stacking in fixed steps has no lower bound.

    The live page already stacks every point into one unbroken column; a few
    more checkpoints and the lowest names fall below the plot and then out of
    the viewBox, where they are not drawn at all. A label dropped without
    saying so is the failure this page otherwise refuses, and the picture
    gains one point per new sweep checkpoint.
    """
    crowd = []
    for i in range(30):
        arm = "w%02d" % i
        crowd += [_record(arm, "big", arm, "greedy", 1.0 + i * 0.001),
                  _record(arm, "big", arm, "sampled", 0.2 + i * 0.0001)]
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"v": crowd}})

    drawn = [t for t in _tags(out, "circle") if t.get("r") == "3.2"]
    assert len(drawn) == 30, "the picture did not draw its input"
    labels = re.findall(r'<text class="lbl-s" x="[\d.]+" y="([\d.]+)" fill=', out)
    assert labels, "nothing was labelled at all"
    assert max(float(y) for y in labels) <= 354, \
        "a label was drawn below the plot floor, or outside the viewBox"
    missing = 30 - len(labels)
    assert missing > 0, "the fixture no longer crowds the label column"
    assert ("%d points are drawn without a name" % missing) in out
    # Every name is still on the page at reading size, in the sweep list.
    assert "<b>big</b> (circle): w00, w01" in out
    assert "w29" in out


@needs_node
def test_a_page_with_no_paired_verdict_draws_no_scatter():
    """Nothing measured both ways is an empty card, not an empty axis.

    And so is a checkpoint measured both ways that records no opponent: those
    are the pre-fix verdicts, and a reading on an unknown scale is not a
    smaller finding than no reading, it is the one thing this axis may not
    carry. A page with no verdicts at all proves only that `modes.recorded`
    was empty, so it never reached the filter's opponent clause.
    """
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)")
    assert "no verdict records both ways of playing" in out
    assert "<svg" not in out

    blind = [dict(r) for r in _BOTH_WAYS[:2]]
    for record in blind:
        record.pop("eval_opponent")
    out = _call_multi(_QUIET, "gvsMarkup(DATA.alltime.modes)",
                      extras={"verdicts": {"v": blind}})
    assert "records no opponent at all, so it is not drawn on an axis" in out
    assert "<svg" not in out and "<circle" not in out


# ---------------------------------------------------------- the matched pair


def _ab_run(readings, opponent="random", control=0.2, episodes=40):
    """Rows for one arm of a matched pair, from (update, steps, lift) triples."""
    return [_row(update, steps, eval_lift_sd=lift, eval_win=0.5,
                 control_win=control, eval_opponent=opponent,
                 eval_episodes=episodes)
            for update, steps, lift in readings]


#: Four shared updates, and a resume that replays two of them with different
#: numbers. The step counter falls at the resume, which is what tells a replay
#: apart from the same row written twice.
_AB_A = _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4), (12, 12000, 1.6),
                 (9, 4500, 0.9), (12, 6000, 1.1), (15, 7500, 1.9)])
_AB_B = _ab_run([(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8), (12, 12000, 0.9)])
_AB_RUNS = [("arm-a", _AB_A, True), ("arm-b", _AB_B, False)]
_AB_CONFIGS = {"arm-a": {"head": "factored", "note": "a", "lr": 3e-4, "seed": 1},
               "arm-b": {"head": "flat", "note": "b", "lr": 3e-4, "seed": 1}}


def test_the_matched_pair_is_paired_at_equal_update_and_not_at_equal_time():
    """The one clean A/B here, and every number the panel prints about it.

    Pairing on the clock would subtract one run's first hour from the other's
    third; pairing on the step count would fold the resumed segment onto the
    readings it replayed. The update index is the only join key both runs
    share, and the resume is carried rather than collapsed.
    """
    _page, payload = render_multi(_AB_RUNS, configs=_AB_CONFIGS)
    ab = payload["alltime"]["ab"]

    assert ab is not None, "two runs matching on every clause were not paired"
    assert (ab["a"]["name"], ab["b"]["name"]) == ("arm-a", "arm-b")
    assert ab["a"]["live"] is True and ab["b"]["live"] is False
    assert [p["update"] for p in ab["points"]] == [3, 6, 9, 12]
    assert [round(p["d"], 6) for p in ab["points"]] == [0.4, 0.5, 0.6, 0.7]
    # The line follows the first write; the replay is carried beside it.
    assert [(p["update"], p["replay_a"]) for p in ab["points"]] == [
        (3, None), (6, None), (9, 0.9), (12, 1.1)]
    assert [p["replay_b"] for p in ab["points"]] == [None, None, None, None]
    assert ab["replayed"] == [9, 12]

    assert round(ab["mean"], 6) == 0.55
    assert round(ab["sd"], 6) == 0.129099
    assert round(ab["se"], 6) == 0.06455
    assert (ab["n"], ab["wins"]) == (4, 4)
    # And the headline moves when the replayed readings are taken instead, so
    # both are computed rather than one being picked silently.
    assert round(ab["alt"]["mean"], 6) == 0.3 and ab["alt"]["wins"] == 4

    # Key names, never values: one of them on disk is an absolute path, and
    # nothing in the payload may vary with the machine.
    assert ab["diff_keys"] == ["head", "note"] and ab["config_keys"] == 4
    assert "factored" not in json.dumps(ab), "a config value reached the payload"

    assert (ab["a_last_update"], ab["b_last_update"]) == (15, 12)
    assert ab["tail"] == [[15, 1.9]], "the longer run's readings past the rule"
    assert ab["tail_b"] == [], "arm-b stopped first, so it has nothing past it"
    assert (ab["a_readings"], ab["b_readings"]) == (7, 4)
    assert ab["episodes"] == 40 and ab["scale"]["opponent"] == "random"
    assert ab["mode"] is None and ab["rule"]
    # Neither run recorded how it played, and two silences are not a match.
    assert ab["mode_recorded"] is False
    assert ab["same_run"] == []


#: The same pair with one update where arm-b leads. Every difference in
#: `_AB_A` is positive, so "N of M favour" reads the same whether `wins`
#: counts the positive differences or simply all of them.
_AB_MIXED = [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 0.5),
                                (12, 12000, 1.6)]), False),
             ("arm-b", _AB_B, False)]


@needs_node
def test_an_update_the_other_arm_won_is_counted_against_the_leader():
    """`wins` is a count of signs, and nothing had ever shown it one."""
    _page, payload = render_multi(_AB_MIXED, configs=_AB_CONFIGS)
    ab = payload["alltime"]["ab"]

    assert [round(p["d"], 6) for p in ab["points"]] == [0.4, 0.5, -0.3, 0.7]
    assert (ab["n"], ab["wins"]) == (4, 3), \
        "the update arm-b won was counted as a win for arm-a"

    out = _call_multi(_AB_MIXED, "abMarkup(DATA.alltime.ab)", configs=_AB_CONFIGS)
    assert "3 of 4 favour arm-a" in out and "4 of 4 favour" not in out
    # And the losing difference is drawn on the losing side of zero.
    below = [t for t in _tags(out, "circle")
             if t.get("r") == "3" and t.get("fill") == "#A2352C"]
    assert len(below) == 1, "the negative difference is not drawn in its own colour"


def test_a_run_reached_through_two_roots_is_never_paired_against_itself():
    """`_run_roots` scans the agent worktrees, so a copy of a run arrives
    under a second label with identical rows. Every clause of the pair's gate
    passes on it, it wins the tie-break on shared updates, and the panel then
    reports a paired mean of exactly zero over every update -- one run
    subtracted from itself, with the most weight anything here can carry.
    """
    copy = [("arm-a", _AB_A, True), ("arm-b", _AB_B, False),
            ("deadbee:arm-a", list(_AB_A), False)]
    configs = dict(_AB_CONFIGS, **{"deadbee:arm-a": dict(_AB_CONFIGS["arm-a"])})
    _page, payload = render_multi(copy, configs=configs)
    ab = payload["alltime"]["ab"]

    assert (ab["a"]["name"], ab["b"]["name"]) == ("arm-a", "arm-b"), \
        "a run was paired against a copy of itself"
    assert round(ab["mean"], 6) == 0.55 and ab["n"] == 4
    assert ab["same_run"] == [["arm-a", "deadbee:arm-a"]]
    assert payload["alltime"]["duplicate_runs"] == [["arm-a", "deadbee:arm-a"]]

    # Two runs that merely agree about one reading are still two runs.
    near = _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4),
                    (12, 12000, 1.7)])
    _page, payload = render_multi(
        [("arm-a", _AB_A, True), ("arm-b", _AB_B, False), ("arm-c", near, False)],
        configs=dict(_AB_CONFIGS, **{"arm-c": {"head": "conv", "note": "c",
                                               "lr": 3e-4, "seed": 1}}))
    assert payload["alltime"]["duplicate_runs"] == []


@needs_node
def test_the_run_that_kept_going_keeps_its_readings_when_it_is_the_losing_one():
    """A and B are oriented by which run scored better, not by which ran on.

    With only an a-tail, a losing run that carried on lost every reading it
    took after the winner stopped -- and the axis still printed A's last
    update at the pixel column belonging to B's.
    """
    short = _ab_run([(3, 3000, 1.2), (6, 6000, 1.3), (9, 9000, 1.25),
                     (12, 12000, 1.35), (15, 15000, 1.30)])
    long_weak = _ab_run(
        [(3, 3000, 0.5), (6, 6000, 0.6), (9, 9000, 0.55), (12, 12000, 0.7),
         (15, 15000, 0.65)]
        + [(u, u * 1000, 1.0 + 0.1 * ((u - 18) // 3)) for u in range(18, 61, 3)])
    runs = [("short-strong", short, False), ("long-weak", long_weak, False)]
    configs = {"short-strong": {"head": "a", "lr": 3e-4},
               "long-weak": {"head": "b", "lr": 3e-4}}

    _page, payload = render_multi(runs, configs=configs)
    ab = payload["alltime"]["ab"]
    assert (ab["a"]["name"], ab["b"]["name"]) == ("short-strong", "long-weak")
    assert (ab["a_last_update"], ab["b_last_update"]) == (15, 60)
    assert ab["tail"] == []
    assert [u for u, _v in ab["tail_b"]] == list(range(18, 61, 3)), \
        "the longer run's 15 later readings are in the payload nowhere"

    out = _call_multi(runs, "abMarkup(DATA.alltime.ab)", configs=configs)
    faded = [t for t in _tags(out, "path") if t.get("opacity") == ".35"]
    assert len(faded) == 1 and faded[0]["stroke"] == "#8A6516", \
        "the tail was not drawn, or was drawn in the other run's colour"
    assert len(_path_points(faded[0]["d"])) == 16

    # The right end of the axis is labelled with the update it maps to.
    ends = re.findall(r'<text class="lbl-s" x="576" y="298" text-anchor="end">'
                      r'(\d+)</text>', out)
    assert ends == ["60"], "the axis printed one run's last update at another's column"
    # The rule stands where the first run stopped, and that is not B here.
    rule = re.findall(r'<text class="lbl-s" x="[-\d.]+" y="24"[^>]*>([^<]*)</text>',
                      out)
    assert rule == ["short-strong stopped"],         "the rule names the run that stopped last, not the one that stopped first"
    axis = [t for t in _tags(out, "line") if t.get("class") == "axis"]
    at = [float(t["x1"]) for t in axis if t["x1"] == t["x2"]]
    assert len(at) == 1 and abs(at[0] - 156.0) < 0.5,         "the rule is not at update 15, where the shared window ends"
    assert ("The faded tail is every reading long-weak took after "
            "short-strong stopped at 15") in out


@needs_node
def test_two_runs_still_level_print_the_shared_update_once():
    """Both runs writing and standing at the same update is the ordinary
    state of the pair this pane exists for. The centred label and the
    right-hand label are then the same number at the same x.
    """
    level = [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4),
                                (12, 12000, 1.6)]), True),
             ("arm-b", _AB_B, True)]
    out = _call_multi(level, "abMarkup(DATA.alltime.ab)", configs=_AB_CONFIGS)
    axis = re.findall(r'<text class="lbl-s"[^>]*y="298"[^>]*>(\d+)</text>', out)
    assert axis == ["3", "12"], "the last shared update is printed twice, overlapping"


def test_two_runs_on_two_scales_are_never_paired():
    """`_same_scale` is three-valued and two of the three answers refuse.

    False is the obvious one. None is the one that matters: two runs that both
    record nothing about their opponent are not thereby comparable, and the
    difference between them is not a smaller finding, it is a different
    quantity.
    """
    named = [("arm-a", _AB_A, True),
             ("arm-b", _ab_run([(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8),
                                (12, 12000, 0.9)], opponent="idle"), False)]
    assert render_multi(named, configs=_AB_CONFIGS)[1]["alltime"]["ab"] is None, \
        "two recorded opponents were subtracted from each other"

    blank = [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4)],
                               opponent=None, control=None), True),
             ("arm-b", _ab_run([(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8)],
                               opponent=None, control=None), False)]
    assert render_multi(blank, configs=_AB_CONFIGS)[1]["alltime"]["ab"] is None, \
        "two runs recording nothing were treated as recording the same thing"

    # The same two runs, once they name the opponent, do pair -- so the refusal
    # above is about the evidence and not about the fixture.
    speak = [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4)],
                               control=None), True),
             ("arm-b", _ab_run([(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8)],
                               control=None), False)]
    assert render_multi(speak, configs=_AB_CONFIGS)[1]["alltime"]["ab"] is not None


def test_a_pair_needs_the_same_battles_the_same_play_and_a_recorded_configuration():
    """Every remaining clause of the gate, one refusal each."""
    def paired(configs=None, **kw):
        left = _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4)])
        right = _ab_run([(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8)], **kw)
        return render_multi([("arm-a", left, True), ("arm-b", right, False)],
                            configs=_AB_CONFIGS if configs is None else configs
                            )[1]["alltime"]["ab"]

    assert paired() is not None
    assert paired(episodes=150) is None, "two evaluation sizes were paired"
    assert paired(control=0.925) is None, "two control rates were paired"

    # A run whose readings are labelled greedy is not the same kind of
    # measurement as one whose readings carry no play mode at all.
    greedy = [_row(u, s, eval_lift_sd=v, eval_win=0.5, control_win=0.2,
                   eval_opponent="random", eval_episodes=40, arm="w1, greedy")
              for u, s, v in [(3, 3000, 0.6), (6, 6000, 0.7), (9, 9000, 0.8)]]
    mixed = render_multi(
        [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2), (9, 9000, 1.4)]), True),
         ("arm-b", greedy, False)], configs=_AB_CONFIGS)[1]["alltime"]["ab"]
    assert mixed is None, "a greedy arm was subtracted from an unlabelled one"

    # Two updates in common is not a paired mean, whatever the rest says.
    short = render_multi(
        [("arm-a", _ab_run([(3, 3000, 1.0), (6, 6000, 1.2)]), True),
         ("arm-b", _ab_run([(3, 3000, 0.6), (6, 6000, 0.7)]), False)],
        configs=_AB_CONFIGS)[1]["alltime"]["ab"]
    assert short is None

    # Five differing keys are two experiments, not one A/B.
    assert paired(configs={"arm-a": {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
                           "arm-b": {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2}}) is None

    # And two runs whose configuration nobody wrote down are not thereby
    # identically configured. An empty diff is not evidence of a match.
    assert paired(configs={}) is None


@needs_node
def test_the_matched_pair_draws_both_readings_of_a_replayed_update():
    """The fold a step axis would have hidden, drawn as two marks instead.

    The line follows the first write and the replay is a hollow marker joined
    to it, so a resume reads as a second measurement of one update rather than
    as a cliff between two.
    """
    out = _call_multi(_AB_RUNS, "abMarkup(DATA.alltime.ab)", configs=_AB_CONFIGS)
    blue = [t for t in _tags(out, "path") if t.get("stroke") == "#2E86AB"]
    solid = [t for t in blue if "opacity" not in t]
    faded = [t for t in blue if t.get("opacity") == ".35"]

    assert len(solid) == 1 and len(_path_points(solid[0]["d"])) == 4, \
        "the line does not follow the four first writes"
    assert len(faded) == 1 and len(_path_points(faded[0]["d"])) == 2, \
        "the readings past the shorter run's last update are not drawn faded"

    # Rising in x and rising in lift, for a run whose readings only climbed:
    # this fails if the series were sorted by value or drawn from the replays.
    points = _path_points(solid[0]["d"])
    assert [p[0] for p in points] == sorted(p[0] for p in points)
    assert [p[1] for p in points] == sorted((p[1] for p in points), reverse=True)

    hollow = [t for t in _tags(out, "circle") if t.get("fill") == "none"]
    assert len(hollow) == 4, "two replayed updates, drawn in both panels"
    assert "Updates 9 and 12 were written twice after a resume" in out
    assert "+0.300 rather than +0.550" in out

    # The lower panel is where the headline number comes from, and none of it
    # was asserted: the band, the mean line and every difference dot could be
    # deleted with the suite green, leaving an empty box under a caption that
    # still described them.
    diffs = [t for t in _tags(out, "circle")
             if t.get("r") == "3" and t.get("fill") == "#26704F"]
    assert len(diffs) == 4, "the four paired differences are not drawn"
    ys = [float(t["cy"]) for t in sorted(diffs, key=lambda t: float(t["cx"]))]
    assert all(200 <= y <= 280 for y in ys), "a difference left its own panel"
    assert ys == sorted(ys, reverse=True), \
        "the differences are not ordered by their own value: 0.4 < 0.5 < 0.6 < 0.7"
    band = [t for t in _tags(out, "rect") if t.get("opacity") == ".14"]
    assert len(band) == 1, "the one-standard-deviation band is not drawn"
    mean = [t for t in _tags(out, "line")
            if t.get("stroke-width") == "1.5" and t.get("stroke") == "#26704F"]
    assert len(mean) == 1, "the paired mean is not drawn"
    top, height = float(band[0]["y"]), float(band[0]["height"])
    assert top < float(mean[0]["y1"]) < top + height, \
        "the mean line does not sit inside its own standard-deviation band"
    # +0.4 and +0.7 are more than one standard deviation from +0.55, so they
    # fall outside the band; +0.5 and +0.6 fall inside it.
    assert sum(1 for y in ys if top <= y <= top + height) == 2

    # The chip is the gate's answer read back, and it was unchecked. These
    # rows record a random opponent and no play mode at all, and the caption
    # may not upgrade that silence into "same play mode".
    assert "vs random (recorded)" in out and "idle" not in out
    assert "40 battles" in out
    assert "mode unknown" in out and "same play mode" not in out
    assert "Neither run recorded how it was played" in out
    # Both panels' domains in the DOM, at reading size.
    assert "The upper panel runs from " in out
    assert "difference panel from " in out and "over updates 3 to 15" in out

    # The rule naming the shorter run is driven by the payload's live flag and
    # never by a file timestamp.
    assert "arm-b stopped" in out and "arm-b still writing" not in out
    assert "3 of 4 configuration keys" not in out and "2 of 4 configuration keys" in out


@needs_node
def test_with_no_matched_pair_the_panel_says_what_a_pair_would_need():
    out = _call_multi(_QUIET, "abMarkup(DATA.alltime.ab)")
    assert "no two runs share a control, an opponent" in out
    assert "<svg" not in out


# ---------------------------------------------------------- the sweep ladders


#: Two sweeps whose arms would be catastrophic on one bar scale: the head
#: family's worst arm beats the observation family's best by 1.9, and both
#: record the same opponent and the same battle count, so no control-based
#: guard could tell them apart.
_SWEEP_RECORDS = [
    _record("factored", "headablate", "factored", "greedy", 2.17, 1.96, 2.38, win=0.96),
    _record("factored", "headablate", "factored", "sampled", 0.96, 0.75, 1.17),
    _record("flat", "headablate", "flat", "greedy", 1.70, 1.44, 1.96, win=0.85),
    _record("flat", "headablate", "flat", "sampled", 0.75, 0.50, 1.00),
    _record("v1", "obsablate", "v1", "greedy", -1.60, -1.86, -1.34, win=0.10),
    _record("v1", "obsablate", "v1", "sampled", -0.30, -0.55, -0.05),
    _record("swarm", "obsablate", "swarm", "greedy", -0.20, -0.46, 0.06, win=0.40),
    _record("swarm", "obsablate", "swarm", "sampled", -0.10, -0.35, 0.15),
]


def test_a_sweep_is_ranked_inside_its_family_and_nowhere_else():
    """The family read off the checkpoint path is the only partition on disk.

    headablate/flat and obsablate/v1 record the same observation, the same
    head, the same opponent and the same battle count, and read +1.70 and
    -1.60. Nothing in the JSON separates them; the path does, and that is what
    keeps them off one bar scale.
    """
    _page, payload = render_multi(
        _QUIET, extras={"verdicts": {"sweep": _SWEEP_RECORDS}})
    sweeps = payload["alltime"]["sweeps"]

    assert [f["family"] for f in sweeps["families"]] == ["headablate", "obsablate"]
    head, obs = sweeps["families"]
    assert head["checkpoints"] == 2 and head["episodes"] == 150
    assert head["scale"]["opponent"] == "random"
    assert head["scale"]["source"] == "recorded"
    # Best first inside a section, one section per way of playing, and never
    # one ordering across both.
    assert [(s["mode"], [(a["arm"], a["lift"]) for a in s["arms"]])
            for s in head["sections"]] == [
        ("greedy", [("factored", 2.17), ("flat", 1.70)]),
        ("sampled", [("factored", 0.96), ("flat", 0.75)])]
    assert [(s["mode"], [a["arm"] for a in s["arms"]]) for s in obs["sections"]] == [
        ("greedy", ["swarm", "v1"]), ("sampled", ["swarm", "v1"])]
    # The interval travels with the arm, which is what licenses the whiskers.
    assert head["sections"][0]["arms"][0]["ci"] == [1.96, 2.38]
    assert (sweeps["singletons"], sweeps["excluded"]) == (0, 0)
    # Every arm carries the scale its own chip will print. Nothing checked
    # that, so the chip could be built from a fabricated opponent and each
    # ladder row would read "vs idle" under a header still saying "vs
    # random" -- the project's signature failure, printed as a label.
    assert [a["scale"]["opponent"] for f in sweeps["families"]
            for s in f["sections"] for a in s["arms"]] == ["random"] * 8
    assert [a["scale"]["source"] for f in sweeps["families"]
            for s in f["sections"] for a in s["arms"]] == ["recorded"] * 8


@needs_node
def test_a_sweep_bar_is_scaled_to_its_own_family_and_not_to_the_page():
    """Bar lengths must not carry from one sweep to the next.

    On one shared scale the drawn lengths would be in the ratio of the lifts:
    the observation family's longest bar would be three-quarters of the head
    family's. Rescaled per family, each family's longest bar is nearly full --
    which is the only reading of a bar that does not invite a comparison the
    data cannot support.
    """
    out = _call_multi(_QUIET, "sweepMarkup(DATA.alltime.sweeps)",
                      extras={"verdicts": {"sweep": _SWEEP_RECORDS}})
    bars = [_percent(t["style"], "width") for t in _tags(out, "i")
            if "width" in t.get("style", "")]
    assert len(bars) == 8, "four sections of two arms"

    # headablate greedy spans 0 .. 2.38 (the widest interval end), so +2.17 is
    # 91.2% of it and +1.70 is 71.4%. obsablate greedy spans -1.86 .. 0.06, so
    # -1.60 is 83.3% and -0.20 is 10.4%.
    head_top, head_next, obs_top, obs_worst = bars[0], bars[1], bars[4], bars[5]
    assert abs(head_top - 91.2) < 0.5 and abs(head_next - 71.4) < 0.5
    assert abs(obs_top - 10.4) < 0.5 and abs(obs_worst - 83.3) < 0.5, \
        "the second family's bars were drawn on the first family's scale"
    assert obs_worst / head_top > 0.85, \
        "the two families' bar lengths are still in the ratio of their lifts"

    # The whiskers are the recorded intervals in that same per-family scale.
    whiskers = [t for t in _tags(out, "u") if "style" in t]
    assert len(whiskers) == 8, "every record in these families carries one"
    assert abs(_percent(whiskers[0]["style"], "width")
               - (2.38 - 1.96) / 2.38 * 100) < 0.5

    assert "nothing is ranked across families" in out

    # The zero marker is the axis every bar is read against, and pinning it
    # to the left edge would make every negative bar read as if it started
    # there. obsablate's greedy section spans -1.86 to 0.06, so zero is at
    # 96.9% of it; headablate's spans 0 to 2.38, so zero is at the left.
    zeros = [_percent(t["style"], "left") for t in _tags(out, "b")
             if "left" in t.get("style", "")]
    assert len(zeros) == 8
    assert all(abs(z) < 0.5 for z in zeros[:4]), "headablate's arms are all positive"
    assert abs(zeros[4] - 96.9) < 0.5 and abs(zeros[5] - 96.9) < 0.5
    assert abs(zeros[6] - 78.6) < 0.5 and abs(zeros[7] - 78.6) < 0.5

    # A bracket means "this weight appears twice in this ladder". Counted
    # across the family it was 2 for every arm, once per play mode, which
    # marks every row and so marks none.
    assert "bracket" not in out, "every row was bracketed, which says nothing"

    # And every chip the section prints is the one the payload computed:
    # two family headers, eight arms, two family captions.
    assert out.count("vs random (recorded)") == 12 and "idle" not in out
    assert out.count('&middot; <span class="chip good">'
                     'vs random (recorded)</span>') == 2
    assert "150 battles each" in out


def test_a_family_that_disagrees_about_its_opponent_is_counted_not_drawn():
    """And a checkpoint belonging to no sweep is a singleton, not a ladder."""
    mixed = [
        _record("a", "mix", "a", "greedy", 1.0, 0.8, 1.2),
        _record("a", "mix", "a", "sampled", 0.5, 0.3, 0.7),
        _record("b", "mix", "b", "greedy", 0.9, 0.7, 1.1, opponent="idle"),
        _record("b", "mix", "b", "sampled", 0.4, 0.2, 0.6, opponent="idle"),
    ]
    lone = [{"name": "_diag", "checkpoint": "runs/_diag/cloned.pt",
             "mode": "greedy", "lift": 1.62, "ci_low": 1.4, "ci_high": 1.85,
             "episodes": 150, "eval_opponent": "random"},
            {"name": "_diag", "checkpoint": "runs/_diag/cloned.pt",
             "mode": "sampled", "lift": 0.71, "ci_low": 0.5, "ci_high": 0.92,
             "episodes": 150, "eval_opponent": "random"}]
    nameless = {"episodes": 150, "greedy": {"lift": 1.0}, "sampled": {"lift": 0.4}}

    _page, payload = render_multi(_QUIET, extras={"verdicts": {
        "mixed": mixed, "lone": lone, "nameless": nameless}})
    sweeps = payload["alltime"]["sweeps"]

    assert sweeps["families"] == [], "a family naming two opponents was ranked"
    assert sweeps["excluded"] == 3, \
        "the mixed family's two checkpoints, and the verdict naming no opponent"
    assert sweeps["singletons"] == 1, "the arm that belongs to no sweep"

    # A family that disagrees only about the battle count is refused too.
    uneven = [_record("a", "size", "a", "greedy", 1.0, 0.8, 1.2),
              _record("a", "size", "a", "sampled", 0.5, 0.3, 0.7),
              _record("b", "size", "b", "greedy", 0.9, 0.7, 1.1, episodes=300),
              _record("b", "size", "b", "sampled", 0.4, 0.2, 0.6, episodes=300)]
    _page, payload = render_multi(_QUIET, extras={"verdicts": {"u": uneven}})
    assert payload["alltime"]["sweeps"]["families"] == []
    assert payload["alltime"]["sweeps"]["excluded"] == 2


@needs_node
def test_a_sweep_holding_one_checkpoint_is_a_singleton_and_not_a_ladder():
    """One arm is not a sweep, and a bar for it would be nearly full length
    for having nothing to be measured against. The guard existed and nothing
    reached it: the only singleton the suite had was a path with no arm level
    at all, which `_sweep_family` refuses one step earlier.
    """
    solo = [_record("only", "solo", "only", "greedy", 1.62, 1.40, 1.85),
            _record("only", "solo", "only", "sampled", 0.71, 0.50, 0.92)]
    _page, payload = render_multi(_QUIET, extras={"verdicts": {"s": solo}})
    sweeps = payload["alltime"]["sweeps"]

    assert sweeps["families"] == [], "a checkpoint was ranked against nothing"
    assert (sweeps["singletons"], sweeps["excluded"]) == (1, 0)

    out = _call_multi(_QUIET, "sweepMarkup(DATA.alltime.sweeps)",
                      extras={"verdicts": {"s": solo}})
    assert '<i style="left:' not in out, "a lone arm was given a bar"
    assert "1 checkpoint belongs to no sweep" in out


@needs_node
def test_a_checkpoint_measured_by_two_verdict_files_is_one_arm():
    """Two evaluation processes are writing verdict files right now.

    The same three checkpoints re-measured under a second file name flattened
    straight into the sections: six peer arms in a three-arm sweep, the same
    name at two different ranks with nothing telling them apart, under a
    header that still counted three checkpoints.
    """
    first = [_record("w0.1", "pw", "w0.1", "greedy", -1.318),
             _record("w0.5", "pw", "w0.5", "greedy", -2.802),
             _record("w1.0", "pw", "w1.0", "greedy", -2.728)]
    again = [_record("w0.1", "pw", "w0.1", "greedy", -1.018),
             _record("w0.5", "pw", "w0.5", "greedy", -2.502),
             _record("w1.0", "pw", "w1.0", "greedy", -2.428)]

    _page, payload = render_multi(
        _QUIET, extras={"verdicts": {"a": first, "b": again}})
    sweeps = payload["alltime"]["sweeps"]
    assert sweeps["families"] == [], \
        "three checkpoints measured twice were drawn as six peer arms"
    assert sweeps["excluded"] == 3, "the disagreeing checkpoints are counted"

    # Read twice and agreeing is one reading, and the ladder keeps its arms.
    _page, payload = render_multi(
        _QUIET, extras={"verdicts": {"a": first, "b": list(first)}})
    family = payload["alltime"]["sweeps"]["families"][0]
    assert family["checkpoints"] == 3
    assert [len(s["arms"]) for s in family["sections"]] == [3], \
        "one checkpoint read twice was drawn as two arms"

    # Two different checkpoints that happen to share an arm name are two
    # arms, and the ladder brackets them so the repeat is visible.
    twins = [_record("w0.5", "pw", "w0.5", "greedy", -2.802),
             _record("w0.5", "pw", "w0.5", "greedy", -2.400),
             _record("w1.0", "pw", "w1.0", "greedy", -2.728)]
    twins[1]["checkpoint"] = "runs/pw/w0.5/other.pt"
    out = _call_multi(_QUIET, "sweepMarkup(DATA.alltime.sweeps)",
                      extras={"verdicts": {"t": twins}})
    assert out.count("bracket") == 2, \
        "two arms sharing a name inside one ladder are not marked"


@needs_node
def test_with_no_sweep_on_disk_the_section_says_so():
    out = _call_multi(_QUIET, "sweepMarkup(DATA.alltime.sweeps)")
    assert "no verdict file records a sweep against a named opponent" in out
    assert 'class="ladder"' not in out


# --------------------------------------------------- what a reading is worth


#: Deliberately not in ascending order, and with neither the widest nor the
#: narrowest at an end. `min` and `max` read the first and last of the sorted
#: list, so a fixture whose write order and sorted order coincide cannot tell
#: a sort from no sort at all: written ascending, dropping the sort left
#: `min` and `max` correct and the whole strip unchanged.
_INTERVALS = [
    _record("a", "f", "a", "greedy", 1.0, 0.74, 1.26),      # half 0.26
    _record("a", "f", "a", "sampled", 0.5, 0.40, 0.60),     # half 0.10
    _record("b", "f", "b", "greedy", 0.9, 0.70, 1.10),      # half 0.20
    _record("b", "f", "b", "sampled", 0.4, 0.20, 0.60),     # half 0.20
    _record("c", "f", "c", "greedy", 0.8, 0.40, 1.20),      # half 0.40
    _record("c", "f", "c", "sampled", 0.3, 0.00, 0.60),     # half 0.30
]


def test_the_precision_strip_is_built_from_the_recorded_intervals_only():
    """Half-widths, a median, and a probe figure that says it is derived."""
    _page, payload = render_multi(_QUIET, extras={"verdicts": {"v": _INTERVALS}})
    pr = payload["alltime"]["precision"]

    assert [round(h, 6) for h in pr["half"]] == [0.1, 0.2, 0.2, 0.26, 0.3, 0.4]
    assert round(pr["median"], 6) == 0.23
    assert (round(pr["min"], 6), round(pr["max"], 6)) == (0.1, 0.4)
    assert pr["population"]["opponent"] == "random"
    assert pr["population"]["episodes"] == 150 and pr["population"]["n"] == 6
    assert pr["population"]["scale"]["source"] == "recorded"
    # Derived from the median and the battle counts, and labelled as derived.
    assert round(pr["probes"]["derived_half"], 9) == round(
        0.23 * (150 / 40) ** 0.5, 9)
    assert pr["probes"]["episodes"] == 40 and pr["probes"]["rule"]
    assert pr["populations_not_drawn"] == []
    # Every in-run reading on the page, none of which carries an interval.
    assert pr["without_interval"] == payload["alltime"]["lift_rows"] == 1
    # And none of them recorded a battle count, so none of them is a probe.
    assert (pr["probes"]["count"], pr["probes"]["unrecorded"]) == (0, 1)


@needs_node
def test_a_reading_that_records_no_battle_count_is_not_called_a_probe():
    """The 40-episode default is an estimate everywhere else on this page.

    `runs/headtohead-aug27`, `runs/passweight-sweep` and `runs/cloned` record
    `episodes: 150` and no `eval_episodes`, and counting them at the probe
    size stated as fact -- the record +1.813 among them -- what the battle
    ledger on the same page carefully calls an estimate, and handed a reader
    a half-width twice what those files record.
    """
    common = dict(eval_win=0.5, control_win=0.26, eval_opponent="random")
    rows = [_row(1, 1000, episodes=150, eval_lift_sd=1.813, **common),
            _row(2, 2000, eval_episodes=40, eval_lift_sd=0.4, **common),
            _row(3, 3000, eval_episodes=150, eval_lift_sd=0.5, **common)]
    runs = [("mixed", rows, False)]

    _page, payload = render_multi(runs, extras={"verdicts": {"v": _INTERVALS}})
    pr = payload["alltime"]["precision"]
    assert pr["probes"]["count"] == 1, \
        "a reading recording 150 battles, or none at all, was counted as a probe"
    assert pr["probes"]["unrecorded"] == 1

    out = _call_multi(runs, "precisionMarkup(DATA.alltime.precision)",
                      extras={"verdicts": {"v": _INTERVALS}})
    assert "1 reading on this page records that size" in out
    assert "1 records no battle count at all" in out


def test_a_second_interval_population_is_counted_and_never_averaged_in():
    """An interval from 300 battles beside one from 150 is not a spread."""
    wide = [_record("x", "g", "x", "greedy", 1.0, 0.0, 2.0, episodes=300),
            _record("x", "g", "x", "sampled", 0.5, -0.5, 1.5, episodes=300)]
    _page, payload = render_multi(_QUIET, extras={
        "verdicts": {"a": _INTERVALS[:4], "b": wide}})
    pr = payload["alltime"]["precision"]

    assert [round(h, 6) for h in pr["half"]] == [0.1, 0.2, 0.2, 0.26], \
        "a 300-battle interval was mixed into the 150-battle strip"
    assert pr["population"]["episodes"] == 150
    assert pr["populations_not_drawn"] == [
        {"opponent": "random", "episodes": 300, "n": 2}]


def test_the_same_weights_are_joined_on_a_bit_equal_reading_and_nothing_else():
    """Greedy on fixed seeds is deterministic; sampled is not.

    So agreement to every digit is evidence of identical weights, and a value
    agreeing to eleven digits and differing in the twelfth is a different
    measurement. An arm name would join two sweeps that both call an arm w0.5.
    """
    same = 1.6230076626442986
    twin = {"episodes": 150, "eval_opponent": "random",
            "greedy": {"lift": same, "ci_low": 1.4, "ci_high": 1.85},
            "sampled": {"lift": 0.7087076246255667, "ci_low": 0.5, "ci_high": 0.92}}
    other = {"episodes": 150, "eval_opponent": "random",
             "greedy": {"lift": same, "ci_low": 1.4, "ci_high": 1.85},
             "sampled": {"lift": 0.7336065585910241, "ci_low": 0.5, "ci_high": 0.95}}
    _page, payload = render_multi(
        _QUIET, extras={"verdicts": {"one": twin, "two": other}})
    reps = payload["alltime"]["precision"]["replicates"]

    assert len(reps) == 1
    assert reps[0]["greedy"] == same and reps[0]["mode"] == "sampled"
    assert reps[0]["values"] == [0.7087076246255667, 0.7336065585910241]
    assert round(reps[0]["spread"], 10) == 0.024898934
    assert reps[0]["sources"] == ["one", "two"]

    # The bar is the drawn half of that finding, and nothing asserted it.
    out = _call_multi(_QUIET, "precisionMarkup(DATA.alltime.precision)",
                      extras={"verdicts": {"one": twin, "two": other}})
    bar = [t for t in _tags(out, "line") if t.get("stroke-width") == "3"]
    assert len(bar) == 1, "the replicate spread bar is not drawn"
    span = float(bar[0]["x2"]) - float(bar[0]["x1"])
    pr = payload["alltime"]["precision"]
    top = max(pr["max"], pr["probes"]["derived_half"], reps[0]["spread"]) * 1.15
    assert abs(span - reps[0]["spread"] / top * (640 - 44 - 56)) < 0.2
    assert "0.025" in out and "greedy repeats to every digit" in out

    nearly = dict(other, greedy=dict(other["greedy"], lift=same + 1e-12))
    _page, payload = render_multi(
        _QUIET, extras={"verdicts": {"one": twin, "two": nearly}})
    assert payload["alltime"]["precision"]["replicates"] == [], \
        "two readings that differ were called the same weights"


@needs_node
def test_the_precision_strip_draws_a_tick_for_every_interval_and_refuses_one():
    out = _call_multi(_QUIET, "precisionMarkup(DATA.alltime.precision)",
                      extras={"verdicts": {"v": _INTERVALS[:4]}})
    ticks = [t for t in _tags(out, "line") if t.get("opacity") == ".55"]
    assert len(ticks) == 4, "one tick per recorded interval"
    xs = sorted(float(t["x1"]) for t in ticks)
    # 0.10, 0.20, 0.20, 0.26 on an axis starting at zero: the gaps between the
    # ticks are in the ratio of the half-widths and of nothing else.
    assert xs[1] == xs[2]
    assert abs((xs[3] - xs[0]) / (xs[1] - xs[0]) - 0.16 / 0.10) < 0.02

    # The blue mark is the derived probe half-width and not the median: they
    # differ by the root of the battle-count ratio, so drawing one at the
    # other's x puts two quantities at one place under two labels.
    _page, payload = render_multi(_QUIET, extras={"verdicts": {"v": _INTERVALS[:4]}})
    pr = payload["alltime"]["precision"]
    top = max(pr["max"], pr["probes"]["derived_half"]) * 1.15
    at = lambda v: 44 + v / top * (640 - 44 - 56)
    derived = [t for t in _tags(out, "line") if t.get("y1") == "70"]
    median = [t for t in _tags(out, "line") if t.get("y1") == "26"]
    assert len(derived) == 1 and len(median) == 1
    assert abs(float(derived[0]["x1"]) - at(pr["probes"]["derived_half"])) < 0.2
    assert abs(float(median[0]["x1"]) - at(pr["median"])) < 0.2
    assert float(derived[0]["x1"]) - float(median[0]["x1"]) > 150, \
        "the derived probe mark was drawn at the median"

    # And the chip reports the population the strip actually drew.
    assert "vs random (recorded)" in out and "idle" not in out
    assert "150 battles" in out
    assert ("The strip runs from 0.00 to %.2f in lift" % top) in out

    # One interval is a reading, not a spread, so no median is drawn over it.
    lone = _call_multi(_QUIET, "precisionMarkup(DATA.alltime.precision)",
                       extras={"verdicts": {"v": [
                           _record("a", "f", "a", "greedy", 1.0, 0.90, 1.10),
                           _record("a", "f", "a", "sampled", 0.5)]}})
    assert len([t for t in _tags(lone, "line") if t.get("opacity") == ".55"]) == 1
    assert "a spread needs two" in lone
    assert not [t for t in _tags(lone, "line") if t.get("stroke") == "#26704F"], \
        "a median was drawn over a single interval"

    none = _call_multi(_QUIET, "precisionMarkup(DATA.alltime.precision)")
    assert "no verdict records an interval" in none and "<svg" not in none


# ------------------------------------------------ the sparkline in each card


def _spark_run(values, **kw):
    """One run's readings, as a group entry of the payload."""
    rows = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=v, eval_win=0.4,
                 control_win=0.26, eval_opponent="random", **kw)
            for i, v in enumerate(values)]
    return render_multi([("r", rows, False)])[1]["alltime"]["groups"][0]["runs"][0]


def test_a_card_sparkline_counts_readings_and_not_steps():
    """A resumed run replays its step counter, and both segments are kept.

    Drawn against steps, updates 1 and 2 of the second segment would land on
    top of the first segment's and the line would fold back on itself. The x
    is the reading's ordinal, so every reading gets a place of its own.
    """
    common = dict(eval_win=0.4, control_win=0.26, eval_opponent="random")
    rows = [_row(1, 1000, eval_lift_sd=0.1, **common),
            _row(2, 2000, eval_lift_sd=0.2, **common),
            _row(3, 3000, eval_lift_sd=0.3, **common),
            # the resume: the step counter falls and the update numbers repeat
            _row(1, 500, eval_lift_sd=0.4, **common),
            _row(2, 1000, eval_lift_sd=0.5, **common)]
    _page, payload = render_multi([("resumed", rows, False)])
    run = payload["alltime"]["groups"][0]["runs"][0]

    assert run["series"] == [[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]], \
        "a resumed run's readings were folded onto the ones they replayed"
    assert payload["alltime"]["resumed"] == 1
    assert run["noise_rule"], "a noise figure with no rule beside it"


def test_a_reading_written_twice_with_the_same_values_is_one_point():
    """The double write that made one run look like it had twice the evidence."""
    same = dict(eval_lift_sd=0.3, eval_win=0.4, control_win=0.26,
                eval_opponent="random")
    rows = [_row(1, 1000, **same), _row(1, 1000, **same),
            _row(2, 2000, eval_lift_sd=0.5, eval_win=0.4, control_win=0.26,
                 eval_opponent="random")]
    _page, payload = render_multi([("twice", rows, False)])
    run = payload["alltime"]["groups"][0]["runs"][0]

    assert run["evals"] == 3 and run["series"] == [[1, 0.3], [2, 0.5]]


def test_the_noise_ruler_is_the_spread_of_consecutive_readings():
    """And it needs four readings, because three differences are not a spread."""
    assert _spark_run([0.0, 0.2, 0.1])["noise"] is None, \
        "a noise figure was invented out of three readings"

    # differences 0.2, -0.1, 0.2: a spread of 0.173205, halved back out of a
    # difference by the root of two, at 95%.
    measured = _spark_run([0.0, 0.2, 0.1, 0.3])["noise"]
    assert abs(measured - 1.96 * 0.1732050808 / 2 ** 0.5) < 1e-9

    # A run that only ever climbed has a wide spread of readings and almost no
    # noise. That is the distinction the ruler exists to draw, and it fails if
    # the figure is the spread of the readings rather than of the differences.
    climbing = _spark_run([0.0, 0.1, 0.2, 0.3])["noise"]
    assert climbing < 1e-8 < measured


@needs_node
def test_a_job_that_ranks_its_arms_draws_no_line_and_no_noise():
    """A table of arms sorted best-first is not a run over time.

    `runs/headtohead-aug27` holds four checkpoints played two ways, written
    best-first, and its own config note says so in as many words. Joined into
    a line it drew a smooth descent that nothing descended; worse, the
    unevenness of that sort's gaps was then published as the card's
    measurement noise -- 13 times the floor the bit-equal replicate actually
    measured, with the two figures on one page.
    """
    arms = [("cloned, greedy", 1.813), ("ppo, greedy", 1.239),
            ("cloned, sampled", 0.775), ("ppo, sampled", 0.422)]
    rows = [_row(i + 1, 0, episodes=150, arm=name, eval_lift_sd=lift,
                 eval_win=0.6, control_win=0.26, eval_opponent="random")
            for i, (name, lift) in enumerate(arms)]
    runs = [("headtohead", rows, False)]

    _page, payload = render_multi(runs, kinds={"headtohead": "job"})
    run = payload["alltime"]["groups"][0]["runs"][0]
    assert run["ranking"] is True, "the fixture is not a ranking job"
    assert run["series"] == [], "a table of arms was built into a trajectory"
    assert run["noise"] is None, "a sort's own gaps were measured as noise"
    assert "separate checkpoints sorted best-first" in run["no_series"]

    out = _call_multi(runs, "groupSpark(DATA.alltime.groups[0])",
                      kinds={"headtohead": "job"})
    assert "<path" not in out and "<svg" not in out
    assert "No line for headtohead" in out


@needs_node
def test_a_run_read_both_ways_draws_no_line_either():
    """Greedy and sampled on one set of weights differ by 2.3x here.

    A run whose readings hold both is two scales in one list, so a line
    through them descends every time the mode changes and `_noise_of` reads
    that step as what one probe moves.
    """
    rows = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=0.40 + i * 0.01,
                 eval_lift_sd_greedy=1.60 + i * 0.01, eval_win=0.5,
                 eval_win_greedy=0.9, control_win=0.26,
                 eval_opponent="random") for i in range(4)]
    runs = [("bothways", rows, False)]

    _page, payload = render_multi(runs)
    run = payload["alltime"]["groups"][0]["runs"][0]
    assert run["modes"] == ["greedy", "sampled"], "the fixture holds one mode"
    assert run["ranking"] is False, "this is a training run, not a job"
    assert run["series"] == [] and run["noise"] is None
    assert "more than one way" in run["no_series"]

    out = _call_multi(runs, "groupSpark(DATA.alltime.groups[0])")
    assert "<path" not in out
    assert "No line for bothways" in out


@needs_node
def test_a_card_sparkline_is_drawn_in_write_order_and_never_sorted():
    """The readings are a sequence, and sorting them draws a climb that never
    happened -- from the run's worst reading to its best, monotonically."""
    rows = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=v, eval_win=0.4,
                 control_win=0.26, eval_opponent="random")
            for i, v in enumerate([0.9, 0.1, 0.8, 0.2, 0.7])]
    out = _call_multi([("saw", rows, False)], "groupSpark(DATA.alltime.groups[0])")

    line = [t for t in _tags(out, "path") if t.get("stroke")]
    assert len(line) == 1
    points = _path_points(line[0]["d"])
    assert len(points) == 5
    assert [p[0] for p in points] == sorted(p[0] for p in points)
    assert [p[1] for p in points] != sorted(p[1] for p in points)
    assert [p[1] for p in points] != sorted((p[1] for p in points), reverse=True)

    # The zero line is the reference that says which readings beat the
    # control, and it can be deleted with nothing noticing.
    assert [t for t in _tags(out, "line") if t.get("class") == "zero"], \
        "no zero line, so nothing says which readings beat the control"


@needs_node
def test_the_noise_ruler_is_read_off_the_longest_trajectory_and_named():
    """Its source, its presence and its height all survived mutation.

    And it may not be painted through the run names: parked 30 units inside
    the right margin while labels were clamped to the viewBox edge, the rule
    split every name on the live page down the middle.
    """
    noisy = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=v, eval_win=0.4,
                  control_win=0.26, eval_opponent="random")
             for i, v in enumerate([0.2, 0.5, 0.3, 0.6, 0.35, 0.7, 0.45, 0.8])]
    calm = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=v, eval_win=0.4,
                 control_win=0.26, eval_opponent="random")
            for i, v in enumerate([0.30, 0.32, 0.31, 0.33])]
    runs = [("noisy", noisy, False), ("calm", calm, False)]

    _page, payload = render_multi(runs)
    drawn = {r["name"]: r for r in payload["alltime"]["groups"][0]["runs"]}
    assert len(drawn["noisy"]["series"]) == 8 and len(drawn["calm"]["series"]) == 4
    assert drawn["noisy"]["noise"] > drawn["calm"]["noise"] > 0

    out = _call_multi(runs, "groupSpark(DATA.alltime.groups[0])")

    # The panel's own domain, computed the way the drawing computes it.
    lo = hi = 0.0
    for run in drawn.values():
        for _at, value in run["series"]:
            lo, hi = min(lo, value), max(hi, value)
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    half = drawn["noisy"]["noise"] / (hi - lo) * (120 - 10 - 18)

    caps = [float(t["y1"]) for t in _tags(out, "line") if t.get("x1") == "603"]
    assert len(caps) == 2, "the ruler's end caps are not drawn"
    assert abs(max(caps) - min(caps) - 2 * half) < 0.1, \
        "the ruler is not the longest run's own noise on this panel's domain"
    assert abs((max(caps) + min(caps)) / 2 - 56) < 0.1
    assert "measured on noisy" in out and "the longest trajectory" in out
    # The domain, in the DOM: a 640-unit chart's own axis numbers render at
    # about 5px at this page's width, so nothing may be stated only there.
    assert ("The lines run from %.2f to %.2f in lift, over readings 1 to 8"
            % (lo, hi)) in out
    assert "measured on calm" not in out, \
        "the ruler was read off the shortest series and named the longest"

    # No name may reach the ruler's column.
    tag = re.search(r'text-anchor="middle">&#177;([\d.]+)</text>', out).group(1)
    left = 606 - (1 + len(tag)) * 6.2 / 2
    ends = re.findall(r'<text class="lbl-s" x="([\d.]+)" y="[-\d.]+" '
                      r'fill="[^"]+">([^<]*)</text>', out)
    assert len(ends) == 2
    for at, text in ends:
        assert float(at) + len(text) * 6.2 <= left, \
            "a run name is painted through the noise ruler"


@needs_node
def test_a_noise_ruler_wider_than_its_plot_is_clamped_and_says_so():
    """`overnight-selfplay`'s noise is 0.85 of its own domain.

    Unbounded, both end caps and the magnitude label are drawn outside the
    120-unit viewBox, where nothing renders them at all -- leaving a bare
    full-height line indistinguishable from a plot border, under a caption
    still calling it a scale mark. These are that run's six readings.
    """
    values = [0.1624, 0.2319, -0.1199, -0.0851, 0.1949, -0.1224]
    rows = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=v, eval_win=0.4,
                 control_win=0.26, eval_opponent="random")
            for i, v in enumerate(values)]
    runs = [("swingy", rows, False)]

    out = _call_multi(runs, "groupSpark(DATA.alltime.groups[0])")
    ruler = [t for t in _tags(out, "line") if t.get("x1") in ("603", "606")]
    assert len(ruler) == 3, "the rule and its two caps"
    ys = [float(t[key]) for t in ruler for key in ("y1", "y2")]
    ys += [float(t["y"]) for t in _tags(out, "text")]
    assert min(ys) >= 0 and max(ys) <= 120, \
        "part of the ruler is drawn outside the viewBox, where it never renders"
    assert "moves further between readings than this picture is tall" in out


@needs_node
def test_two_controls_never_meet_on_one_sparkline():
    """The panel's container is the scale group, so this is structural.

    An idle-scale run's control wins 92% of its matches and a random-scale
    one's 26%. They land in different cards, so their lines cannot reach each
    other however the drawing code is written.
    """
    idle = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=0.9 + i * 0.1,
                 eval_win=0.95, control_win=0.925) for i in range(4)]
    rand = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=0.1 + i * 0.1,
                 eval_win=0.5, control_win=0.26) for i in range(4)]

    out = _call_multi([("idler", idle, False), ("randomer", rand, False)],
                      "JSON.stringify(DATA.alltime.groups.map(groupSpark))")
    panels = json.loads(out)

    assert len(panels) == 2
    lines = [[_path_points(t["d"]) for t in _tags(p, "path")] for p in panels]
    assert [len(l) for l in lines] == [1, 1], \
        "two control rates were drawn on one axis"
    assert all(len(l[0]) == 4 for l in lines)
    assert "idler" in panels[0] and "randomer" not in panels[0]
    assert "randomer" in panels[1] and "idler" not in panels[1]


@needs_node
def test_a_card_that_records_no_control_gets_no_line():
    """Readings recording no control are not known to share an axis.

    That is exactly why the card is already not ordered, and drawing a picture
    inside it would put back the comparison the card refuses to make.
    """
    rows = [_row(i + 1, (i + 1) * 1000, eval_lift_sd=0.1 * i) for i in range(4)]
    out = _call_multi([("blind", rows, False)],
                      "groupSpark(DATA.alltime.groups[0])")
    assert "<svg" not in out and "<path" not in out
    assert "not known to share one axis" in out

    lonely = _call_multi(
        [("one", [_row(1, 1000, eval_lift_sd=0.2, eval_win=0.4,
                       control_win=0.26, eval_opponent="random")], False)],
        "groupSpark(DATA.alltime.groups[0])")
    assert "<svg" not in lonely and "a trend needs two" in lonely


@needs_node
def test_the_new_pictures_register_no_chart_and_need_no_scrub():
    """CHARTS drives wireScrub, which the all-time branch of draw() never calls.

    A picture that pushed into it would register a scrub line nothing ever
    wires up, and every value it held would be readable only by hovering a
    chart that does not respond. So the values are drawn in, or printed in the
    DOM beneath.
    """
    out = _call_multi(
        _AB_RUNS, "(function(){var m=allTimeMarkup(DATA.alltime);"
                  "return JSON.stringify({charts:CHARTS.length,"
                  "svgs:(m.match(/<svg/g)||[]).length,"
                  "scrub:m.indexOf('class=\"scrub\"')>=0,"
                  "nan:m.indexOf('NaN')>=0,"
                  "undef:m.indexOf('undefined')>=0});})()",
        configs=_AB_CONFIGS,
        extras={"verdicts": {"v": _BOTH_WAYS, "s": _SWEEP_RECORDS}})
    drawn = json.loads(out)

    assert drawn["charts"] == 0, "a chart was registered on a view with no time axis"
    assert drawn["svgs"] >= 4, "the pictures were not drawn at all"
    assert drawn["scrub"] is False
    assert drawn["nan"] is False, "a non-finite number reached the drawn markup"
    assert drawn["undef"] is False


# ------------------------------------------------- the licence to rank at all
#
# `_block_of` is the only thing on this page that produces an ordering, and
# every clause of its gate is what stops that ordering being a cross-scale
# ranking. Four of the five used to be deletable with the whole suite still
# green, and deleting the shared-control clause produced exactly the failure
# the view exists to prevent: an idle-scale lift crowned as the record and
# stamped "vs random (stated)".


def _gate_rows(**patch):
    """Three arms of one job that qualify, before `patch` breaks one thing."""
    rows = [
        _row(1, 0, episodes=150, arm="alpha, greedy",
             eval_lift_sd=0.55, eval_win=0.61, control_win=0.26),
        _row(2, 0, episodes=150, arm="beta, greedy",
             eval_lift_sd=0.40, eval_win=0.52, control_win=0.26),
        _row(3, 0, episodes=150, arm="gamma, greedy",
             eval_lift_sd=0.22, eval_win=0.44, control_win=0.26),
    ]
    for key, value in patch.items():
        index, field = key.split("_", 1)
        rows[int(index)][field] = value
    return rows


_GATE_NOTE = ("All three arms against the SAME random opponent on the SAME "
              "150 paired seeds.")


def _gate_block(rows, note=_GATE_NOTE, kind="job"):
    _page, payload = render_multi([("j", rows, False)],
                                  notes={"j": note},
                                  kinds=({"j": kind} if kind else {}))
    return payload["alltime"]["block"]


def test_the_ranking_gate_needs_a_job():
    """A training run's rows are a trajectory, not a table of arms.

    Sorting them by value ranks a run against its own past selves and calls
    the best moment the result.
    """
    assert _gate_block(_gate_rows()) is not None, "the fixture does not qualify"
    assert _gate_block(_gate_rows(), kind=None) is None,         "a run that no config called a job was ranked as one"


def test_the_ranking_gate_needs_every_row_to_name_its_arm():
    """One unlabelled row and the ladder has a bar nobody can attribute."""
    assert _gate_block(_gate_rows(**{"2_arm": ""})) is None
    assert _gate_block(_gate_rows(**{"2_arm": "   "})) is None,         "whitespace was accepted as an arm name"


def test_the_ranking_gate_needs_one_control_rate_across_every_arm():
    """The clause whose removal reproduces the original disaster exactly.

    A job with one arm read against a control that wins 92.5% of its own
    matches and two read against one winning 26% is two measurements, and
    ranking them puts an idle-scale number at the top of a random-scale
    ladder wearing a "(stated)" chip.
    """
    mixed = _gate_rows(**{"0_control_win": 0.925, "0_eval_lift_sd": 2.90})
    assert _gate_block(mixed) is None,         "arms on two different scales were ranked against each other"


def test_the_ranking_gate_needs_a_control_that_was_actually_measured():
    """A shared rate matching nothing measured is a shared unknown.

    Three of the five rates on disk match neither the idle control nor the
    random one, and a note claiming one opponent does not turn an
    unidentified scale into an identified one.
    """
    unknown = _gate_rows(**{"0_control_win": 0.44, "1_control_win": 0.44,
                            "2_control_win": 0.44})
    assert _gate_block(unknown) is None
    # Unless the rows say who they faced, which is evidence and not inference.
    named = _gate_rows(**{"0_control_win": 0.44, "1_control_win": 0.44,
                          "2_control_win": 0.44,
                          "0_eval_opponent": "expert",
                          "1_eval_opponent": "expert",
                          "2_eval_opponent": "expert"})
    block = _gate_block(named)
    assert block is not None and block["scale"]["named"] == "expert"


def test_the_ranking_gate_needs_the_arms_to_have_run_the_same_battles():
    """Arms over different numbers of battles were not on one seed set.

    Without this the block still built, and the record card printed "0 paired
    seeds" above the words "one opponent, one seed set" -- fabricating the
    provenance claim the page exists to protect.
    """
    assert _gate_block(_gate_rows(**{"2_episodes": 90})) is None,         "arms that ran different numbers of battles were called one seed set"
    stripped = _gate_rows()
    for row in stripped:
        row.pop("episodes")
    assert _gate_block(stripped) is None,         "a block with no seed count at all still claimed one"


def test_the_ranking_gate_needs_the_note_to_state_the_conditions():
    """Equal control rates do not license a ranking; a stated method does."""
    assert _gate_block(_gate_rows(), note="Four arms, plotted best first.") is None
    assert _gate_block(_gate_rows(), note="") is None


def test_a_note_that_denies_its_conditions_is_not_a_licence_to_rank():
    """The word SAME twice is not a claim, and counting it is not a gate.

    "Arms did NOT face the SAME opponent and were NOT played on the SAME
    seeds" says SAME twice and carries no caveat. A substring count read it
    as a licence, and the page printed a record, a sorted ladder and a "vs
    random (stated)" chip on top of a note denying every one of them.
    """
    from cr_sim.train.watch import _states_shared_conditions

    denial = ("Arms did NOT face the SAME opponent and were NOT played on "
              "the SAME seeds. Do not rank these against each other.")
    assert _states_shared_conditions(denial) is False
    assert _gate_block(_gate_rows(), note=denial) is None

    # And the absence of the word CAVEAT is not the other half either: the
    # claim has to be made, and has to name both things it is claiming.
    assert _states_shared_conditions("Ran the sweep overnight.") is False
    assert _states_shared_conditions("The SAME opponent throughout.") is False,         "one mention of one condition was read as both"
    assert _states_shared_conditions(
        "The SAME opponent, twice, on the SAME afternoon.") is False,         "a claim about one opponent alone licensed a ranking on seeds too"
    assert _states_shared_conditions(
        "The SAME 150 seeds, run on the SAME afternoon.") is False
    assert _states_shared_conditions(
        "Every arm met the SAME opponent on the SAME seeds.") is True
    assert _states_shared_conditions(
        "Every arm met the SAME opponent on the SAME seeds.\n"
        "CAVEAT: the targets changed too.") is False


def test_the_gate_that_is_open_produces_a_page_that_says_so_consistently():
    """The record chip, the ladder header and the note must agree.

    The "(stated)" chip and the seed count are claims about method. They are
    only ever printed where the gate passed, and the gate's own note is
    quoted underneath so the claim can be checked against its source.
    """
    block = _gate_block(_gate_rows())
    assert block["scale"]["stated"] is True and block["seeds"] == 150
    assert block["note"] == _GATE_NOTE


# ------------------------------------------------------ reading what is there


def test_an_opponent_written_on_the_row_is_read_off_the_row():
    """`eval_opponent` is mandatory on every new lift row and was never read.

    `selfplay.check_lift_is_named` refuses to write `eval_lift_sd` without it,
    so a lift that names a third opponent is not hypothetical. Inferring
    "idle" from a control rate that happens to sit at 92.5% puts a
    confidently wrong opponent on a number that names its own.
    """
    rows = [_row(1, 1000, eval_lift_sd=1.4, eval_win=0.62, control_win=0.925,
                 eval_opponent="expert")]
    _page, payload = render_multi([("expert-probe", rows, False)])
    alltime = payload["alltime"]

    scale = alltime["demoted"]["scale"]
    assert scale["named"] == "expert" and scale["source"] == "recorded"
    assert scale["opponent"] == "expert", "a recorded opponent was overruled by a guess"
    # The disagreement is carried rather than resolved: the rate does match
    # the idle control, and that is worth saying and not worth acting on.
    assert scale["anchor"] == "idle" and scale["conflict"] is True
    # A row that names its opponent is not on an unidentified scale.
    assert alltime["unidentified"] == 0
    assert alltime["named_rows"] == 1
    # And the group it lands in is headed by the name, not the inference.
    assert alltime["groups"][0]["scale"]["named"] == "expert"


@needs_node
def test_a_page_whose_rows_name_their_opponents_stops_saying_none_of_them_do():
    """Two sentences on the page assert that not one lift names its opponent.

    They were hardcoded, and stayed on the page while every row in the
    payload named one.
    """
    rows = [_row(1, 1000, eval_lift_sd=1.4, eval_win=0.62, control_win=0.925,
                 eval_opponent="expert")]
    runs = [("expert-probe", rows, False)]

    out = _call_multi(runs, "allTimeMarkup(DATA.alltime)")
    assert "Not one lift on disk names the opponent" not in out
    assert "1 of 1 readings name the opponent" in out
    assert "vs expert (recorded)" in out
    assert "vs idle (inferred)" not in out,         "an opponent was inferred over the one the row records"


def test_both_arms_of_a_row_that_holds_two_are_read():
    """`rotating_probe` writes greedy and sampled on one row.

    `eval_lift_sd` is the sampled arm by that function's own contract, and
    `eval_lift_sd_greedy` is the argmax beside it. Selecting rows on
    `"eval_lift_sd" in row` reads the sampled number, shows no mode and drops
    the greedy arm off the page -- including out of the largest-number panel,
    which is where it would have been.
    """
    rows = [_row(1, 1000, eval_lift_sd=-0.2, eval_win=0.3, control_win=0.26,
                 eval_opponent="random", eval_lift_sd_greedy=2.9,
                 eval_win_greedy=0.9)]
    _page, payload = render_multi([("rotating-probe-run", rows, False)])
    alltime = payload["alltime"]

    assert alltime["lift_rows"] == 2, "one row holding two readings counted as one"
    readings = sorted((r["mode"], r["best"]) for r in
                      [{"mode": alltime["demoted"]["mode"],
                        "best": alltime["demoted"]["lift"]}])
    assert readings == [("greedy", 2.9)],         "the largest number in the payload never reached the page"
    assert alltime["modes"]["with_mode"] == 2,         "neither arm was given the mode its writer recorded"
    group = alltime["groups"][0]["runs"][0]
    assert group["modes"] == ["greedy", "sampled"]


def test_readings_that_record_no_control_at_all_are_not_one_group():
    """`register_job.py` skips the naming guard, so such rows are reachable.

    Bucketing every one of them under a single `None` key and then sorting
    inside that card prints an idle-scale reading above a random-scale one
    under identical "scale unidentified" chips -- a ranking across the widest
    scale gap on the machine, drawn as though it were a card.
    """
    idle = [_row(1, 0, eval_lift_sd=2.5, eval_opponent="idle"),
            _row(2, 0, eval_lift_sd=2.1, eval_opponent="idle")]
    rand = [_row(1, 0, eval_lift_sd=0.9, eval_opponent="random")]
    for row in idle + rand:
        row.pop("control_win", None)

    _page, payload = render_multi([("idle-probe-run", idle, False),
                                   ("random-verdict-run", rand, False)])
    groups = payload["alltime"]["groups"]

    assert [g["scale"]["named"] for g in groups] == ["idle", "random"],         "two opponents were merged into one nameless card"
    assert all(len(g["runs"]) == 1 for g in groups)

    # And where nothing at all was recorded, the card is not ordered.
    blank = [_row(1, 0, eval_lift_sd=2.5), _row(2, 0, eval_lift_sd=0.9)]
    for row in blank:
        row.pop("control_win", None)
    _page, payload = render_multi([("b", blank[:1], False), ("a", blank[1:], False)])
    group = payload["alltime"]["groups"][0]
    assert group["rankable"] is False
    assert [r["name"] for r in group["runs"]] == ["a", "b"],         "readings with no recorded scale were ordered by size"


def test_a_half_transcribed_checkpoint_is_attributed_by_its_checkpoint_path():
    """Two sweeps on this machine both call an arm "w0.5".

    Joining a verdict record to a metrics row on the arm name alone
    attributes one sweep's reading to the other, which is the mistake the
    whole view exists to prevent -- and it named several runs while quoting
    only the first one's number.
    """
    verdict = [
        {"checkpoint": "runs/sweepA/w0.1/cloned.pt", "name": "w0.1",
         "mode": "greedy", "lift": -1.3, "episodes": 150,
         "eval_opponent": "random"},
        {"checkpoint": "runs/sweepA/w0.1/cloned.pt", "name": "w0.1",
         "mode": "sampled", "lift": 0.05, "episodes": 150,
         "eval_opponent": "random"},
    ]
    other = [_row(1, 0, episodes=150, arm="w0.1 hard, greedy",
                  eval_lift_sd=0.9, control_win=0.26)]
    third = [_row(1, 0, episodes=150, arm="w0.1, greedy",
                  eval_lift_sd=1.7, control_win=0.26)]

    _page, payload = render_multi(
        [("sweepB-sweep", other, False), ("sweepC-sweep", third, False)],
        extras={"verdicts": {"sweepA_verdict": verdict}})
    recorded = payload["alltime"]["modes"]["recorded"]

    assert len(recorded) == 1
    assert "half_transcribed" not in recorded[0],         "a checkpoint from one sweep was attributed to two unrelated runs"

    # With the run that actually holds those weights on the page, the join is
    # made -- and every run named is quoted with its own number.
    mine = [_row(1, 0, episodes=150, arm="w0.1, greedy",
                 eval_lift_sd=-1.3, control_win=0.26)]
    _page, payload = render_multi(
        [("sweepA/w0.1", mine, False), ("sweepC-sweep", third, False)],
        extras={"verdicts": {"sweepA_verdict": verdict}})
    half = payload["alltime"]["modes"]["recorded"][0]["half_transcribed"]
    assert half["mode"] == "greedy"
    assert half["runs"] == [{"run": "sweepA/w0.1", "lift": -1.3}]
    assert half["hidden"] == 0.05


def test_the_greedy_only_transcription_of_a_paired_verdict_is_caught():
    """runs/cloned is the case the file's own docstring calls dangerous.

    Its metrics rows carry no arm label and its verdict is paired, so the
    guard that exists to stop a greedy-only number being read as the answer
    never fired for it -- while the run tab reported +1.623 for a checkpoint
    that reads +0.709 the other way.
    """
    rows = [_row(1, 0, episodes=150, eval_lift_sd=1.623, eval_win=0.83,
                 control_win=0.26)]
    extras = {"verdicts": {"cloned": {
        "episodes": 150,
        "greedy": {"lift": 1.623, "win": 0.83},
        "sampled": {"lift": 0.709, "win": 0.55},
        "lift": 1.623}}}

    _page, payload = render_multi([("cloned", rows, False)], extras=extras)
    recorded = payload["alltime"]["modes"]["recorded"]

    assert [r["weight"] for r in recorded] == ["cloned"]
    half = recorded[0]["half_transcribed"]
    # Identified by an exact match against one of the two measured lifts, so
    # the mode is read off the evidence rather than assumed.
    assert half["mode"] == "greedy"
    assert half["runs"] == [{"run": "cloned", "lift": 1.623}]
    assert half["hidden"] == 0.709


def test_a_gap_is_not_printed_across_two_different_opponents():
    """The one place an opponent is on the object, and it was thrown away.

    A greedy record measured against one opponent paired with a sampled
    record measured against another was subtracted, printed in a column
    headed Gap, chipped "sign flips", and labelled "opponent recorded"
    without ever saying which.
    """
    verdict = [
        {"checkpoint": "runs/x/cloned.pt", "name": "w0.1", "mode": "greedy",
         "lift": -1.3, "episodes": 150, "eval_opponent": "random"},
        {"checkpoint": "runs/x/cloned.pt", "name": "w0.1", "mode": "sampled",
         "lift": 0.05, "episodes": 150, "eval_opponent": "idle"},
    ]
    _page, payload = render_multi([("x", [_row(1, 1000)], False)],
                                  extras={"verdicts": {"v": verdict}})
    item = payload["alltime"]["modes"]["recorded"][0]

    assert item["opponent_mismatch"] is True
    assert item["gap"] is None, "a 1.35 difference across two opponents was called a play-mode gap"
    assert item["opponent"] is None
    assert (item["opponent_greedy"], item["opponent_sampled"]) == ("random", "idle")


@needs_node
def test_the_modes_table_says_which_opponent_each_row_faced():
    """"opponent recorded" without the name is not a provenance chip."""
    verdict = [
        {"checkpoint": "runs/x/cloned.pt", "name": "w0.1", "mode": "greedy",
         "lift": -1.3, "episodes": 150, "eval_opponent": "random"},
        {"checkpoint": "runs/x/cloned.pt", "name": "w0.1", "mode": "sampled",
         "lift": 0.05, "episodes": 150, "eval_opponent": "random"},
    ]
    runs = [("x", [_row(1, 1000)], False)]
    out = _call_multi(runs, "modesMarkup(DATA.alltime)",
                      extras={"verdicts": {"v": verdict}})
    assert "vs random" in out and "opponent recorded" not in out


def test_two_measurements_are_only_the_same_measurement_on_the_same_weights():
    """The exhibit about re-running a reading joined on the arm name alone.

    Two verdicts holding an arm called "clone" at two different checkpoints
    were called "the same weights", their unknown mode was bucketed under the
    string "None", and the caption then asserted that mode was greedy and
    therefore deterministic. Three collapses in one sentence.
    """
    def record(checkpoint, mode=None):
        return [{"name": "clone", "checkpoint": checkpoint, "mode": mode,
                 "lift": 1.2345678, "episodes": 150, "eval_opponent": "random"}]

    _page, payload = render_multi(
        [("a", [_row(1, 1000)], False)],
        extras={"verdicts": {"v_one": record("runs/a/cloned.pt"),
                             "v_two": record("runs/b/cloned.pt")}})
    assert "resolution" not in payload["alltime"]["exhibits"],         "two checkpoints with one arm name were called the same weights"

    # The same checkpoint in two files is the real case, and it keeps working.
    _page, payload = render_multi(
        [("a", [_row(1, 1000)], False)],
        extras={"verdicts": {"v_one": record("runs/a/cloned.pt", "greedy"),
                             "v_two": record("runs/a/cloned.pt", "greedy")}})
    identical = payload["alltime"]["exhibits"]["resolution"]["identical"]
    assert identical["mode"] == "greedy" and identical["deterministic"] is True
    assert identical["checkpoint"] == "runs/a/cloned.pt"
    assert identical["sources"] == ["v_one", "v_two"]

    # The same checkpoint with no mode recorded is still not a re-run: only
    # greedy play on fixed seeds is the deterministic thing this exhibit is
    # about, and an unknown mode read as greedy is the assumption that hid a
    # working fine-tune for a day.
    _page, payload = render_multi(
        [("a", [_row(1, 1000)], False)],
        extras={"verdicts": {"v_one": record("runs/a/cloned.pt"),
                             "v_two": record("runs/a/cloned.pt")}})
    assert "resolution" not in payload["alltime"]["exhibits"],         "an unrecorded play mode was read as greedy and called deterministic"

    # And a record that names no checkpoint at all identifies no weights.
    def anonymous():
        return [{"name": "clone", "mode": "greedy", "lift": 1.2345678,
                 "episodes": 150}]

    _page, payload = render_multi(
        [("a", [_row(1, 1000)], False)],
        extras={"verdicts": {"v_one": anonymous(), "v_two": anonymous()}})
    assert "resolution" not in payload["alltime"]["exhibits"]


def test_a_selected_peak_is_only_compared_with_a_replay_of_the_same_thing():
    """Exhibit (b) attributed an opponent change to selection bias.

    runs/poc-vs-random's in-run readings sit at control 0.30 and its verdict
    reports 0.04, and the verdict measures final.pt while the peak came from
    best.pt -- which the note says replays at -0.033, not the +0.141 the
    panel printed. The payload knew about the control disagreement and
    printed it in a card far below instead.
    """
    rows = [_row(1, 1000, eval_lift_sd=0.375, eval_win=0.5, control_win=0.30),
            _row(2, 2000, eval_lift_sd=0.1, eval_win=0.4, control_win=0.30)]
    extras = {"verdicts": {"poc": {
        "episodes": 300, "checkpoint": "final.pt", "lift": 0.141,
        "ci_low": -0.018, "ci_high": 0.299, "control_win": 0.04,
        "control_draw": 0.913,
        "note": "best.pt, the highest of 19 readings, evaluates at -0.033 over 300."}}}

    _page, payload = render_multi([("poc", rows, False)], extras=extras)
    selection = payload["alltime"]["exhibits"]["selection"]

    assert selection["same_scale"] is False,         "0.30 and 0.04 were treated as one scale"
    assert selection["verdict_checkpoint"] == "final.pt"
    assert selection["same_checkpoint"] is None,         "two checkpoints were asserted to be the same weights"
    assert selection["best_scale"]["control"] == 0.30
    assert selection["verdict_scale"]["control"] == 0.04


@needs_node
def test_the_selection_exhibit_carries_its_scales_on_the_page():
    """Both numbers were bare, in the one panel that subtracts two lifts."""
    rows = [_row(1, 1000, eval_lift_sd=0.375, eval_win=0.5, control_win=0.30)]
    extras = {"verdicts": {"poc": {
        "episodes": 300, "checkpoint": "final.pt", "lift": 0.141,
        "control_win": 0.04, "note": "the peak replays at -0.033."}}}
    out = _call_multi([("poc", rows, False)], "exhibitsMarkup(DATA.alltime)",
                      extras=extras)

    assert '+0.375<span class="chip warn">scale unidentified (control 0.30)</span>' in out
    assert '+0.141<span class="chip warn">scale unidentified (control 0.04)</span>' in out
    assert "were not read against the same control" in out
    assert "final.pt" in out


# ------------------------------------------------------------- the ledger


def test_an_evaluation_run_is_not_counted_as_training_and_as_a_verdict():
    """runs/cloned and runs/search-expert have steps=0 and no gradients.

    Their `episodes` are the battles of the evaluation their own verdict file
    already counts, bit-identically, so adding them to the training line
    counts 190 battles twice under a heading that says "counted exactly".
    """
    trained = [_row(1, 4096, episodes=200)]
    cloned = [_row(1, 0, episodes=150, eval_lift_sd=1.623,
                   eval_win=0.83, control_win=0.26)]
    extras = {"verdicts": {"cloned": {"episodes": 150,
                                      "greedy": {"lift": 1.623},
                                      "sampled": {"lift": 0.709}}}}

    _page, payload = render_multi([("trained", trained, False),
                                   ("cloned", cloned, False)], extras=extras)
    ever = payload["alltime"]["ever"]
    counted = {c["what"]: c["n"] for c in ever["battles"]["counted"]}

    assert counted["training episodes"] == 200,         "an evaluation run's battles were counted as training as well"
    assert counted["paired verdict battles"] == 300
    assert ever["battles"]["total"] == 500
    # Not dropped: reported on its own, beside the runs it belongs to.
    assert ever["episodes"]["untrained"] == 150
    assert ever["untrained"] == ["cloned"]


def test_a_job_row_is_its_own_quantity_and_the_rows_add():
    """`_segment_total` is a cumulative-counter rule, and a job has no counter.

    Seven arms of 150 battles each is 1,050 battles; the cumulative rule read
    the largest row and reported 150, so the job the record is drawn from
    contributed a seventh of itself to the excluded line.
    """
    job = [_row(i, 0, episodes=150) for i in range(1, 8)]
    _page, payload = render_multi([("headtohead", job, False)],
                                  kinds={"headtohead": "job"})
    battles = payload["alltime"]["ever"]["battles"]

    assert payload["alltime"]["ever"]["episodes"]["jobs"] == 1050
    assert battles["excluded"]["n"] == 1050
    # And rows with no `what` of their own are named rather than invisible.
    assert battles["excluded"]["items"][0]["job"] == "headtohead"
    assert battles["excluded"]["items"][0]["n"] == 1050


def test_a_reading_that_records_its_own_size_is_counted_at_it():
    """The estimate's rule text claimed every reading used the 40 default.

    `rotating_probe` writes `eval_episodes`, and 14 rows on this machine
    record 150 while being counted at 40.
    """
    recorded = [_row(1, 1000, eval_lift_sd=0.4, control_win=0.26,
                     eval_episodes=150)]
    plain = [_row(1, 1000, eval_lift_sd=0.4, control_win=0.26)]

    _page, payload = render_multi([("a", recorded, False), ("b", plain, False)])
    estimated = payload["alltime"]["ever"]["battles"]["estimated"]

    # 150 recorded + 40 estimated + one control run per evaluating run.
    assert estimated["n"] == 150 + 40 + 80
    assert estimated["recorded"] == 1
    assert "1 of 2 readings record their own eval_episodes" in estimated["rule"]


def test_the_hours_tile_counts_the_same_population_it_sums():
    """A job logging elapsed time would raise a denominator it cannot move."""
    model = [_row(1, 1000, elapsed_seconds=600)]
    job = [_row(1, 0, elapsed_seconds=99, what="a benchmark")]

    _page, payload = render_multi([("m", model, False), ("j", job, False)],
                                  kinds={"j": "job"})
    ever = payload["alltime"]["ever"]

    assert ever["seconds"] == 600, "a job's elapsed time entered the total"
    assert ever["reporting_elapsed"] == 1,         "a job was counted as a run reporting elapsed time it does not contribute"
    assert ever["models"] == 1


@needs_node
def test_the_page_states_no_fact_about_the_data_it_has_not_counted():
    """Two sentences were prose about one afternoon's payload.

    "it takes five values. Two match a control that was actually measured.
    The other three do not" printed unchanged above six group cards, and
    "two of which are one evaluation written twice" printed above a tile
    whose own two numbers were equal.
    """
    runs = [("a", [_row(1, 1000, eval_lift_sd=0.5, control_win=0.44)], False),
            ("b", [_row(1, 1000, eval_lift_sd=0.3, control_win=0.26)], False)]

    out = _call_multi(runs, "groupsMarkup(DATA.alltime)+everMarkup(DATA.alltime)")
    assert "it takes five values" not in out
    assert "The other three do not" not in out
    assert "2 groups" in out and "1 where the opponent can be put a name to" in out
    assert "two of which are one evaluation written twice" not in out
    assert "none of them a repeat of another" in out


# ---------------------------------------------------- what reaches the page


def test_a_verdict_and_a_soak_summary_are_read_off_the_disk(tmp_path):
    """Nothing tested that the evidence layer reads anything at all.

    Every other test injects `extras=` by hand, so `_extras_of` could return
    an empty dict with the whole suite green -- and with it the intervals,
    the only recorded opponents on the machine, and ten thousand matches.
    """
    from cr_sim.train.watch import _extras_of

    runs = tmp_path / "runs"
    (runs / "cloned").mkdir(parents=True)
    (runs / "cloned" / "verdict.json").write_text(
        json.dumps({"episodes": 150, "greedy": {"lift": 1.6},
                    "sampled": {"lift": 0.7}}), encoding="utf-8")
    (runs / "soak-spells").mkdir()
    (runs / "soak-spells" / "summary.json").write_text(
        json.dumps({"matches": 10000, "mean_ticks": 1770.15}), encoding="utf-8")
    (runs / "half").mkdir()
    (runs / "half" / "verdict.json").write_text("{\"episodes\":", encoding="utf-8")

    extras = _extras_of([runs])
    assert list(extras["verdicts"]) == ["cloned"],         "a half-written verdict took the reader down with it"
    assert extras["verdicts"]["cloned"]["greedy"]["lift"] == 1.6
    assert extras["soak"]["matches"] == 10000
    assert extras["soak"]["run"] == "soak-spells"


def test_one_verdict_reachable_through_two_roots_is_counted_once(tmp_path):
    """A worktree holding a copy added a second set of battles for one file.

    The duplicate also matched no run label, so its own disputed-control
    check was skipped in silence.
    """
    from cr_sim.train.watch import _extras_of

    payload = {"episodes": 150, "lift": 1.6}
    roots = []
    for where in ("main", "wt"):
        root = tmp_path / where / "runs" / "cloned"
        root.mkdir(parents=True)
        (root / "verdict.json").write_text(json.dumps(payload), encoding="utf-8")
        roots.append(tmp_path / where / "runs")

    extras = _extras_of(roots)
    assert list(extras["verdicts"]) == ["cloned"]
    assert extras["duplicate_verdicts"] and "wt" in extras["duplicate_verdicts"][0]

    # A genuinely different file under the same name is still kept, renamed.
    (tmp_path / "wt" / "runs" / "cloned" / "verdict.json").write_text(
        json.dumps({"episodes": 300, "lift": 0.4}), encoding="utf-8")
    extras = _extras_of(roots)
    assert sorted(extras["verdicts"]) == ["cloned", "cloned/verdict"]
    assert extras["duplicate_verdicts"] == []


def test_the_job_split_is_read_from_config_and_survives_a_bad_one(tmp_path):
    """`_kind_of` gates the census, every counter and the ranking.

    It also took the watcher down on a config that is valid JSON and not an
    object, or that is not UTF-8 -- and the refresh loop catches only
    KeyboardInterrupt, so the served page froze exactly as it did for the NaN
    bug, with no sign it had stopped.
    """
    from cr_sim.train.watch import _kind_of, _note_of

    run = tmp_path / "job"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"kind": "job", "note": "hi"}),
                                     encoding="utf-8")
    assert _kind_of(run) == "job" and _note_of(run) == "hi"

    (run / "config.json").write_text(json.dumps({"note": "a model"}),
                                     encoding="utf-8")
    assert _kind_of(run) is None, "a trainer's config was called a job"

    for bad in ("null", "[1,2,3]", '"job"', "42", "{not json"):
        (run / "config.json").write_text(bad, encoding="utf-8")
        assert _kind_of(run) is None and _note_of(run) == "", bad
    (run / "config.json").write_bytes(
        json.dumps({"kind": "job"}).encode("utf-16"))
    assert _kind_of(run) is None and _note_of(run) == ""


def test_the_watcher_writes_a_page_and_names_the_runs_it_could_not(tmp_path):
    """End to end, through `main`, which no test had run before.

    A run directory with a config and no rows contributes nothing to any
    total and is exactly what a census is asked about, so it is named rather
    than dropped without a word.
    """
    from cr_sim.train.watch import main

    runs = tmp_path / "runs"
    (runs / "real").mkdir(parents=True)
    _write(runs / "real" / "metrics.jsonl", [_row(1, 1000, episodes=40)])
    (runs / "real" / "config.json").write_text("{}", encoding="utf-8")
    (runs / "empty").mkdir()
    (runs / "empty" / "metrics.jsonl").write_text("", encoding="utf-8")
    (runs / "empty" / "config.json").write_text(
        json.dumps({"total_steps": 400000}), encoding="utf-8")

    out = tmp_path / "progress.html"
    assert main([str(runs / "real"), str(runs / "empty"),
                 "--once", "--out", str(out)]) == 0
    body = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))

    assert list(body["runs"]) == ["real"]
    assert body["alltime"]["census"] == 1
    assert body["alltime"]["skipped"] == ["empty"],         "a run that started and wrote nothing vanished from the census"


def test_the_same_run_path_twice_is_still_one_run(tmp_path):
    """The de-duplication was applied to discovery and not to the arguments.

    Every model total doubled while `body["runs"]` collapsed to one key, so
    the census identity the page prints in its own footer broke.
    """
    from cr_sim.train.watch import main

    run = tmp_path / "runs" / "cloned"
    run.mkdir(parents=True)
    _write(run / "metrics.jsonl", [_row(1, 4096, episodes=150),
                                   _row(2, 8192, episodes=150)])

    out = tmp_path / "progress.html"
    assert main([str(run), str(run), "--once", "--out", str(out)]) == 0
    body = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))

    assert body["alltime"]["census"] == len(body["runs"]) == 1
    assert body["alltime"]["ever"]["episodes"]["models"] == 150
    assert body["alltime"]["lift_rows"] == 0
    assert body["order"] == ["cloned"], "one run, two tabs"
    # The duplicate never reaches the aggregate at all, rather than reaching
    # it and being absorbed: the same path twice is one run, not a collision.
    assert body["alltime"]["collisions"] == []


def test_a_resume_that_replays_its_update_numbers_keeps_both_segments(tmp_path):
    """The dedupe in `once()` deleted a whole pre-resume segment.

    A resume replays the same update numbers with the counters reset, so a
    dict keyed on `updates` across the whole file overwrites every pre-resume
    row with its post-resume namesake. The two runs on disk that actually
    resumed lost 816 real training battles and eleven minutes that way, under
    a tile claiming resumes are added segment by segment.
    """
    from cr_sim.train.watch import main

    run = tmp_path / "runs" / "resumed"
    run.mkdir(parents=True)
    _write(run / "metrics.jsonl", [
        _row(1, 1000, episodes=100, elapsed_seconds=60),
        _row(2, 2000, episodes=200, elapsed_seconds=120),
        # The same evaluation written twice under one update number, which is
        # what the dedupe is for and must keep doing.
        _row(2, 2000, episodes=200, elapsed_seconds=120),
        # Resumed: update numbers replay from 2 with the counters reset.
        _row(2, 2100, episodes=50, elapsed_seconds=30),
        _row(3, 3000, episodes=120, elapsed_seconds=75),
    ])

    out = tmp_path / "progress.html"
    assert main([str(run), "--once", "--out", str(out)]) == 0
    body = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    ever = body["alltime"]["ever"]

    # 200 across the first segment plus 120 across the second. Keying on
    # `updates` gives 120, and the largest row alone gives 200.
    assert ever["episodes"]["models"] == 320
    assert ever["seconds"] == 195
    assert len(body["runs"]["resumed"]["series"]["steps"]) == 4,         "the adjacent double-write survived, or a real row was dropped"


# -------------------------------------------------------- the page as read


@needs_node
def test_the_two_headline_panels_say_whether_they_are_on_one_scale():
    """They sit one above the other, so the second has to answer it first.

    An idle-anchored number printed under a random-scale record with no such
    line rebuilds the 92%-vs-26% collapse in the page's own headline panels.
    """
    name, ladder, note = _block()
    idler = [_row(1, 1000, eval_lift_sd=2.9, eval_win=0.97, control_win=0.925)]
    runs = [("old-idle-run", idler, False), (name, ladder, False)]
    kw = {"notes": {name: note}, "kinds": {name: "job"}}

    out = _call_multi(runs, "demotedHeading(DATA.alltime)+'|'"
                            "+demotedMarkup(DATA.alltime)", **kw)
    heading, markup = out.split("|", 1)

    assert "different scale" in heading,         "the heading still invites the comparison it exists to refuse"
    assert "Not the record" in markup, "the panel never says they differ"
    assert "26%" in markup and ("92%" in markup or "93%" in markup),         "the two control rates are never put side by side"
    assert "cannot be compared at all" in markup


@needs_node
def test_the_ladder_never_puts_a_sampled_arm_above_a_greedy_one():
    """A change can leave the argmax untouched and move the distribution.

    Ranking both modes in one list orders checkpoints by how they were
    played: greedy beats sampled for every paired checkpoint on this machine,
    so the bars are driven by the mode more than by the weights.
    """
    rows = [
        _row(1, 0, episodes=150, arm="A, greedy", eval_lift_sd=1.9,
             eval_win=0.8, control_win=0.26),
        _row(2, 0, episodes=150, arm="B, sampled", eval_lift_sd=2.4,
             eval_win=0.85, control_win=0.26),
        _row(3, 0, episodes=150, arm="B, greedy", eval_lift_sd=0.3,
             eval_win=0.4, control_win=0.26),
    ]
    runs = [("sweep-x", rows, False)]
    kw = {"notes": {"sweep-x": _GATE_NOTE}, "kinds": {"sweep-x": "job"}}

    _page, payload = render_multi(runs, **kw)
    assert [(g["mode"], [a["arm"] for a in g["arms"]])
            for g in payload["alltime"]["block"]["groups"]] == [
        ("greedy", ["A, greedy", "B, greedy"]),
        ("sampled", ["B, sampled"])]
    # And the record slot is not `max(greedy, sampled)` over the block.
    assert [(m["mode"], m["top"]["arm"]) for m in payload["alltime"]["record"]["modes"]] == [
        ("greedy", "A, greedy"), ("sampled", "B, sampled")]

    drawn = _call_multi(runs, "ladderMarkup(DATA.alltime)", **kw)
    assert drawn.index("+1.900") < drawn.index("+0.300") < drawn.index("+2.400"),         "one ranking over two play modes"
    assert "greedy play" in drawn and "sampled play" in drawn


def test_the_greedy_and_sampled_columns_are_both_readable_on_a_phone():
    """The page is built for a 390px screen and is served to one.

    Four columns in a 301px scroll box parked the sampled figure off the edge
    and the gap fully off-screen, so the default state of the page's own
    anti-greedy-bias table showed the greedy number alone -- including on the
    rows flagged "sign flips", where the hidden half is the other sign.
    """
    page = render([_row(1, 1000)], "run")
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]

    assert 'class="ledger modes"' in page, "the table cannot be targeted"
    stacked = [b for b in style.split("@media") if ".ledger.modes" in b]
    assert stacked, "the modes table has no narrow-screen layout at all"
    block = stacked[0].replace(" ", "").replace(chr(10), "")
    assert "max-width:700px" in block
    assert ".ledger.modestr{display:block" in block, "the rows do not stack"
    assert "content:attr(data-h)" in block, "a stacked cell with no header"
    for header in ("Greedy", "Sampled", "Gap"):
        assert 'data-h="' + header + '"' in page,             header + " has no label once the table stacks"


def test_the_page_asks_nothing_of_a_network_it_may_not_have():
    """It is read over the LAN from a phone whose wifi often has no route out.

    Three render-blocking requests to a font host buy nothing there, and
    everything else in the file -- the manifest, the icon -- is already a
    data: URI for exactly that reason.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.4, control_win=0.26)], "run")
    for host in ("fonts.googleapis.com", "fonts.gstatic.com"):
        assert host not in page, host + " is fetched at render time"
    # Every absolute URL in the file, except the SVG namespace, which names a
    # namespace and is never fetched.
    external = [u for u in re.findall(r"https?://[^\"' )]+", page)
                if not u.startswith("http://www.w3.org/")]
    assert external == [], "the page reaches outside itself: " + repr(external)
    for attribute in ("src=", "href="):
        for value in re.findall(attribute + r"\"([^\"]*)\"", page):
            assert value.startswith(("data:", "#", ".")) or "://" not in value,                 "a remote " + attribute + value
    # And every font it does ask for degrades to something the device has.
    for declaration in re.findall(r"font-family:([^;}]+)", page):
        assert any(g in declaration for g in ("sans-serif", "monospace", "ui-")),             "a font with no fallback: " + declaration


@needs_node
def test_the_run_tab_shows_both_arms_of_a_row_that_holds_two():
    """The all-time view is not the only place the greedy arm went missing.

    `rotating_probe` writes the sampled arm to `eval_lift_sd`, so a run using
    it draws its distribution's trajectory under the unqualified heading
    "lift vs control" while the argmax beside it appears nowhere.
    """
    rows = [_row(1, 1000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.26,
                 eval_lift_sd_greedy=1.2, eval_win_greedy=0.7),
            _row(2, 2000, eval_lift_sd=0.6, eval_win=0.55, control_win=0.26,
                 eval_lift_sd_greedy=1.25, eval_win_greedy=0.72)]
    _page, payload = render_multi([("probe", rows, True)])
    series = payload["runs"]["probe"]["series"]
    summary = payload["runs"]["probe"]["summary"]

    assert series["lift_greedy"] == [[1000, 1.2], [2000, 1.25]]
    assert summary["latest_lift"] == 0.6 and summary["latest_lift_greedy"] == 1.25
    assert summary["modes_recorded"] is True

    drawn = _call_multi([("probe", rows, True)],
                        "(function(){var o=[];var q=function(){return {innerHTML:''}};"
                        "return JSON.stringify(DATA.runs.probe.series.lift_greedy);})()")
    assert "1.25" in drawn

    # A run whose probe records one arm says nothing about a second.
    plain = [_row(1, 1000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.26)]
    _page, payload = render_multi([("probe", plain, True)])
    assert payload["runs"]["probe"]["series"]["lift_greedy"] == []
    assert payload["runs"]["probe"]["summary"]["modes_recorded"] is False
