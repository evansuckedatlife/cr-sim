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
                 "Recorded hours", "Distinct evaluations"):
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
        result = subprocess.run([node, str(script)], capture_output=True,
                                text=True, timeout=60)
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
    # Best-first, and computed rather than copied: the fixture is not in this
    # order and beta appears twice under two different ways of playing.
    assert [(a["arm"], a["lift"]) for a in block["arms"]] == [
        ("beta, greedy", 0.55), ("alpha, greedy", 0.40), ("beta, sampled", 0.22)]
    assert block["seeds"] == 150

    ranked = [a["arm"] for a in block["arms"]]
    assert not any("idler" in arm for arm in ranked), \
        "an idle-scale lift was ranked against random-scale ones"
    # And it is not simply dropped -- the largest number on the machine is
    # shown, in the one place it cannot be mistaken for the record.
    assert payload["alltime"]["demoted"]["name"] == "idler"
    assert payload["alltime"]["demoted"]["lift"] == 0.90
    assert payload["alltime"]["record"]["top"]["lift"] == 0.55, \
        "the record was taken from the biggest number rather than the best comparable one"


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
    assert record["top"]["mode"] == "greedy" and record["twin"]["mode"] == "sampled", \
        "the record is not a pair, so one of the two numbers can be quoted alone"
    assert record["twin"]["lift"] == 0.22


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
    assert ever["episodes"]["jobs"] == 9000, "a job leaked into the model tally"
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
    assert battles["estimated"]["n"] not in (battles["total"],), "the estimate was added in"
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
    assert len(recorded) == 1
    # Renamed for the reader: "_diag" is a scratch directory, not an arm.
    assert recorded[0]["weight"] == "baseline clone"
    assert recorded[0]["opponent"] == "random", \
        "the one file that records its opponent was not read for it"
    assert recorded[0]["gap"] == pytest.approx(1.623 - 0.734)
    assert recorded[0]["straddles_zero"] is True

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
    _page, payload = render_multi([(name, rows, False)],
                                  notes={name: note}, kinds={name: "job"},
                                  extras={"verdicts": {
                                      "expert": {"episodes": 40, "lift": 2.7,
                                                 "ci_low": 2.4, "ci_high": 3.1}}})
    alltime = payload["alltime"]

    for arm in alltime["block"]["arms"]:
        assert "ci" not in arm, "a ladder row grew an interval from nowhere"
    for pair in alltime["modes"]["pairs"]:
        assert pair["greedy"]["ci"] is None and pair["sampled"]["ci"] is None
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
    for block in hidden:
        assert ".view-toggle" not in block, \
            "the only way into the all-time view is hidden at some width"


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
    runs = [("mystery", unknown, False), (name, ladder, False)]
    kw = {"notes": {name: note}, "kinds": {name: "job"}}

    drawn = _call_multi(runs, "ladderMarkup(DATA.alltime)", **kw)
    assert '+0.550<span class="chip good">vs random (stated)</span>' in drawn, \
        "the number and its scale are not in the same element"

    groups = _call_multi(runs, "groupsMarkup(DATA.alltime)", **kw)
    assert '+0.610<span class="chip warn">scale unidentified (control 0.85)</span>' in groups
    assert "vs idle" not in groups, "an unmatched control rate was given a name"


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
