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
                 "Pass rate"):
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


def _call(rows, expression):
    """Evaluate `expression` against the page's own chart code."""
    body = _script_body(rows)
    result = subprocess.run(
        [node, "-e", body + chr(10) + "process.stdout.write(String(" + expression + "));"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


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
    result = subprocess.run(
        [node, "-e", body + stub + chr(10) + "process.stdout.write(String(" + expression + "));"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _run_js(rows, harness):
    """Run arbitrary JS `harness` statements after the page's function
    definitions, for tests that need more than one expression -- a scripted
    sequence, or a fake DOM/fetch/localStorage the page's functions call
    into. Returns whatever the harness wrote to stdout."""
    body = _script_body(rows)
    result = subprocess.run(
        [node, "-e", body + harness], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


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
