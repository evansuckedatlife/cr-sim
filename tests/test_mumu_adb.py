"""AdbBridge must be fully testable without a running emulator: every method
that shells out takes an injectable ``runner``, so these tests check argv
construction, stdout parsing, and error handling against a fake process
rather than real adb. The one test that needs an actual MuMu instance is
gated on a connection probe so the suite stays green on a machine with
neither adb nor MuMu installed -- which, as of writing, is this one (see
cr_sim/mumu/adb.py's module docstring for what was actually checked).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import cr_sim.mumu.adb as adb_mod

#: Deliberately not a real path -- explicit adb_path is trusted verbatim and
#: never touches the filesystem, so any string does, but routing it through
#: Path() keeps the expected argv consistent with what AdbBridge itself
#: builds regardless of the host OS's path separator.
FAKE_ADB = str(Path("C:/fake/adb.exe"))


def _completed(stdout="", stderr="", returncode=0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRunner:
    """Stands in for cr_sim.mumu.adb._run_subprocess. Records every call so
    tests can assert on the exact argv AdbBridge built, and plays back one
    canned response (or exception) per call, in order."""

    def __init__(self, responses):
        self.calls: list[dict] = []
        self._responses = list(responses)

    def __call__(self, args, *, timeout, binary):
        self.calls.append({"args": list(args), "timeout": timeout, "binary": binary})
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class TestFindAdbBinary:
    """Isolated from whatever is actually installed on the machine running
    this suite -- shutil.which and the module's candidate-path list are both
    monkeypatched, so these pass identically whether or not MuMu happens to
    be installed here."""

    def test_prefers_path_over_bundled_candidates(self, monkeypatch, tmp_path):
        on_path = tmp_path / "path-adb.exe"
        on_path.write_text("")
        monkeypatch.setattr(adb_mod.shutil, "which", lambda name: str(on_path))
        monkeypatch.setattr(adb_mod, "_CANDIDATE_ADB_PATHS", (str(tmp_path / "bundled-adb.exe"),))
        assert adb_mod.find_adb_binary() == on_path

    def test_falls_back_to_candidate_paths_when_not_on_path(self, monkeypatch, tmp_path):
        bundled = tmp_path / "shell" / "adb.exe"
        bundled.parent.mkdir()
        bundled.write_text("")
        monkeypatch.setattr(adb_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(adb_mod, "_CANDIDATE_ADB_PATHS", (str(bundled),))
        assert adb_mod.find_adb_binary() == bundled

    def test_extra_candidates_are_checked_before_bundled_ones(self, monkeypatch, tmp_path):
        extra = tmp_path / "extra-adb.exe"
        extra.write_text("")
        bundled = tmp_path / "bundled-adb.exe"
        bundled.write_text("")
        monkeypatch.setattr(adb_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(adb_mod, "_CANDIDATE_ADB_PATHS", (str(bundled),))
        assert adb_mod.find_adb_binary(extra_candidates=[str(extra)]) == extra

    def test_returns_none_when_nothing_exists(self, monkeypatch):
        monkeypatch.setattr(adb_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(adb_mod, "_CANDIDATE_ADB_PATHS", ())
        assert adb_mod.find_adb_binary() is None


class TestAdbPathResolution:
    def test_explicit_adb_path_bypasses_search_entirely(self, monkeypatch):
        # If this ever called find_adb_binary, the test below would fail --
        # asserting that it does not is the point.
        def _boom():
            raise AssertionError("find_adb_binary should not be called when adb_path is explicit")

        monkeypatch.setattr(adb_mod, "find_adb_binary", _boom)
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB)
        assert bridge.adb_path == Path(FAKE_ADB)

    def test_raises_adb_not_found_error_when_nothing_is_found(self, monkeypatch):
        monkeypatch.setattr(adb_mod, "find_adb_binary", lambda: None)
        bridge = adb_mod.AdbBridge()
        with pytest.raises(adb_mod.AdbNotFoundError):
            bridge.adb_path

    def test_resolution_is_cached_after_the_first_lookup(self, monkeypatch):
        calls = {"n": 0}

        def _find():
            calls["n"] += 1
            return Path(FAKE_ADB)

        monkeypatch.setattr(adb_mod, "find_adb_binary", _find)
        bridge = adb_mod.AdbBridge()
        assert bridge.adb_path == Path(FAKE_ADB)
        assert bridge.adb_path == Path(FAKE_ADB)
        assert calls["n"] == 1


class TestSerial:
    def test_serial_is_host_colon_port(self):
        bridge = adb_mod.AdbBridge(host="127.0.0.1", port=7555, adb_path=FAKE_ADB)
        assert bridge.serial == "127.0.0.1:7555"

    def test_serial_reflects_a_multi_instance_port(self):
        bridge = adb_mod.AdbBridge(port=adb_mod.instance_port(1), adb_path=FAKE_ADB)
        assert bridge.serial == "127.0.0.1:16416"


class TestInstancePort:
    def test_sequence_matches_mumu_player_12s_own_numbering(self):
        assert adb_mod.instance_port(0) == 16384
        assert adb_mod.instance_port(1) == 16416
        assert adb_mod.instance_port(2) == 16448

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError):
            adb_mod.instance_port(-1)


class TestDevices:
    def test_parses_multiple_lines_and_skips_the_header(self):
        runner = FakeRunner(
            [_completed(stdout="List of devices attached\n127.0.0.1:7555\tdevice\nemulator-5554\toffline\n\n")]
        )
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.devices() == (
            adb_mod.AdbDevice("127.0.0.1:7555", "device"),
            adb_mod.AdbDevice("emulator-5554", "offline"),
        )

    def test_builds_the_expected_argv(self):
        runner = FakeRunner([_completed(stdout="List of devices attached\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        bridge.devices()
        assert runner.calls[0]["args"] == [FAKE_ADB, "devices"]
        assert runner.calls[0]["binary"] is False

    def test_empty_list_returns_empty_tuple(self):
        runner = FakeRunner([_completed(stdout="List of devices attached\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.devices() == ()


class TestIsConnected:
    def test_true_when_this_bridges_serial_is_state_device(self):
        runner = FakeRunner([_completed(stdout="List of devices attached\n127.0.0.1:7555\tdevice\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.is_connected() is True

    def test_false_when_serial_absent(self):
        runner = FakeRunner([_completed(stdout="List of devices attached\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.is_connected() is False

    def test_false_when_serial_present_but_offline(self):
        runner = FakeRunner([_completed(stdout="List of devices attached\n127.0.0.1:7555\toffline\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.is_connected() is False


class TestConnect:
    def test_succeeds_when_the_device_appears_afterward(self):
        runner = FakeRunner(
            [
                _completed(stdout="connected to 127.0.0.1:7555\n"),
                _completed(stdout="List of devices attached\n127.0.0.1:7555\tdevice\n"),
            ]
        )
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        bridge.connect()  # must not raise
        assert runner.calls[0]["args"] == [FAKE_ADB, "connect", "127.0.0.1:7555"]

    def test_raises_when_the_device_never_appears(self):
        # Models adb's own quirk of exiting 0 even on a refused connection --
        # connect() must not trust that and must check devices() itself.
        runner = FakeRunner(
            [
                _completed(stdout="cannot connect to 127.0.0.1:7555: refused\n", returncode=0),
                _completed(stdout="List of devices attached\n"),
            ]
        )
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        with pytest.raises(adb_mod.AdbCommandError):
            bridge.connect()


class TestShell:
    def test_builds_expected_argv_and_returns_stdout(self):
        runner = FakeRunner([_completed(stdout="hello\n")])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.shell("echo hello") == "hello\n"
        call = runner.calls[0]
        assert call["args"] == [FAKE_ADB, "-s", bridge.serial, "shell", "echo hello"]
        assert call["binary"] is False


class TestExecOut:
    def test_builds_expected_argv_and_uses_binary_mode(self):
        payload = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        runner = FakeRunner([_completed(stdout=payload)])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        result = bridge.exec_out("screencap -p")
        assert result == payload
        call = runner.calls[0]
        assert call["args"] == [FAKE_ADB, "-s", bridge.serial, "exec-out", "screencap -p"]
        assert call["binary"] is True

    def test_bytes_are_returned_unmodified_no_crlf_translation(self):
        # The whole point of exec-out over `adb shell`: 0x0A must survive.
        payload = b"\x00\x0a\x0d\x0a\xff" * 10
        runner = FakeRunner([_completed(stdout=payload)])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        assert bridge.exec_out("screencap -p") == payload


class TestErrorHandling:
    def test_timeout_raises_adb_timeout_error(self):
        def raising_runner(args, *, timeout, binary):
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=raising_runner, timeout=0.01)
        with pytest.raises(adb_mod.AdbTimeoutError):
            bridge.devices()

    def test_nonzero_returncode_raises_adb_command_error_with_stderr(self):
        runner = FakeRunner([_completed(stderr="error: no devices/emulators found", returncode=1)])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        with pytest.raises(adb_mod.AdbCommandError) as excinfo:
            bridge.shell("echo hi")
        assert excinfo.value.returncode == 1
        assert "no devices/emulators found" in excinfo.value.stderr
        assert "no devices/emulators found" in str(excinfo.value)

    def test_exec_out_failure_decodes_bytes_stderr(self):
        runner = FakeRunner([_completed(stdout=b"", stderr=b"boom", returncode=1)])
        bridge = adb_mod.AdbBridge(adb_path=FAKE_ADB, runner=runner)
        with pytest.raises(adb_mod.AdbCommandError) as excinfo:
            bridge.exec_out("screencap -p")
        assert excinfo.value.stderr == "boom"

    def test_adb_not_found_and_command_errors_are_both_adb_errors(self):
        assert issubclass(adb_mod.AdbNotFoundError, adb_mod.AdbError)
        assert issubclass(adb_mod.AdbTimeoutError, adb_mod.AdbError)
        assert issubclass(adb_mod.AdbCommandError, adb_mod.AdbError)


def _probe_real_mumu_device() -> bool:
    """True only if adb is actually reachable and a MuMu instance is
    connected on the default endpoint. Evaluated once at collection time so
    the gated test below never touches a subprocess unless both are true."""
    try:
        bridge = adb_mod.AdbBridge()
        bridge.adb_path  # raises AdbNotFoundError if adb isn't installed anywhere findable
        return bridge.is_connected()
    except adb_mod.AdbError:
        return False


@pytest.mark.skipif(
    not _probe_real_mumu_device(), reason="no MuMu/adb device connected on this machine"
)
def test_real_device_shell_echo_round_trips():
    bridge = adb_mod.AdbBridge()
    bridge.connect()
    assert bridge.shell("echo cr-sim-mumu-probe").strip() == "cr-sim-mumu-probe"
