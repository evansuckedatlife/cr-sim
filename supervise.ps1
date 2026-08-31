# Keep a training run alive across process death.
#
# This machine bugchecks. A run that dies at hour four should cost one
# checkpoint interval, not four hours -- so this restarts the trainer with
# --resume, which continues from checkpoint.pt with the optimiser state and
# step count intact.
#
# It cannot survive a bugcheck: the machine goes down and takes this with it.
# After a reboot, run this same command again -- it will find the checkpoint
# and pick up where the run stopped.
#
#   powershell -File supervise.ps1 -Name my-run -Steps 500000
#
# -Name has no default on purpose. It used to be "projected-v2", which is an
# existing 20-row run directory, so launching this with no arguments -- the
# thing you do after a bugcheck, from muscle memory -- quietly resumed
# somebody else's experiment from its checkpoint and wrote into its metrics.
# Refused with a message rather than declared Mandatory, because a mandatory
# parameter prompts, and the caller here is as often an agent as a person.

param(
    [string]$Name,
    [int]$Steps      = 500000,
    [int]$MaxRestarts = 50
)

if (-not $Name) {
    Write-Error "-Name is required: it is the run directory under runs\ and the label on the progress page. There is no safe default, because every default is some other run's name."
    exit 2
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDir = Join-Path $root "runs\$Name"
$log = Join-Path $root "$Name.log"

$common = @(
    "-m", "cr_sim.train.run",
    "--steps", "$Steps",
    "--envs", "6",
    "--horizon", "256",
    "--tps", "20",
    "--frame-skip", "30",
    "--match-seconds", "120",
    "--reward", "projected",
    "--horizon-seconds", "3",
    # Both spelled out, and neither is optional. They are the two settings the
    # handoff calls load-bearing, they used to be left to run.py's defaults of
    # 11 and 0.3, and this launcher is the one you reach for after a bugcheck
    # -- so every crash-resilient run silently trained at the level where 92%
    # of matches draw, charged for elixir it was never rewarded for spending.
    # runs/learn-1m-factored-lvl11 is 557,056 steps of that. Written here even
    # though run.py now defaults the same way, because a launcher that relies
    # on a default records nothing, and config.json is where a run says what
    # it was.
    "--tower-level", "5",
    "--elixir-weight", "0.0",
    "--opponent", "self",
    "--eval-every", "10",
    # Frequent on purpose. A checkpoint every ten updates is roughly half an
    # hour of work to lose; every three is under ten minutes.
    "--save-every", "3",
    "--seed", "0",
    "--name", $Name
)

for ($attempt = 0; $attempt -le $MaxRestarts; $attempt++) {
    $resume = Test-Path (Join-Path $runDir "checkpoint.pt")
    $argv = if ($resume) { $common + @("--resume") } else { $common }

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $mode = if ($resume) { "resuming" } else { "starting fresh" }
    Add-Content -Path $log -Value "[$stamp] attempt $attempt, $mode" -Encoding utf8

    $p = Start-Process -FilePath "python" -ArgumentList $argv `
        -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput "$root\$Name.out" `
        -RedirectStandardError  "$root\$Name.err"
    $p.WaitForExit()
    $code = $p.ExitCode

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($code -eq 0) {
        Add-Content -Path $log -Value "[$stamp] finished cleanly" -Encoding utf8
        break
    }
    Add-Content -Path $log -Value "[$stamp] exited $code, restarting in 15s" -Encoding utf8

    # Without a checkpoint there is nothing to resume from, so a crash before
    # the first save would restart from zero forever. Stop instead and leave
    # the error where someone will read it.
    if (-not (Test-Path (Join-Path $runDir "checkpoint.pt"))) {
        Add-Content -Path $log -Value "[$stamp] died before any checkpoint; giving up" -Encoding utf8
        break
    }
    Start-Sleep -Seconds 15
}
