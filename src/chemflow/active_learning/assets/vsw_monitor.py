#!/usr/bin/env python3
"""Launch + monitor a Schrodinger VSW job, then run a follow-up command.

Polls `$SCHRODINGER/jobcontrol -status` until the job reaches a terminal state,
prints heartbeat/status transitions, retries on failure (with -RESTART), and on
success runs an optional `--after` command (e.g. extract scores + train).

Run through Schrodinger's Python (so $SCHRODINGER is set), or any python3 as long
as $SCHRODINGER is exported:

  python3 vsw_monitor.py --inp vsw.inp \
      --vsw-args "-host_glide localhost:1 -host_ligprep localhost:1 \
                  -host_mmod localhost:1 -host_qikprop localhost:1 -adjust -NJOBS 1 -TMPLAUNCHDIR" \
      --interval 60 --retries 1 \
      --after "bash parse_and_train.sh"

  # or monitor an already-running job:
  python3 vsw_monitor.py --jobid <JobId> --interval 60
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

SCHRO = os.environ.get("SCHRODINGER")

# jobcontrol -status terminal strings (lowercased, substring match).
TERMINAL_OK = ("completed", "finished", "incorporated")
TERMINAL_BAD = ("died", "fizzled", "stopped", "killed", "failed")


def _sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def launch(inp: str, vsw_args: list[str], restart: bool = False) -> str:
    """Start VSW (no -WAIT) and return its JobId by parsing the launcher output."""
    cmd = [f"{SCHRO}/vsw", inp, "-OVERWRITE"]
    if restart:
        cmd.append("-RESTART")
    cmd += vsw_args
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    p = _sh(cmd)
    out = (p.stdout or "") + (p.stderr or "")
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    m = re.search(r"JobId:\s*(\S+)", out)
    if not m:
        raise SystemExit("ERROR: no JobId in vsw output — launch failed (check license/hosts).")
    jid = m.group(1)
    print(f"Launched JobId: {jid}", flush=True)
    return jid


def job_state(jobid: str) -> tuple[str, str]:
    """Return (state, raw) where state in {'ok','bad','running'}."""
    raw = _sh([f"{SCHRO}/jobcontrol", "-status", jobid]).stdout.strip()
    low = raw.lower()
    if any(k in low for k in TERMINAL_OK):
        return "ok", raw
    if any(k in low for k in TERMINAL_BAD):
        return "bad", raw
    return "running", raw


def monitor(jobid: str, interval: int, timeout: int) -> bool:
    start = time.time()
    last = None
    while True:
        state, raw = job_state(jobid)
        elapsed = int(time.time() - start)
        if raw != last:
            print(f"[{elapsed:>6}s] {jobid}: {raw.splitlines()[0] if raw else '(no status)'}", flush=True)
            last = raw
        if state == "ok":
            print(f"[{elapsed}s] COMPLETED", flush=True)
            return True
        if state == "bad":
            print(f"[{elapsed}s] FAILED", flush=True)
            return False
        if timeout and elapsed > timeout:
            print(f"[{elapsed}s] TIMEOUT after {timeout}s (job still {state})", flush=True)
            return False
        time.sleep(interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Launch + monitor a VSW job, then run --after.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inp", help="VSW .inp to launch")
    g.add_argument("--jobid", help="monitor an already-running JobId instead of launching")
    ap.add_argument("--vsw-args", default="", help="extra args passed to vsw (quoted)")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds (default 60)")
    ap.add_argument("--timeout", type=int, default=0, help="give up after N s (0 = never)")
    ap.add_argument("--retries", type=int, default=1, help="relaunch with -RESTART on failure")
    ap.add_argument("--after", default="", help="shell command to run on success (e.g. extract + train)")
    args = ap.parse_args(argv)

    if not SCHRO:
        print("ERROR: $SCHRODINGER not set (module load schrodinger).", file=sys.stderr)
        return 2

    vsw_args = shlex.split(args.vsw_args)

    if args.jobid:
        ok = monitor(args.jobid, args.interval, args.timeout)
    else:
        ok = False
        for attempt in range(args.retries + 1):
            jid = launch(args.inp, vsw_args, restart=(attempt > 0))
            ok = monitor(jid, args.interval, args.timeout)
            if ok:
                break
            if attempt < args.retries:
                print(f"Retry {attempt + 1}/{args.retries} with -RESTART …", flush=True)

    if not ok:
        print("Job did not complete successfully — NOT running --after.", file=sys.stderr)
        return 1

    if args.after:
        print(f"$ {args.after}", flush=True)
        rc = subprocess.run(args.after, shell=True).returncode
        if rc != 0:
            print(f"--after command failed (exit {rc}).", file=sys.stderr)
            return rc
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
