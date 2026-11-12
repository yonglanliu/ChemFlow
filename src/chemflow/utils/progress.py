"""Shared progress reporting: "N / total done" with rate and ETA.

Used by the docking paths (``alchemfold dock`` batches, ``alchemfold screen``
shards) and the active-learning oracle, so they all report the same way.

Writes a live bar when stdout is a terminal and periodic plain lines when it is
not (a SLURM log must not fill with carriage returns).
"""

from __future__ import annotations

import sys
import time

__all__ = ["Progress", "format_clock"]


def format_clock(seconds: float) -> str:
    """Seconds -> H:MM:SS (or M:SS when under an hour)."""
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


class Progress:
    """Count completed items out of a known total.

    ``total`` may be None when the count is not known up front; the bar then
    just reports how many are done, with no percentage or ETA.
    """

    def __init__(self, total: int | None, label: str = "progress",
                 unit: str = "compound", stream=None, log_every: float = 0.10):
        self.total = total or None
        self.label = label
        self.unit = unit
        self.done = 0
        self.failed = 0
        self.start = time.time()
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._log_every = log_every
        self._next_log = 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def rate(self) -> float:
        return self.done / self.elapsed if self.elapsed > 0 else 0.0

    def update(self, n: int = 1, failed: int = 0, note: str = "") -> None:
        self.done += n
        self.failed += failed
        self._render(note)

    def _render(self, note: str = "") -> None:
        suffix = f"  {note}" if note else ""
        fails = f"  {self.failed} failed" if self.failed else ""
        if self.total:
            frac = self.done / self.total
            eta = (self.total - self.done) / self.rate if self.rate > 0 else 0.0
            if self.tty:
                width = 28
                filled = int(width * frac)
                self.stream.write(
                    f"\r{self.label} [{'#' * filled}{'.' * (width - filled)}] "
                    f"{self.done}/{self.total} {self.unit}s ({frac * 100:5.1f}%)"
                    f"{fails}  {self.rate:.2f}/s  ETA {format_clock(eta)}{suffix}"
                )
                self.stream.flush()
            elif frac >= self._next_log or self.done == self.total:
                print(f"{self.label}: {self.done}/{self.total} {self.unit}s "
                      f"({frac * 100:.0f}%){fails} {self.rate:.2f}/s "
                      f"ETA {format_clock(eta)}{suffix}", file=self.stream, flush=True)
                self._next_log = frac + self._log_every
        elif self.tty:
            self.stream.write(f"\r{self.label} {self.done} {self.unit}s{fails} "
                              f"({self.rate:.2f}/s){suffix}")
            self.stream.flush()

    def close(self) -> str:
        """Finish the line and return a one-line summary (also printed)."""
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
        total = f"/{self.total}" if self.total else ""
        fails = f", {self.failed} failed" if self.failed else ""
        summary = (f"{self.label}: {self.done}{total} {self.unit}s finished"
                   f"{fails} in {format_clock(self.elapsed)} "
                   f"({self.rate:.2f} {self.unit}s/s)")
        print(summary, file=self.stream, flush=True)
        return summary

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
