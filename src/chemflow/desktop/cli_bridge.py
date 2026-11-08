"""Qt-free helpers for the desktop → ``alchemforge`` CLI bridge (testable without PySide6)."""

from __future__ import annotations

from pathlib import Path

ALCHEMFORGE_MODULE = "alchemforge.cli"


def render_step(argv_template: list[str], config: str) -> list[str]:
    """Fill ``{config}`` / ``{name}`` placeholders in a CLI argv template.

    ``{config}`` → the config path as given; ``{name}`` → its filename stem
    (AlchemForge's ``generate`` commands take a project name, not a path).
    """
    stem = Path(config).stem if config else ""
    return [token.replace("{config}", config).replace("{name}", stem) for token in argv_template]
