"""
mdp_tools.py

Utilities for parsing and writing GROMACS .mdp files while preserving
comments, blank lines, and section headers. Supports generating a full
set of per-lambda-window .mdp files for FEP/RBFE calculations.

Usage examples are at the bottom of this file (under `if __name__ == "__main__"`).
"""

import os
import re
from pathlib import Path
from typing import Optional


class MDPFile:
    """
    Represents a GROMACS .mdp file as an ordered sequence of lines, each
    classified as either a parameter line, a comment line, or a blank line.

    Parameters are stored in a dict for easy lookup/modification, while the
    original line order (including comments and blank lines) is preserved
    for writing back out, so round-tripping a file doesn't scramble its
    formatting or strip its documentation.
    """

    def __init__(self):
        # Ordered list of "tokens" describing the file structure.
        # Each token is a dict: {"type": "param"|"comment"|"blank", ...}
        self._lines = []
        # Quick-lookup dict: parameter name -> index into self._lines
        self._param_index = {}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @classmethod
    def read(cls, filepath: str) -> "MDPFile":
        """Parse an existing .mdp file into an MDPFile object."""
        mdp = cls()
        with open(filepath, "r") as f:
            raw_lines = f.readlines()

        for raw_line in raw_lines:
            mdp._parse_line(raw_line.rstrip("\n"))

        return mdp

    def _parse_line(self, line: str):
        stripped = line.strip()

        if stripped == "":
            self._lines.append({"type": "blank"})
            return

        if stripped.startswith(";"):
            self._lines.append({"type": "comment", "text": line})
            return

        # Parameter line: "key = value ; optional inline comment"
        # Split off inline comment first, if present
        if ";" in line:
            code_part, _, inline_comment = line.partition(";")
            inline_comment = ";" + inline_comment
        else:
            code_part, inline_comment = line, None

        if "=" not in code_part:
            # Malformed/unexpected line — keep it verbatim rather than
            # silently dropping or crashing, so nothing is lost on write.
            self._lines.append({"type": "comment", "text": line})
            return

        key, _, value = code_part.partition("=")
        key = key.strip()
        value = value.strip()

        entry = {
            "type": "param",
            "key": key,
            "value": value,
            "inline_comment": inline_comment,
        }
        self._param_index[key] = len(self._lines)
        self._lines.append(entry)

    # ------------------------------------------------------------------
    # Access / modification
    # ------------------------------------------------------------------
    def get(self, key: str, default=None) -> Optional[str]:
        """Return the raw string value for a parameter, or `default`."""
        idx = self._param_index.get(key)
        if idx is None:
            return default
        return self._lines[idx]["value"]

    def set(self, key: str, value, comment: Optional[str] = None):
        """
        Set a parameter's value, creating it if it doesn't already exist.
        `value` is converted to a string automatically (so ints/floats
        can be passed directly). If `comment` is given and the parameter
        is newly created, it's added as an inline comment.
        """
        value = str(value)
        idx = self._param_index.get(key)

        if idx is not None:
            self._lines[idx]["value"] = value
            if comment is not None:
                self._lines[idx]["inline_comment"] = (
                    comment if comment.strip().startswith(";") else f"; {comment}"
                )
        else:
            entry = {
                "type": "param",
                "key": key,
                "value": value,
                "inline_comment": (
                    comment
                    if (comment is None or comment.strip().startswith(";"))
                    else f"; {comment}"
                ),
            }
            self._param_index[key] = len(self._lines)
            self._lines.append(entry)

    def remove(self, key: str):
        """Remove a parameter entirely (line is deleted, not just blanked)."""
        idx = self._param_index.pop(key, None)
        if idx is not None:
            self._lines[idx] = {"type": "blank"}  # keep spacing, drop content

    def as_dict(self) -> dict:
        """Return all parameters as a plain {key: value} dict."""
        return {
            entry["key"]: entry["value"]
            for entry in self._lines
            if entry["type"] == "param"
        }

    def append_section(self, header: str, params: dict):
        """
        Append a new commented section header followed by a block of
        parameters — useful for adding, e.g., a full free-energy block
        onto a base .mdp that doesn't have one yet.
        """
        self._lines.append({"type": "blank"})
        self._lines.append({"type": "comment", "text": f"; {'=' * 64}"})
        self._lines.append({"type": "comment", "text": f";{header.center(64)}"})
        self._lines.append({"type": "comment", "text": f"; {'=' * 64}"})
        for key, value in params.items():
            self.set(key, value)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def render(self, key_width: int = 26) -> str:
        """Render the current state back into .mdp file text."""
        out_lines = []
        for entry in self._lines:
            if entry["type"] == "blank":
                out_lines.append("")
            elif entry["type"] == "comment":
                out_lines.append(entry["text"])
            elif entry["type"] == "param":
                key_str = entry["key"].ljust(key_width)
                line = f"{key_str} = {entry['value']}"
                if entry["inline_comment"]:
                    line += f"  {entry['inline_comment']}"
                out_lines.append(line)
        return "\n".join(out_lines) + "\n"

    def write(self, filepath: str, key_width: int = 26):
        """Write the current state out to a .mdp file."""
        with open(filepath, "w") as f:
            f.write(self.render(key_width=key_width))

    def copy(self) -> "MDPFile":
        """Return a deep-enough copy safe to modify independently."""
        new = MDPFile()
        new._lines = [dict(line) for line in self._lines]
        new._param_index = dict(self._param_index)
        return new

    def __repr__(self):
        n_params = sum(1 for l in self._lines if l["type"] == "param")
        return f"<MDPFile: {n_params} parameters>"


# ----------------------------------------------------------------------
# FEP-specific helper: generate one .mdp per lambda window
# ----------------------------------------------------------------------
def generate_lambda_mdps(
    template_path: str,
    output_dir: str,
    n_windows: int,
    output_filename: str = None,
    lambda_param: str = "init-lambda-state",
):
    """
    Read a template .mdp (e.g. em_FEP_AMBER.mdp) and write out one copy
    per lambda window, with `lambda_param` set to 0, 1, ..., n_windows-1.

    Files are written to  output_dir/lambda_<N>/<output_filename>

    Parameters
    ----------
    template_path : path to the template .mdp file
    output_dir    : base directory under which lambda_0, lambda_1, ... live
    n_windows     : how many lambda windows to generate (0-indexed)
    output_filename : name to give the written file in each lambda_N folder
                       (defaults to the template's own filename)
    lambda_param  : the .mdp key to set per window (default: init-lambda-state)
    """
    template = MDPFile.read(template_path)

    if output_filename is None:
        output_filename = Path(template_path).name

    written_paths = []
    for lam in range(n_windows):
        window_dir = os.path.join(output_dir, f"lambda_{lam}")
        os.makedirs(window_dir, exist_ok=True)

        mdp = template.copy()
        mdp.set(lambda_param, lam)

        out_path = os.path.join(window_dir, output_filename)
        mdp.write(out_path)
        written_paths.append(out_path)

    return written_paths


def verify_lambda_vectors_consistent(mdp_paths: list, vector_keys: list = None):
    """
    Sanity check: confirm that lambda vector parameters (coul-lambdas,
    vdw-lambdas, etc.) are byte-identical across a list of .mdp files,
    as they must be — only init-lambda-state should differ between windows.

    Returns a dict of {key: True/False} indicating consistency per key.
    Raises no exceptions; prints a human-readable report.
    """
    if vector_keys is None:
        vector_keys = [
            "coul-lambdas",
            "vdw-lambdas",
            "bonded-lambdas",
            "restraint-lambdas",
            "sc-alpha",
            "sc-power",
            "sc-sigma",
        ]

    parsed = [MDPFile.read(p) for p in mdp_paths]
    results = {}

    for key in vector_keys:
        values = {p: mdp.get(key) for p, mdp in zip(mdp_paths, parsed)}
        unique_values = set(values.values())
        consistent = len(unique_values) <= 1
        results[key] = consistent
        if not consistent:
            print(f"[INCONSISTENT] '{key}' differs across files:")
            for p, v in values.items():
                print(f"    {p}: {v}")

    if all(results.values()):
        print(f"All {len(vector_keys)} lambda-vector parameters are consistent "
              f"across {len(mdp_paths)} files.")

    return results


def check_no_placeholders(mdp_paths: list, placeholder_pattern: str = r"\{.*?\}"):
    """
    Confirm no unreplaced template placeholders (e.g. leftover
    '{LAMBDA_STATE}') remain in a set of generated .mdp files.
    """
    pattern = re.compile(placeholder_pattern)
    problems = []

    for path in mdp_paths:
        with open(path) as f:
            content = f.read()
        matches = pattern.findall(content)
        if matches:
            problems.append((path, matches))

    if problems:
        print("Unreplaced placeholders found:")
        for path, matches in problems:
            print(f"    {path}: {matches}")
    else:
        print(f"No unreplaced placeholders found in {len(mdp_paths)} files.")

    return problems


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # --- Example 1: read a file, inspect/modify a few parameters ---
    mdp = MDPFile.read("npt_FEP_AMBER.mdp")
    print(mdp)
    print("Current ref_t:", mdp.get("ref_t"))
    print("Current pcoupl:", mdp.get("pcoupl"))

    mdp.set("ref_t", "310   310", comment="raised to 310 K for this test")
    mdp.write("npt_FEP_AMBER_310K.mdp")

    # --- Example 2: generate all 13 lambda windows for EM/NVT/NPT ---
    n_windows = 13
    stage_templates = {
        "em_FEP_AMBER.mdp": "em.mdp",
        "nvt_FEP_AMBER.mdp": "nvt.mdp",
        "npt_FEP_AMBER.mdp": "npt.mdp",
    }

    all_written = []
    for template_file, out_name in stage_templates.items():
        written = generate_lambda_mdps(
            template_path=template_file,
            output_dir="rbfe_run",
            n_windows=n_windows,
            output_filename=out_name,
            lambda_param="init-lambda-state",
        )
        all_written.extend(written)
        print(f"Generated {len(written)} files from {template_file}")

    # --- Example 3: QC checks before submitting any jobs ---
    check_no_placeholders(all_written)

    npt_files = [p for p in all_written if p.endswith("npt.mdp")]
    verify_lambda_vectors_consistent(npt_files)

    # Confirm init-lambda-state actually differs and is correctly indexed
    for lam in range(n_windows):
        mdp = MDPFile.read(f"rbfe_run/lambda_{lam}/npt.mdp")
        actual = mdp.get("init-lambda-state")
        assert str(lam) == actual, (
            f"Mismatch in lambda_{lam}/npt.mdp: "
            f"expected init-lambda-state={lam}, found {actual}"
        )
    print(f"init-lambda-state correctly set for all {n_windows} windows.")