#!/usr/bin/env python3
"""Install one dependency profile from the repository's requirements.txt."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def read_profiles(path: Path) -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"# \[profile:([a-z0-9-]+)\]", line)
        if match:
            current = match.group(1)
            if current in profiles:
                raise ValueError(f"Duplicate requirements profile: {current}")
            profiles[current] = []
        elif current is not None:
            if line.startswith("#| "):
                profiles[current].append(line[3:])
            elif line.strip() and not line.lstrip().startswith("#"):
                profiles[current].append(line)
    return profiles


def main() -> None:
    profiles = read_profiles(ROOT / "requirements.txt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(profiles), required=True)
    parser.add_argument("--python", default=sys.executable, help="Target environment's Python executable")
    parser.add_argument("--installer", choices=["pip", "uv"], default="pip")
    parser.add_argument("--pin", action="append", default=[], help="Additional requirement constraint, repeatable")
    parser.add_argument("--print", action="store_true", dest="print_only", help="Print requirements without installing")
    args = parser.parse_args()
    content = "\n".join(profiles[args.profile]) + "\n"
    if args.print_only:
        print(content, end="")
        return
    if not profiles[args.profile]:
        parser.error(f"Empty dependency profile: {args.profile}")
    with tempfile.TemporaryDirectory(prefix="robot101-requirements-") as directory:
        path = Path(directory) / "selected.txt"
        path.write_text(content, encoding="utf-8")
        if args.installer == "uv":
            command = ["uv", "pip", "install", "--python", args.python]
        else:
            command = [args.python, "-m", "pip", "install"]
        subprocess.run(command + ["-r", str(path), *args.pin], cwd=ROOT, check=True)
        # Make python -m robot101... entrypoints available in the target environment.
        subprocess.run(command + ["--no-deps", "-e", str(ROOT)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
