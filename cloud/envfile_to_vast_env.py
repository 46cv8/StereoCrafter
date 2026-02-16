#!/usr/bin/env python3
"""Convert an env file into a Vast-compatible --env payload.

Usage:
  python cloud/envfile_to_vast_env.py --env-file cloud/hf.env

Output example:
  -e HF_TOKEN=hf_xxx -e HF_HOME=/opt/hf-cache

Then use:
  vastai create instance ... --env "$(python cloud/envfile_to_vast_env.py --env-file cloud/hf.env)"
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Dict

_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    out: Dict[str, str] = {}
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env line {idx}: {raw}")

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if not _VAR_RE.match(key):
            raise ValueError(f"Invalid env var name on line {idx}: {key}")

        # Remove matching single/double wrapper quotes if present.
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]

        out[key] = val

    return out


def build_vast_env_arg(env_vars: Dict[str, str]) -> str:
    parts = []
    for key, value in env_vars.items():
        parts.append("-e")
        parts.append(shlex.quote(f"{key}={value}"))
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render env-file values for vastai --env.")
    parser.add_argument("--env-file", required=True, help="Path to env file.")
    args = parser.parse_args()

    env_vars = parse_env_file(Path(args.env_file).expanduser().resolve())
    if not env_vars:
        raise ValueError("No env vars found in file.")
    print(build_vast_env_arg(env_vars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
