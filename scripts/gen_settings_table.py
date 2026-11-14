#!/usr/bin/env python3
"""Generates the Markdown table of configuration settings from agent.config.ENV_VARS."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.config import ENV_VARS  # noqa: E402


def generate_markdown_table() -> str:
    lines = [
        "| Environment Variable | Description | Default | Required |",
        "|---|---|---|---|",
    ]
    for name, meta in sorted(ENV_VARS.items()):
        desc = meta["description"]
        default = meta["default"]
        req = "Yes" if meta["required"] else "No"
        def_str = f"`{default}`" if default != "" else "*(empty)*"
        lines.append(f"| `{name}` | {desc} | {def_str} | {req} |")
    return "\n".join(lines)


def main() -> None:
    print(generate_markdown_table())


if __name__ == "__main__":
    main()
