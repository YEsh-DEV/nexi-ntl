"""Standalone offline validator for Nexi Phase 3 telephony configuration.

Uses only the Python standard library (json, argparse, re, pathlib). No external
packages, no network, no live servers.

Usage:
    python scripts/validate_telephony_config.py [--dir telephony]

Exits 0 when every telephony/*.json file is valid; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXPECTED_TRUNK_NAME = "nexi-inbound-trunk"
EXPECTED_AGENT_NAME = "nexi"

# E.164: a leading '+' followed by digits only (country code first).
E164_RE = re.compile(r"^\+\d{7,15}$")


class ValidationError(Exception):
    """Raised when a telephony configuration file fails validation."""


def load_json_file(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path.name}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"{path.name}: cannot read file: {exc}") from exc


def validate_numbers(value, path: Path, production: bool = False) -> None:
    if not isinstance(value, list):
        raise ValidationError(f"{path.name}: 'numbers' must be a list")
    if not value:
        raise ValidationError(f"{path.name}: 'numbers' cannot be empty")
    for entry in value:
        if not isinstance(entry, str) or not E164_RE.match(entry):
            raise ValidationError(
                f"{path.name}: number {entry!r} is not in E.164 format "
                "(must start with a leading country code, e.g. +919876543210)"
            )
        if production and (entry.startswith("+910000") or entry.replace("+", "").replace("0", "") == ""):
            raise ValidationError(
                f"{path.name}: number {entry!r} is a template placeholder, invalid for production"
            )


def validate_inbound_trunk(data: dict, path: Path, production: bool = False) -> None:
    trunk = data.get("trunk")
    if not isinstance(trunk, dict):
        raise ValidationError(f"{path.name}: missing 'trunk' object")
    name = trunk.get("name")
    if name != EXPECTED_TRUNK_NAME:
        raise ValidationError(
            f"{path.name}: trunk name is {name!r}, expected {EXPECTED_TRUNK_NAME!r}"
        )
    validate_numbers(trunk.get("numbers", []), path, production=production)


def validate_dispatch_rule(data: dict, path: Path) -> None:
    trunk = data.get("trunk")
    if trunk != EXPECTED_TRUNK_NAME:
        raise ValidationError(
            f"{path.name}: dispatch trunk is {trunk!r}, expected {EXPECTED_TRUNK_NAME!r}"
        )
    dispatch = data.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValidationError(f"{path.name}: missing 'dispatch' object")
    agent = dispatch.get("agent_name")
    if agent != EXPECTED_AGENT_NAME:
        raise ValidationError(
            f"{path.name}: dispatch agent_name is {agent!r}, expected {EXPECTED_AGENT_NAME!r}"
        )


def validate_file(path: Path, production: bool = False) -> list[str]:
    """Return a list of human-readable OK lines for a valid file."""
    data = load_json_file(path)
    if not isinstance(data, dict):
        raise ValidationError(f"{path.name}: top-level JSON must be an object")

    file_stem = path.stem
    if file_stem == "inbound_trunk":
        validate_inbound_trunk(data, path, production=production)
    elif file_stem == "dispatch_rule":
        validate_dispatch_rule(data, path)
    else:
        raise ValidationError(
            f"{path.name}: unknown telephony config file (expected inbound_trunk or dispatch_rule)"
        )

    return [f"{path.name}: valid"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "telephony",
        help="Directory containing telephony *.json files (default: telephony/)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Enforce strict production checks (rejects template placeholder numbers)",
    )
    args = parser.parse_args()

    telephony_dir: Path = args.dir
    if not telephony_dir.is_dir():
        print(f"error: telephony directory not found: {telephony_dir}", file=sys.stderr)
        return 1

    json_files = sorted(telephony_dir.glob("*.json"))
    if not json_files:
        print(f"error: no *.json files found in {telephony_dir}", file=sys.stderr)
        return 1

    if args.production:
        expected_files = {"inbound_trunk.json", "dispatch_rule.json"}
        present_files = {p.name for p in json_files}
        missing = expected_files - present_files
        if missing:
            print(f"error: missing required production files: {missing}", file=sys.stderr)
            return 1

    failures: list[str] = []
    for path in json_files:
        try:
            for line in validate_file(path, production=args.production):
                print(line)
        except ValidationError as exc:
            failures.append(str(exc))

    if failures:
        print(f"\n{len(failures)} validation failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(json_files)} telephony configuration file(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
