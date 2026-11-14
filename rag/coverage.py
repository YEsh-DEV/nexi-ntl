"""Coverage audit, report generator, and approval binding for knowledge artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.config import load_settings
from rag.manifest import resolve_artifact_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def generate_coverage_report(
    artifact_dir: Path,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    expected_path = expected_path or (REPO_ROOT / "eval" / "coverage_expected.yaml")
    expected_rules: list[dict[str, Any]] = []
    if expected_path.is_file():
        parsed = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict) and "expected_sources" in parsed:
            expected_rules = parsed["expected_sources"]

    manifest_file = artifact_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found in {artifact_dir}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])

    source_map: dict[str, dict[str, Any]] = {}
    for s in sources:
        src_name = s.get("source", "")
        source_map[src_name] = s

    zero_chunk_files: list[str] = []
    needs_ocr_pages: dict[str, list[int]] = {}
    undated_files: list[str] = []
    missing_expected: list[str] = []
    below_threshold: list[dict[str, Any]] = []

    for s in sources:
        name = s.get("source", "")
        chunks = s.get("chunks", 0)
        empty_pages = s.get("empty_pages", [])
        if chunks == 0:
            zero_chunk_files.append(name)
        if empty_pages:
            needs_ocr_pages[name] = empty_pages
        if not s.get("effective_date"):
            undated_files.append(name)

    for rule in expected_rules:
        pattern = rule.get("pattern", "")
        min_chunks = rule.get("min_chunks", 1)
        found = False
        for name, info in source_map.items():
            if pattern in name or name in pattern:
                found = True
                if info.get("chunks", 0) < min_chunks:
                    below_threshold.append({
                        "source": name,
                        "found_chunks": info.get("chunks", 0),
                        "required_chunks": min_chunks,
                    })
                break
        if not found and rule.get("required", False):
            missing_expected.append(pattern)

    is_partial = bool(zero_chunk_files or missing_expected or below_threshold)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": manifest.get("version"),
        "total_sources": len(sources),
        "total_chunks": manifest.get("chunks", 0),
        "zero_chunk_files": zero_chunk_files,
        "needs_ocr_pages": needs_ocr_pages,
        "undated_files": undated_files,
        "missing_expected_sources": missing_expected,
        "below_threshold_sources": below_threshold,
        "partial": is_partial,
    }

    report_path = artifact_dir / "coverage_report.json"
    report_json = json.dumps(report, indent=2)
    report_path.write_text(report_json, encoding="utf-8")

    # Update manifest partial and coverage_report pointer
    manifest["partial"] = is_partial
    manifest["coverage_report"] = "coverage_report.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return report


def approve_coverage(artifact_dir: Path, approved_by: str) -> dict[str, Any]:
    approved_by = approved_by.strip()
    if not approved_by:
        raise ValueError("Approver name cannot be empty")

    report_path = artifact_dir / "coverage_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Coverage report not found at {report_path}. Run report first.")

    report_sha256 = compute_sha256(report_path.read_bytes())
    manifest_file = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    approval_record = {
        "approved_by": approved_by,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": report_sha256,
    }

    manifest["coverage_approval"] = approval_record
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return approval_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge coverage reporter and approver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Generate coverage report for version")
    report_parser.add_argument("version", help="Artifact version (hex or staging/version)")

    approve_parser = subparsers.add_parser("approve", help="Approve coverage report for version")
    approve_parser.add_argument("version", help="Artifact version (hex or staging/version)")
    approve_parser.add_argument("--by", dest="approved_by", required=True, help="Human approver name")

    args = parser.parse_args()
    settings = load_settings()
    root = settings.chroma_dir.resolve()

    version = args.version
    if (root / "staging" / version).is_dir():
        artifact_dir = root / "staging" / version
    else:
        artifact_dir = resolve_artifact_path(root, version)

    if not artifact_dir.is_dir():
        print(f"Error: artifact directory not found for version '{version}'", file=sys.stderr)
        sys.exit(1)

    if args.command == "report":
        report = generate_coverage_report(artifact_dir)
        print(json.dumps(report, indent=2))
        print(f"\nCoverage report written to {artifact_dir / 'coverage_report.json'}")
        if report["partial"]:
            print("WARNING: Artifact is marked PARTIAL due to missing/zero-chunk sources.", file=sys.stderr)
    elif args.command == "approve":
        approval = approve_coverage(artifact_dir, args.approved_by)
        print(f"Coverage approved by {approval['approved_by']} (hash: {approval['report_sha256']})")


if __name__ == "__main__":
    main()
