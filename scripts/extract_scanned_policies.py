"""Offline OCR extraction for scanned SRM AP university policy PDFs.

Converts scanned bitmap PDFs into searchable PDFs using native Linux utilities
(pdftoppm, tesseract, and pdfunite). Runs completely offline: zero python OCR
dependencies, zero runtime latency/pressure on the voice assistant.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "uni_pdfs"
BACKUP_DIR = REPO / ".local-baseline" / "scanned_originals"

SCANNED_FILES = [
    "13.Student Internship Policy.pdf",
    "17. Financial Assistance Policy of SRM University - AP to UG and PG Students for Attending Conferences, Workshops and Competitions.pdf",
    "31. Seed Grant Assistance Policy of SRM University - AP to Full Time UG & PG Students for Research Projects (2).pdf",
    "student-code-of-conduct-policy-of-SRM-university-AP-.pdf",
    "Academic-Programmes-Ordinances.pdf",
]


def check_prerequisites() -> None:
    for tool in ("pdftoppm", "tesseract", "pdfunite"):
        if not shutil.which(tool):
            sys.exit(f"Error: Required tool '{tool}' not found in PATH.")


def ocr_pdf(source_path: Path, output_path: Path) -> None:
    print(f"\nProcessing: {source_path.name}")
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nexi_ocr_") as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Render PDF pages to PNG at 150 DPI (optimal balance of OCR accuracy and speed)
        render_prefix = tmp / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "150", str(source_path), str(render_prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        png_files = sorted(tmp.glob("page-*.png"))
        if not png_files:
            raise RuntimeError(f"No pages rendered for {source_path.name}")

        print(f"  Rendered {len(png_files)} pages. Running Tesseract OCR...")

        # 2. Run OCR on each page to produce a 1-page searchable PDF
        page_pdfs = []
        for index, png in enumerate(png_files, start=1):
            out_base = tmp / f"ocr_{index:03d}"
            subprocess.run(
                ["tesseract", str(png), str(out_base), "--oem", "1", "-l", "eng", "pdf"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            page_pdf = tmp / f"ocr_{index:03d}.pdf"
            if not page_pdf.exists():
                raise RuntimeError(f"OCR failed for page {index} of {source_path.name}")
            page_pdfs.append(str(page_pdf))
            print(f"    Page {index}/{len(png_files)} complete", end="\r", flush=True)

        print()

        # 3. Combine 1-page searchable PDFs into one final searchable PDF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_combined = tmp / "combined.pdf"
        subprocess.run(["pdfunite", *page_pdfs, str(temp_combined)], check=True)
        shutil.copy2(temp_combined, output_path)

    elapsed = time.perf_counter() - start_time
    print(f"  Successfully wrote searchable PDF to {output_path.name} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        choices=SCANNED_FILES,
        help="Process a specific scanned PDF instead of all five.",
    )
    args = parser.parse_args()

    check_prerequisites()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    targets = [args.file] if args.file else SCANNED_FILES

    for filename in targets:
        src = PDF_DIR / filename
        if not src.exists():
            print(f"Skipping {filename}: not found in {PDF_DIR}")
            continue

        # Backup original scanned PDF if not already backed up
        backup = BACKUP_DIR / filename
        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"Backed up original scanned PDF to {backup.relative_to(REPO)}")

        # Write searchable PDF directly to uni_pdfs
        ocr_pdf(src, src)

    print("\nOffline OCR conversion completed successfully.")


if __name__ == "__main__":
    main()
