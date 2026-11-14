"""Record and table chunking with parent context propagation, table header repetition and date derivation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    source_type: str  # 'pdf' or 'json'
    location: str  # page number or json path
    source_hash: str
    context_prefix: str = ""
    effective_date: str | None = None
    valid_until: str | None = None
    doc_version: str | None = None
    topic_key: str | None = None
    ocr: bool = False

    @property
    def full_document(self) -> str:
        ref = f"Reference: {self.source} | {self.location}"
        if self.context_prefix:
            return f"{ref}\nContext: {self.context_prefix}\n{self.text}"
        return f"{ref}\n{self.text}"


def parse_date_str(val: Any) -> str | None:
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None
    # YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val_str):
        return val_str
    # ISO datetime
    try:
        dt = datetime.fromisoformat(val_str)
        return dt.date().isoformat()
    except ValueError:
        pass
    # Academic year format: 2025-26 or 2025-2026
    m = re.fullmatch(r"(\d{4})[-/](\d{2,4})", val_str)
    if m:
        return f"{m.group(1)}-07-01"
    return None


def derive_dates_and_topic(
    source_path: Path,
    record: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, str | None, str]:
    """Derive effective_date, valid_until, doc_version, and topic_key.

    Priority:
    1. Inside record
    2. Sidecar <file>.meta.yaml
    3. Filename convention
    """
    effective_date: str | None = None
    valid_until: str | None = None
    doc_version: str | None = None
    topic_key: str | None = None

    # 1. From record
    if record and isinstance(record, dict):
        effective_date = parse_date_str(record.get("effective_date") or record.get("date") or record.get("start_date"))
        valid_until = parse_date_str(record.get("valid_until") or record.get("expiry_date") or record.get("end_date"))
        doc_version = str(record.get("doc_version") or record.get("version")) if record.get("doc_version") or record.get("version") else None
        if record.get("topic_key"):
            topic_key = str(record["topic_key"])

    # 2. Sidecar <file>.meta.yaml
    sidecar = source_path.with_name(f"{source_path.name}.meta.yaml")
    if sidecar.is_file():
        try:
            meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                effective_date = effective_date or parse_date_str(meta.get("effective_date"))
                valid_until = valid_until or parse_date_str(meta.get("valid_until"))
                doc_version = doc_version or (str(meta.get("doc_version")) if meta.get("doc_version") else None)
                topic_key = topic_key or meta.get("topic_key")
        except Exception as sidecar_exc:
            logger.debug("Non-fatal sidecar read error for %s: %s", source_path, sidecar_exc)

    # 3. Filename convention: name__YYYY-MM-DD or AY YYYY-YY
    stem = source_path.stem
    if "__" in stem:
        parts = stem.split("__", 1)
        topic_key = topic_key or parts[0].strip()
        date_cand = parse_date_str(parts[1].strip())
        if date_cand:
            effective_date = effective_date or date_cand

    # Match AY 2025-26 or similar in stem
    ay_match = re.search(r"AY\s*(\d{4})[-_](\d{2,4})", stem, re.IGNORECASE)
    if ay_match:
        ay_year = ay_match.group(1)
        effective_date = effective_date or f"{ay_year}-07-01"
        cleaned_stem = re.sub(r"AY\s*\d{4}[-_]\d{2,4}", "", stem, flags=re.IGNORECASE).strip(" -_")
        topic_key = topic_key or (cleaned_stem if cleaned_stem else stem)

    if not topic_key:
        topic_key = stem

    return effective_date, valid_until, doc_version, topic_key


def _format_parent_context(ancestors: list[tuple[str, dict[str, Any]]]) -> str:
    parts = []
    for path_label, fields in ancestors:
        ident_parts = []
        for k in ("name", "title", "id", "code"):
            if k in fields:
                ident_parts.append(f"{k}: {fields[k]}")
        if ident_parts:
            parts.append(f"{path_label} ({', '.join(ident_parts)})")
        else:
            parts.append(path_label)
    return " > ".join(parts)


def chunk_json_node(
    node: Any,
    path: str = "root",
    ancestors: list[tuple[str, dict[str, Any]]] | None = None,
    min_chars: int = 200,
    max_chars: int = 1000,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Recursively chunk JSON node carrying parent context.

    Returns list of (location_path, context_prefix, text, record_scalars).
    """
    ancestors = ancestors or []
    results: list[tuple[str, str, str, dict[str, Any]]] = []

    if isinstance(node, dict):
        scalars: dict[str, Any] = {}
        nested: dict[str, Any] = {}
        for k, v in node.items():
            if v is not None and not isinstance(v, (dict, list)):
                scalars[k] = v
            else:
                nested[k] = v

        current_ancestors = ancestors
        if scalars:
            context_prefix = _format_parent_context(ancestors)
            # Encode scalars
            text = json.dumps(scalars, ensure_ascii=False)
            if len(text) > max_chars:
                # Split large records by sentences or keys
                sub_parts = []
                cur_dict: dict[str, Any] = {}
                for k, v in scalars.items():
                    cur_dict[k] = v
                    cand = json.dumps(cur_dict, ensure_ascii=False)
                    if len(cand) >= min_chars or len(cand) >= max_chars:
                        sub_parts.append(cand)
                        cur_dict = {}
                if cur_dict:
                    sub_parts.append(json.dumps(cur_dict, ensure_ascii=False))
                for idx, part in enumerate(sub_parts):
                    results.append((f"{path}#part{idx+1}", context_prefix, part, scalars))
            else:
                results.append((path, context_prefix, text, scalars))

            current_ancestors = ancestors + [(path, scalars)]

        # Recurse into nested children separately (never flattened into parent line)
        for k, v in nested.items():
            results.extend(
                chunk_json_node(
                    v,
                    path=f"{path} > {k}",
                    ancestors=current_ancestors,
                    min_chars=min_chars,
                    max_chars=max_chars,
                )
            )

    elif isinstance(node, list):
        for index, item in enumerate(node):
            item_label = None
            if isinstance(item, dict):
                for id_key in ("name", "title", "id", "code"):
                    if item.get(id_key):
                        item_label = f"{id_key}:{item[id_key]}"
                        break
            item_path = f"{path} > {item_label or index}"
            results.extend(
                chunk_json_node(
                    item,
                    path=item_path,
                    ancestors=ancestors,
                    min_chars=min_chars,
                    max_chars=max_chars,
                )
            )
    elif node is not None and str(node).strip():
        context_prefix = _format_parent_context(ancestors)
        text = str(node).strip()
        results.append((path, context_prefix, text, {}))

    return results


def is_table_block(lines: list[str]) -> bool:
    """Detect if consecutive lines form a table (>= 2 columns separated by tabs or >=2 spaces, or numbers)."""
    if len(lines) < 2:
        return False
    table_rows = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split on tabs or 2+ spaces
        cols = re.split(r"\t+|\s{2,}", line)
        if len(cols) >= 2:
            table_rows += 1
    return table_rows >= max(2, int(len(lines) * 0.6))


def chunk_text_and_tables(
    raw_text: str,
    page: int,
    min_chars: int = 200,
    max_chars: int = 1000,
) -> list[tuple[str, str]]:
    """Split page text and tables keeping table rows intact and repeating header.

    Returns list of (location, chunk_text).
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    # Split into blocks separated by 2+ newlines
    raw_blocks = re.split(r"\n\s*\n+", raw_text)
    chunks: list[tuple[str, str]] = []

    for block_idx, block in enumerate(raw_blocks):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if is_table_block(lines):
            header_row = lines[0]
            current_table_lines = [header_row]
            for row in lines[1:]:
                # If adding row exceeds max_chars, emit chunk and repeat header
                cand = "\n".join(current_table_lines + [row])
                if len(cand) > max_chars and len(current_table_lines) > 1:
                    chunks.append((f"page {page} (table)", "\n".join(current_table_lines)))
                    current_table_lines = [header_row, row]
                else:
                    current_table_lines.append(row)
            if current_table_lines:
                chunks.append((f"page {page} (table)", "\n".join(current_table_lines)))
        else:
            # Regular text: split on sentence boundaries if large
            block_text = " ".join(lines)
            if len(block_text) <= max_chars:
                chunks.append((f"page {page}", block_text))
            else:
                sentences = re.split(r"(?<=[.!?])\s+", block_text)
                cur_chunk: list[str] = []
                cur_len = 0
                for s in sentences:
                    if cur_len + len(s) + 1 > max_chars and cur_chunk:
                        chunks.append((f"page {page}", " ".join(cur_chunk)))
                        cur_chunk = [s]
                        cur_len = len(s)
                    else:
                        cur_chunk.append(s)
                        cur_len += len(s) + 1
                if cur_chunk:
                    chunks.append((f"page {page}", " ".join(cur_chunk)))

    # Merge small sibling chunks if under max_chars
    merged: list[tuple[str, str]] = []
    buffer_text = ""
    buffer_loc = ""
    for loc, text in chunks:
        if not buffer_text:
            buffer_text = text
            buffer_loc = loc
        elif len(buffer_text) + len(text) + 2 <= max_chars and buffer_loc == loc:
            buffer_text = f"{buffer_text}\n\n{text}"
        else:
            merged.append((buffer_loc, buffer_text))
            buffer_text = text
            buffer_loc = loc
    if buffer_text:
        merged.append((buffer_loc, buffer_text))

    return merged
