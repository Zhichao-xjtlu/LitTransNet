"""Shared deterministic I/O and scalar helpers for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            yield value


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_lines(path, (text,))


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(
    path: Path | str,
    value: object,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
    ) + "\n"
    _atomic_write_text(Path(path), text)


def atomic_write_jsonl(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
) -> None:
    lines = (
        json.dumps(dict(row), ensure_ascii=False, sort_keys=sort_keys) + "\n"
        for row in rows
    )
    _atomic_write_lines(Path(path), lines)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def safe_float(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_values(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [
        item
        for item in (clean_text(part) for part in re.split(r"[;|]+", text))
        if item
    ]


def join_unique(values: Iterable[object], separator: str = ";") -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        for item in split_values(value):
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                output.append(item)
    return separator.join(output)
