"""Content-addressed cache manifests for deterministic pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from rag.core.io_utils import atomic_write_json, read_json, sha256_file


CACHE_SCHEMA_VERSION = "stage-cache/1.0"


@dataclass(frozen=True)
class CacheDecision:
    hit: bool
    reason: str
    key: str = ""


class StageCache:
    """Validate completed outputs against exact inputs and parameters."""

    @staticmethod
    def _records(paths: Mapping[str, Path | str]) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        for label, raw_path in sorted(paths.items()):
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records[label] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        return records

    def build_key(
        self,
        *,
        stage: str,
        version: str,
        params: Mapping[str, object],
        inputs: Mapping[str, Path | str],
    ) -> str:
        value = {
            "stage": stage,
            "version": version,
            "params": dict(params),
            "inputs": self._records(inputs),
        }
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(
        self,
        record_path: Path | str,
        *,
        stage: str,
        version: str,
        params: Mapping[str, object],
        inputs: Mapping[str, Path | str],
        outputs: Mapping[str, Path | str],
    ) -> CacheDecision:
        record_file = Path(record_path)
        if not record_file.is_file():
            return CacheDecision(False, "missing_record")
        try:
            record = read_json(record_file)
        except (OSError, ValueError, json.JSONDecodeError):
            return CacheDecision(False, "invalid_record")
        if not isinstance(record, dict) or record.get("schema_version") != CACHE_SCHEMA_VERSION:
            return CacheDecision(False, "invalid_record")
        try:
            key = self.build_key(
                stage=stage,
                version=version,
                params=params,
                inputs=inputs,
            )
        except FileNotFoundError:
            return CacheDecision(False, "missing_input")
        if record.get("key") != key:
            return CacheDecision(False, "input_or_parameter_mismatch", key)
        stored_outputs = record.get("outputs")
        if not isinstance(stored_outputs, dict):
            return CacheDecision(False, "invalid_record", key)
        for label, raw_path in sorted(outputs.items()):
            path = Path(raw_path).resolve()
            if not path.is_file():
                return CacheDecision(False, "missing_output", key)
            expected = stored_outputs.get(label)
            if not isinstance(expected, dict):
                return CacheDecision(False, "output_set_mismatch", key)
            if sha256_file(path) != expected.get("sha256"):
                return CacheDecision(False, "output_digest_mismatch", key)
        if set(stored_outputs) != set(outputs):
            return CacheDecision(False, "output_set_mismatch", key)
        return CacheDecision(True, "hit", key)

    def commit(
        self,
        record_path: Path | str,
        *,
        stage: str,
        version: str,
        params: Mapping[str, object],
        inputs: Mapping[str, Path | str],
        outputs: Mapping[str, Path | str],
    ) -> None:
        key = self.build_key(
            stage=stage,
            version=version,
            params=params,
            inputs=inputs,
        )
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "stage_version": version,
            "key": key,
            "params": dict(params),
            "inputs": self._records(inputs),
            "outputs": self._records(outputs),
            "complete": True,
        }
        atomic_write_json(Path(record_path), record)
