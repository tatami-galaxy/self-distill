"""Content-addressed, resumable scores for hint-generator comparisons."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def model_identity(model: str) -> dict:
    """Track local model replacements without reading gigabytes of weight data."""
    path = Path(model).expanduser()
    if not path.is_dir():
        return {"model": model}
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}:
            stat = item.stat()
            files.append((item.name, stat.st_size, stat.st_mtime_ns))
    return {"model": str(path.resolve()), "files": files}


class ConditionCache:
    """Each atomic JSON entry is keyed by all semantic scoring inputs.

    Request inputs are hashed, rather than duplicated on disk for every hint and
    long rollout. The saved result has its own checksum to reject damaged entries.
    Execution controls (GPU placement and batch size) do not belong in the key.
    """

    def __init__(self, root: Path, kind: str, config: dict, force: bool = False):
        self.root = root / "score_cache" / kind
        self.config = config
        self.force = force
        self.hits = 0
        self.misses = 0

    def key(self, request: Any) -> str:
        return digest({"version": CACHE_VERSION, "config": self.config, "request": request})

    def load(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if self.force or not path.exists():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text())
            if (
                record["version"] != CACHE_VERSION
                or record["key"] != key
                or record["result_digest"] != digest(record["result"])
            ):
                raise ValueError("entry identity or checksum differs")
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid condition cache {path}; use --force to recompute.") from error
        self.hits += 1
        return record["result"]

    def save(self, key: str, result: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        record = {
            "version": CACHE_VERSION,
            "key": key,
            "result_digest": digest(result),
            "result": result,
        }
        temporary.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(temporary, path)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}
