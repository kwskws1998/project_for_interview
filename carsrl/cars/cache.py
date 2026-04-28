"""Persistent JSONL cache for SLM appraisals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from carsrl.cars.schema import Appraisal


def stable_state_hash(serialized_state: str) -> str:
    return hashlib.sha256(serialized_state.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class AppraisalCache:
    """Append-only JSONL cache keyed by SHA-256 of serialized state text."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._items: dict[str, Appraisal] = {}
        self.stats = CacheStats()
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                appraisal = record.get("appraisal")
                if isinstance(key, str) and isinstance(appraisal, dict):
                    self._items[key] = Appraisal.from_cache_record(appraisal)

    def get(self, serialized_state: str) -> Appraisal | None:
        key = stable_state_hash(serialized_state)
        item = self._items.get(key)
        if item is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return item

    def put(self, serialized_state: str, appraisal: Appraisal) -> str:
        key = stable_state_hash(serialized_state)
        self._items[key] = appraisal
        self.stats.writes += 1
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "serialized_state": serialized_state,
                "appraisal": appraisal.to_dict(),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return key

    def __contains__(self, serialized_state: str) -> bool:
        return stable_state_hash(serialized_state) in self._items

    def __len__(self) -> int:
        return len(self._items)

    def values(self) -> Iterable[Appraisal]:
        return self._items.values()
