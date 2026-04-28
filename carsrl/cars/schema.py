"""Data model and JSON parsing for cognitive appraisals."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


NUMERIC_FIELDS = ("phi", "confidence", "affordance", "novelty", "risk")


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class Appraisal:
    phi: float
    confidence: float
    subgoal: str
    affordance: float
    novelty: float
    risk: float
    raw_text: str = ""
    parse_error: str | None = None

    @classmethod
    def fallback(cls, raw_text: str = "", parse_error: str | None = None) -> "Appraisal":
        return cls(
            phi=0.0,
            confidence=0.0,
            subgoal="unknown",
            affordance=0.0,
            novelty=0.0,
            risk=0.0,
            raw_text=raw_text,
            parse_error=parse_error,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], raw_text: str = "") -> "Appraisal":
        return cls(
            phi=clamp01(data.get("phi")),
            confidence=clamp01(data.get("confidence")),
            subgoal=str(data.get("subgoal", "unknown"))[:160],
            affordance=clamp01(data.get("affordance")),
            novelty=clamp01(data.get("novelty")),
            risk=clamp01(data.get("risk")),
            raw_text=raw_text,
            parse_error=None,
        )

    @classmethod
    def from_json_text(cls, text: str) -> "Appraisal":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                return cls.fallback(raw_text=text, parse_error="no_json_object")
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                return cls.fallback(raw_text=text, parse_error=f"json_decode_error: {exc}")

        if not isinstance(data, dict):
            return cls.fallback(raw_text=text, parse_error="json_not_object")
        return cls.from_dict(data, raw_text=text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi": self.phi,
            "confidence": self.confidence,
            "subgoal": self.subgoal,
            "affordance": self.affordance,
            "novelty": self.novelty,
            "risk": self.risk,
            "raw_text": self.raw_text,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_cache_record(cls, data: dict[str, Any]) -> "Appraisal":
        return cls(
            phi=clamp01(data.get("phi")),
            confidence=clamp01(data.get("confidence")),
            subgoal=str(data.get("subgoal", "unknown"))[:160],
            affordance=clamp01(data.get("affordance")),
            novelty=clamp01(data.get("novelty")),
            risk=clamp01(data.get("risk")),
            raw_text=str(data.get("raw_text", "")),
            parse_error=data.get("parse_error"),
        )
