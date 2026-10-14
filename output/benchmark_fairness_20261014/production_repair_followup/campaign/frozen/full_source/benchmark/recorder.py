from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(obj: Any):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if is_dataclass(obj):
        return {k: _safe(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return _safe(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(v) for v in obj]
    return str(obj)


class BenchmarkRecorder:
    """On-disk benchmark recorder that writes JSONL/JSON incrementally."""

    def __init__(self, run_dir: Path, metadata: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = _safe(metadata)
        self._stage_started: dict[str, float] = {}
        self.stage_durations_ms: dict[str, float] = {}
        self.llm_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.started = time.perf_counter()
        self.write_json("metadata.json", {"created_at": _ts(), "metadata": self.metadata})

    def _append_jsonl(self, filename: str, payload: dict) -> None:
        path = self.run_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_safe(payload), ensure_ascii=False) + "\n")

    def write_json(self, filename: str, payload: Any) -> None:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_safe(payload), handle, indent=2, ensure_ascii=False)

    def save_snapshot(self, name: str, payload: Any) -> None:
        filename = self.snapshots_dir / f"{name}.json"
        with filename.open("w", encoding="utf-8") as handle:
            json.dump(_safe(payload), handle, indent=2, ensure_ascii=False)

    def log_event(self, kind: str, payload: dict) -> None:
        self._append_jsonl("events.jsonl", {"timestamp": _ts(), "kind": kind, **_safe(payload)})

    def start_stage(self, name: str, details: dict | None = None) -> None:
        self._stage_started[name] = time.perf_counter()
        self._append_jsonl("stage_events.jsonl", {
            "timestamp": _ts(),
            "event": "start",
            "stage": name,
            "details": _safe(details or {}),
        })

    def end_stage(self, name: str, details: dict | None = None, status: str = "completed") -> None:
        started = self._stage_started.pop(name, None)
        duration_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
        if duration_ms is not None:
            self.stage_durations_ms[name] = self.stage_durations_ms.get(name, 0.0) + duration_ms
        self._append_jsonl("stage_events.jsonl", {
            "timestamp": _ts(),
            "event": "end",
            "stage": name,
            "status": status,
            "duration_ms": duration_ms,
            "details": _safe(details or {}),
        })

    def observe_llm(self, event: dict) -> None:
        payload = _safe(event)
        usage = payload.get("usage", {}) or {}
        derived_total = int(
            usage.get("total_tokens", 0)
            or ((usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0))
            or ((usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0))
        )
        self.llm_call_count += 1
        self.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.total_output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += derived_total
        self._append_jsonl("llm_events.jsonl", {"timestamp": _ts(), **payload})

    def finalize(self, status: str = "completed", extra: dict | None = None) -> dict:
        summary = {
            "status": status,
            "metadata": self.metadata,
            "runtime_s": round(time.perf_counter() - self.started, 3),
            "stage_durations_ms": {k: round(v, 2) for k, v in self.stage_durations_ms.items()},
            "llm_call_count": self.llm_call_count,
            "token_totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "finished_at": _ts(),
        }
        if extra:
            summary.update(_safe(extra))
        self.write_json("run_summary.json", summary)
        return summary
