from __future__ import annotations

import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "pathagent_trace_v2"
FORBIDDEN_RAW_KEYS = {
    "answer_zh",
    "gold_answer",
    "gold_answer_zh",
    "ground_truth",
    "raw_pathology_report",
    "source_report_evidence",
    "source_report_fields",
    "术后病理",
    "主要病理诊断",
    "结构化金标",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def assert_blind_raw_trace(trace: Dict[str, Any]) -> None:
    def walk(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in FORBIDDEN_RAW_KEYS:
                    raise ValueError(f"raw trace contains forbidden gold field at {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(trace)


class TraceRecorder:
    """Crash-tolerant event recorder for one PathAgent rollout.

    Every external model or retrieval call is represented by a `before` event and
    a matching `after` event sharing the same `call_id`. Events are appended to a
    per-rollout JSONL file immediately; `finalize` additionally writes one complete
    raw trace JSON object and appends it to the run-level raw JSONL.
    """

    def __init__(
        self,
        trace_root: str | os.PathLike[str],
        run_id: str,
        trace_id: str,
        rollout_id: int,
        task_input: Dict[str, Any],
        runtime: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace_root = Path(trace_root)
        self.run_id = run_id
        self.trace_id = trace_id
        self.rollout_id = rollout_id
        self.started_at = time.time()
        self.events: list[Dict[str, Any]] = []
        self.final_output: Dict[str, Any] = {}
        self.execution: Dict[str, Any] = {"status": "running", "error": None}
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.event_dir = self.trace_root / "logs"
        self.raw_dir = self.trace_root / "raw"
        self.event_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.raw_path = self.raw_dir / f"{trace_id}.json"
        if self.raw_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing raw trace: {self.raw_path}"
            )
        self.event_log_path = self.event_dir / f"{trace_id}.events.jsonl"
        if self.event_log_path.exists():
            incomplete_path = self.event_log_path.with_suffix(
                f".incomplete.{int(self.started_at)}.jsonl"
            )
            self.event_log_path.replace(incomplete_path)
        self.runtime = {
            "python": platform.python_version(),
            **_json_safe(runtime or {}),
        }
        self.task_input = _json_safe(task_input)
        assert_blind_raw_trace({"task_input": self.task_input})

    def before_call(
        self,
        component: str,
        operation: str,
        request: Dict[str, Any],
        step_id: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> str:
        call_id = uuid.uuid4().hex
        self._append_event(
            {
                "call_id": call_id,
                "phase": "before",
                "component": component,
                "operation": operation,
                "step_id": step_id,
                "attempt": attempt,
                "request": request,
            }
        )
        return call_id

    def after_call(
        self,
        call_id: str,
        component: str,
        operation: str,
        response: Dict[str, Any],
        status: str = "ok",
        error: Optional[str] = None,
        step_id: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> None:
        self._append_event(
            {
                "call_id": call_id,
                "phase": "after",
                "component": component,
                "operation": operation,
                "step_id": step_id,
                "attempt": attempt,
                "status": status,
                "response": response,
                "error": error,
            }
        )

    def record_state(
        self,
        event_type: str,
        payload: Dict[str, Any],
        step_id: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> None:
        self._append_event(
            {
                "phase": "state",
                "component": "pathagent",
                "operation": event_type,
                "step_id": step_id,
                "attempt": attempt,
                "payload": payload,
            }
        )

    def finalize(
        self,
        final_output: Dict[str, Any],
        status: str = "completed",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.final_output = _json_safe(final_output)
        self.execution = {
            "status": status,
            "error": error,
            "latency_ms": round((time.time() - self.started_at) * 1000),
            "event_count": len(self.events),
            "event_log": str(self.event_log_path),
        }
        trace = self.to_dict()
        assert_blind_raw_trace(trace)
        raw_temporary = self.raw_path.with_suffix(".json.tmp")
        raw_temporary.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raw_temporary.replace(self.raw_path)
        aggregate_path = self.trace_root / "raw_trace.jsonl"
        temporary_path = aggregate_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for completed_path in sorted(self.raw_dir.glob("*.json")):
                completed_trace = json.loads(completed_path.read_text(encoding="utf-8"))
                handle.write(json.dumps(completed_trace, ensure_ascii=False) + "\n")
        temporary_path.replace(aggregate_path)
        return trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "rollout_id": self.rollout_id,
            "runtime": self.runtime,
            "task_input": self.task_input,
            "events": self.events,
            "final_output": self.final_output,
            "execution": self.execution,
        }

    def _append_event(self, event: Dict[str, Any]) -> None:
        safe_event = {
            "event_id": len(self.events) + 1,
            "timestamp": time.time(),
            **_json_safe(event),
        }
        assert_blind_raw_trace(safe_event)
        self.events.append(safe_event)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_event, ensure_ascii=False) + "\n")
