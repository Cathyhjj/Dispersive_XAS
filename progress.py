"""Progress reporting helpers for long-running DXAS batch workflows."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_progress_line(
    *,
    label: str = "DXAS",
    stage: str,
    status: str = "running",
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    message: str | None = None,
    stage_elapsed_s: float | None = None,
    eta_s: float | None = None,
) -> str:
    pieces = [f"[{label}]", status, stage.replace("_", " ")]
    if current is not None and total is not None and total > 0:
        percent = (100.0 * float(current)) / float(total)
        noun = unit or "items"
        pieces.append(f"{int(current)}/{int(total)} {noun} ({percent:.1f}%)")
    elif current is not None:
        noun = unit or "items"
        pieces.append(f"{int(current)} {noun}")
    if stage_elapsed_s is not None:
        pieces.append(f"elapsed {_format_duration(stage_elapsed_s)}")
    if eta_s is not None:
        pieces.append(f"ETA {_format_duration(eta_s)}")
    if message:
        pieces.append(message)
    return " | ".join(pieces)


class BatchProgressReporter:
    """Emit progress updates to the console and optionally to a JSON file."""

    def __init__(
        self,
        json_path: str | Path | None = None,
        *,
        stream: TextIO | None = None,
        enabled: bool = True,
        label: str = "DXAS",
    ) -> None:
        self.json_path = Path(json_path).expanduser().resolve() if json_path else None
        self.stream = sys.stdout if stream is None else stream
        self.enabled = bool(enabled)
        self.label = str(label)
        self.run_started_at = time.time()
        self._stage_started_at = self.run_started_at
        self._current_stage: str | None = None
        self._context: dict[str, Any] = {}
        self.last_payload: dict[str, Any] | None = None

    def set_context(self, **kwargs: Any) -> None:
        self._context = {k: v for k, v in kwargs.items() if v is not None}

    def update(
        self,
        stage: str,
        *,
        status: str = "running",
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        if stage != self._current_stage or status == "started":
            self._current_stage = stage
            self._stage_started_at = now

        run_elapsed_s = now - self.run_started_at
        stage_elapsed_s = now - self._stage_started_at
        percent = None
        eta_s = None
        if current is not None and total is not None and total > 0:
            percent = (100.0 * float(current)) / float(total)
            if current > 0 and current < total:
                eta_s = stage_elapsed_s * (float(total - current) / float(current))

        payload: dict[str, Any] = {
            "label": self.label,
            "status": status,
            "stage": stage,
            "message": message,
            "current": None if current is None else int(current),
            "total": None if total is None else int(total),
            "unit": unit,
            "percent": percent,
            "run_started_at": _utc_iso(self.run_started_at),
            "updated_at": _utc_iso(now),
            "run_elapsed_s": run_elapsed_s,
            "stage_elapsed_s": stage_elapsed_s,
            "eta_s": eta_s,
            **self._context,
        }
        if extra:
            payload["extra"] = extra

        self.last_payload = payload

        if self.enabled:
            print(
                format_progress_line(
                    label=self.label,
                    stage=stage,
                    status=status,
                    current=current,
                    total=total,
                    unit=unit,
                    message=message,
                    stage_elapsed_s=stage_elapsed_s,
                    eta_s=eta_s,
                ),
                file=self.stream,
                flush=True,
            )

        if self.json_path is not None:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            tmp_path.replace(self.json_path)

        return payload


def emit_progress(
    progress: BatchProgressReporter | None,
    stage: str,
    *,
    status: str = "running",
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
    label: str = "DXAS",
) -> dict[str, Any] | None:
    if progress is None:
        print(
            format_progress_line(
                label=label,
                stage=stage,
                status=status,
                current=current,
                total=total,
                unit=unit,
                message=message,
                stage_elapsed_s=None,
                eta_s=None,
            ),
            flush=True,
        )
        return None
    return progress.update(
        stage,
        status=status,
        current=current,
        total=total,
        unit=unit,
        message=message,
        extra=extra,
    )


__all__ = ["BatchProgressReporter", "emit_progress", "format_progress_line"]
