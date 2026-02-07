"""
Metrics tracking and monitoring for the OCR pipeline.
"""

from __future__ import annotations

import base64
import gzip
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger

from iwe_pipeline.utils.utils import SYSTEM_PROMPT, get_pdf_num_pages


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_rate(rate_per_sec: float) -> str:
    if rate_per_sec <= 0:
        return "0.0 items/s"
    if rate_per_sec >= 100:
        return f"{rate_per_sec:.0f} items/s"
    if rate_per_sec >= 10:
        return f"{rate_per_sec:.1f} items/s"
    return f"{rate_per_sec:.2f} items/s"


def _format_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _make_record_id(doc_id: str, page_index: int, timestamp: float) -> str:
    raw = f"{doc_id}|{page_index}|{int(timestamp)}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _truncate_text(value: str | None, limit: int | None) -> str:
    if value is None:
        return ""
    if limit is None or limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"... (truncated, {len(value)} chars)"


def _json_preview(value: Any, limit: int | None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return _truncate_text(text, limit)


def _build_request_payload_preview(
    request_payloads: Any,
    media: list[dict[str, Any]] | None,
    page_index: int,
    max_chars: int | None,
) -> str:
    if isinstance(request_payloads, list) and page_index < len(request_payloads):
        payload = request_payloads[page_index]
        if isinstance(payload, str):
            return _truncate_text(payload, max_chars)
        return _json_preview(payload, max_chars)
    if isinstance(request_payloads, dict):
        return _json_preview(request_payloads, max_chars)
    if media and page_index < len(media):
        media_bytes = media[page_index].get("media_bytes")
        if media_bytes:
            media_preview = _truncate_text(str(media_bytes), max_chars)
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{media_preview}"},
                            },
                        ],
                    }
                ],
            }
            return _json_preview(payload, max_chars)
    return "request payload unavailable"


class _MonitorState:
    def __init__(self, output_dir: Path, input_dir: Path | None):
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "status": "starting",
            "start_time": time.time(),
            "last_update": None,
            "completed": 0,
            "total": 0,
            "total_known": False,
            "rate_per_sec": 0.0,
            "eta_seconds": None,
            "output_dir": str(output_dir),
            "input_dir": str(input_dir) if input_dir else None,
            "records": [],
            "records_total": 0,
            "records_index": {},
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._data)

        total = data.get("total", 0)
        total_known = data.get("total_known", False)
        completed = data.get("completed", 0)
        rate_per_sec = data.get("rate_per_sec", 0.0)
        eta_seconds = data.get("eta_seconds", None)

        percentage = None
        if total_known and total > 0:
            percentage = min(100.0, (completed / total) * 100.0)

        start_time = data.get("start_time")
        last_update = data.get("last_update")
        elapsed = None
        if start_time is not None:
            end_time = last_update if last_update is not None else time.time()
            elapsed = max(0.0, end_time - start_time)

        return {
            "status": data.get("status"),
            "completed": completed,
            "total": total if total_known else None,
            "total_known": total_known,
            "percentage": percentage,
            "rate_per_sec": rate_per_sec,
            "rate_human": _format_rate(rate_per_sec),
            "eta_seconds": eta_seconds,
            "eta_human": _format_duration(eta_seconds),
            "elapsed_seconds": elapsed,
            "elapsed_human": _format_duration(elapsed),
            "started_at": _format_timestamp(start_time),
            "last_update": _format_timestamp(last_update),
            "output_dir": data.get("output_dir"),
            "input_dir": data.get("input_dir"),
        }

    def records_snapshot(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        with self._lock:
            records = list(self._data.get("records", []))
            total = self._data.get("records_total", len(records))

        start = max(0, offset or 0)
        if limit is not None and limit > 0:
            records = records[start : start + limit]
        elif start:
            records = records[start:]

        return {"records": records, "records_total": total}

    def record_by_id(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            records_index = self._data.get("records_index", {})
            record = records_index.get(record_id)
            return dict(record) if record else None


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _make_handler(state: _MonitorState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = _load_template("index.html").encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return

            if parsed.path == "/record":
                body = _load_template("record.html").encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return

            if parsed.path == "/api/progress":
                payload = json.dumps(state.snapshot()).encode("utf-8")
                self._send_bytes(payload, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/records":
                params = parse_qs(parsed.query)
                limit_param = params.get("limit", [None])[0]
                offset_param = params.get("offset", [None])[0]
                limit = None
                offset = None
                if limit_param:
                    try:
                        limit = int(limit_param)
                    except ValueError:
                        limit = None
                if offset_param:
                    try:
                        offset = int(offset_param)
                    except ValueError:
                        offset = None
                payload = json.dumps(state.records_snapshot(limit, offset)).encode("utf-8")
                self._send_bytes(payload, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/record":
                params = parse_qs(parsed.query)
                record_id = params.get("record_id", [None])[0]
                if not record_id:
                    self.send_error(400, "record_id required")
                    return
                record = state.record_by_id(record_id)
                if record is None:
                    self.send_error(404, "record not found")
                    return
                payload = json.dumps(record).encode("utf-8")
                self._send_bytes(payload, "application/json; charset=utf-8")
                return

            self.send_error(404, "Not Found")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return _Handler


def _glob_paths(root: Path, pattern: str) -> list[Path]:
    if "**" in pattern:
        return sorted(root.rglob(pattern))
    return sorted(root.glob(pattern))


def _find_checkpoint_files(output_dir: Path) -> list[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return _glob_paths(checkpoints_dir, "**/chunk_*.jsonl")


def _find_main_output_files(output_dir: Path) -> list[Path]:
    for pattern in ("*.parquet", "*.parquet.gz", "*.jsonl", "*.jsonl.gz"):
        files = sorted(output_dir.glob(pattern))
        if files:
            return files
    return []


def _is_jsonl_like(path: Path) -> bool:
    return (
        path.suffix == ".jsonl"
        or path.name.endswith(".jsonl.gz")
        or path.name.endswith(".parquet.gz")
    )


def _count_jsonl_lines(path: Path) -> int:
    try:
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle)
        with path.open("rt", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except Exception as exc:
        logger.warning(f"Failed to count lines in {path}: {exc}")
        return 0


def _count_parquet_rows(path: Path) -> int:
    try:
        parquet_file = pq.ParquetFile(path)
        return parquet_file.metadata.num_rows
    except Exception as exc:
        logger.warning(f"Failed to read parquet metadata from {path}: {exc}")
        return 0


class _OutputCounter:
    def __init__(self) -> None:
        self._counts: dict[Path, int] = {}
        self._mtimes: dict[Path, float] = {}

    def _count_paths(self, paths: list[Path], seen: set[Path]) -> int:
        total = 0
        for path in paths:
            seen.add(path)
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue

            cached_mtime = self._mtimes.get(path)
            if cached_mtime is not None and cached_mtime == mtime:
                total += self._counts.get(path, 0)
                continue

            if path.suffix == ".parquet":
                count = _count_parquet_rows(path)
            elif (
                path.name.endswith(".jsonl.gz")
                or path.suffix == ".jsonl"
                or path.name.endswith(".parquet.gz")
            ):
                count = _count_jsonl_lines(path)
            else:
                count = 0

            self._mtimes[path] = mtime
            self._counts[path] = count
            total += count
        return total

    def count(self, output_dir: Path, output_glob: str | None) -> int:
        seen: set[Path] = set()

        if output_glob:
            files = _glob_paths(output_dir, output_glob)
            total = self._count_paths(files, seen)
        else:
            main_files = _find_main_output_files(output_dir)
            checkpoint_files = _find_checkpoint_files(output_dir)

            main_count = self._count_paths(main_files, seen)
            checkpoint_count = self._count_paths(checkpoint_files, seen)

            has_parquet = any(path.suffix == ".parquet" for path in main_files)
            if checkpoint_files and has_parquet:
                total = main_count + checkpoint_count
            elif checkpoint_files and main_files:
                total = max(main_count, checkpoint_count)
            elif checkpoint_files:
                total = checkpoint_count
            else:
                total = main_count

        for cached in list(self._counts.keys()):
            if cached not in seen:
                self._counts.pop(cached, None)
                self._mtimes.pop(cached, None)

        return total


def _count_input_documents(input_dir: Path, input_glob: str, page_level: bool) -> int:
    files = sorted(input_dir.glob(input_glob))
    if not page_level:
        return len(files)

    total_pages = 0
    for path in files:
        try:
            total_pages += get_pdf_num_pages(str(path))
        except Exception as exc:
            logger.warning(f"Failed to count pages in {path}: {exc}")
    return total_pages


def _iter_documents_from_file(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning(f"Failed to read records from {path}: {exc}")
        return []
    return documents


def _extract_page_records(
    document: dict[str, Any],
    *,
    file_timestamp: float,
    request_payload_max_chars: int | None,
) -> list[dict[str, Any]]:
    doc_id = document.get("id", "")
    metadata = document.get("metadata") or {}
    media = document.get("media") or []
    request_page_indices = metadata.get("request_page_indices")
    inference_results = metadata.get("inference_results") or []
    request_payloads = metadata.get("request_payloads")

    if not request_page_indices:
        request_page_indices = list(range(len(inference_results)))

    records: list[dict[str, Any]] = []
    for idx, page_index in enumerate(request_page_indices):
        result = inference_results[idx] if idx < len(inference_results) else None
        error = None
        output_text = ""

        if isinstance(result, dict):
            output_text = result.get("text") or ""
            error = result.get("error") or result.get("error_message")
        elif result is not None:
            output_text = str(result)

        status = "pending"
        if error:
            status = "error"
        elif output_text:
            status = "success"

        request_payload = _build_request_payload_preview(
            request_payloads, media, page_index, request_payload_max_chars
        )

        record_timestamp = metadata.get("timestamp")
        if not isinstance(record_timestamp, int | float):
            record_timestamp = file_timestamp

        record_id = _make_record_id(doc_id, int(page_index), float(record_timestamp))

        records.append(
            {
                "doc_id": doc_id,
                "page_index": page_index,
                "record_id": record_id,
                "status": status,
                "request_payload": request_payload,
                "model_output": output_text,
                "error": error or "",
                "timestamp": _format_timestamp(record_timestamp),
                "timestamp_epoch": record_timestamp,
            }
        )

    return records


class _RecordCache:
    def __init__(self) -> None:
        self._mtimes: dict[Path, float] = {}
        self._records: dict[Path, list[dict[str, Any]]] = {}

    def read_records(
        self, path: Path, *, request_payload_max_chars: int | None
    ) -> list[dict[str, Any]]:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return []

        cached_mtime = self._mtimes.get(path)
        if cached_mtime is not None and cached_mtime == mtime:
            return self._records.get(path, [])

        documents = _iter_documents_from_file(path)
        records: list[dict[str, Any]] = []
        for doc in documents:
            records.extend(
                _extract_page_records(
                    doc,
                    file_timestamp=mtime,
                    request_payload_max_chars=request_payload_max_chars,
                )
            )

        self._mtimes[path] = mtime
        self._records[path] = records
        return records

    def prune(self, active_paths: set[Path]) -> None:
        for cached in list(self._records.keys()):
            if cached not in active_paths:
                self._records.pop(cached, None)
                self._mtimes.pop(cached, None)


_TEMPLATES: dict[str, str] = {}
_TEMPLATES_DIR = Path(__file__).with_name("ui")


def _load_template(name: str) -> str:
    cached = _TEMPLATES.get(name)
    if cached is not None:
        return cached
    path = _TEMPLATES_DIR / name
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = f"<html><body>Missing template: {name}</body></html>"
    _TEMPLATES[name] = content
    return content


@dataclass
class OCRInferenceProgressMonitor(PipelineStep):
    """
    Expose a lightweight web UI to track OCR inference progress.

    The monitor:
    1. Starts an HTTP server on the requested host/port
    2. Scans the output directory to count processed items
    3. Computes throughput, ETA, and completion status
    4. Serves an HTML page plus a JSON progress endpoint
    """

    output_dir: str
    port: int = 8040
    host: str = "127.0.0.1"
    update_interval: int = 5
    output_glob: str | None = None
    stats_path: str | None = None
    input_dir: str | None = None
    input_glob: str = "*.pdf"
    page_level: bool = True
    total_documents: int | None = None
    inference_job_id: str | None = None
    request_payload_max_chars: int | None = 1200
    max_records: int = -1
    ignore_existing_outputs: bool = True

    name: str = "OCRInferenceProgressMonitor"
    type: str = "Monitor"

    def _is_job_running(self, job_id: str) -> bool:
        try:
            import subprocess

            result = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True)
            return bool(result.stdout.strip())
        except Exception as exc:
            logger.warning(f"Failed to check Slurm job status: {exc}")
            return True

    def _resolve_total(self) -> tuple[int, bool]:
        if self.total_documents is not None:
            return self.total_documents, True

        if self.input_dir:
            input_dir = Path(self.input_dir)
            if input_dir.exists():
                total = _count_input_documents(input_dir, self.input_glob, self.page_level)
                return total, True

        return 0, False

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        if rank != 0:
            if data:
                yield from data
            return

        if data:
            yield from data

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_dir = Path(self.input_dir) if self.input_dir else None
        stats_path = Path(self.stats_path) if self.stats_path else None

        state = _MonitorState(output_dir=output_dir, input_dir=input_dir)
        total_docs, total_known = self._resolve_total()
        state.update(total=total_docs, total_known=total_known)

        handler = _make_handler(state)
        server = _ReusableThreadingHTTPServer((self.host, self.port), handler)
        server_thread = threading.Thread(
            target=server.serve_forever, name="ocr-progress-server", daemon=True
        )
        server_thread.start()

        logger.info(f"OCR progress monitor running at http://{self.host}:{self.port}")
        logger.info(f"Output directory: {output_dir}")
        if input_dir:
            logger.info(f"Input directory: {input_dir}")
        if total_known:
            logger.info(f"Expected total items: {total_docs}")

        counter = _OutputCounter()
        record_cache = _RecordCache()
        start_time = time.time()
        state.update(status="running", start_time=start_time, last_update=start_time)
        baseline_count = 0
        if self.ignore_existing_outputs:
            baseline_count = counter.count(output_dir, self.output_glob)
            if baseline_count > 0:
                logger.warning(
                    f"Found {baseline_count} existing outputs; tracking new progress only."
                )
        if stats_path and stats_path.exists():
            try:
                stats_path.stat().st_mtime
                logger.warning("stats.json exists from a previous run; ignoring until it updates.")
            except OSError as exc:
                logger.warning(f"Failed to stat stats.json: {exc}")

        try:
            while True:
                if stats_path and stats_path.exists():
                    try:
                        stats_mtime = stats_path.stat().st_mtime
                    except OSError:
                        stats_mtime = None
                    if stats_mtime is not None and stats_mtime < start_time:
                        logger.info(
                            "stats.json is older than this monitor start; waiting for update."
                        )
                    else:
                        state.update(status="complete")
                        logger.info("stats.json detected - marking job as complete.")
                        break

                if self.inference_job_id and not self._is_job_running(self.inference_job_id):
                    state.update(status="stopped")
                    logger.info(
                        f"Inference job {self.inference_job_id} no longer running; stopping."
                    )
                    break

                current_count = counter.count(output_dir, self.output_glob)
                completed_docs = (
                    max(0, current_count - baseline_count)
                    if self.ignore_existing_outputs
                    else current_count
                )
                current_time = time.time()
                elapsed = max(0.0, current_time - start_time)
                rate = completed_docs / elapsed if elapsed > 0 else 0.0

                eta_seconds = None
                if total_known and total_docs > 0 and completed_docs < total_docs and rate > 0:
                    eta_seconds = (total_docs - completed_docs) / rate

                record_paths: set[Path] = set()
                if self.output_glob:
                    record_paths.update(
                        path
                        for path in _glob_paths(output_dir, self.output_glob)
                        if _is_jsonl_like(path)
                    )
                else:
                    record_paths.update(_find_checkpoint_files(output_dir))
                    record_paths.update(
                        path for path in _find_main_output_files(output_dir) if _is_jsonl_like(path)
                    )

                records: list[dict[str, Any]] = []
                for path in sorted(record_paths):
                    records.extend(
                        record_cache.read_records(
                            path, request_payload_max_chars=self.request_payload_max_chars
                        )
                    )
                record_cache.prune(record_paths)

                records.sort(
                    key=lambda item: item.get("timestamp_epoch") or 0,
                    reverse=True,
                )
                records_total = len(records)
                if self.max_records is not None and self.max_records > 0:
                    records = records[: self.max_records]

                records_index = {
                    record_id: record
                    for record in records
                    if (record_id := record.get("record_id"))
                }

                state.update(
                    completed=completed_docs,
                    last_update=current_time,
                    rate_per_sec=rate,
                    eta_seconds=eta_seconds,
                    records=records,
                    records_total=records_total,
                    records_index=records_index,
                )

                logger.info(
                    f"Progress: {completed_docs} / " f"{total_docs if total_known else 'unknown'}"
                )

                if total_known and total_docs > 0 and completed_docs >= total_docs:
                    state.update(status="complete")
                    logger.info("Completed items reached expected total; stopping monitor.")
                    break

                time.sleep(self.update_interval)
        finally:
            server.shutdown()
            server.server_close()
            logger.info("OCR progress monitor server stopped.")
