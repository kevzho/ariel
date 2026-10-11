"""Append-only, hash-chained JSONL ledger with fsync on every write.

`prev_sha256` links each row to the previous line, detecting edits, reordering
and interior deletions. Failed writes raise RecordingError and must stop training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class RecordingError(RuntimeError):
    """Persistence failed. Training must stop rather than run unrecorded."""


def to_jsonable(value):
    """Convert NumPy values and namespaces for json.dumps(default=...)."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:                       # pragma: no cover
        pass
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def dumps(value, **kwargs) -> str:
    """Strict JSON (no NaN/inf) with numpy support; raises RecordingError on bad values."""
    try:
        return json.dumps(value, allow_nan=False, default=to_jsonable, **kwargs)
    except (TypeError, ValueError) as exc:
        raise RecordingError(f"value cannot be recorded as JSON: {exc}") from exc


def line_sha256(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def read_ledger(path: Path) -> tuple[list[dict], str | None]:
    """Return (rows, head_sha). Raises RecordingError if the chain is broken.

    A trailing partial line (process killed mid-write) is ignored: it was never
    acknowledged to the caller.
    """
    path = Path(path)
    if not path.exists():
        return [], None
    rows, head = [], None
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        lines.pop()                          # unterminated final line: never fsynced as complete
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordingError(f"{path}:{number}: unreadable ledger row") from exc
        if row.get("prev_sha256") != head:
            raise RecordingError(f"{path}:{number}: hash chain broken")
        head = line_sha256(line)
        rows.append(row)
    return rows, head


class Ledger:
    """Single-writer append-only ledger. Open one per worker process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.poisoned = False
        self.rows, self.head = read_ledger(self.path)
        if self.path.exists():
            self._truncate_partial_tail()

    def _truncate_partial_tail(self) -> None:
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            keep = data.rfind(b"\n") + 1
            with self.path.open("r+b") as stream:
                stream.truncate(keep)
                stream.flush()
                os.fsync(stream.fileno())

    def append(self, row: dict) -> dict:
        """Append and fsync one row, rolling back the file size on failure.

        A failed rollback poisons the ledger and prevents further writes.
        """
        if self.poisoned:
            raise RecordingError(f"ledger {self.path} is unusable after an earlier failed write")
        row = {**row, "prev_sha256": self.head}
        line = dumps(row, sort_keys=True)
        size = self.path.stat().st_size if self.path.exists() else 0
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:
            try:
                with self.path.open("r+b") as stream:
                    stream.truncate(size)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                self.poisoned = True
            if isinstance(exc, OSError):
                raise RecordingError(f"ledger write failed: {self.path}: {exc}") from exc
            raise
        self.head = line_sha256(line)
        self.rows.append(row)
        return row


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file + fsync + rename: the file is either complete or absent."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise RecordingError(f"write failed: {path}: {exc}") from exc


def atomic_write_json(path: Path, value: dict) -> None:
    atomic_write_bytes(path, (dumps(value, indent=2, sort_keys=True) + "\n").encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
