"""Console logging plus a metrics sink that fans out to CSV, JSON and TensorBoard.

The trainer records one row per episode. Rather than coupling the trainer to a
particular backend, it writes to :class:`MetricsLogger`, which appends rows to a
CSV incrementally (so a crashed run still leaves usable data), mirrors scalars
into TensorBoard when available, and dumps a JSON summary on close.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["configure_logging", "get_logger", "MetricsLogger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Install a console handler (and optional file handler) on the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # PyBullet and matplotlib are chatty at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


class MetricsLogger:
    """Append-only metrics sink writing CSV + JSON, optionally to TensorBoard.

    Parameters
    ----------
    output_dir:
        Directory that will hold ``metrics.csv`` and ``metrics_summary.json``.
    tensorboard_dir:
        When given and ``tensorboard`` is installed, scalars are mirrored here.
    fieldnames:
        Column order for the CSV. Keys logged later that are absent from this
        list are appended to the header on first sight; the CSV is rewritten in
        that (rare) case so the file stays well-formed.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tensorboard_dir: str | Path | None = None,
        fieldnames: Sequence[str] | None = None,
    ) -> None:
        self._log = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.json_path = self.output_dir / "metrics_summary.json"
        self._rotate_existing_csv()

        self._fieldnames: list[str] = list(fieldnames or [])
        self._rows: list[dict[str, Any]] = []
        self._header_written = False
        self._writer = None

        self._tb_writer = None
        if tensorboard_dir is not None:
            self._tb_writer = self._make_tensorboard_writer(tensorboard_dir)

    def _rotate_existing_csv(self) -> None:
        """Move a pre-existing ``metrics.csv`` aside before starting a new run.

        Rows are appended incrementally so a crashed run still leaves usable
        data, but that makes re-running an experiment under the same name
        dangerous: without this, the new run appends onto the old file and
        produces a CSV with two header rows and two runs interleaved, which
        pandas then reads as a single corrupt frame. Rotating preserves the old
        data instead of destroying it.
        """
        if not self.csv_path.exists():
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = self.csv_path.with_name(f"metrics_{timestamp}.csv")
        try:
            self.csv_path.replace(archived)
        except OSError as exc:  # pragma: no cover - file locked by another process
            self._log.warning(
                "Could not rotate %s (%s). Refusing to append to it; "
                "metrics for this run may be incomplete.", self.csv_path, exc
            )
            return
        self._log.info("Existing metrics.csv archived to %s", archived.name)

    def _make_tensorboard_writer(self, tensorboard_dir: str | Path):
        """Create a SummaryWriter, degrading gracefully if TensorBoard is absent."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:  # pragma: no cover - depends on optional install
            self._log.warning("TensorBoard not available; skipping TensorBoard logging.")
            return None
        path = Path(tensorboard_dir)
        path.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(path))

    def log(self, row: Mapping[str, Any], step: int | None = None) -> None:
        """Record one row of metrics and flush it to the CSV immediately."""
        record = dict(row)
        self._rows.append(record)

        new_keys = [k for k in record if k not in self._fieldnames]
        if new_keys and self._header_written:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv()
        elif new_keys:
            self._fieldnames.extend(new_keys)

        self._append_csv(record)

        if self._tb_writer is not None:
            tb_step = step if step is not None else len(self._rows)
            for key, value in record.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, tb_step)

    def _append_csv(self, record: Mapping[str, Any]) -> None:
        write_header = not self._header_written
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(record)

    def _rewrite_csv(self) -> None:
        """Rewrite the whole CSV after the schema widened mid-run."""
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows[:-1]:
                writer.writerow(row)
        self._header_written = True

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        """Write a JSON summary blob (aggregate stats, config echo, timings)."""
        with self.json_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(summary), handle, indent=2, default=str)
        return self.json_path

    @property
    def rows(self) -> list[dict[str, Any]]:
        """All rows logged so far."""
        return list(self._rows)

    def close(self) -> None:
        """Flush and release the TensorBoard writer."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
