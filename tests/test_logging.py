"""Tests for the metrics sink.

The rotation test covers a bug that silently corrupted a real training run:
re-running an experiment under the same name appended the new run onto the old
``metrics.csv``, producing a file with two header rows and two runs interleaved.
pandas reads that back as a single frame in which boolean columns have become
strings, so every downstream aggregate breaks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from drone_rl.utils.logging import MetricsLogger


def read_rows(path) -> list[dict]:
    return pd.read_csv(path).to_dict("records")


def test_logs_rows_to_csv(tmp_path):
    with MetricsLogger(tmp_path, fieldnames=["episode", "reward"]) as logger:
        logger.log({"episode": 1, "reward": -3.5})
        logger.log({"episode": 2, "reward": 7.25})

    frame = pd.read_csv(logger.csv_path)
    assert list(frame.columns) == ["episode", "reward"]
    assert frame.episode.tolist() == [1, 2]
    assert frame.reward.tolist() == pytest.approx([-3.5, 7.25])


def test_rerun_does_not_append_to_previous_run(tmp_path):
    """A second run must start a fresh CSV, not extend the first one."""
    with MetricsLogger(tmp_path, fieldnames=["episode", "success"]) as first:
        for episode in range(1, 6):
            first.log({"episode": episode, "success": episode > 3})

    with MetricsLogger(tmp_path, fieldnames=["episode", "success"]) as second:
        second.log({"episode": 1, "success": True})

    frame = pd.read_csv(second.csv_path)
    assert len(frame) == 1, "New run appended onto the previous run's CSV"
    # The decisive symptom of the original bug: a header row read as data,
    # which turns the boolean column into strings.
    assert frame.success.dtype != object or set(frame.success) <= {True, False}
    assert frame.episode.tolist() == [1]


def test_previous_run_is_archived_not_destroyed(tmp_path):
    """Rotation must preserve the old data, not silently delete it."""
    with MetricsLogger(tmp_path, fieldnames=["episode"]) as first:
        first.log({"episode": 1})
        first.log({"episode": 2})

    MetricsLogger(tmp_path, fieldnames=["episode"]).close()

    archived = sorted(tmp_path.glob("metrics_*.csv"))
    assert len(archived) == 1, "Previous metrics.csv was not archived"
    assert pd.read_csv(archived[0]).episode.tolist() == [1, 2]


def test_header_widens_without_corrupting_earlier_rows(tmp_path):
    """A key seen only later must trigger a clean rewrite, not a second header."""
    with MetricsLogger(tmp_path, fieldnames=["episode"]) as logger:
        logger.log({"episode": 1})
        logger.log({"episode": 2, "loss": 0.5})
        logger.log({"episode": 3, "loss": 0.25})

    with logger.csv_path.open(encoding="utf-8") as handle:
        header_count = sum(1 for line in handle if line.startswith("episode"))
    assert header_count == 1

    frame = pd.read_csv(logger.csv_path)
    assert frame.episode.tolist() == [1, 2, 3]
    assert list(frame.columns) == ["episode", "loss"]


def test_summary_is_written(tmp_path):
    with MetricsLogger(tmp_path) as logger:
        logger.log({"episode": 1, "reward": 1.0})
        path = logger.write_summary({"episodes": 1, "note": "ok"})
    assert path.is_file()
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["episodes"] == 1
