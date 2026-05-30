"""Tests for src.utils.atomic_write."""
import os
from pathlib import Path
from unittest.mock import patch

from src.utils.atomic_write import atomic_write


def test_atomic_write_creates_file(tmp_path):
    """Normal write produces file with correct content."""
    target = tmp_path / "state.json"
    data = '{"key": "value"}'
    atomic_write(target, data)
    assert target.read_text() == data


def test_atomic_write_overwrites_existing(tmp_path):
    """Overwriting an existing file replaces content atomically."""
    target = tmp_path / "state.json"
    target.write_text("old content")
    new_data = '{"new": true}'
    atomic_write(target, new_data)
    assert target.read_text() == new_data


def test_atomic_write_cleans_up_on_failure(tmp_path):
    """If the write raises, temp file is cleaned up and target is untouched."""
    target = tmp_path / "state.json"
    target.write_text("original")

    with patch("src.utils.atomic_write.os.fdopen", side_effect=IOError("disk full")):
        try:
            atomic_write(target, "bad data")
        except IOError:
            pass

    # Original file should be untouched
    assert target.read_text() == "original"
    # No leftover .tmp files
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_atomic_write_raises_on_missing_parent():
    """Caller must ensure parent exists; atomic_write does not mkdir."""
    target = Path("/nonexistent_dir_xyz/state.json")
    try:
        atomic_write(target, "data")
        assert False, "Should have raised"
    except (FileNotFoundError, OSError):
        pass
