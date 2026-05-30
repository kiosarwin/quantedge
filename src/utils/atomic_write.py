"""Atomic file write utility - prevents state corruption on crash."""
import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, data: str) -> None:
    """Write data atomically using write-to-temp-then-rename.

    Creates a temporary file in the same directory as the target,
    writes data to it, then atomically replaces the target file.
    The caller must ensure path.parent exists before calling.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
