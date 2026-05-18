from pathlib import Path

from src.main import cleanup_workspace_artifacts


def test_cleanup_workspace_artifacts_removes_only_safe_junk(tmp_path: Path):
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__" / "main.cpython-312.pyc").write_text("x")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml.bak").write_text("backup")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "final_test.log").write_text("temp")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "state.json").write_text("{}")

    out = cleanup_workspace_artifacts(tmp_path)

    assert out["dirs"] >= 2
    assert out["files"] >= 2
    assert not (tmp_path / "src" / "__pycache__").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "config" / "config.yaml.bak").exists()
    assert not (tmp_path / "logs" / "final_test.log").exists()
    assert (tmp_path / "data" / "state.json").exists()
