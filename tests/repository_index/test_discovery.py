from __future__ import annotations

from pathlib import Path
from threading import Event

from git import Repo
import pytest

from vibe.core.repository_index.discovery import (
    DiscoveryCancelledError,
    RepositoryDiscovery,
)


def _repository(root: Path) -> Repo:
    return Repo.init(root, initial_branch="main")


def test_discovers_tracked_and_non_ignored_untracked_text_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    repository.index.add([tracked.name])
    (tmp_path / "untracked.md").write_text("# Notes\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    result = RepositoryDiscovery().discover(tmp_path)

    assert [file.path for file in result.files] == [
        ".gitignore",
        "tracked.py",
        "untracked.md",
    ]
    assert result.files[1].language == "python"


def test_resolves_root_from_attached_subdirectory(tmp_path: Path) -> None:
    _repository(tmp_path)
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("pass\n", encoding="utf-8")

    result = RepositoryDiscovery().discover(nested)

    assert result.root == tmp_path.resolve()
    assert [file.path for file in result.files] == ["src/package/module.py"]


def test_excludes_vibeignore_sensitive_binary_oversized_and_symlink_files(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    (tmp_path / ".vibeignore").write_text("ignored/\n*.scratch\n", encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "module.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "notes.scratch").write_text("private\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"hello\0world")
    (tmp_path / "large.txt").write_text("x" * 40, encoding="utf-8")
    target = tmp_path / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(target)

    result = RepositoryDiscovery(max_file_size=32).discover(tmp_path)

    assert [file.path for file in result.files] == [".vibeignore", "target.txt"]
    assert result.skipped_binary == 1
    assert result.skipped_oversized == 1
    assert result.skipped_sensitive == 3
    assert result.skipped_symlink == 1


def test_cancellation_prevents_discovery(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "module.py").write_text("pass\n", encoding="utf-8")
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(DiscoveryCancelledError):
        RepositoryDiscovery().discover(tmp_path, cancel_event=cancel_event)


def test_excludes_git_submodules(tmp_path: Path) -> None:
    child_root = tmp_path / "child"
    parent_root = tmp_path / "parent"
    child_root.mkdir()
    parent_root.mkdir()
    child = _repository(child_root)
    child.config_writer().set_value("user", "name", "Tester").release()
    child.config_writer().set_value("user", "email", "t@example.com").release()
    (child_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    child.index.add(["module.py"])
    child.index.commit("child")
    parent = _repository(parent_root)
    parent.create_submodule("dependency", "vendor/dependency", url=str(child_root))

    result = RepositoryDiscovery().discover(parent_root)

    assert [file.path for file in result.files] == [".gitmodules"]
