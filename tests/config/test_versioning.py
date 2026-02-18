"""Tests for Config Versioning."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.versioning import ConfigVersionManager, VersionEntry, VersionManifest

# ---------------------------------------------------------------------------
# TestVersionEntry
# ---------------------------------------------------------------------------


class TestVersionEntry:
    """VersionEntry 모델 검증."""

    def test_creation(self) -> None:
        """기본 생성."""
        entry = VersionEntry(
            version=1,
            timestamp="2026-02-18T12:00:00Z",
            sha256="abc123",
            description="initial",
        )
        assert entry.version == 1
        assert entry.sha256 == "abc123"
        assert entry.description == "initial"

    def test_frozen(self) -> None:
        """frozen 모델은 수정 불가."""
        entry = VersionEntry(
            version=1,
            timestamp="2026-02-18T12:00:00Z",
            sha256="abc",
        )
        with pytest.raises(ValidationError):
            entry.version = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestConfigVersionManager
# ---------------------------------------------------------------------------


class TestConfigVersionManager:
    """ConfigVersionManager 검증."""

    def test_snapshot_creates_version_file(self, tmp_path: Path) -> None:
        """snapshot -> 버전 파일 생성 + 매니페스트 업데이트."""
        mgr = ConfigVersionManager(base_dir=tmp_path)
        data = {"strategy": "tsmom", "lookback": 30}

        entry = mgr.snapshot("default", data, description="initial config")

        assert entry.version == 1
        assert entry.description == "initial config"

        # 파일 존재 확인
        config_dir = tmp_path / "default"
        assert config_dir.exists()
        yaml_files = list(config_dir.glob("v001_*.yaml"))
        assert len(yaml_files) == 1

        # 매니페스트 확인
        manifest = mgr._load_manifest("default")
        assert manifest.current_version == 1
        assert len(manifest.entries) == 1

    def test_snapshot_skip_duplicate(self, tmp_path: Path) -> None:
        """동일 데이터 중복 스냅샷 -> 1개만 생성."""
        mgr = ConfigVersionManager(base_dir=tmp_path)
        data = {"strategy": "tsmom", "lookback": 30}

        entry1 = mgr.snapshot("default", data)
        entry2 = mgr.snapshot("default", data)

        assert entry1.version == entry2.version
        assert entry1.sha256 == entry2.sha256

        manifest = mgr._load_manifest("default")
        assert len(manifest.entries) == 1

    def test_list_versions(self, tmp_path: Path) -> None:
        """2회 스냅샷 -> 2개 엔트리, version 오름차순."""
        mgr = ConfigVersionManager(base_dir=tmp_path)

        mgr.snapshot("default", {"v": 1})
        mgr.snapshot("default", {"v": 2})

        versions = mgr.list_versions("default")
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    def test_load_version(self, tmp_path: Path) -> None:
        """snapshot -> load_version 으로 동일 데이터 복원."""
        mgr = ConfigVersionManager(base_dir=tmp_path)
        original = {"strategy": "tsmom", "lookback": 30, "vol_target": 0.35}

        mgr.snapshot("default", original)
        loaded = mgr.load_version("default", version=1)

        assert loaded == original

    def test_diff(self, tmp_path: Path) -> None:
        """서로 다른 데이터 diff -> added/removed/changed 확인."""
        mgr = ConfigVersionManager(base_dir=tmp_path)

        mgr.snapshot("default", {"a": 1, "b": 2, "c": 3})
        mgr.snapshot("default", {"a": 1, "b": 99, "d": 4})

        result = mgr.diff("default", v1=1, v2=2)

        assert result["added"] == {"d": 4}
        assert result["removed"] == {"c": 3}
        assert result["changed"] == {"b": {"old": 2, "new": 99}}

    def test_rollback_to(self, tmp_path: Path) -> None:
        """v1 데이터 -> v2 변경 -> rollback to v1 -> v3는 v1 데이터."""
        mgr = ConfigVersionManager(base_dir=tmp_path)
        data_v1 = {"strategy": "tsmom", "lookback": 30}
        data_v2 = {"strategy": "tsmom", "lookback": 60, "extra": True}

        mgr.snapshot("default", data_v1, description="v1")
        mgr.snapshot("default", data_v2, description="v2")

        entry = mgr.rollback_to("default", version=1)

        # rollback 시 v2 백업(v3) + v1 복원(v4) 또는 중복 skip
        assert entry.version >= 3
        restored = mgr.load_version("default", entry.version)
        assert restored == data_v1

    def test_empty_manifest(self, tmp_path: Path) -> None:
        """존재하지 않는 설정 -> 빈 목록."""
        mgr = ConfigVersionManager(base_dir=tmp_path)

        versions = mgr.list_versions("nonexistent")
        assert versions == []

    def test_version_manifest_model(self) -> None:
        """VersionManifest 기본 생성."""
        manifest = VersionManifest(config_name="test")
        assert manifest.config_name == "test"
        assert manifest.entries == []
        assert manifest.current_version == 0
