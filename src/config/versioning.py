"""Config Versioning -- 설정 버전 관리.

설정 변경 이력을 추적하고, 이전 버전으로 안전한 롤백을 지원합니다.

Directory structure:
    config/versions/{config_name}/
    ├── v001_20260218_120000.yaml
    ├── v002_20260219_090000.yaml
    └── manifest.yaml

Rules Applied:
    - #11 Pydantic Modeling: frozen models
    - SHA-256 hash for change detection
    - Safe rollback (snapshot current before restore)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# -- Models ----------------------------------------------------------------


class VersionEntry(BaseModel):
    """단일 버전 엔트리."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(description="버전 번호 (1-based)")
    timestamp: datetime = Field(description="스냅샷 생성 시각 (UTC)")
    sha256: str = Field(description="설정 데이터 SHA-256 해시")
    description: str = Field(default="", description="변경 설명")


class VersionManifest(BaseModel):
    """설정 버전 매니페스트."""

    model_config = ConfigDict(frozen=True)

    config_name: str = Field(description="설정 이름")
    entries: list[VersionEntry] = Field(default_factory=list, description="버전 엔트리 목록")
    current_version: int = Field(default=0, description="현재 활성 버전")


# -- Manager ---------------------------------------------------------------


class ConfigVersionManager:
    """설정 버전 관리자.

    설정의 스냅샷 생성, 버전 조회, diff, 안전한 롤백을 지원합니다.
    """

    def __init__(self, base_dir: Path = Path("config/versions")) -> None:
        self._base_dir = base_dir

    @staticmethod
    def _compute_hash(config_data: dict[str, Any]) -> str:
        """설정 데이터의 SHA-256 해시 계산."""
        raw = yaml.dump(config_data, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _version_filename(version: int, ts: datetime) -> str:
        """버전 파일명 생성: v{NNN}_{YYYYMMDD}_{HHMMSS}.yaml."""
        return f"v{version:03d}_{ts.strftime('%Y%m%d')}_{ts.strftime('%H%M%S')}.yaml"

    def _config_dir(self, config_name: str) -> Path:
        """설정별 디렉토리 경로."""
        return self._base_dir / config_name

    # -- Manifest I/O ------------------------------------------------------

    def _load_manifest(self, config_name: str) -> VersionManifest:
        """매니페스트 로드 (없으면 빈 매니페스트 생성)."""
        manifest_path = self._config_dir(config_name) / "manifest.yaml"
        if not manifest_path.exists():
            return VersionManifest(config_name=config_name)

        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        return VersionManifest(**raw)

    def _save_manifest(self, config_name: str, manifest: VersionManifest) -> None:
        """매니페스트 YAML 저장."""
        config_dir = self._config_dir(config_name)
        config_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = config_dir / "manifest.yaml"
        data = manifest.model_dump(mode="json")
        manifest_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    # -- Public API --------------------------------------------------------

    def snapshot(
        self,
        config_name: str,
        config_data: dict[str, Any],
        description: str = "",
    ) -> VersionEntry:
        """현재 설정 스냅샷 저장.

        SHA-256 해시가 최신 버전과 동일하면 중복 스냅샷을 건너뜁니다.

        Args:
            config_name: 설정 이름 (예: "default", "paper")
            config_data: 설정 딕셔너리
            description: 변경 설명

        Returns:
            생성된 (또는 기존) VersionEntry
        """
        sha = self._compute_hash(config_data)
        manifest = self._load_manifest(config_name)

        # 중복 검사: 최신 버전과 해시 동일 → skip
        if manifest.entries:
            latest = manifest.entries[-1]
            if latest.sha256 == sha:
                return latest

        now = datetime.now(tz=UTC)
        new_version = manifest.current_version + 1

        entry = VersionEntry(
            version=new_version,
            timestamp=now,
            sha256=sha,
            description=description,
        )

        # 버전 파일 저장
        config_dir = self._config_dir(config_name)
        config_dir.mkdir(parents=True, exist_ok=True)
        filename = self._version_filename(new_version, now)
        version_path = config_dir / filename
        version_path.write_text(yaml.dump(config_data, sort_keys=True), encoding="utf-8")

        # 매니페스트 업데이트
        updated_manifest = VersionManifest(
            config_name=config_name,
            entries=[*manifest.entries, entry],
            current_version=new_version,
        )
        self._save_manifest(config_name, updated_manifest)

        return entry

    def list_versions(self, config_name: str) -> list[VersionEntry]:
        """버전 목록 반환 (version 오름차순).

        Args:
            config_name: 설정 이름

        Returns:
            VersionEntry 리스트 (version 오름차순 정렬)
        """
        manifest = self._load_manifest(config_name)
        return sorted(manifest.entries, key=lambda e: e.version)

    def load_version(self, config_name: str, version: int) -> dict[str, Any]:
        """특정 버전의 설정 데이터 로드.

        Args:
            config_name: 설정 이름
            version: 로드할 버전 번호

        Returns:
            설정 딕셔너리

        Raises:
            FileNotFoundError: 버전 파일을 찾을 수 없을 때
        """
        manifest = self._load_manifest(config_name)
        entry = self._find_entry(manifest, version)

        filename = self._version_filename(entry.version, entry.timestamp)
        version_path = self._config_dir(config_name) / filename
        if not version_path.exists():
            msg = f"Version file not found: {version_path}"
            raise FileNotFoundError(msg)

        result: dict[str, Any] = yaml.safe_load(version_path.read_text(encoding="utf-8"))
        return result

    def diff(self, config_name: str, v1: int, v2: int) -> dict[str, Any]:
        """두 버전 간 차이 비교.

        Args:
            config_name: 설정 이름
            v1: 기준 버전
            v2: 비교 버전

        Returns:
            {"added": {...}, "removed": {...}, "changed": {...}}
        """
        data1 = self.load_version(config_name, v1)
        data2 = self.load_version(config_name, v2)
        return self._compute_diff(data1, data2)

    def rollback_to(self, config_name: str, version: int) -> VersionEntry:
        """안전한 롤백: 현재 상태를 스냅샷한 후 대상 버전으로 복원.

        Args:
            config_name: 설정 이름
            version: 롤백 대상 버전

        Returns:
            복원된 설정의 새 VersionEntry
        """
        # 현재 최신 데이터를 먼저 스냅샷 (안전 백업)
        manifest = self._load_manifest(config_name)
        if manifest.entries:
            latest = manifest.entries[-1]
            current_data = self.load_version(config_name, latest.version)
            self.snapshot(
                config_name,
                current_data,
                description=f"Auto-backup before rollback to v{version:03d}",
            )

        # 대상 버전 로드 후 새 스냅샷으로 저장
        target_data = self.load_version(config_name, version)
        return self.snapshot(
            config_name,
            target_data,
            description=f"Rollback to v{version:03d}",
        )

    # -- Private helpers ---------------------------------------------------

    @staticmethod
    def _find_entry(manifest: VersionManifest, version: int) -> VersionEntry:
        """매니페스트에서 특정 버전 엔트리 검색."""
        for entry in manifest.entries:
            if entry.version == version:
                return entry
        msg = f"Version {version} not found in manifest '{manifest.config_name}'"
        raise KeyError(msg)

    @staticmethod
    def _compute_diff(
        data1: dict[str, Any],
        data2: dict[str, Any],
    ) -> dict[str, Any]:
        """두 딕셔너리 간 flat diff 계산."""
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())

        added = {k: data2[k] for k in keys2 - keys1}
        removed = {k: data1[k] for k in keys1 - keys2}
        changed: dict[str, dict[str, Any]] = {}

        for k in keys1 & keys2:
            if data1[k] != data2[k]:
                changed[k] = {"old": data1[k], "new": data2[k]}

        return {"added": added, "removed": removed, "changed": changed}
