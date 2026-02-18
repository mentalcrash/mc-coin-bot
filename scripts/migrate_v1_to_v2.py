"""Migrate v1 (Gate) strategy YAML files to v2 (Phase) format.

Usage:
    uv run python scripts/migrate_v1_to_v2.py               # Migrate all
    uv run python scripts/migrate_v1_to_v2.py --dry-run      # Preview changes
    uv run python scripts/migrate_v1_to_v2.py --no-backup    # Skip .bak creation

Changes applied:
    - gates: → phases: (GATE_TO_PHASE mapping)
    - G1+G2 → P4 merge, G2H+G3 → P5 merge
    - decisions[].gate → decisions[].phase
    - version: 2 added
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.pipeline.store import StrategyStore


def migrate_file(
    yaml_path: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> bool:
    """Migrate a single strategy YAML from v1 to v2.

    The StrategyStore._deserialize() already handles v1→v2 conversion
    (including gate merging). We just load and re-save.

    Args:
        yaml_path: Path to strategy YAML file.
        dry_run: If True, only print what would change.
        backup: If True, create .bak before overwriting.

    Returns:
        True if file was migrated, False if already v2 or skipped.
    """
    import yaml

    raw_text = yaml_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)

    if raw is None:
        return False

    # Already v2?
    if "phases" in raw and "gates" not in raw:
        return False

    # Not a v1 file (no gates key)?
    if "gates" not in raw:
        return False

    name = yaml_path.stem
    if dry_run:
        gate_ids = list(raw.get("gates", {}).keys())
        n_decisions = len(raw.get("decisions", []))
        print(f"  [DRY-RUN] {name}: gates={gate_ids}, decisions={n_decisions}")
        return True

    # Create backup
    if backup:
        bak_path = yaml_path.with_suffix(".yaml.bak")
        shutil.copy2(yaml_path, bak_path)

    # Load through store (handles v1→v2 conversion) and re-save
    base_dir = yaml_path.parent
    store = StrategyStore(base_dir=base_dir)
    record = store.load(name)
    store.save(record)

    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point for migration script."""
    parser = argparse.ArgumentParser(description="Migrate v1 gate YAML to v2 phase format")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("strategies"),
        help="Directory containing strategy YAML files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip .bak file creation",
    )
    args = parser.parse_args(argv)

    strategies_dir: Path = args.dir
    if not strategies_dir.exists():
        print(f"Error: {strategies_dir} does not exist")
        return 1

    yaml_files = sorted(strategies_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {strategies_dir}")
        return 0

    print(f"Scanning {len(yaml_files)} YAML files in {strategies_dir}/")
    if args.dry_run:
        print("[DRY-RUN MODE — no files will be modified]\n")

    migrated = 0
    skipped = 0
    errors = 0

    for path in yaml_files:
        try:
            if migrate_file(path, dry_run=args.dry_run, backup=not args.no_backup):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {path.stem}: {e}")
            errors += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped (already v2), {errors} errors")
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
