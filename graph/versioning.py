"""
graph/versioning.py — Graph snapshot and version management.

Implements:
  1. Snapshot creation (copy of Kùzu DB + metadata JSON)
  2. Snapshot listing and rollback
  3. DVC-compatible data tracking (writes .dvc stub files)
  4. Semantic version bumping

Snapshots are stored in: data/snapshots/<version_tag>/
DVC stub files are written to: data/snapshots/<version_tag>.dvc
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


_DEFAULT_DB_PATH       = os.getenv("GRAPH_DB_PATH",      "./data/graph.kuzu")
_DEFAULT_SNAPSHOT_DIR  = os.getenv("SNAPSHOT_DIR",       "./data/snapshots")
_METADATA_FILENAME     = "snapshot_metadata.json"


# ---------------------------------------------------------------------------
# Snapshot creation
# ---------------------------------------------------------------------------

def create_snapshot(
    version_tag: str,
    release_notes: str = "",
    db_path: str = _DEFAULT_DB_PATH,
    snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR,
) -> dict:
    """
    Create a versioned snapshot of the current graph database.

    Args:
        version_tag:   Semantic version string, e.g. "0.2.0"
        release_notes: Human-readable change description
        db_path:       Source Kùzu DB path
        snapshot_dir:  Parent directory for snapshots

    Returns:
        {version_tag, snapshot_path, metadata_path, created_at}
    """
    snap_dir = Path(snapshot_dir) / version_tag
    snap_dir.mkdir(parents=True, exist_ok=True)

    db_source = Path(db_path)
    if db_source.exists():
        if db_source.is_dir():
            dest = snap_dir / "graph.kuzu"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(db_source, dest)
        else:
            shutil.copy2(db_source, snap_dir / "graph.kuzu")
    else:
        # No DB yet — create empty marker
        (snap_dir / "graph.kuzu.empty").touch()

    # Metadata
    metadata = {
        "version_tag":    version_tag,
        "release_notes":  release_notes,
        "created_at":     datetime.now(tz=timezone.utc).isoformat(),
        "db_source":      str(db_source.resolve()),
        "snapshot_path":  str(snap_dir),
    }
    meta_path = snap_dir / _METADATA_FILENAME
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Write DVC-compatible stub (plain text, tracks the snapshot path)
    dvc_path = Path(snapshot_dir) / f"{version_tag}.dvc"
    dvc_stub = {
        "outs": [{"path": str(snap_dir / "graph.kuzu"), "md5": None}],
        "meta": {"version": version_tag},
    }
    with dvc_path.open("w") as f:
        json.dump(dvc_stub, f, indent=2)

    return {
        "version_tag":    version_tag,
        "snapshot_path":  str(snap_dir),
        "metadata_path":  str(meta_path),
        "dvc_stub_path":  str(dvc_path),
        "created_at":     metadata["created_at"],
    }


# ---------------------------------------------------------------------------
# Snapshot listing
# ---------------------------------------------------------------------------

def list_snapshots(snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR) -> list[dict]:
    """
    List all available snapshots sorted by creation time (newest first).

    Returns:
        List of metadata dicts, or empty list if no snapshots exist.
    """
    snap_dir = Path(snapshot_dir)
    if not snap_dir.exists():
        return []

    snapshots: list[dict] = []
    for item in snap_dir.iterdir():
        if item.is_dir():
            meta_path = item / _METADATA_FILENAME
            if meta_path.exists():
                with meta_path.open() as f:
                    try:
                        snapshots.append(json.load(f))
                    except json.JSONDecodeError:
                        pass

    snapshots.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return snapshots


def get_latest_snapshot(snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR) -> dict | None:
    """Return the most recent snapshot metadata, or None if none exist."""
    snapshots = list_snapshots(snapshot_dir)
    return snapshots[0] if snapshots else None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback_to_snapshot(
    version_tag: str,
    db_path: str = _DEFAULT_DB_PATH,
    snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR,
) -> dict:
    """
    Restore the graph database from a specific snapshot.

    Args:
        version_tag: Which snapshot to restore
        db_path:     Target DB path to overwrite
        snapshot_dir: Parent snapshot directory

    Returns:
        {status, version_tag, restored_at}
    """
    snap_dir = Path(snapshot_dir) / version_tag
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot {version_tag!r} not found in {snapshot_dir}")

    snap_db = snap_dir / "graph.kuzu"
    db_target = Path(db_path)

    if snap_db.exists():
        if db_target.exists():
            if db_target.is_dir():
                shutil.rmtree(db_target)
            else:
                db_target.unlink()

        if snap_db.is_dir():
            shutil.copytree(snap_db, db_target)
        else:
            shutil.copy2(snap_db, db_target)
    else:
        return {
            "status":       "WARN_EMPTY_SNAPSHOT",
            "version_tag":  version_tag,
            "restored_at":  datetime.now(tz=timezone.utc).isoformat(),
        }

    return {
        "status":       "OK",
        "version_tag":  version_tag,
        "restored_at":  datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Semantic version bumping
# ---------------------------------------------------------------------------

def bump_version(current_version: str, bump: str = "patch") -> str:
    """
    Bump a semantic version string.

    Args:
        current_version: e.g. "0.1.3"
        bump: "major" | "minor" | "patch"

    Returns:
        New version string, e.g. "0.1.4"
    """
    parts = current_version.strip().lstrip("v").split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0

    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


# ---------------------------------------------------------------------------
# Current version helper
# ---------------------------------------------------------------------------

def get_current_version(snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR) -> str:
    """Return the current graph version from the latest snapshot, or '0.1.0'."""
    latest = get_latest_snapshot(snapshot_dir)
    return latest["version_tag"] if latest else "0.1.0"
