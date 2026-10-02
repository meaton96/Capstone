"""
@file player_manifest.py
@brief Check a Unity player build against its BUILD_MANIFEST.json (thesis section 7.2).

The build script (Capstone/Assets/Editor/BuildManifest.cs) records the git commit a player was built
from, whether the source had uncommitted changes, and the SHA-256 of every file the player loads.
verify_player recomputes those hashes: a changed, missing, or extra file in the player means the
binary is not the one the manifest describes, and training / evaluation refuse to launch it.

A player without a manifest (built before 2026-09-30) is reported, not refused.

Command line:  python env/player_manifest.py linux_server/capstone.x86_64
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "BUILD_MANIFEST.json"
## @brief Keep in step with BuildManifest.cs: build-folder entries that are not part of the player.
EXCLUDED_TOP = {MANIFEST_NAME, "Results", "BatchConfigs", "__pycache__"}
EXCLUDED_PREFIXES = ("capstone_Data/ML-Agents/",)


class PlayerIntegrityError(RuntimeError):
    """@brief The player's files do not match its manifest."""


def included(rel_path: str) -> bool:
    top = rel_path.split("/")[0]
    if top in EXCLUDED_TOP or top.endswith("_DoNotShip"):
        return False
    return not rel_path.startswith(EXCLUDED_PREFIXES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_player(executable: str) -> dict:
    """@brief Verify the player folder containing @p executable against its manifest.

    @return Summary: status "verified" (with source_commit, source_dirty, built_at, unity_version, files,
            manifest_sha256) or "no_manifest".
    @throws PlayerIntegrityError listing every changed, missing and extra file.
    """
    build_dir = Path(executable).resolve().parent
    manifest_path = build_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"status": "no_manifest", "build_dir": str(build_dir)}

    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    actual = {p.relative_to(build_dir).as_posix() for p in build_dir.rglob("*") if p.is_file()}
    actual = {p for p in actual if included(p)}

    problems = []
    for rel, digest in sorted(expected.items()):
        path = build_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"changed: {rel}")
    problems += [f"extra (not in manifest): {rel}" for rel in sorted(actual - set(expected))]
    if problems:
        shown = "\n  - ".join(problems[:20]) + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else "")
        raise PlayerIntegrityError(
            f"{build_dir} does not match its {MANIFEST_NAME} (commit {manifest.get('source_commit')}):\n  - {shown}")

    return {
        "status": "verified",
        "build_dir": str(build_dir),
        "source_commit": manifest.get("source_commit"),
        "source_dirty": manifest.get("source_dirty"),
        "source_diff_sha256": manifest.get("source_diff_sha256"),
        "built_at": manifest.get("built_at"),
        "unity_version": manifest.get("unity_version"),
        "files": len(expected),
        "manifest_sha256": sha256_file(manifest_path),
    }


def check_player(executable: Optional[str], allow_unverified: bool = False, record_dir: Optional[Path] = None) -> dict:
    """@brief Verify before launching; print a one-line result and optionally save it as player_manifest.json.

    @param allow_unverified  Launch anyway when the files do not match (prints the mismatch instead of raising).
    @param record_dir        Run / output folder to save the summary in, so results name the exact player.
    """
    if executable is None:        # Unity Editor session: nothing to verify
        return {"status": "editor"}
    try:
        summary = verify_player(executable)
    except PlayerIntegrityError as e:
        if not allow_unverified:
            raise
        print(f"[player] WARNING (--allow-unverified-player): {e}")
        summary = {"status": "mismatch_allowed", "build_dir": str(Path(executable).resolve().parent), "error": str(e)}

    if summary["status"] == "verified":
        dirty = " + uncommitted changes" if summary["source_dirty"] else ""
        print(f"[player] verified {summary['files']} files against {MANIFEST_NAME}: commit "
              f"{summary['source_commit']}{dirty}, built {summary['built_at']}")
    elif summary["status"] == "no_manifest":
        print(f"[player] WARNING: {summary['build_dir']} has no {MANIFEST_NAME} (built before 2026-09-30); "
              "rebuild it to make the player verifiable.")
    if record_dir is not None:
        Path(record_dir, "player_manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python env/player_manifest.py <path to capstone.x86_64>")
    try:
        result = verify_player(sys.argv[1])
    except PlayerIntegrityError as err:
        sys.exit(str(err))
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["status"] == "verified" else 2)
