"""
@file player_manifest.py
@brief Check a Unity player build against its BUILD_MANIFEST.json (thesis section 7.2).

The build script (Capstone/Assets/Editor/BuildManifest.cs) records the git commit a player was built
from, whether the source had uncommitted changes, and the SHA-256 of every file the player loads.
verify_player recomputes those hashes: a changed, missing, or extra file in the player means the
binary is not the one the manifest describes, and training / evaluation refuse to launch it.

The player's files are an allowlist (included): the executable, the shared libraries beside it, and
capstone_Data/ except ML-Agents/, where the player writes timers while it runs. The libraries count because
the executable's RUNPATH is $ORIGIN, so the loader looks in the player folder before the system folders and
a library dropped there would be loaded. Nothing else in the folder is part of the player (runner scripts,
slurm/, logs/, Results/, BatchConfigs/, Burst's *_DoNotShip debug output), and editing it does not
invalidate the build.

Manifests written before 2026-10-02 (schema 1) hashed every file in the folder except Results/,
BatchConfigs/, __pycache__/, *_DoNotShip/ and ML-Agents/, so they also list helper scripts and logs.
Only their allowlisted entries are compared. The old rule covered everything the allowlist covers, so
those entries are what a schema-2 manifest of the same build would list, and existing builds verify
without a rebuild.

A player without a manifest (built before 2026-09-30) is reported, not refused.

Command line:  python env/player_manifest.py linux_server/capstone.x86_64
"""

import hashlib
import json
import re
import sys
from itertools import chain
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "BUILD_MANIFEST.json"
## @brief Manifest format written by BuildManifest.cs; schema 1 (before 2026-10-02) hashed the whole folder.
SCHEMA = 2
## @brief Keep in step with BuildManifest.cs: the player's own files (see included).
EXECUTABLE = "capstone.x86_64"
DATA_DIR = "capstone_Data/"
RUNTIME_DIR = "capstone_Data/ML-Agents/"                    # written by the player while it runs
SHARED_LIBRARY = re.compile(r"^[^/]+\.so(\.[0-9]+)*$")      # top level: UnityPlayer.so, libdecor-0.so.0, ...


class PlayerIntegrityError(RuntimeError):
    """@brief The player's files do not match its manifest."""


def included(rel_path: str) -> bool:
    """@brief Whether @p rel_path (relative to the player folder, '/'-separated) is one of the player's files."""
    if rel_path.startswith(DATA_DIR):
        return not rel_path.startswith(RUNTIME_DIR)
    return rel_path == EXECUTABLE or SHARED_LIBRARY.match(rel_path) is not None


def player_files(build_dir: Path) -> set:
    """@brief The player's files present in @p build_dir. Only the top level and capstone_Data/ are walked,
    so the result folders running players write into are never scanned."""
    paths = chain(build_dir.iterdir(), (build_dir / DATA_DIR).rglob("*"))
    return {rel for rel in (p.relative_to(build_dir).as_posix() for p in paths if p.is_file()) if included(rel)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_player(executable: str) -> dict:
    """@brief Verify the player folder containing @p executable against its manifest.

    @return Summary: status "verified" (with source_commit, source_dirty, built_at, unity_version, files,
            manifest_schema, skipped_entries, manifest_sha256) or "no_manifest".
    @throws PlayerIntegrityError listing every changed, missing and extra file, or if @p executable is not
            the player's executable.
    """
    build_dir = Path(executable).resolve().parent
    manifest_path = build_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"status": "no_manifest", "build_dir": str(build_dir)}

    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema", 1)
    expected = manifest["files"]
    if schema < 2:              # the whole folder was hashed: compare only the player's own entries
        expected = {rel: digest for rel, digest in expected.items() if included(rel)}
    actual = player_files(build_dir)

    problems = []
    exe = Path(executable).resolve().name
    if not included(exe):       # another program in the folder, which the allowlist does not cover
        problems.append(f"not the player executable: {exe}")
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
        "manifest_schema": schema,
        "skipped_entries": len(manifest["files"]) - len(expected),     # schema 1: helper files, not checked
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
        skipped = (f" (old-format manifest: {summary['skipped_entries']} entries for non-player files skipped)"
                   if summary["skipped_entries"] else "")
        print(f"[player] verified {summary['files']} files against {MANIFEST_NAME}: commit "
              f"{summary['source_commit']}{dirty}, built {summary['built_at']}{skipped}")
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
