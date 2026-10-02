"""
@file test_config_safety.py
@brief Thesis chapter 7 fixes: config schema/bounds, rejection instead of fallback, loopback gRPC,
       config hashes in the episode log (7.1), weights-only checkpoint loading and the player manifest (7.2).
"""

import ast
import copy
import json
import re
import socket
from pathlib import Path

import pytest

from channels.channels import EpisodeConfigChannel
from channels.config_schema import (
    ConfigValidationError, max_agv_count, validate_config, validate_scenario,
)
from env_wrappers.loopback import LoopbackRpcCommunicator
from env_wrappers.unity_env import UnitySchedulingEnv
from scenarios import compound_generator, randomized_generator

SCENARIO = {
    "name": "tiny",
    "seed": 3,
    "agvCount": 7,
    "machineTypeLayout": ["Mill"] * 3 + ["Lathe"] * 3 + ["Weld"] * 3 + ["Inspect"] * 3 + ["Assemble"] * 3,
    "reservationProtocol": "releasePrevious",
    "jobs": [
        {"id": 0, "arrivalTime": 0.0, "operations": [
            {"machineType": "Mill", "machineIndex": "any", "duration": 40.0},
            {"machineType": "Weld", "machineIndex": [0, 2], "duration": [50.0, 70.0]},
            {"machineType": "Inspect", "machineIndex": 1, "duration": 20.0},
        ]},
    ],
    "stochastic": {"machineFailuresEnabled": True, "weibullK": 1.5, "weibullLambda": 900.0},
    "_comment": "underscore keys are comments",
}

CONFIG = {
    "name": "30j_15m", "seed": 42, "jobCount": 30, "machinesPerType": 3,
    "machineTypes": ["Mill", "Lathe", "Weld", "Inspect", "Assemble"],
    "minProcTime": 15.0, "maxProcTime": 60.0, "minOpsPerJob": 4, "maxOpsPerJob": 6, "agvCount": 10,
}


def with_changes(base, **changes):
    out = copy.deepcopy(base)
    out.update(changes)
    return out


def errors_of(fn, item):
    with pytest.raises(ConfigValidationError) as exc:
        fn(item)
    return exc.value.errors


class TestSchema:
    def test_valid_scenario_and_config_pass(self):
        validate_scenario(SCENARIO)
        validate_config(CONFIG)

    def test_channel_roundtrip_config_passes(self):
        """@brief The manual round-trip script's CONFIG matches the schema (it once failed it with 16 errors).
        Parsed, not imported: that script launches a player, and pytest must never run it."""
        source = (Path(__file__).resolve().parents[1] / "scripts" / "channel_roundtrip.py").read_text()
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets)]
        assert len(nodes) == 1, "channel_roundtrip.py must define CONFIG once, as a top-level literal"
        validate_config(ast.literal_eval(nodes[0].value))

    @pytest.mark.parametrize("make", [
        lambda: compound_generator(2700.0, random_warmup=True),
        lambda: compound_generator(variant="compound_v2"),
        lambda: randomized_generator(2700.0, random_warmup=True),
    ])
    def test_training_generators_pass(self, make):
        generator = make()
        for seed in range(10_000, 10_010):
            validate_scenario(generator(seed))

    def test_fleet_bound_depends_on_protocol_and_floor(self):
        assert max_agv_count(15, "releasePrevious") == 15
        assert max_agv_count(15, "holdPrevious") == 8
        assert max_agv_count(105, "releasePrevious") == 105
        assert max_agv_count(3, "holdPrevious") == 1
        errs = errors_of(validate_scenario, with_changes(SCENARIO, agvCount=9, reservationProtocol="holdPrevious"))
        assert any("gridlock-safe maximum 8" in e for e in errs)
        errs = errors_of(validate_scenario, with_changes(SCENARIO, agvCount=16))
        assert any("gridlock-safe maximum 15" in e for e in errs)
        errs = errors_of(validate_config, with_changes(CONFIG, agvCount=0))
        assert any("agvCount" in e for e in errs)

    def test_unknown_keys_are_errors_not_ignored(self):
        assert any("agvcount" in e for e in errors_of(validate_scenario, with_changes(SCENARIO, agvcount=5)))
        s = with_changes(SCENARIO, stochastic={"machineFailuresEnabled": True, "weibulLambda": 450.0})
        assert any("weibulLambda" in e for e in errors_of(validate_scenario, s))
        assert any("maxArrivalTime" in e for e in errors_of(validate_config, with_changes(CONFIG, maxArrivalTime=0.0)))

    def test_physical_bounds(self):
        s = copy.deepcopy(SCENARIO)
        s["jobs"][0]["operations"][0]["duration"] = -5.0
        s["jobs"][0]["operations"][1]["machineIndex"] = [0, 3]
        s["jobs"][0]["operations"][2]["machineType"] = "Drill"
        s["stochastic"]["weibullLambda"] = 0.0
        s["dispatchingRule"] = "SPT_FOO"
        errs = errors_of(validate_scenario, s)
        assert len(errs) == 5, errs      # every problem reported at once
        errs = errors_of(validate_config, with_changes(CONFIG, minProcTime=60.0, maxProcTime=15.0,
                                                       machineFlexibilityProbability=1.5))
        assert len(errs) == 2, errs

    def test_agv_failure_parameters(self):
        good = {"agvFailuresEnabled": True, "agvWeibullK": 1.5, "agvWeibullLambda": 8400.0,
                "agvRepairLogMu": 4.6, "agvRepairLogSigma": 0.5}
        validate_scenario(with_changes(SCENARIO, stochastic=good))
        validate_config(with_changes(CONFIG, stochastic=good))
        bad = dict(good, agvWeibullK=0.0, agvWeibullLambda=-1.0, agvRepairLogSigma=-0.1)
        errs = errors_of(validate_scenario, with_changes(SCENARIO, stochastic=bad))
        assert len(errs) == 3, errs

    def test_nan_and_bool_are_not_numbers(self):
        errors_of(validate_config, with_changes(CONFIG, minProcTime=float("nan")))
        errors_of(validate_scenario, with_changes(SCENARIO, agvCount=True))


class TestChannelRejects:
    def test_invalid_scenario_is_not_queued(self):
        channel = EpisodeConfigChannel()
        with pytest.raises(ConfigValidationError):
            channel.queue_scenarios([SCENARIO, with_changes(SCENARIO, agvCount=99)])
        assert channel.message_queue == []      # all or nothing: nothing reached Unity

    def test_scenario_file_is_validated(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(with_changes(SCENARIO, layout="Z")))
        channel = EpisodeConfigChannel()
        with pytest.raises(ConfigValidationError):
            channel.queue_scenarios([str(path)])
        with pytest.raises(FileNotFoundError):
            channel.queue_scenarios([str(tmp_path / "missing.json")])
        assert channel.message_queue == []

    def test_valid_items_are_sent(self):
        channel = EpisodeConfigChannel()
        channel.queue_scenarios([SCENARIO])
        channel.send_config(CONFIG)
        assert len(channel.message_queue) == 2

    def test_invalid_config_is_not_sent(self):
        channel = EpisodeConfigChannel()
        with pytest.raises(ConfigValidationError):
            channel.send_config(with_changes(CONFIG, machineTypes=["Mill", "Lathes"]))
        assert channel.message_queue == []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _non_loopback_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))     # TEST-NET address: no packet is sent, only picks the outbound interface
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


class TestLoopback:
    def test_server_listens_on_loopback_only(self):
        port = _free_port()
        comm = LoopbackRpcCommunicator(worker_id=0, base_port=port, timeout_wait=1)
        try:
            assert comm.bound_addresses and all(
                a.startswith(("127.0.0.1:", "[::1]:")) for a in comm.bound_addresses)
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
            ip = _non_loopback_ip()
            if ip is None:
                pytest.skip("no non-loopback interface to test against")
            with pytest.raises(OSError):
                socket.create_connection((ip, port), timeout=2).close()
        finally:
            comm.close()


class TestConfigHashInEpisodeLog:
    def test_summary_carries_hashes_from_telemetry(self):
        env = UnitySchedulingEnv.__new__(UnitySchedulingEnv)
        env._episode_return, env._episode_length, env._episode_terms = 1.0, 3, {}
        telemetry = {"events": [], "result": {"configHash": "abc", "instanceHash": "def", "appliedConfig": "x=1"}}
        summary = env._episode_summary(None, interrupted=False, telemetry=telemetry)
        assert (summary["config_hash"], summary["instance_hash"], summary["applied_config"]) == ("abc", "def", "x=1")
        summary = env._episode_summary(None, interrupted=False, telemetry=None)   # player built before the fix
        assert summary["config_hash"] is None

    def test_train_logs_hash_and_applied_config(self, tmp_path):
        import csv
        import io
        from collections import deque
        from unittest.mock import MagicMock
        from train import EPISODE_CSV_FIELDS, log_episodes

        out = io.StringIO()
        episode_csv = csv.DictWriter(out, fieldnames=EPISODE_CSV_FIELDS, extrasaction="ignore")
        applied = io.StringIO()
        episode = {"return": 1.0, "length": 2, "config_hash": "abc", "instance_hash": "def", "applied_config": "x=1"}
        infos = [{"episode": episode}, {"episode": dict(episode, applied_config=None)}]
        assert log_episodes(MagicMock(), infos, 10, deque(), episode_csv, applied) == 2
        assert out.getvalue().count("abc,def") == 2
        lines = applied.getvalue().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["config"] == "x=1"


class _Payload:
    """Unpickling this runs code: the attack weights_only blocks."""

    def __reduce__(self):
        import os
        return (os.system, ("touch PWNED_MARKER",))


class TestCheckpointLoading:
    def test_round_trip_through_save_checkpoint(self, tmp_path):
        import torch
        from config import PPOConfig
        from train import load_checkpoint, save_checkpoint

        net = torch.nn.Linear(3, 2)
        opt = torch.optim.Adam(net.parameters())
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, net, opt, 123, PPOConfig(), "flow_time", (15, 64))
        ckpt = load_checkpoint(path, "cpu")
        assert ckpt["global_step"] == 123
        assert torch.equal(ckpt["model_state_dict"]["weight"], net.weight)

    def test_pickled_code_is_refused_not_run(self, tmp_path, monkeypatch):
        import pickle
        import torch
        from train import load_checkpoint

        monkeypatch.chdir(tmp_path)
        path = tmp_path / "evil.pt"
        torch.save({"model_state_dict": {}, "extra": _Payload()}, path)
        with pytest.raises(pickle.UnpicklingError):
            load_checkpoint(path, "cpu")
        assert not (tmp_path / "PWNED_MARKER").exists()


BUILD_MANIFEST_CS = Path(__file__).resolve().parents[2] / "Capstone" / "Assets" / "Editor" / "BuildManifest.cs"


class TestPlayerManifest:
    """env/player_manifest.py against a fake build folder with a manifest in BuildManifest.cs's format."""

    PLAYER = {"capstone.x86_64": b"\x7fELF launcher", "UnityPlayer.so": b"engine", "libdecor-0.so.0": b"wayland",
              "libdecor-cairo.so": b"wayland", "capstone_Data/Managed/Simulation.dll": b"sim code"}
    # Written by the player while it runs, or kept beside it but never loaded.
    RUNTIME = {"capstone_Data/ML-Agents/Timers/SimulationGrid_timers.json": b"{}",
               "BatchConfigs/scenario.json": b"{}", "Results/results.csv": b"a,b",
               "Capstone_BurstDebugInformation_DoNotShip/Data/Plugins/lib_burst_generated.txt": b"symbols"}
    # Helper scripts and logs, as in linux_server/ on 2026-10-02.
    HELPERS = {"slurm/oracle.sbatch": b"#!/bin/bash", "logs/smoke_heads_run1.out": b"ok",
               "run_batch.sh": b"#!/bin/bash", "run_experiment_queue.py": b"print()", "repro_compare.py": b"print()",
               "run_agvr_0929.log": b"log"}

    @classmethod
    def make_build(cls, root, schema=2):
        """Schema 1 lists the helpers too, as BuildManifest.cs did before 2026-10-02."""
        from player_manifest import sha256_file
        for rel, data in {**cls.PLAYER, **cls.RUNTIME, **cls.HELPERS}.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
        listed = {**cls.PLAYER, **cls.HELPERS} if schema == 1 else cls.PLAYER
        manifest = {"schema": schema, "executable": "capstone.x86_64", "source_commit": "abc123",
                    "source_dirty": False, "built_at": "2026-10-02T00:00:00Z",
                    "files": {rel: sha256_file(root / rel) for rel in sorted(listed)}}
        (root / "BUILD_MANIFEST.json").write_text(json.dumps(manifest))
        return root / "capstone.x86_64"

    def test_allowlist_is_the_player_files(self):
        from player_manifest import included
        assert {rel for rel in {**self.PLAYER, **self.RUNTIME, **self.HELPERS} if included(rel)} == set(self.PLAYER)
        # The executable's RUNPATH is $ORIGIN, so a library beside it is loaded before the system's.
        assert included("libm.so.6") and included("libstdc++.so.6.0.30")
        assert not included("UnityPlayer.so.orig") and not included("slurm/libx.so")

    def test_matching_player_is_verified_and_other_files_ignored(self, tmp_path):
        from player_manifest import check_player
        exe = self.make_build(tmp_path)
        (tmp_path / "Results" / "new_run.csv").write_text("x")           # written by the player at runtime
        (tmp_path / "capstone_Data/ML-Agents/Timers/SimulationGrid_timers.json").write_text('{"t": 1}')
        (tmp_path / "slurm/oracle.sbatch").write_text("#!/bin/bash\n# edited after the build")
        (tmp_path / "slurm/setup_rit_python.sh").write_text("#!/bin/bash")
        (tmp_path / "run_agvr_0929.log").unlink()
        summary = check_player(str(exe), record_dir=tmp_path)
        assert summary["status"] == "verified" and summary["files"] == len(self.PLAYER)
        assert summary["source_commit"] == "abc123" and summary["skipped_entries"] == 0
        assert json.loads((tmp_path / "player_manifest.json").read_text())["status"] == "verified"

    def test_old_manifest_checks_only_player_entries(self, tmp_path):
        from player_manifest import PlayerIntegrityError, check_player
        exe = self.make_build(tmp_path, schema=1)
        (tmp_path / "slurm/oracle.sbatch").write_text("#!/bin/bash\n# edited after the build")
        (tmp_path / "run_agvr_0929.log").unlink()
        summary = check_player(str(exe))
        assert summary["status"] == "verified" and summary["manifest_schema"] == 1
        assert summary["files"] == len(self.PLAYER) and summary["skipped_entries"] == len(self.HELPERS)
        (tmp_path / "UnityPlayer.so").write_bytes(b"patched")
        with pytest.raises(PlayerIntegrityError, match="changed: UnityPlayer.so"):
            check_player(str(exe))

    def test_changed_missing_or_extra_player_file_is_refused(self, tmp_path):
        from player_manifest import PlayerIntegrityError, check_player
        exe = self.make_build(tmp_path)
        (tmp_path / "capstone_Data/Managed/Simulation.dll").write_bytes(b"patched")
        (tmp_path / "libdecor-cairo.so").unlink()
        (tmp_path / "libm.so.6").write_bytes(b"dropped in")
        (tmp_path / "capstone_Data/Managed/Simulation.dll.config").write_text("<configuration/>")   # Mono reads it
        with pytest.raises(PlayerIntegrityError) as exc:
            check_player(str(exe))
        assert "changed: capstone_Data/Managed/Simulation.dll" in str(exc.value)
        assert "missing: libdecor-cairo.so" in str(exc.value)
        assert "extra (not in manifest): libm.so.6" in str(exc.value)
        assert "extra (not in manifest): capstone_Data/Managed/Simulation.dll.config" in str(exc.value)
        assert check_player(str(exe), allow_unverified=True)["status"] == "mismatch_allowed"

    def test_other_program_in_the_folder_is_refused(self, tmp_path):
        from player_manifest import PlayerIntegrityError, check_player
        exe = self.make_build(tmp_path)
        (tmp_path / "other.x86_64").write_bytes(b"\x7fELF not built by Unity")
        assert check_player(str(exe))["status"] == "verified"           # not part of the player
        with pytest.raises(PlayerIntegrityError, match="not the player executable: other.x86_64"):
            check_player(str(tmp_path / "other.x86_64"))

    @pytest.mark.skipif(not BUILD_MANIFEST_CS.exists(), reason="C# sources not present (e.g. on the cluster)")
    def test_allowlist_matches_build_manifest_cs(self):
        import player_manifest as pm
        src = BUILD_MANIFEST_CS.read_text()
        assert re.search(r'Executable = "([^"]+)"', src).group(1) == pm.EXECUTABLE
        assert re.search(r'DataDir = "([^"]+)"', src).group(1) == pm.DATA_DIR
        assert re.search(r'RuntimeDir = "([^"]+)"', src).group(1) == pm.RUNTIME_DIR
        assert re.search(r'SharedLibrary = new Regex\(@"([^"]+)"\)', src).group(1) == pm.SHARED_LIBRARY.pattern
        assert int(re.search(r"Schema = (\d+);", src).group(1)) == pm.SCHEMA

    def test_missing_manifest_is_reported_not_refused(self, tmp_path):
        from player_manifest import check_player
        (tmp_path / "capstone.x86_64").write_bytes(b"old build")
        assert check_player(str(tmp_path / "capstone.x86_64"))["status"] == "no_manifest"
