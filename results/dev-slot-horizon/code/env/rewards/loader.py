"""
@file loader.py
@brief Load a reward function from a JSON spec, without rebuilding Unity.
"""

import importlib
import importlib.util
import inspect
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from rewards.base import RewardFunction

## @brief The env/ directory; relative reward paths and package imports resolve against it.
ENV_ROOT = Path(__file__).resolve().parent.parent


class FunctionReward(RewardFunction):
    """@brief Adapts a plain function @c fn(prev, curr, ctx, **params) to @ref RewardFunction."""

    def __init__(self, fn: Callable, **params):
        super().__init__(**params)
        self.fn = fn

    def compute(self, prev, curr, ctx):
        out = self.fn(prev, curr, ctx, **self.params)
        return dict(out) if isinstance(out, Mapping) else {"reward": float(out)}


@dataclass
class LoadedReward:
    """@brief A resolved reward spec: builds per-env instances and archives its sources."""

    name: str
    spec: Dict[str, Any]
    ## @brief The class or function named by the spec's @c entry.
    target: Any
    ## @brief File the entry was loaded from.
    source_path: Optional[Path]
    spec_path: Optional[Path] = None

    def build(self) -> RewardFunction:
        """@brief Create a fresh reward instance with the spec's params (one per env)."""
        params = dict(self.spec.get("params", {}))
        if inspect.isclass(self.target):
            if not issubclass(self.target, RewardFunction):
                raise TypeError(f"{self.target.__name__} must subclass rewards.RewardFunction")
            return self.target(**params)
        if callable(self.target):
            return FunctionReward(self.target, **params)
        raise TypeError(f"Reward entry {self.spec['entry']!r} is not a class or function")

    def archive(self, run_dir: Union[str, Path]) -> Path:
        """@brief Copy the spec and reward source into @p run_dir/reward so the run records
        exactly which reward it trained with."""
        out = Path(run_dir) / "reward"
        out.mkdir(parents=True, exist_ok=True)
        (out / "spec.json").write_text(json.dumps(self.spec, indent=2))
        if self.source_path is not None:
            shutil.copy2(self.source_path, out / self.source_path.name)
        return out


def load_reward(spec: Union[str, Path, Mapping]) -> LoadedReward:
    """@brief Resolve a reward spec (JSON file path or dict) to a @ref LoadedReward.

    @details Builds one instance immediately so bad params fail here, not mid-training.
    """
    spec_path = None
    if isinstance(spec, (str, Path)):
        spec_path = Path(spec).resolve()
        spec = json.loads(spec_path.read_text())
    spec = dict(spec)
    if "entry" not in spec:
        raise KeyError("Reward spec needs an 'entry', e.g. 'rewards/functions/flow_time.py:FlowTimeReward'")

    target_ref, sep, attr = spec["entry"].rpartition(":")
    if not sep or not target_ref or not attr:
        raise ValueError(f"Reward entry {spec['entry']!r} must be 'file.py:Name' or 'module:Name'")

    _ensure_env_root_importable()
    if target_ref.endswith(".py"):
        source_path = _resolve_file(target_ref, spec_path)
        module = _import_file(source_path)
    else:
        module = importlib.import_module(target_ref)
        source_path = Path(module.__file__) if getattr(module, "__file__", None) else None

    if not hasattr(module, attr):
        raise AttributeError(f"{source_path or target_ref} has no attribute {attr!r}")

    loaded = LoadedReward(name=spec.get("name", attr), spec=spec, target=getattr(module, attr),
                          source_path=source_path, spec_path=spec_path)
    loaded.build()
    return loaded


def _resolve_file(ref: str, spec_path: Optional[Path]) -> Path:
    path = Path(ref)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = ([spec_path.parent / path] if spec_path else []) + [ENV_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Reward file {ref!r} not found (looked in: {', '.join(str(c) for c in candidates)})")


def _ensure_env_root_importable() -> None:
    if str(ENV_ROOT) not in sys.path:
        sys.path.insert(0, str(ENV_ROOT))


def _import_file(path: Path):
    module_name = f"_reward_{path.stem}_{abs(hash(str(path)))}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module
