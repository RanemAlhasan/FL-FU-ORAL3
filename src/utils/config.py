"""Config loading + CLI --set override support (same design as the
oral-cancer framework's src/utils/config.py, reused here for consistency)."""
from __future__ import annotations

import copy
import datetime
import os
from typing import Any, Dict, List, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving \`defaults: <relative_or_absolute_path>\`
    inheritance recursively. A child config's own keys always override
    whatever the same key resolves to in its defaults chain. This restores
    the oral-cancer framework's config-inheritance behavior (e.g.
    configs/fl_fedavg.yaml says \`defaults: base.yaml\` to inherit
    dataset_path, logs_root, checkpoints_root, etc. from base.yaml, then
    overrides algorithm/global_epochs/local_epochs on top). The CIFAR-10
    reproduction's config does not use \`defaults:\` at all, so this is
    purely additive and has no effect on that project's single flat
    config file."""
    path = os.path.abspath(path)
    with open(path, "r") as f:
        own_config = yaml.safe_load(f) or {}

    defaults_ref = own_config.pop("defaults", None)
    if defaults_ref is None:
        return own_config

    defaults_path = defaults_ref if os.path.isabs(defaults_ref)         else os.path.join(os.path.dirname(path), defaults_ref)
    parent_config = load_config(defaults_path)

    merged = dict(parent_config)
    merged.update(own_config)
    return merged


def save_config_snapshot(config: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def make_run_id(prefix: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}"


def resolve_run_dirs(run_id: str, logs_root: str, checkpoints_root: str, outputs_root: str) -> Dict[str, str]:
    dirs = {
        "log_dir": os.path.join(logs_root, run_id),
        "tb_dir": os.path.join(logs_root, run_id, "tb"),
        "checkpoint_dir": os.path.join(checkpoints_root, run_id),
        "output_dir": os.path.join(outputs_root, run_id),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _coerce_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def parse_set_overrides(set_args: Optional[List[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not set_args:
        return overrides
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"Invalid --set argument '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = _coerce_value(raw_value.strip())
    return overrides


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(config)
    merged.update(overrides)
    return merged


def load_source_run_config(source_run_id: str, logs_root: str = "logs") -> Dict[str, Any]:
    """Read-only load of a previous run's config snapshot. Used by FU/retrain
    scripts to inspect (but never modify) the FL run they are forking from."""
    import os
    import yaml
    snapshot_path = os.path.join(logs_root, source_run_id, "config.snapshot.yaml")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"Could not find config snapshot for source_run '{source_run_id}' "
            f"at '{snapshot_path}'. Has that run completed Phase 1 (FL training)?"
        )
    with open(snapshot_path, "r") as f:
        return yaml.safe_load(f)
