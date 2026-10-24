"""Resolve saved adapters and their recorded training settings without model loading."""
import json
from pathlib import Path


def is_adapter(path):
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))


def resolve_checkpoint(environ):
    explicit = environ.get("CKPT")
    if explicit:
        if explicit.strip().lower() in ("base", "none"):
            return None, None
        path = Path(explicit).expanduser()
        if not is_adapter(path):
            raise ValueError(f"CKPT is not a saved adapter: {path}")
        return path.name, str(path.resolve())
    root = environ.get("CKPT_DIR")
    if root:
        candidates = [p for p in Path(root).expanduser().glob("step_*")
                      if p.name[5:].isdigit() and is_adapter(p)]
        if not candidates:
            raise ValueError(f"No saved adapters in CKPT_DIR: {root}")
        path = max(candidates, key=lambda p: int(p.name[5:]))
        return path.name, str(path.resolve())
    return None, None


def training_preset(checkpoint):
    if checkpoint is None:
        return None
    path = Path(checkpoint).parent / "run_config.json"
    if path.exists():
        return json.loads(path.read_text())["preset"]
    return None


def infer_training_approach(checkpoint):
    if checkpoint is None:
        return "base"
    preset = training_preset(checkpoint)
    if preset is not None:
        return preset["rl_config"]["approach"]
    # Older adapters have no run configuration. This fallback is optional.
    name = Path(checkpoint).parent.name
    for approach in ("gaussian_noise", "token_sampling", "dirichlet_sampling", "discrete"):
        if approach in name:
            return approach
    raise ValueError(f"No recorded training approach for {checkpoint}")
