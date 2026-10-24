"""Shared data, checkpoint and result locations.

Defaults live under data/<experiment>, checkpoints/<experiment>, and
results/<experiment>. Environment overrides select alternative storage roots.
Importing this module creates no files or directories.
"""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "src"
DATA_ROOT = Path(os.environ.get("THESIS_DATA_ROOT", REPOSITORY / "data")).expanduser()
ARTIFACT_ROOT = Path(os.environ.get(
    "THESIS_EXPERIMENT_ROOT", REPOSITORY / "checkpoints"
)).expanduser()
RESULT_ROOT = Path(os.environ.get("THESIS_RESULTS_ROOT", REPOSITORY / "results")).expanduser()


def data_dir(experiment, *parts):
    """Locate a dataset; callers may still explicitly select external data."""
    return DATA_ROOT / experiment / Path(*parts)


def checkpoint_dir(experiment, *parts):
    return ARTIFACT_ROOT / experiment / Path(*parts)


def result_dir(experiment, *parts):
    return RESULT_ROOT / experiment / Path(*parts)


def artifact_dir(source_file):
    """Locate the checkpoint directory belonging to a maintained source module."""
    relative = Path(source_file).resolve().relative_to(SOURCE_ROOT)
    return ARTIFACT_ROOT / relative.parent

