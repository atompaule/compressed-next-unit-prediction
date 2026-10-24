"""Register ablation without QR projection and its checkpoint loader."""

from __future__ import annotations
from pathlib import Path
import torch
from .config import ModelConfig
from .dataset import Vocabulary, load_data_config
from .model import Model

NONORTHOGONAL_VARIANT = "dense_nonorthogonal"
NONORTHOGONAL_VERSION = 1


class NonOrthogonalModel(Model):
    """Preserve the original model and initialization; omit only projection.

    Readout still uses Q_r.T, not an inverse. Feedback uses Q_r and the existing
    normalization. Both operations train the same unconstrained dense matrix.
    """

    def __init__(self, condition: str, vocabulary: Vocabulary, config: ModelConfig):
        if condition != "continuous":
            raise ValueError(
                "The non-orthogonal experiment supports continuous feedback only"
            )
        super().__init__(condition, vocabulary, config)

    @torch.no_grad()
    def project_role_maps_(self):
        pass


@torch.no_grad()
def assert_finite_role_maps(model):
    if not isinstance(model, NonOrthogonalModel) or model.role_maps is None:
        raise AssertionError("Expected a non-orthogonal dense role-map model")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError("Non-finite model parameters")


def nonorthogonal_metadata():
    return dict(
        name=NONORTHOGONAL_VARIANT,
        version=NONORTHOGONAL_VERSION,
        transform="Independent freely learned dense Q_r matrices",
        initialization="Exactly the baseline seed, including its QR-initialized orthogonal matrices",
        orthogonality="Initial only; no QR projection or constraint during optimization",
        readout="E.T @ Q_r.T @ h; transpose retained, no matrix inverse",
        feedback="normalize(sum_r Q_r @ E[value_r]) to detached mean value-embedding norm",
        parameter_count="Identical to the independent-Q orthogonal baseline",
    )


@torch.no_grad()
def role_map_diagnostics(model):
    maps = model.role_maps.detach().float().cpu()
    error = maps.transpose(-1, -2) @ maps - torch.eye(model.d_model)
    singular = torch.linalg.svdvals(maps)
    return dict(
        maximum_orthogonality_error=float(error.abs().max()),
        per_role_maximum_orthogonality_error=error.abs().flatten(1).amax(1).tolist(),
        per_role_frobenius_orthogonality_error=error.flatten(1).norm(dim=1).tolist(),
        per_role_minimum_singular_value=singular.amin(1).tolist(),
        per_role_maximum_singular_value=singular.amax(1).tolist(),
    )


def load_nonorthogonal_checkpoint(path: str | Path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_variant") != nonorthogonal_metadata():
        raise ValueError("Not a supported non-orthogonal register checkpoint")
    config = ModelConfig(**checkpoint["model_config"])
    vocabulary = Vocabulary(load_data_config(checkpoint["data_manifest"]))
    if list(vocabulary.tokens) != checkpoint["vocabulary"]:
        raise ValueError("Checkpoint vocabulary mismatch")
    model = NonOrthogonalModel(checkpoint["condition"], vocabulary, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert_finite_role_maps(model)
    return model.eval()
