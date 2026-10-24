"""Register decoder and feedback controls. Parameter names preserve checkpoints."""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from .config import ModelConfig
from .cot2 import target_distribution

COT2 = "continuous_cot2"
CONDITIONS = (
    "continuous",
    "continuous_unbound",
    "cache_only_full_head",
    "discrete_conventional",
    COT2,
)


class Model(nn.Module):
    """Small causal decoder with tied node logits and orthogonal role maps."""

    def __init__(
        self,
        condition: str,
        vocabulary: Any,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        config.validate()
        self.model_config = config
        self.num_registers = len(vocabulary.roles)
        self.condition = condition
        self.vocabulary = vocabulary
        self.d_model = config.d_model
        transformer_config = GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=config.max_positions,
            n_ctx=config.max_positions,
            n_embd=config.d_model,
            n_layer=config.layers,
            n_head=config.heads,
            n_inner=4 * config.d_model,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            activation_function="gelu_new",
            initializer_range=0.02,
            layer_norm_epsilon=1e-5,
            bos_token_id=vocabulary.id("<BOS>"),
            pad_token_id=vocabulary.id("<PAD>"),
            use_cache=True,
        )
        self.transformer = GPT2Model(transformer_config)
        node_mask = torch.zeros(len(vocabulary), dtype=torch.bool)
        node_mask[list(vocabulary.node_ids)] = True
        self.register_buffer("node_mask", node_mask, persistent=False)
        self.register_buffer(
            "node_ids",
            torch.tensor(vocabulary.node_ids, dtype=torch.long),
            persistent=False,
        )

        if condition in ("continuous", "continuous_unbound", "cache_only_full_head"):
            maps = []
            for _ in range(self.num_registers):
                matrix = torch.randn(config.d_model, config.d_model)
                orthogonal, upper = torch.linalg.qr(matrix)
                signs = torch.sign(torch.diagonal(upper))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                maps.append(orthogonal * signs.unsqueeze(0))
            self.role_maps = nn.Parameter(torch.stack(maps, dim=0))
        else:
            self.register_parameter("role_maps", None)

    @property
    def embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def full_hidden(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state

    def cached_hidden(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Process one new chunk and return its hidden states plus mutated cache."""

        output = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if output.past_key_values is None:
            raise AssertionError("Cached decoding returned no cache")
        return output.last_hidden_state, output.past_key_values

    def language_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,vd->...v", hidden, self.embeddings.weight)

    def node_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.language_logits(hidden)
        return logits.masked_fill(
            ~self.node_mask,
            torch.finfo(logits.dtype).min,
        )

    def role_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.role_maps is None:
            raise RuntimeError(f"{self.condition} has no role maps")
        unbound = torch.einsum("...d,rde->...re", hidden, self.role_maps)
        logits = torch.einsum("...re,ve->...rv", unbound, self.embeddings.weight)
        return logits.masked_fill(
            ~self.node_mask.view(*(1,) * (logits.ndim - 1), -1),
            torch.finfo(logits.dtype).min,
        )

    def _target_node_norm(self) -> torch.Tensor:
        return self.embeddings.weight[self.node_ids].norm(dim=-1).mean().detach()

    def bind_embeddings(self, role_embeddings: torch.Tensor) -> torch.Tensor:
        if self.role_maps is None:
            raise RuntimeError(f"{self.condition} has no role maps")
        if role_embeddings.shape[-2] != self.num_registers:
            raise ValueError("Wrong state width")
        bound_roles = torch.einsum(
            "...re,rde->...rd",
            role_embeddings,
            self.role_maps,
        )
        combined = bound_roles.sum(dim=-2)
        return F.normalize(combined, dim=-1, eps=1e-8) * self._target_node_norm()

    def bind_token_ids(self, state_ids: torch.Tensor) -> torch.Tensor:
        return self.bind_embeddings(self.embeddings(state_ids))

    def fixed_latent(self, batch_size: int) -> torch.Tensor:
        latent_id = torch.full(
            (batch_size,),
            self.vocabulary.id("<LATENT>"),
            device=self.embeddings.weight.device,
            dtype=torch.long,
        )
        return self.embeddings(latent_id)

    @torch.no_grad()
    def project_role_maps_(self) -> None:
        if self.role_maps is None:
            return
        orthogonal, upper = torch.linalg.qr(self.role_maps)
        signs = torch.sign(torch.diagonal(upper, dim1=-2, dim2=-1))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        self.role_maps.copy_(orthogonal * signs.unsqueeze(-2))

    def distribution_feedback(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.condition != COT2:
            raise ValueError(
                "Distribution feedback is only defined for continuous_cot2"
            )
        # All vocabulary probabilities contribute, including non-value tokens.
        combined = probabilities @ self.embeddings.weight
        return (
            F.normalize(combined, dim=-1, eps=1e-8) * self._target_node_norm()
        ).unsqueeze(-2)

    def feedback(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.num_registers:
            raise ValueError("Wrong state width")
        if self.condition == "continuous":
            return self.bind_token_ids(states).unsqueeze(1)
        if self.condition == "continuous_unbound":
            # Position-specific readout is retained; only feedback omits binding.
            combined = self.embeddings(states).sum(dim=-2)
            return (
                F.normalize(combined, dim=-1, eps=1e-8) * self._target_node_norm()
            ).unsqueeze(1)
        if self.condition == "cache_only_full_head":
            return self.fixed_latent(states.shape[0]).unsqueeze(1)
        if self.condition == COT2:
            return self.distribution_feedback(
                target_distribution(
                    states, len(self.vocabulary), self.embeddings.weight.dtype
                )
            )
        return self.embeddings(states)


def assert_orthogonal_role_maps(
    model: Model,
    tolerance: float = 1e-4,
) -> None:
    if model.role_maps is None:
        return
    maps = model.role_maps.detach().float().cpu()
    products = maps @ maps.transpose(-1, -2)
    identity = torch.eye(model.d_model).expand_as(products)
    maximum_error = float((products - identity).abs().max())
    if maximum_error > tolerance:
        raise AssertionError(
            f"Role-map orthogonality error {maximum_error:.3e} exceeds "
            f"{tolerance:.1e}"
        )
