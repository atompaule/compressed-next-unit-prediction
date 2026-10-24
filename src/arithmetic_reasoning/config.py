"""Architecture configuration shared across register widths."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 96
    layers: int = 3
    heads: int = 4
    max_positions: int = 1024
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.d_model, self.layers, self.heads, self.max_positions) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("d_model must divide into heads; dropout must be in [0,1)")
