"""GSM8K preset: 1,000 training batches and 250 optimizer updates."""
from dataclasses import dataclass, field

MODELS = ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct")
APPROACHES = ("discrete", "gaussian_noise", "token_sampling", "dirichlet_sampling")


@dataclass
class RLConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    group_size: int = 4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    approach: str = "dirichlet_sampling"
    dirichlet_gamma: float = 4.0
    length_penalty_alpha: float = 0.0
    learning_rate: float = 1e-5
    temperature: float = 1.0
    max_tokens: int = 128
    max_batches: int = 1000


@dataclass
class Config:
    model_id: str = MODELS[0]
    min_p: float = 0.03
    rl_config: RLConfig = field(default_factory=RLConfig)


CFG = Config()
