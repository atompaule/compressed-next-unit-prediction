from dataclasses import dataclass, field


@dataclass
class SolverConfig:
    steps: int = 300
    lr: float = 0.05

    # Loss 1: Target ranking loss
    # Pure ordering — fires when any pair (i, j) with p_i > p_j has s_j > s_i
    ranking_weight: float = 3.0

    # Loss 2: Target margin loss
    # Adjacent pairs only — demands similarity gap proportional to probability gap
    margin_weight: float = 2.0
    margin_min: float = 0.02
    margin_scale: float = 0.5

    # Loss 3: Interloper similarity loss
    # Pushes interloper similarities below the weakest target's similarity
    interloper_weight: float = 5.0
    interloper_margin_threshold: float = 0.2

    # Loss 4: Target similarity loss
    # Pulls the embedding toward all target tokens (unweighted)
    target_sim_weight: float = 1.0


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 1.0
    top_p: float = 0.97
    num_interlopers: int = 10
    solver_config: SolverConfig = field(default_factory=SolverConfig)


CFG = Config()
