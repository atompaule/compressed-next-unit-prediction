"""Stable run names and configuration checks for the logical sentence study."""

PROMPT_MODES = ("token", "chunk")


def rotation_mode(kind, no_token_pe=False, fixed_rotation_rates=False):
    if no_token_pe and fixed_rotation_rates:
        raise ValueError("Fixed nonzero rotation rates and --no-token-pe are exclusive")
    if kind != "sentence":
        if no_token_pe or fixed_rotation_rates:
            raise ValueError("Rotation controls require sentence kind")
        return None
    return "none" if no_token_pe else "fixed" if fixed_rotation_rates else "learned"


def run_name(kind, conclusions, seed, layers, no_token_pe=False, prompt_mode="token",
             fixed_rotation_rates=False):
    rotation_mode(kind, no_token_pe, fixed_rotation_rates)
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    if kind != "sentence" and (no_token_pe or prompt_mode != "token"):
        raise ValueError("Chunked prompts and --no-token-pe require sentence kind")
    return (
        f"{kind}_{conclusions}_seed{seed}"
        + (f"_{layers}l" if layers != 2 else "")
        + ("_nope" if no_token_pe else "")
        + ("_fixed_rates" if fixed_rotation_rates else "")
        + ("_chunked_prompt" if prompt_mode == "chunk" else "")
    )


def check_resume_prompt(saved_config, prompt_mode):
    # Historical runs used token prompts and had no prompt_mode field.
    if saved_config.get("prompt_mode", "token") != prompt_mode:
        raise ValueError("Cannot resume with a different premise representation")


def check_resume_rotation(saved, requested):
    """Reject condition switches, including accidental freezing of a trained run.

    Legacy runs have no rotation_mode field. Their config or alpha state still
    distinguishes learned rotations, the zero-rate control and discrete models.
    A legacy checkpoint can never count as a fixed-rate training run.
    """
    if "rotation_mode" in saved:
        actual = saved["rotation_mode"]
    elif "state" in saved:
        alpha = saved["state"].get("alpha")
        actual = None if alpha is None else "learned" if alpha.count_nonzero() else "none"
    elif saved.get("kind") == "discrete":
        actual = None
    else:
        actual = "none" if saved.get("sentence_composition") == "unrotated_sum" else "learned"
    if actual != requested:
        raise ValueError(f"Cannot resume rotation mode {actual!r} as {requested!r}")


def add_rotation_argument(parser):
    parser.add_argument(
        "--fixed-rotation-rates", action="store_true",
        help="Freeze the seed-matched within-sentence rotation rates; keep the optimizer LR schedule",
    )


def add_prompt_argument(parser):
    parser.add_argument(
        "--prompt-mode", choices=PROMPT_MODES, default="token",
        help="CoNUP premise inputs: original tokens or one composition per premise",
    )
