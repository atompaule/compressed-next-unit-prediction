import re

def load_training_dataset():
    """Load the training split only when RL training is explicitly started."""
    from src.common.datasets import load_dataset
    return load_dataset("gsm8k", "main", split="train")


def get_reward(generated: str, gt_answer: str) -> float:
    def _last_number(text: str) -> float | None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return float(nums[-1].replace(",", "")) if nums else None

    m = re.search(r"####\s*([\d,]+)", gt_answer)
    if m is None:
        return 0.0
    gt = float(m.group(1).replace(",", ""))
    boxed = re.search(r"\\boxed\{([\d,.\-]+)\}", generated)
    if boxed:
        try:
            pred = float(boxed.group(1).replace(",", ""))
        except ValueError:
            pred = _last_number(boxed.group(1))
    else:
        pred = _last_number(generated)
    return 1.0 if pred is not None and abs(pred - gt) < 1e-3 else 0.0


def make_prompt(question: str, tokenizer) -> str:
    prompt = (
        "Answer the following question. "
        "Give your final "
        r"numeric answer under 'answer: ' in the format: answer: $\boxed{N}$"
        " where N is the number.\n\n"
        f"Question: {question}\n"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
