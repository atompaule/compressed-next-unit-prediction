"""Dataset loading for preliminary experiments with one explicit cache location."""
from src.common.experiment_paths import data_dir


def load_dataset(*args, **kwargs):
    """Keep Hugging Face dataset/split semantics and allow a cache override."""
    from datasets import load_dataset as huggingface_load_dataset
    kwargs.setdefault("cache_dir", str(data_dir("preliminary_experiments", "huggingface")))
    return huggingface_load_dataset(*args, **kwargs)
