"""Order-invariant completion metrics, including explicit novelty denominators."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from collections import Counter


def score(sentences, world, done, unfinished=False):
    gold = set(world["conclusions"])
    unseen = set(world["unseen_conclusions"])
    reserved = set(world["reserved_conclusions"])
    counts = Counter(sentences)
    unique = set(sentences)
    valid = sum(n for s, n in counts.items() if s in gold)
    return dict(
        exact=bool(
            done and not unfinished and len(sentences) == len(gold) and unique == gold
        ),
        generated=len(sentences),
        valid_occurrences=valid,
        correct_unique=len(unique & gold),
        required=len(gold),
        precision=valid / len(sentences) if sentences else 0.0,
        recall=len(unique & gold) / len(gold),
        duplicate_count=sum(n - 1 for n in counts.values()),
        done=bool(done),
        unseen_correct=len(unique & unseen),
        unseen_required=len(unseen),
        unseen_recall=len(unique & unseen) / len(unseen) if unseen else None,
        reserved_correct=len(unique & reserved),
        reserved_required=len(reserved),
        reserved_recall=len(unique & reserved) / len(reserved) if reserved else None,
    )


def aggregate(scores):
    sums = {
        k: sum(s[k] for s in scores)
        for k in [
            "generated",
            "valid_occurrences",
            "correct_unique",
            "required",
            "duplicate_count",
            "unseen_correct",
            "unseen_required",
            "reserved_correct",
            "reserved_required",
        ]
    }
    ratio = lambda a, b: sums[a] / sums[b] if sums[b] else None
    return dict(
        samples=len(scores),
        exact_rate=sum(s["exact"] for s in scores) / len(scores),
        precision=ratio("valid_occurrences", "generated"),
        recall=ratio("correct_unique", "required"),
        unseen_recall=ratio("unseen_correct", "unseen_required"),
        reserved_recall=ratio("reserved_correct", "reserved_required"),
        duplicate_fraction=ratio("duplicate_count", "generated"),
        done_rate=sum(s["done"] for s in scores) / len(scores),
        counts=sums,
    )
