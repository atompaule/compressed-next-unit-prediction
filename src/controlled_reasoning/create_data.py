"""EOS-only logical data; familiar structures and reserved complete sentences.

No model execution. Generate data once and share it across paired model seeds.
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from src.common.experiment_paths import artifact_dir
from pathlib import Path
from collections import Counter
import argparse, hashlib, importlib, itertools, json, random, sys

ROOT = artifact_dir(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from logic import LEXICON, SPECIAL, TEMPLATES, digest, closure, independent_closure
from signatures import SIGNATURES

COUNTS = {
    "carry_region": 160,
    "contain_chain": 160,
    "team_region_adjacency": 160,
    "type_property": 160,
    "subtype_chain": 64,
    "adjacency_chain": 128,
    "adjacency_star": 128,
    "permission_location": 64,
}


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def jsonl(p, rows):
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))


def sentence(f):
    return TEMPLATES[f[0]].format(a=f[1], b=f[2]).removesuffix(" .")


def partition(f):
    # Whole directed fact, not unordered lexical pair or held-out entity type.
    k = int(digest(["sentence-holdout-v1", *f])[:16], 16) % 17
    return "validation" if k == 0 else "test" if k == 1 else "train"


def vocabulary():
    words = {t for pool in LEXICON.values() for t in pool}
    for template in TEMPLATES.values():
        words.update(
            template.replace("{a}", "").replace("{b}", "").replace(".", "").split()
        )
    return {t: i for i, t in enumerate(SPECIAL + sorted(words))}


def catalogue(v):
    result = []
    for rel, (aa, bb) in SIGNATURES.items():
        for a, b in itertools.product(
            [w for t in aa for w in LEXICON[t]], [w for t in bb for w in LEXICON[t]]
        ):
            f = (rel, a, b)
            s = sentence(f)
            result.append(
                dict(
                    id=len(result),
                    fact=f,
                    sentence=s,
                    token_ids=[v[t] for t in s.split()] + [2],
                    partition=partition(f),
                )
            )
    result.append(
        dict(
            id=len(result),
            fact=["DONE", "", ""],
            sentence="<DONE>",
            token_ids=[3],
            partition="train",
        )
    )
    return result


def counts(n):
    assert (
        n % 32 == 0
    ), "Sample count must preserve all eight family proportions (multiple of32)"
    return {f: c * n // 1024 for f, c in COUNTS.items()}


def make_split(
    ncon, split, size, seed, seen, v, allow_repeats=False, excluded_worlds=None
):
    motif = importlib.import_module(f"motifs{ncon}")
    rng = random.Random(seed)
    ordering = random.Random(seed + 900000)
    rows = []
    worlds = []
    attempts = Counter()
    for family, n in counts(size).items():
        for _ in range(n):
            for trial in range(100000):
                attempts[family] += 1
                f, con, types, depths, proofs = motif.candidate(family, rng)
                wid = digest(sorted(f))[:20]
                if wid in (excluded_worlds if allow_repeats else seen):
                    continue
                pp = [partition(z) for z in f + con]
                if split == "train":
                    if any(p != "train" for p in pp):
                        continue
                else:
                    other = "test" if split == "validation" else "validation"
                    if other in pp or not any(partition(z) == split for z in con):
                        continue
                seen.add(wid)
                break
            else:
                raise RuntimeError(f"Exhausted {ncon}/{split}/{family}")
            assert set(con) == independent_closure(f) - set(f)
            ordering.shuffle(f)
            order = list(range(ncon))
            ordering.shuffle(order)
            premises = [sentence(z) for z in f]
            gold = [sentence(z) for z in con]
            pids = [t for s in premises for t in [*[v[w] for w in s.split()], 2]] + [4]
            targets = [[*[v[w] for w in gold[i].split()], 2] for i in order] + [[3]]
            reserved = [
                sentence(z) for z in con if partition(z) == split and split != "train"
            ]
            common = dict(
                world_id=wid,
                split=split,
                family=family,
                prompt=" ".join(s + " <EOS>" for s in premises) + " <QUERY>",
            )
            rows.append(
                dict(
                    **common,
                    prompt_ids=pids,
                    targets=targets,
                    order=order,
                    conclusions=[gold[i] for i in order],
                    reserved_conclusions=reserved,
                )
            )
            worlds.append(
                dict(
                    **common,
                    premises=f,
                    conclusion_facts=con,
                    conclusions=gold,
                    types=types,
                    proof_depths=[depths[z] for z in con],
                    proofs=[proofs[z] for z in con],
                    reserved_conclusions=reserved,
                )
            )
    ordering.shuffle(rows)
    return rows, worlds, dict(attempts)


def build(ncon, out, args):
    out.mkdir(parents=True, exist_ok=False)
    v = vocabulary()
    cat = catalogue(v)
    write(out / "vocabulary.json", v)
    write(out / "catalogue.json", cat)
    masks = {
        "train": [c["id"] for c in cat if c["partition"] == "train"],
        "validation": [c["id"] for c in cat if c["partition"] != "test"],
        "test": [c["id"] for c in cat],
    }
    write(out / "candidate_ids_by_phase.json", masks)
    seen = set()
    splits = {}
    ww = {}
    attempts = {}
    for split, size, offset in [
        ("validation", args.validation, 10),
        ("test", args.test, 20),
    ]:
        splits[split], ww[split], attempts[split] = make_split(
            ncon, split, size, args.seed + 100000 * ncon + offset, seen, v
        )
    excluded_worlds = set(seen)
    world_counts = Counter()
    observed = set()
    target_seen = set()
    lexcounts = Counter()
    epochs = []
    rules = Counter()
    depths = {}
    maxprompt = maxcompletion = 0
    for epoch in range(1, args.epochs + 1):
        rows, worlds, attempts[f"epoch{epoch:02d}"] = make_split(
            ncon,
            "train",
            args.per_epoch,
            args.seed + 100000 * ncon + 1000 + epoch,
            seen,
            v,
            allow_repeats=args.allow_repeats_after is not None
            and epoch > args.allow_repeats_after,
            excluded_worlds=excluded_worlds,
        )
        for w in worlds:
            world_counts[w["world_id"]] += 1
            assert w["world_id"] not in excluded_worlds
            observed.update(sentence(f) for f in w["premises"] + w["conclusion_facts"])
            target_seen.update(w["conclusions"])
            rules.update(x["rule"] for x in w["proofs"])
            for f in w["premises"] + w["conclusion_facts"]:
                lexcounts.update(f[1:])
            signature = (w["family"], tuple(sorted(w["proof_depths"])))
            depths[str(signature)] = depths.get(str(signature), 0) + 1
        for r in rows:
            assert len(r["targets"]) == ncon + 1 and all(
                t != v.get(".") for s in r["targets"] for t in s
            )
            maxprompt = max(maxprompt, 1 + len(r["prompt_ids"]))
            maxcompletion = max(maxcompletion, sum(map(len, r["targets"])))
        p = out / f"epoch{epoch:02d}.jsonl"
        jsonl(p, rows)
        epochs.append(
            dict(
                epoch=epoch,
                worlds=len(rows),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                target_tokens=sum(sum(map(len, r["targets"])) for r in rows),
            )
        )
        print(ncon, "epoch", epoch, flush=True)
    evalstats = {}
    evalworlds = []
    train_ids = set(masks["train"])
    index = {tuple(c["token_ids"]): c["id"] for c in cat}
    for split, rows in splits.items():
        byid = {w["world_id"]: w for w in ww[split]}
        unseen = reserved = total = 0
        for r in rows:
            w = byid[r["world_id"]]
            novel = [s for s in w["conclusions"] if s not in observed]
            assert w["reserved_conclusions"] and set(w["reserved_conclusions"]) <= set(
                novel
            )
            for s in w["reserved_conclusions"]:
                cid = index[tuple([v[t] for t in s.split()] + [2])]
                assert cid not in train_ids
                if split == "test":
                    assert cid not in masks["validation"]
            assert str((w["family"], tuple(sorted(w["proof_depths"])))) in depths
            r["unseen_conclusions"] = novel
            w["unseen_conclusions"] = novel
            unseen += len(novel)
            reserved += len(w["reserved_conclusions"])
            total += len(w["conclusions"])
            assert all(
                t < 5
                or any(
                    t == v[x] and lexcounts[x] > 0
                    for pool in LEXICON.values()
                    for x in pool
                )
                or t
                in {
                    v[x]
                    for template in TEMPLATES.values()
                    for x in template.replace("{a}", "")
                    .replace("{b}", "")
                    .replace(".", "")
                    .split()
                }
                for t in r["prompt_ids"]
            )
        evalstats[split] = dict(
            worlds=len(rows),
            conclusions=total,
            reserved_conclusions=reserved,
            unseen_in_training_text=unseen,
            families=dict(Counter(r["family"] for r in rows)),
        )
        jsonl(out / f"{split}.jsonl", rows)
        evalworlds += ww[split]
    assert all(lexcounts[x] > 0 for pool in LEXICON.values() for x in pool)
    jsonl(out / "worlds.jsonl", evalworlds)
    write(out / "training_seen_sentences.json", sorted(observed))
    audit = dict(
        passed=True,
        conclusions_per_world=ncon,
        training_worlds=args.epochs * args.per_epoch,
        globally_unique_worlds=len(seen),
        unique_training_worlds=len(world_counts),
        repeated_training_presentations=sum(world_counts.values()) - len(world_counts),
        maximum_world_presentations=max(world_counts.values()),
        allow_repeats_after=args.allow_repeats_after,
        vocabulary_size=len(v),
        no_period_token="." not in v,
        premises_and_targets_use_EOS=True,
        DONE_retained=True,
        independent_closure_checked_every_world=True,
        all_evaluation_families_and_proof_depth_profiles_seen_in_training=True,
        all_entities_seen_in_training=True,
        reserved_absent_from_training_inputs_targets_and_training_catalogue=True,
        test_reserved_excluded_from_validation_inputs_targets_and_catalogue=True,
        candidate_counts={k: len(x) for k, x in masks.items()},
        evaluation=evalstats,
        minimum_entity_occurrences=min(lexcounts.values()),
        rule_final_step_counts=dict(rules),
        max_train_prompt_tokens=maxprompt,
        max_train_completion_tokens=maxcompletion,
        attempts=attempts,
    )
    write(out / "audit.json", audit)
    write(
        out / "manifest.json",
        dict(
            epochs=epochs,
            data_seed=args.seed,
            conclusions=ncon,
            vocabulary_size=len(v),
            scope="Identical family proportions and role/type templates in train/validation/test; sentence novelty only. Three vs six retains prior structural differences.",
            novelty="Whole directed fact hash: val0,test1,train2..16 modulo17; exclude both holdouts from all training facts; each eval world includes at least one split-reserved conclusion and no other split-reserved fact.",
            candidate_mask_required=True,
            allow_repeats_after=args.allow_repeats_after,
            training_sampling="Random worlds; global uniqueness enforced only through allow_repeats_after, if specified. Evaluation worlds always excluded.",
            pilot_epochs=args.epochs,
            evalstats=evalstats,
            files={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.glob("*.jsonl")
            },
        ),
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "attempts"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conclusions", type=int, nargs="+", default=[3, 6], choices=[3, 6])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--per-epoch", type=int, default=1024)
    p.add_argument("--validation", type=int, default=512)
    p.add_argument("--test", type=int, default=1024)
    p.add_argument("--seed", type=int, default=170917)
    p.add_argument("--allow-repeats-after", type=int, default=None)
    a = p.parse_args()
    for n in a.conclusions:
        build(n, a.output / f"{n}_conclusions", a)
