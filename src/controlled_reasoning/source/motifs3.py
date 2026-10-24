
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from collections import defaultdict
from logic import LEXICON, closure, independent_closure


def connected(facts):
    adj = defaultdict(set)
    for _, a, b in facts:
        adj[a].add(b)
        adj[b].add(a)
    todo = [next(iter(adj))]
    seen = set()
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        todo.extend(adj[x] - seen)
    return len(seen) == len(adj)


def candidate(family, rng):
    pools = {k: rng.sample(v, len(v)) for k, v in LEXICON.items()}
    types = {}

    def take(k):
        x = pools[k].pop()
        types[x] = k
        return x

    if family == "carry_region":
        p, o, l, r = take("person"), take("object"), take("place"), take("region")
        f = [("carries", p, o), ("in", p, l), ("in", l, r)]
    elif family == "contain_chain":
        o, b, l, r = take("object"), take("container"), take("place"), take("region")
        f = [("in", o, b), ("in", b, l), ("in", l, r)]
    elif family == "team_region_adjacency":
        p, t, l, r, a = (
            take("person"),
            take("team"),
            take("place"),
            take("region"),
            take("place"),
        )
        f = [("member", p, t), ("based", t, l), ("in", l, r), ("adjacent", l, a)]
    elif family == "type_property":
        o, a, b, q = (
            take("object"),
            take("category"),
            take("category"),
            take("property"),
        )
        f = [("isa", o, a), ("subtype", a, b), ("has", b, q)]
    elif family == "subtype_chain":
        a, b, c, d = [take("category") for _ in range(4)]
        f = [("subtype", a, b), ("subtype", b, c), ("subtype", c, d)]
    elif family in ["adjacency_chain", "adjacency_star"]:
        a, b, c, d = [take("place") for _ in range(4)]
        f = [
            ("adjacent", a, b),
            ("adjacent", b if family == "adjacency_chain" else a, c),
            ("adjacent", c if family == "adjacency_chain" else a, d),
        ]
    elif family == "permission_location":
        p, m, l, r = take("person"), take("machine"), take("place"), take("region")
        f = [
            ("trained", p, m),
            ("licensed", p, m),
            ("in", m, l),
            ("in", p, l),
            ("in", l, r),
        ]
        # Same actor, missing prerequisite: block a trained-only/licensed-only shortcut.
        x, y = take("machine"), take("machine")
        f += [("trained", p, x), ("licensed", p, y)]
    elif family == "carried_container":
        p, b, o, q, l = (
            take("person"),
            take("container"),
            take("object"),
            take("object"),
            take("place"),
        )
        f = [("carries", p, b), ("in", p, l), ("in", o, b), ("in", q, b)]
    elif family == "inherit_chain":
        o, a, b, c = (
            take("object"),
            take("category"),
            take("category"),
            take("category"),
        )
        f = [("isa", o, a), ("subtype", a, b), ("subtype", b, c)]
    else:
        raise ValueError(family)
    depths, proofs = closure(f)
    derived = sorted(set(depths) - set(f))
    assert (
        len(derived) == 3
        and independent_closure(f) - set(f) == set(derived)
        and connected(f)
    )
    return f, derived, types, depths, proofs
