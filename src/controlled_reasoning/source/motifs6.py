
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
        p, o, l, r, z = (
            take("person"),
            take("object"),
            take("place"),
            take("region"),
            take("region"),
        )
        f = [("carries", p, o), ("in", p, l), ("in", l, r), ("in", r, z)]
    elif family == "contain_chain":
        o, b, l, r, z = (
            take("object"),
            take("container"),
            take("place"),
            take("region"),
            take("region"),
        )
        f = [("in", o, b), ("in", b, l), ("in", l, r), ("in", r, z)]
    elif family == "team_region_adjacency":
        p, q, t, l, r, a, b = (
            take("person"),
            take("person"),
            take("team"),
            take("place"),
            take("region"),
            take("place"),
            take("place"),
        )
        f = [
            ("member", p, t),
            ("member", q, t),
            ("based", t, l),
            ("in", l, r),
            ("adjacent", l, a),
            ("adjacent", a, b),
        ]
    elif family == "type_property":
        o, a, b, c, q = (
            take("object"),
            take("category"),
            take("category"),
            take("category"),
            take("property"),
        )
        f = [("isa", o, a), ("subtype", a, b), ("subtype", b, c), ("has", c, q)]
    elif family == "subtype_chain":
        a, b, c, d, e = [take("category") for _ in range(5)]
        f = [("subtype", a, b), ("subtype", b, c), ("subtype", c, d), ("subtype", d, e)]
    elif family in ["adjacency_chain", "adjacency_star"]:
        places = [take("place") for _ in range(7)]
        f = [
            (
                "adjacent",
                places[i] if family == "adjacency_chain" else places[0],
                places[i + 1],
            )
            for i in range(6)
        ]
    elif family == "permission_location":
        p = take("person")
        m, n, z, x, y = [take("machine") for _ in range(5)]
        l, r = take("place"), take("region")
        f = (
            [("trained", p, a) for a in [m, n, z, x]]
            + [("licensed", p, a) for a in [m, n, z, y]]
            + [("in", m, l), ("in", n, l), ("in", p, l), ("in", l, r)]
        )
    elif family == "carried_container":
        p, b, o, l, r, a = (
            take("person"),
            take("container"),
            take("object"),
            take("place"),
            take("region"),
            take("place"),
        )
        f = [
            ("carries", p, b),
            ("in", p, l),
            ("in", l, r),
            ("in", o, b),
            ("adjacent", l, a),
        ]
    elif family == "inherit_chain":
        o = take("object")
        a, b, c, d, e = [take("category") for _ in range(5)]
        q = take("property")
        f = [
            ("isa", o, a),
            ("subtype", a, b),
            ("subtype", a, c),
            ("subtype", a, d),
            ("subtype", a, e),
            ("has", b, q),
        ]
    else:
        raise ValueError(family)
    depths, proofs = closure(f)
    derived = sorted(set(depths) - set(f))
    assert (
        len(derived) == 6
        and independent_closure(f) - set(f) == set(derived)
        and connected(f)
    )
    return f, derived, types, depths, proofs
