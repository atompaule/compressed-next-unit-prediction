"""Rule semantics and independent closure checks for the current logical data."""

from collections import defaultdict
import hashlib
import json

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<DONE>", "<QUERY>"]
LEXICON = {
    "person": "Mira Noah Ada Theo Lena Omar Iris Finn Nora Ravi Vera Leon Alma Hugo Cleo Ivan Esme Luca Maya Nico Rosa Sami Tara Zane Elia Remy Anya Kian Lyra Otto Daria Emil".split(),
    "object": "key coin compass lantern ribbon whistle hammer brush mirror bottle notebook camera spoon marble feather shell button candle ticket ring badge pencil cup scarf bell token glove map flute drum fork lens".split(),
    "container": "case chest crate basket pouch trunk locker barrel cabinet drawer suitcase hamper".split(),
    "place": "observatory workshop library kitchen studio hangar garden tower cabin museum station garage cellar gallery chapel office attic bakery school theatre".split(),
    "region": "northland southland eastland westland highland lowland lakeside riverside coastland woodland upland midland".split(),
    "team": "Orion Vega Atlas LyraTeam Nova Sirius Comet Eclipse Zenith Horizon Apollo Helios".split(),
    "category": "mineral metal crystal alloy fabric fibre ceramic polymer liquid fuel solvent pigment".split(),
    "property": "shiny fragile heavy light rough smooth flexible rigid porous dense soft hard".split(),
    "machine": "lathe crane kiln press drill saw pump mill scanner printer furnace mixer".split(),
}
RULES = [
    ("carry_location", [("carries", "x", "o"), ("in", "x", "l")], ("in", "o", "l")),
    ("containment", [("in", "x", "y"), ("in", "y", "z")], ("in", "x", "z")),
    (
        "team_assignment",
        [("member", "p", "t"), ("based", "t", "s")],
        ("assigned", "p", "s"),
    ),
    (
        "assignment_region",
        [("assigned", "p", "s"), ("in", "s", "r")],
        ("assigned", "p", "r"),
    ),
    ("type_inheritance", [("isa", "x", "c"), ("subtype", "c", "d")], ("isa", "x", "d")),
    (
        "subtype_transitivity",
        [("subtype", "a", "b"), ("subtype", "b", "c")],
        ("subtype", "a", "c"),
    ),
    (
        "category_property",
        [("subtype", "c", "d"), ("has", "d", "q")],
        ("has", "c", "q"),
    ),
    (
        "instance_property",
        [("isa", "x", "c"), ("has", "c", "q")],
        ("property", "x", "q"),
    ),
    (
        "conjunctive_permission",
        [("trained", "p", "m"), ("licensed", "p", "m")],
        ("may", "p", "m"),
    ),
    ("adjacency_symmetry", [("adjacent", "a", "b")], ("adjacent", "b", "a")),
]
TEMPLATES = {
    "in": "{a} is in {b} .",
    "carries": "{a} carries {b} .",
    "member": "{a} belongs to {b} .",
    "based": "{a} is based in {b} .",
    "assigned": "{a} is assigned to {b} .",
    "isa": "{a} is a {b} .",
    "subtype": "{a} is a kind of {b} .",
    "has": "{a} has property {b} .",
    "property": "{a} is {b} .",
    "trained": "{a} is trained for {b} .",
    "licensed": "{a} is licensed for {b} .",
    "may": "{a} may operate {b} .",
    "adjacent": "{a} is adjacent to {b} .",
}


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def closure(premises):
    """Semi-naive-sized generic unification; shortest proof height is retained."""
    depths = {tuple(f): 0 for f in premises}
    proofs = {}
    while True:
        changed = False
        by = defaultdict(list)
        for f in depths:
            by[f[0]].append(f)
        for name, body, head in RULES:
            bindings = [({}, [])]
            for pattern in body:
                new = []
                for env, parents in bindings:
                    for fact in by[pattern[0]]:
                        e = dict(env)
                        for var, val in zip(pattern[1:], fact[1:]):
                            if var in e and e[var] != val:
                                break
                            e[var] = val
                        else:
                            new.append((e, parents + [fact]))
                bindings = new
            for env, parents in bindings:
                fact = (head[0], env[head[1]], env[head[2]])
                if fact[0] in ("in", "subtype", "adjacent") and fact[1] == fact[2]:
                    continue
                depth = 1 + max(depths[p] for p in parents)
                if fact not in depths or depth < depths[fact]:
                    depths[fact] = depth
                    proofs[fact] = {"rule": name, "parents": [list(p) for p in parents]}
                    changed = True
        if not changed:
            return depths, proofs


def independent_closure(premises):
    """Independent explicit relational implementation, used to audit every world."""
    fs = set(map(tuple, premises))
    while True:
        old = set(fs)
        d = defaultdict(set)
        for r, a, b in old:
            d[r].add((a, b))
        for a, b in d["adjacent"]:
            fs.add(("adjacent", b, a))
        for person, obj in d["carries"]:
            for subject, loc in d["in"]:
                if person == subject:
                    fs.add(("in", obj, loc))
        joins = [
            ("in", "in", "in"),
            ("member", "based", "assigned"),
            ("assigned", "in", "assigned"),
            ("isa", "subtype", "isa"),
            ("subtype", "subtype", "subtype"),
            ("subtype", "has", "has"),
            ("isa", "has", "property"),
        ]
        for left, right, out in joins:
            for a, b in d[left]:
                for c, z in d[right]:
                    if b == c and not (out in ("in", "subtype") and a == z):
                        fs.add((out, a, z))
        for a, b in d["trained"] & d["licensed"]:
            fs.add(("may", a, b))
        if fs == old:
            return fs
