"""Lossless dependency groups; the ordered specification is in PHRASE_RULES.md."""

VERSION = "grammatical_groups"
MERGE_WORDS = 7
VERB_PARTS = {"aux", "auxpass", "neg", "prt", "advmod"}
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "expl"}
CLAUSES = {"advcl", "ccomp", "parataxis", "acl", "relcl"}
INTRODUCTIONS = (
    "on the other hand",
    "for example",
    "in this connection",
    "in conclusion",
)
FIXED_LINKS = ("as well as", "in order for", "prior to", "because of", "such as")


def split_sentence(text, doc, tokenizer, max_tokens=16, merge_words=MERGE_WORDS):
    """merge_words is the sole boundary size limit; max_tokens only flags output.

    Apply the canonical GPT-2 boundary alignment in prepare.py afterwards to conserve whole-text BPE tokenization.
    Deterministic for a fixed learned spaCy parse, not a semantic oracle.
    """
    if text != doc.text or max_tokens < 1 or merge_words < 1:
        raise ValueError("Require matching parser text and positive size parameters")
    if not text:
        return []
    protected, proposed, barriers = [], {0, len(doc)}, set()
    bracket_pairs, stack, fixed_links = [], [], []
    nominal_links = []
    opening_quotes, quoted = set(), False
    for t in doc:
        if t.text == '"':
            if not quoted:
                opening_quotes.add(t.i)
            quoted = not quoted
        elif t.text == "“":
            opening_quotes.add(t.i)

    def words(a, b):
        return sum(not t.is_punct and not t.is_space for t in doc[a:b])

    def protect(a, b):
        if a < b:
            protected.append((a, b))

    def predicate_complement(t):
        return (
            t.dep_ == "xcomp"
            or (
                t.dep_ == "ccomp"
                and t.tag_ == "VB"
                and any(
                    (c.tag_ == "TO" and c.dep_ == "aux")
                    or (c.dep_ in SUBJECTS and "Acc" in c.morph.get("Case"))
                    for c in t.children
                )
                and not any(c.dep_ == "mark" for c in t.children)
            )
            or (t.dep_ in {"ccomp", "oprd"} and t.pos_ == "ADJ")
        )

    def shared_predicate(t):
        return (
            t.dep_ == "conj"
            and t.pos_ == t.head.pos_ == "VERB"
            and t.head.i < t.i
            and not any(
                c.dep_ in SUBJECTS | {"mark", "aux", "auxpass"} for c in t.children
            )
            and not any(
                c.dep_ in {"dobj", "obj", "dative", "ccomp", "xcomp"}
                for c in t.head.children
            )
            and all(
                c.dep_ in {"cc", "advmod", "neg", "prt"}
                for c in doc[t.head.i + 1 : t.i]
            )
        )

    def partitive_head(t):
        return t.pos_ in {"PRON", "DET", "NUM"} and any(
            c.dep_ == "prep" and c.lower_ == "of" for c in t.children
        )

    def propose(i, strong=False):
        # Introducers/opening quotes right; closing punctuation left.
        while i > 0 and (
            doc[i - 1].dep_ in {"cc", "mark"}
            or doc[i - 1].pos_ == "SCONJ"
            or i - 1 in opening_quotes
            or doc[i - 1].text in {"(", "[", "{"}
        ):
            i -= 1
        proposed.add(i)
        if strong and i:
            barriers.add(i)

    # 1. Base groups: NP/entities, PP objects, verb cores and adjective groups.
    nouns = list(doc.noun_chunks)
    noun_by_root = {n.root.i: n for n in nouns}
    for span in [*nouns, *doc.ents]:
        protect(span.start, span.end)
        propose(span.start)
    for n in nouns:
        if n.root.dep_ == "pobj":
            protect(n.root.head.i, n.end)
        for child in n.root.children:
            other = noun_by_root.get(child.i)
            if (
                child.dep_ == "conj"
                and other is not None
                and all(
                    c.is_punct or c.dep_ in {"cc", "preconj"}
                    for c in doc[n.end : other.start]
                )
            ):
                protect(n.start, other.end)
    # Compact nominal dependencies outrank parser-proposed clause boundaries.
    # noun_chunks omits some temporal NPs and postnominal numeric labels.
    nominal_deps = {"det", "predet", "amod", "compound", "nummod", "poss", "case"}
    for t in doc:
        if t.head.pos_ in {"NOUN", "PROPN", "NUM"} and t.dep_ in nominal_deps:
            a, b = sorted((t.i, t.head.i))
            if not any(x.text in {",", ";", ":", "(", ")"} for x in doc[a : b + 1]):
                nominal_links.append((a, b + 1))
        # Adjacent nominal heads form a contiguous compound, even if the parser
        # attaches them to different predicates. No verb/noun lexical exceptions.
        if (
            t.i
            and t.pos_ in {"NOUN", "PROPN"}
            and doc[t.i - 1].pos_ in {"NOUN", "PROPN"}
            and t.dep_ != "npadvmod"
            and doc[t.i - 1].dep_ not in {"npadvmod", "pobj"}
        ):
            nominal_links.append((t.i - 1, t.i + 1))
        if (
            t.pos_ == "NUM"
            and t.dep_ == "conj"
            and t.head.pos_ in {"NOUN", "PROPN", "NUM"}
        ):
            a, b = sorted((t.i, t.head.i))
            if all(
                x.pos_ in {"NOUN", "PROPN", "NUM", "CCONJ", "PUNCT"}
                for x in doc[a : b + 1]
            ):
                nominal_links.append((a, b + 1))
    # Numeric enumerations are compact groups, not one chunk per number.
    # Use the existing word budget; punctuation stays with preceding numbers.
    a = 0
    while a < len(doc):
        if doc[a].pos_ != "NUM":
            a += 1
            continue
        b = a + 1
        while b < len(doc) and (doc[b].pos_ in {"NUM", "CCONJ"} or doc[b].text == ","):
            if words(a, b + 1) > merge_words:
                break
            b += 1
        while b > a + 1 and doc[b - 1].pos_ != "NUM":
            b -= 1
        if sum(t.pos_ == "NUM" for t in doc[a:b]) >= 2:
            nominal_links.append((a, b))
        a = max(a + 1, b)
    protected.extend(nominal_links)
    for t in doc:
        if t.pos_ == t.head.pos_ == "ADP" and t.head.i == t.i - 1:
            protect(t.head.i, t.i + 1)
        if partitive_head(t):
            propose(t.left_edge.i)
            for c in t.children:
                if c.dep_ == "prep" and c.lower_ == "of":
                    protect(t.left_edge.i, c.i + 1)
        # Bare nominal heads in a PP: as part of, by means of, in front of.
        if t.dep_ == "prep" and t.lower_ == "of" and t.i >= 2:
            noun, prep = doc[t.i - 1], doc[t.i - 2]
            if (
                noun.pos_ == "NOUN"
                and noun.dep_ == "pobj"
                and noun.head == prep
                and prep.pos_ == "ADP"
                and t.head == noun
            ):
                protect(prep.i, t.i + 1)
        if shared_predicate(t):
            protect(t.head.i, t.i + 1)
        if t.pos_ in {"SCONJ", "ADP"} and t.i > 0:
            modifier = doc[t.i - 1]
            if modifier.dep_ == "advmod" and (
                modifier.head == t or (t.pos_ == "SCONJ" and modifier.head == t.head)
            ):
                protect(modifier.i, t.i + 1)
        if t.dep_ in {"pobj", "pcomp"}:
            protect(min(t.head.i, t.left_edge.i), t.i + 1)
        if t.pos_ in {"VERB", "AUX"} and t.dep_ not in {"aux", "auxpass"}:
            core = [
                t,
                *[
                    c
                    for c in t.children
                    if c.dep_ in VERB_PARTS
                    and (
                        c.dep_ != "advmod"
                        or (
                            c.i < t.i
                            and not any(
                                x.text == ","
                                or (x.dep_ in SUBJECTS and x.pos_ != "PRON")
                                for x in doc[c.i + 1 : t.i]
                            )
                        )
                    )
                ],
            ]
            for c in t.children:
                if c.dep_ in {"acomp", "attr"} and (
                    c.pos_ == "ADJ"
                    or (c.dep_ == "acomp" and "Part" in c.morph.get("VerbForm"))
                ):
                    core.append(c)
                if c.dep_ in SUBJECTS and c.pos_ == "PRON":
                    core.append(c)
            if t.lemma_ == "be" and t.i + 1 < len(doc) and doc[t.i + 1].pos_ == "ADJ":
                core.append(doc[t.i + 1])
            lo, hi = min(c.i for c in core), max(c.i for c in core) + 1
            protect(lo, hi)
            propose(lo)
            proposed.add(hi)
        if t.pos_ in {"ADJ", "ADV"}:
            core = [
                t,
                *[c for c in t.children if c.dep_ in {"advmod", "neg", "preconj"}],
            ]
            core.extend(
                c
                for c in t.conjuncts
                if c.pos_ == t.pos_ and not any(s.dep_ in SUBJECTS for s in c.children)
            )
            protect(min(c.i for c in core), max(c.i for c in core) + 1)
        if t.dep_ in {"prep", "agent", "npadvmod"}:
            propose(t.left_edge.i if t.dep_ == "npadvmod" else t.i)

    # 2. Barriers: clauses, coordinated clauses/PPs, punctuation, introductions,
    # explanatory parentheticals. Relative/participial modifiers stay mergeable.
    for t in doc:
        if t.dep_ in CLAUSES:
            start = t.left_edge.i
            propose(
                start,
                strong=(t.dep_ not in {"acl", "relcl"} and not predicate_complement(t))
                or (start > 0 and doc[start - 1].text == ","),
            )
        if t.dep_ == "conj" and t.pos_ in {"VERB", "AUX", "ADP"}:
            propose(t.left_edge.i, strong=not shared_predicate(t))
        if t.dep_ == "conj" and t.pos_ in {"NOUN", "PROPN", "PRON"}:
            start = t.left_edge.i
            j = start - 1
            while j >= 0 and (doc[j].is_punct or doc[j].dep_ == "cc"):
                if doc[j].text == ",":
                    # Resume the outer coordination after a relative clause;
                    # ordinary noun lists remain protected base groups.
                    previous = doc[j - 1] if j > 0 else doc[j]
                    relatives = [
                        v for v in [previous, *previous.ancestors] if v.dep_ == "relcl"
                    ]
                    if any(v not in t.ancestors for v in relatives):
                        propose(start, strong=True)
                    break
                j -= 1
        if t.text in {";", "--", "—"} and t.i + 1 < len(doc):
            propose(t.i + 1, strong=True)
        if t.text in {"(", "[", "{"}:
            stack.append(t.i)
        elif t.text in {")", "]", "}"} and stack:
            a = stack[-1]
            if {"(": ")", "[": "]", "{": "}"}[doc[a].text] != t.text:
                continue
            stack.pop()
            bracket_pairs.append((a, t.i + 1))
    # Unclosed brackets must not protect the entire remaining sentence.
    lower = [t.lower_ for t in doc]
    for phrase in FIXED_LINKS:
        tokens = phrase.split()
        for a in range(len(doc) - len(tokens) + 1):
            if lower[a : a + len(tokens)] == tokens:
                protect(a, a + len(tokens))
                fixed_links.append((a, a + len(tokens)))
    for phrase in INTRODUCTIONS:
        tokens = phrase.split()
        for a in range(len(doc) - len(tokens) + 1):
            b = a + len(tokens)
            if lower[a:b] == tokens:
                protect(a, b)
                while b < len(doc) and doc[b].is_punct:
                    b += 1
                proposed.add(b)
                barriers.add(b)
    asides, compact_brackets = [], []
    for a, b in bracket_pairs:
        if words(a + 1, b - 1) <= merge_words:
            protect(a, b)
            compact_brackets.append((a, b))
        else:
            # An enclosing NP/entity must not swallow a long aside. Local
            # groups wholly inside/outside remain governed by the same rules.
            protected = [(x, y) for x, y in protected if not (x <= a and y >= b)]
            proposed.update({a, b})
            barriers.update({a, b})
        inner = doc[a + 1 : b - 1]
        explanation = any(t.pos_ in {"VERB", "AUX"} for t in inner) or any(
            t.lower_ in {"perhaps", "example", "namely", "not"} for t in inner
        )
        if doc[a].text == "(" and explanation:
            while b < len(doc) and doc[b].text in {".", ",", ";", ":", "!", "?"}:
                b += 1
            proposed.update({a, b})
            barriers.update({a, b})
            asides.append((a, b))

    # 3. Protected dependency links: infinitival chains, comparisons,
    # pronominal or nominalized adjective objects.
    for t in doc:
        if predicate_complement(t) and t.head.pos_ in {"VERB", "AUX", "ADJ"}:
            protect(min(t.head.i, t.left_edge.i), t.i + 1)
        comparison = t.head
        while comparison.dep_ in {"prep", "pobj"} and comparison.head != comparison:
            comparison = comparison.head
        if (
            t.dep_ == "prep"
            and t.lower_ in {"than", "as"}
            and comparison.pos_ in {"ADJ", "ADV"}
        ):
            for c in t.children:
                if c.dep_ in {"pobj", "pcomp"}:
                    end = noun_by_root[c.i].end if c.i in noun_by_root else c.i + 1
                    protect(comparison.i, end)
        if (
            t.dep_ in {"dobj", "obj"}
            and t.pos_ in {"PRON", "ADJ"}
            and not partitive_head(t)
        ):
            protect(min(t.head.i, t.left_edge.i), t.i + 1)

    # Only compact brackets are atomic; long interiors retain normal boundaries.
    atomic = [*compact_brackets, *fixed_links, *nominal_links]
    barriers = {i for i in barriers if not any(a < i < b for a, b in atomic)}
    points = sorted(
        i for i in proposed if i in barriers or not any(a < i < b for a, b in protected)
    )
    points = [
        i
        for i in points
        if i in {0, len(doc)}
        or i in {a for a, _ in asides}
        or (
            not doc[i].is_punct
            and (bool(doc[i - 1].whitespace_) or doc[i - 1].text in {"--", "—"})
        )
    ]
    barriers.intersection_update(points)
    groups = list(zip(points, points[1:]))

    # 4. One bounded merge, prioritized by dependency role, then left-to-right.
    def priority(left, right):
        a, b = left
        _, c = right
        if b in barriers or words(a, c) > merge_words:
            return None
        ranks = []
        for t in doc[a:c]:
            cross = (a <= t.i < b <= t.head.i < c) or (a <= t.head.i < b <= t.i < c)
            if not cross:
                continue
            # Complete predicative adjective phrases before attaching subjects.
            if (
                t.dep_ == "prep"
                and t.head.pos_ in {"ADJ", "NOUN", "PROPN"}
                and t.head.dep_ in {"acomp", "attr", "oprd"}
            ):
                ranks.append(-2)
            elif t.dep_ == "attr":
                ranks.append(-1)
            elif t.dep_ in SUBJECTS:
                ranks.append(0)
            elif t.dep_ in {"dobj", "obj", "dative", "attr", "oprd"}:
                ranks.append(1)
            elif (
                t.dep_ == "appos"
                and not any(
                    x.dep_ in {"prep", "acl", "relcl", "conj", "ccomp"}
                    for x in t.children
                )
                and not any(
                    x.text == ":" for x in doc[min(t.i, t.head.i) : max(t.i, t.head.i)]
                )
            ):
                ranks.append(2)
            elif t.dep_ in {"prep", "acl", "relcl"} and t.head.pos_ in {
                "NOUN",
                "PROPN",
                "PRON",
            }:
                ranks.append(2)
            elif t.dep_ in {"prep", "agent", "npadvmod", "advmod"} and t.head.pos_ in {
                "VERB",
                "AUX",
                "ADJ",
            }:
                ranks.append(3)
        return min(ranks) if ranks else None

    while True:
        choices = [
            (p, i)
            for i in range(len(groups) - 1)
            if (p := priority(groups[i], groups[i + 1])) is not None
        ]
        if not choices:
            break
        _, i = min(choices)
        groups[i : i + 2] = [(groups[i][0], groups[i + 1][1])]

    # 5. Original character slices. No rewritten text or generated punctuation.
    offsets = {0: 0, len(doc): len(text)}
    for a, b in groups:
        for i in (a, b):
            if i not in offsets:
                offsets[i] = len(text[: doc[i].idx].rstrip())
    spans = [(offsets[a], offsets[b]) for a, b in groups]
    ids = tokenizer([text[a:b] for a, b in spans], add_special_tokens=False)[
        "input_ids"
    ]
    result = [
        dict(
            text=text[a:b],
            start=a,
            end=b,
            tokens=len(seq),
            boundary="structural" if end in barriers else "group",
            over_cap=len(seq) > max_tokens,
            policy=VERSION,
        )
        for (a, b), (_, end), seq in zip(spans, groups, ids)
    ]
    assert "".join(r["text"] for r in result) == text
    return result
