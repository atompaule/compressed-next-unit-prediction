# Grammatical chunking: ordered specification

Policy: `grammatical_groups`, selected through `policy="phrases"`.
The input is one source sentence and its fixed `en_core_web_sm` dependency parse.
The rules are deterministic given that parse. The parser itself is learned.
These are context-dependent grammatical groups, not independent propositions.

## 1. Form base groups

Protect base noun phrases and named entities. Also protect compact determiner,
attributive, compound, possessive and numerical dependencies on nominal heads,
including temporal nouns omitted by spaCy noun_chunks. Adjacent nominal heads
remain contiguous even when attached differently by the parser; numerical
coordination is protected when its intervening span is entirely nominal,
numeric, conjunctive or punctuation. Numeric enumerations additionally form
compact groups using the same seven-word budget. Adjacent-noun fallback excludes
time adjuncts and a preceding prepositional object, preventing merges such as
`machine tomorrow` or `In recent years support`. Protect a preposition together
with its nominal object or the head of its gerund complement. Coordinate noun
phrases remain together when only conjunctions/punctuation intervene.

Protect the adjacent pattern ADP + bare nominal object + `of` attached to
that noun (`as part of`, `by means of`, `in front of`). This is a structural
heuristic, not a dictionary of relational nouns. Its object remains protected
by the ordinary PP rule; the resulting group can exceed the merge limit.

Protect coordinated verbs as one core when the later verb has no subject,
clause marker or auxiliary, the earlier verb has no separate object or clausal
complement, and only conjunctions/adverbs/negation/particles intervene. This
approximates a shared predicate (`promote and uphold`). Clauses with their own
subjects and verbs with separate objects retain their boundaries.

Protect partitive pronoun/determiner/numeral heads with their attached `of`
phrase (`some of the examples`). Such heads are not glued to the preceding
verb by the short-pronominal-object rule.

Protect adjacent dependency-linked prepositions (`due to`). Extend comparative
protection to adverbs (`as late as`), using the same rule as for adjectives.

Protect verb cores: auxiliaries, negation, particles, preceding adverb modifiers,
pronominal subjects, and adjectival complements. A complement tagged `acomp`
with participial morphology is included even when its POS is VERB
(`became grown`, `remained crowded`). Preceding adverbs do not
expand the core across a comma or a nominal subject; pronominal subjects
remain allowed inside the core. Adverb groups include their own modifiers. An immediately following
adjective also joins copular `be`. Protect coordinated adjective groups without
an independent subject. A protected core may exceed the size limit below.

Propose boundaries at group starts, verb-core ends and prepositional/adverbial
group starts. Overlapping protected intervals suppress their internal proposals.

## 2. Mark structural boundaries

- Dependent clauses and coordinated verbs or prepositional groups.
- Semicolons and standalone double/em dashes; no cut inside an unspaced word.
- A comma closing a relative clause can separate a following coordinated noun
  group outside that relative clause. Ordinary noun lists stay protected.
- Relative and participial modifiers have mergeable boundaries, except when
  introduced by a comma.
- Keep clause introducers and opening quotation marks on the right, closing
  punctuation on the left. Straight double quotes alternate opening/closing;
  closing quotes are not moved onto the next group. An immediately preceding adverbial modifier stays
  with its subordinating conjunction when it modifies that conjunction or the
  same clause head (`even when`, `only if`, `just as`). Also protect an
  immediately preceding adverb that modifies a preposition itself (`even in`,
  `especially for`). No lexical list is used.
- Keep compact bracket contents intact only up to the existing seven-word merge
  limit. Longer bracket interiors use ordinary grammatical boundaries; enclosing
  noun/entity spans cannot protect the entire aside. Bracket types must match.
  An unclosed bracket does not protect the remainder of the sentence. Opening
  brackets stay with the following content and closing brackets with preceding
  content. Explanatory round-bracket spans retain outer aside boundaries.
- Four explicit discourse introductions end a group: `on the other hand`,
  `for example`, `in this connection`, `in conclusion`.

The aside criterion is a small, incomplete heuristic. It does not recognize all
possible explanatory parentheses.

## 3. Protect grammatical links

Keep infinitival predicate chains together (`going to have finished`). This
also covers `ccomp` complements with a base-form verb, infinitival `to` and no
clause marker (`allowed him to construct`); a base-form complement with an
accusative subject (`help us move`, `saw her cross`) also qualifies without
`to`. Finite `that` clauses and unmarked nominative-subject clauses stay separate.
An adjective tagged `ccomp` or `oprd` also forms a predicate complement
(`make the app better`); its intervening subject/object stays in that core. Protect
adjective comparisons with `than`/`as`, including a dimension PP such as
`narrower in diameter than the plate`. Protect pronominal and nominalized-adjective
objects (`said nothing`, `bring out your dead`).

There are five explicit fixed links: `as well as`, `in order for`, `prior to`,
`because of`, `such as`. No sentence-specific output table, lexical verb lists,
or parser reanalysis is used.

Precedence: compact bracket interiors, fixed links and compact nominal
dependency links suppress internal boundaries; otherwise structural boundaries
take precedence over protected base groups. Long bracket interiors are not atomic.
Only whitespace-separated boundaries are allowed, except before a detached aside
or after a double/em dash.
This keeps compounds and units such as `pressure–stress` and `bit/s` intact.

## 4. Apply one bounded merge rule

Merge adjacent groups only if they share a dependency link, no structural
boundary intervenes, and the resulting group has **at most seven non-punctuation,
non-space spaCy tokens** (`merge_words=7`). These are grammatical tokens, not
GPT-2 tokens or a universal definition of words.

Choose the highest-priority eligible merge globally, then repeat:

1. A predicative adjective or noun with its PP (`aware of this`, `the heart of
   this event`). Its dependency must be `acomp`, `attr` or `oprd`.
2. A nominal predicate complement (`attr`) with its predicate.
3. Subject with predicate.
4. Object or dative/recipient complement with predicate (`provided to us`).
5. Nominal modifier with its noun (PP, relative or participial modifier),
   or a simple apposition without its own PP, clause or coordination.
   Apposition merges cannot cross a colon; the usual size limit applies.
6. Other verbal/adjectival adjunct with its head.

Complete the predicate before spending the merge budget on the subject.

Ties are resolved left to right. The same size limit also distinguishes compact from long bracket contents; there is no weighted cost function, length penalty or dynamic program.
Protected groups can be longer than seven tokens. `max_tokens` only marks
oversized outputs for reporting; changing it does not change phrase boundaries.

## 5. Preserve the source

Emit original character slices and offsets, including original whitespace.
Never normalize spelling or generate punctuation. Before catalogue construction,
`prepare.py` merges any boundaries that would change the
concatenated GPT-2 token IDs. Text preservation alone does not guarantee token
preservation. This final safeguard can merge grammatical groups beyond the
seven-token limit; it never rewrites the source.

## Verification and scope


Validation checks text and GPT-2 token preservation, model capacity, catalogue references, and separation between partitions.
