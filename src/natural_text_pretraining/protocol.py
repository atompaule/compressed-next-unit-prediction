"""Fixed experimental conditions. Catalogue positives are additional to K."""

NEGATIVE_CANDIDATES = 18000
SEEDS = (101, 102, 103)
CONDITIONS = {
    "discrete": ("discrete", "learned", 18000),
    "learned_6000": ("sentence", "learned", 6000),
    "learned_12000": ("sentence", "learned", 12000),
    "learned_18000": ("sentence", "learned", 18000),
    "fixed_18000": ("sentence", "fixed_random", 18000),
}
