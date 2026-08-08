from collections.abc import Sequence
from itertools import product

def compute_fighter_feature_names(
    *,
    fighter_1_prefix: str,
    fighter_2_prefix: str,
    feature_names: list[str],
) -> Sequence[str]:
    return [
        f"{prefix}{feature}"
        for prefix, feature in product(
            [fighter_1_prefix, fighter_2_prefix],
            feature_names
        )
    ]
