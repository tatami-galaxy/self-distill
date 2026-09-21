"""Print random question–hint pairs from an SDFT hint cache.

Example:
    python -m eval.inspect_hints \
        data/pi/hint/deepmath/Qwen3-1.7B-t0.7_g6_lora_r16-checkpoint-20 \
        -n 5 --seed 42
"""

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hint_cache", type=Path, help="Path to a saved Hugging Face hint dataset.")
    parser.add_argument("-n", "--n", type=int, default=5,
                        help="Number of pairs to print, capped at the cache size (default: 5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed; omit for a fresh sample each time.")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be at least 1")
    if not args.hint_cache.is_dir():
        parser.error(f"Hint cache directory does not exist: {args.hint_cache}")
    try:
        cache = load_from_disk(str(args.hint_cache))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(cache, Dataset):
        parser.error("Expected a single Dataset; point to a hint cache or an individual split.")
    missing = {"question", "hint"}.difference(cache.column_names)
    if missing:
        parser.error(f"Hint cache is missing columns: {', '.join(sorted(missing))}")
    if not len(cache):
        parser.error("Hint cache is empty")

    indices = random.Random(args.seed).sample(range(len(cache)), min(args.n, len(cache)))
    print(f"Cache: {args.hint_cache}\nShowing {len(indices)} of {len(cache)} rows.")
    for number, index in enumerate(indices, start=1):
        row = cache[index]
        print(f"\n{'=' * 80}\nSample {number} (cache row {index})")
        print(f"\nQuestion:\n{row['question']}\n\nHint:\n{row['hint']}")


if __name__ == "__main__":
    main()
