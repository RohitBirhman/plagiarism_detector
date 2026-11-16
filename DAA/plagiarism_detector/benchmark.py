"""Measure how running time grows with document length (shows the O(m x n) cost)."""
import random

from preprocess import preprocess
from lcs import lcs
from edit_distance import edit_distance
from compare import _timed

VOCAB = ("data algorithm string table dynamic program sequence token memory "
         "matrix cost value input output search sort graph tree node path "
         "similar document copy source report system method result").split()


def make_pair(length: int, change_ratio: float = 0.2, seed: int = 1):
    """Make a random token list and a copy with `change_ratio` of tokens replaced."""
    rng = random.Random(seed)
    a = [rng.choice(VOCAB) for _ in range(length)]
    b = [rng.choice(VOCAB) if rng.random() < change_ratio else t for t in a]
    return a, b


def run_benchmark(sizes=(50, 100, 200, 400, 800)):
    rows = []
    for size in sizes:
        a, b = make_pair(size)
        _, lcs_ms = _timed(lcs, a, b, repeats=1)
        _, ed_ms = _timed(edit_distance, a, b, repeats=1)
        rows.append({"tokens": size, "LCS (ms)": round(lcs_ms, 2),
                     "Edit Distance (ms)": round(ed_ms, 2)})
    return rows


if __name__ == "__main__":
    print(f"{'tokens':>8} {'LCS ms':>10} {'EditDist ms':>13}")
    for r in run_benchmark():
        print(f"{r['tokens']:>8} {r['LCS (ms)']:>10} {r['Edit Distance (ms)']:>13}")
