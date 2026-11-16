"""Run both algorithms on the SAME pair, with the SAME preprocessing and threshold."""
import time
import tracemalloc

from preprocess import preprocess
from lcs import lcs, lcs_length_two_row
from edit_distance import edit_distance, edit_distance_two_row
from similarity import lcs_similarity, edit_similarity, verdict


def _timed(fn, *args, repeats: int = 3):
    """Run fn a few times and return (result, best time in milliseconds)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        best = min(best, (time.perf_counter() - start) * 1000)
    return result, best


def _peak_kb(fn, *args) -> float:
    """Peak extra memory (KB) used while fn runs. Measured in a separate pass,
    because tracemalloc slows code down and would distort the timings."""
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


FULL_TABLE_LIMIT = 6_000_000   # m x n cells; above this the memory-saving version is used


def compare_tokens(a: list[str], b: list[str], repeats: int = 3,
                   measure_memory: bool = False,
                   full_table_limit: int = FULL_TABLE_LIMIT) -> dict:
    """Core comparison on two token lists (scores + timings, no verdict).

    Small pairs use the full m x n table (as on the slides, and it lets us return the
    common words). Very large pairs use the two-row version: identical scores,
    same O(m x n) time, but far less memory, so nothing has to be cut off.
    """
    if not a or not b:
        raise ValueError("One of the documents has no comparable words after preprocessing.")
    m, n = len(a), len(b)
    big = m * n > full_table_limit

    if big:
        lcs_len, lcs_ms = _timed(lcs_length_two_row, a, b, repeats=repeats)
        dist, ed_ms = _timed(edit_distance_two_row, a, b, repeats=repeats)
        common, lcs_fn, ed_fn = [], lcs_length_two_row, edit_distance_two_row
    else:
        (lcs_len, common), lcs_ms = _timed(lcs, a, b, repeats=repeats)
        dist, ed_ms = _timed(edit_distance, a, b, repeats=repeats)
        lcs_fn, ed_fn = lcs, edit_distance

    result = {
        "tokens_a": m, "tokens_b": n,
        "lcs_length": lcs_len, "common_tokens": common,
        "edit_distance": dist,
        "lcs_score": lcs_similarity(lcs_len, m, n),
        "edit_score": edit_similarity(dist, m, n),
        "lcs_ms": lcs_ms, "edit_ms": ed_ms,
        "memory_saving": big,
    }
    if measure_memory:
        result["lcs_kb"] = _peak_kb(lcs_fn, a, b)
        result["edit_kb"] = _peak_kb(ed_fn, a, b)
    return result


def compare_texts(text_a: str, text_b: str, threshold: float = 60.0,
                  remove_stop_words: bool = True, max_tokens: int | None = None,
                  measure_memory: bool = False) -> dict:
    a = preprocess(text_a, remove_stop_words)
    b = preprocess(text_b, remove_stop_words)
    truncated = bool(max_tokens) and (len(a) > max_tokens or len(b) > max_tokens)
    if max_tokens:
        a, b = a[:max_tokens], b[:max_tokens]

    r = compare_tokens(a, b, measure_memory=measure_memory)
    r.update({
        "lcs_verdict": verdict(r["lcs_score"], threshold),
        "edit_verdict": verdict(r["edit_score"], threshold),
        "threshold": threshold,
        "truncated": truncated,
    })
    return r
