"""Compare many documents at once: every pair is checked with both algorithms."""
from itertools import combinations

from preprocess import preprocess
from compare import compare_tokens


def compare_many(docs: dict[str, str], remove_stop_words: bool = True,
                 max_tokens: int | None = None, progress=None) -> dict:
    """docs = {name: text}.  Returns {"names", "rows", "skipped", "truncated"}.

    Each row holds the scores and timings of one pair. Flags (plagiarism or not)
    are decided later from the score and the threshold, so changing the
    threshold never needs a re-run.
    """
    tokens, skipped, truncated = {}, [], []
    for name, text in docs.items():
        t = preprocess(text, remove_stop_words)
        if not t:
            skipped.append(name)
            continue
        if max_tokens and len(t) > max_tokens:
            truncated.append(name)
            t = t[:max_tokens]
        tokens[name] = t

    names = list(tokens)
    pairs = list(combinations(names, 2))
    rows = []
    for k, (x, y) in enumerate(pairs, start=1):
        r = compare_tokens(tokens[x], tokens[y], repeats=1)
        rows.append({
            "doc_a": x, "doc_b": y,
            "tokens_a": r["tokens_a"], "tokens_b": r["tokens_b"],
            "lcs_length": r["lcs_length"], "edit_distance": r["edit_distance"],
            "lcs_score": r["lcs_score"], "edit_score": r["edit_score"],
            "lcs_ms": r["lcs_ms"], "edit_ms": r["edit_ms"],
            "memory_saving": r["memory_saving"],
        })
        if progress:
            progress(k, len(pairs), f"{x} vs {y}")
    return {"names": names, "rows": rows, "skipped": skipped, "truncated": truncated}


def score_matrix(names: list[str], rows: list[dict], key: str) -> list[list]:
    """Square matrix of scores (None on the diagonal) for heat-map style tables."""
    idx = {n: i for i, n in enumerate(names)}
    mat = [[None] * len(names) for _ in names]
    for r in rows:
        i, j = idx[r["doc_a"]], idx[r["doc_b"]]
        mat[i][j] = mat[j][i] = r[key]
    return mat
