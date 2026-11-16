"""Step 6: turn raw algorithm output into comparable similarity percentages."""


def lcs_similarity(lcs_len: int, m: int, n: int) -> float:
    longest = max(m, n)
    return 100.0 if longest == 0 else 100.0 * lcs_len / longest


def edit_similarity(distance: int, m: int, n: int) -> float:
    longest = max(m, n)
    return 100.0 if longest == 0 else 100.0 * (1 - distance / longest)


def verdict(score: float, threshold: float) -> str:
    return "PLAGIARISM SUSPECTED" if score >= threshold else "Looks original"


def is_flagged(score: float, threshold: float) -> bool:
    return score >= threshold


def band(overall_percent: float) -> str:
    """Overall document verdict used by the web check (bands are adjustable)."""
    if overall_percent < 20:
        return "Low"
    if overall_percent < 40:
        return "Moderate"
    return "High"
