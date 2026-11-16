"""Edit (Levenshtein) Distance using dynamic programming on token sequences.

Operations: insertion, deletion, substitution (each cost 1).
Time  : O(m x n)
Space : O(m x n)
"""


def edit_distance(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    # dp[i][j] = min edits to turn a[:i] into b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i            # delete everything
    for j in range(n + 1):
        dp[0][j] = j            # insert everything

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )
    return dp[m][n]


def edit_distance_two_row(a: list[str], b: list[str]) -> int:
    """Same distance as the full table, but keeps only two rows: O(min(m, n)) space.
    Used automatically for very large documents so the whole text can still be scanned."""
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            d = prev[j - 1] + (x != y)
            u = prev[j] + 1
            if u < d:
                d = u
            l = cur[j - 1] + 1
            if l < d:
                d = l
            cur.append(d)
        prev = cur
    return prev[-1]
