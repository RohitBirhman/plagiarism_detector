"""Longest Common Subsequence using dynamic programming.

Time  : O(m x n)
Space : O(m x n)  (the full table is kept so we can backtrack the actual subsequence)
"""


def _table(a: list[str], b: list[str]) -> list[list[int]]:
    m, n = len(a), len(b)
    # dp[i][j] = LCS length of a[:i] and b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def lcs_alignment(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Backtrack the table: return [(i, j), ...] with a[i] == b[j], in order."""
    dp = _table(a, b)
    pairs, i, j = [], len(a), len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def lcs(a: list[str], b: list[str]) -> tuple[int, list[str]]:
    """Return (length of LCS, the LCS tokens themselves)."""
    dp = _table(a, b)
    length = dp[len(a)][len(b)]
    # backtrack (same logic as lcs_alignment, reusing the table we already built)
    common, i, j = [], len(a), len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            common.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    common.reverse()
    return length, common


def lcs_length_two_row(a: list[str], b: list[str]) -> int:
    """Same LCS length as the full table, but keeps only two rows: O(min(m, n)) space.
    Used automatically for very large documents so the whole text can still be scanned."""
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                up, left = prev[j], cur[j - 1]
                cur.append(up if up > left else left)
        prev = cur
    return prev[-1]
