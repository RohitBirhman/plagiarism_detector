"""Show the original text with the words that belong to the LCS highlighted."""
import html

from preprocess import preprocess_with_spans
from lcs import lcs_alignment


MAX_HIGHLIGHT_CELLS = 25_000_000   # highlighting needs the full table


def highlighted_pair(text_a: str, text_b: str, remove_stop_words: bool = True,
                     max_tokens: int | None = None):
    """Return (html_a, html_b), or (None, None) if the pair is too large to highlight."""
    ta, sa = preprocess_with_spans(text_a, remove_stop_words)
    tb, sb = preprocess_with_spans(text_b, remove_stop_words)
    if max_tokens:
        ta, sa, tb, sb = ta[:max_tokens], sa[:max_tokens], tb[:max_tokens], sb[:max_tokens]
    if len(ta) * len(tb) > MAX_HIGHLIGHT_CELLS:
        return None, None
    pairs = lcs_alignment(ta, tb)
    hit_a = {i for i, _ in pairs}
    hit_b = {j for _, j in pairs}
    return _render(text_a, sa, hit_a, max_tokens is not None and len(ta) == max_tokens), \
           _render(text_b, sb, hit_b, max_tokens is not None and len(tb) == max_tokens)


def _render(text: str, spans, hits: set[int], cut: bool) -> str:
    out, pos = [], 0
    last_end = spans[-1][1] if (cut and spans) else len(text)
    for k, (s, e) in enumerate(spans):
        out.append(html.escape(text[pos:s]))
        word = html.escape(text[s:e])
        out.append(f"<mark>{word}</mark>" if k in hits else word)
        pos = e
    out.append(html.escape(text[pos:last_end]))
    body = "".join(out).replace("\n", "<br>")
    return ("<div style='border:1px solid rgba(128,128,128,.4);border-radius:6px;"
            "padding:10px;max-height:320px;overflow:auto;line-height:1.6'>" + body + "</div>")
