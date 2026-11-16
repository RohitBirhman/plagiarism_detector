"""Single-file mode: check ONE document against the web, scanning the ENTIRE document.

Pipeline (the same two-stage idea real plagiarism checkers use):
  1. Split the whole document into chunks (sentences; very short ones are merged with
     their neighbours and very long ones are split, so every word is covered).
  2. Search the web for each chunk              -> candidate source pages.
  3. Download each page and split it into sentences.
  4. Compare chunk vs page sentences with LCS and with Edit Distance (our DAA part).
  5. Keep the best match per chunk; flag it if the score reaches the threshold.

The search engine only supplies candidates; the similarity scores come from the
two algorithms, so the LCS vs Edit Distance comparison still holds.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

from preprocess import preprocess
from lcs import lcs
from edit_distance import edit_distance
from similarity import lcs_similarity, edit_similarity, band

UA = {"User-Agent": "Mozilla/5.0 (compatible; StudentPlagiarismChecker/1.0)"}
LONG_WORDS = 70      # a piece longer than this many words is cut up ...
PIECE_WORDS = 40     # ... into pieces of this many words
MAX_FAILS_IN_A_ROW = 3


# ---------- text helpers ----------
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def _split_long(s: str) -> list[str]:
    words = s.split()
    if len(words) <= LONG_WORDS:
        return [s]
    return [" ".join(words[i:i + PIECE_WORDS]) for i in range(0, len(words), PIECE_WORDS)]


def make_chunks(text: str, remove_stop_words: bool = True, min_tokens: int = 6,
                max_chunks: int | None = None) -> list[str]:
    """Cover the whole text: merge short sentences, split very long ones.
    If max_chunks is given and there are more chunks, pick evenly spread ones."""
    chunks, buf, buf_n = [], "", 0
    for s in split_sentences(text):
        n = len(preprocess(s, remove_stop_words))
        if n == 0:
            continue
        buf = f"{buf} {s}".strip()
        buf_n += n
        if buf_n >= min_tokens:
            chunks.append(buf)
            buf, buf_n = "", 0
    if buf:                                    # leftover short tail: attach it, don't drop it
        if chunks:
            chunks[-1] += " " + buf
        else:
            chunks.append(buf)
    chunks = [piece for c in chunks for piece in _split_long(c)]
    if max_chunks and len(chunks) > max_chunks:
        step = (len(chunks) - 1) / (max_chunks - 1)
        chunks = [chunks[round(i * step)] for i in range(max_chunks)]
    return chunks


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------- search back-ends ----------
def search_duckduckgo(query: str, n: int = 4) -> list[str]:
    try:
        from ddgs import DDGS                      # pip install ddgs
    except ImportError:
        from duckduckgo_search import DDGS         # older package name
    results = DDGS().text(query, max_results=n)
    return [r["href"] for r in results if r.get("href")]


def make_google_search(api_key: str, cx: str):
    """Google Custom Search JSON API (needs your own key and search-engine id)."""
    def search(query: str, n: int = 4) -> list[str]:
        import requests
        r = requests.get("https://www.googleapis.com/customsearch/v1",
                         params={"key": api_key, "cx": cx, "q": query, "num": min(n, 10)},
                         timeout=10)
        r.raise_for_status()
        return [it["link"] for it in r.json().get("items", []) if it.get("link")]
    return search


def fetch_page_text(url: str, timeout: int = 8) -> str:
    """Download a web page and return its visible text ('' for non-HTML)."""
    import requests
    from bs4 import BeautifulSoup                  # pip install beautifulsoup4
    r = requests.get(url, headers=UA, timeout=timeout)
    if "text/html" not in r.headers.get("Content-Type", ""):
        return ""
    soup = BeautifulSoup(r.text[:600_000], "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    return soup.get_text(" ")


def _search_with_retry(search, query, n, retry_delay):
    try:
        return search(query, n)
    except Exception:
        if retry_delay:
            time.sleep(retry_delay)                # rate limits often clear after a pause
        return search(query, n)


def _page_pieces(body: str, remove_stop_words: bool):
    out = []
    for s in split_sentences(body):
        for piece in _split_long(s):
            t = preprocess(piece, remove_stop_words)
            if t:
                out.append((piece, t, set(t)))
    return out


# ---------- main routine ----------
def check_against_web(text: str, threshold: float = 60.0, search=search_duckduckgo,
                      fetch=fetch_page_text, max_chunks: int | None = None,
                      results_per_query: int = 4, remove_stop_words: bool = True,
                      delay: float = 1.0, retry_delay: float = 3.0, workers: int = 4,
                      progress=None) -> dict:
    """max_chunks=None (default) scans the entire document."""
    all_chunks = make_chunks(text, remove_stop_words)
    chunks = make_chunks(text, remove_stop_words, max_chunks=max_chunks)
    doc_tokens = len(preprocess(text, remove_stop_words))
    pages: dict[str, list] = {}
    errors: list[str] = []
    findings = []
    lcs_ms_total = edit_ms_total = 0.0
    fails_in_a_row = 0
    stopped_early = False

    for idx, chunk in enumerate(chunks, start=1):
        if progress:
            progress(idx, len(chunks), chunk[:70])
        ctoks = preprocess(chunk, remove_stop_words)
        cset = set(ctoks)
        query = " ".join(chunk.split()[:15])

        try:
            urls = list(dict.fromkeys(_search_with_retry(search, query, results_per_query, retry_delay)))
            fails_in_a_row = 0
        except Exception as e:                         # network, rate limit, no library...
            errors.append(f"Search failed: {type(e).__name__}: {e}")
            urls = []
            fails_in_a_row += 1
            if fails_in_a_row >= MAX_FAILS_IN_A_ROW:
                stopped_early = True
                break
        if idx < len(chunks) and delay:
            time.sleep(delay)

        new = [u for u in urls if u not in pages]
        if new:                                        # download the new pages in parallel
            def grab(u):
                try:
                    return u, fetch(u), None
                except Exception as e:
                    return u, "", f"Could not read {u}: {type(e).__name__}"
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for u, body, err in ex.map(grab, new):
                    if err:
                        errors.append(err)
                    pages[u] = _page_pieces(body, remove_stop_words)

        best = {"lcs": (0.0, "", ""), "edit": (0.0, "", "")}   # (score, url, matched text)
        for url in urls:
            for sent, stoks, sset in pages.get(url, []):
                if _jaccard(cset, sset) < 0.2:         # cheap filter: skip hopeless pairs
                    continue
                t0 = time.perf_counter()
                l_len, _ = lcs(ctoks, stoks)
                t1 = time.perf_counter()
                dist = edit_distance(ctoks, stoks)
                t2 = time.perf_counter()
                lcs_ms_total += (t1 - t0) * 1000
                edit_ms_total += (t2 - t1) * 1000
                ls = lcs_similarity(l_len, len(ctoks), len(stoks))
                es = edit_similarity(dist, len(ctoks), len(stoks))
                if ls > best["lcs"][0]:
                    best["lcs"] = (ls, url, sent)
                if es > best["edit"][0]:
                    best["edit"] = (es, url, sent)

        findings.append({
            "sentence": chunk, "tokens": len(ctoks),
            "lcs_score": best["lcs"][0], "lcs_url": best["lcs"][1], "lcs_match": best["lcs"][2],
            "edit_score": best["edit"][0], "edit_url": best["edit"][1], "edit_match": best["edit"][2],
        })

    meta = {"total_chunks": len(all_chunks), "checked_chunks": len(findings),
            "doc_tokens": doc_tokens, "stopped_early": stopped_early}
    return summarize(findings, threshold, errors, lcs_ms_total, edit_ms_total, len(pages), meta)


def summarize(findings: list[dict], threshold: float, errors=None, lcs_ms: float = 0.0,
              edit_ms: float = 0.0, pages: int = 0, meta: dict | None = None) -> dict:
    """Overall % = share of checked words that sit in flagged sentences.
    Recomputed from stored scores, so moving the threshold slider needs no new search."""
    total = sum(f["tokens"] for f in findings) or 1
    meta = dict(meta or {})
    doc_tokens = meta.get("doc_tokens") or total
    meta["coverage_percent"] = min(100.0, 100.0 * sum(f["tokens"] for f in findings) / doc_tokens)
    out = {"findings": findings, "errors": errors or [], "lcs_ms": lcs_ms,
           "edit_ms": edit_ms, "pages_read": pages, "meta": meta}
    for algo in ("lcs", "edit"):
        flagged = [f for f in findings if f[f"{algo}_score"] >= threshold]
        pct = 100.0 * sum(f["tokens"] for f in flagged) / total
        out[f"{algo}_flagged"] = len(flagged)
        out[f"{algo}_percent"] = pct
        out[f"{algo}_band"] = band(pct)
    return out


def rescore(out: dict, threshold: float) -> dict:
    """Re-apply a new threshold to a finished check (no new searching)."""
    return summarize(out["findings"], threshold, out["errors"], out["lcs_ms"],
                     out["edit_ms"], out["pages_read"], out["meta"])
