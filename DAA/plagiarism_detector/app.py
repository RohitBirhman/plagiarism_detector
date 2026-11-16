"""Streamlit demo UI.  Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmark import run_benchmark
from evaluation import (CATEGORIES, category_summary, conclusion, metrics,
                        run_evaluation, sweep)
from file_reader import read_bytes
from highlight import highlighted_pair
from multi import compare_many, score_matrix
from preprocess import preprocess
from similarity import is_flagged, verdict
import web_check

DATA = Path(__file__).parent / "data"
SAMPLE_FILES = ["original.txt", "copied.txt", "lightly_edited.txt", "reordered.txt", "unrelated.txt"]
TYPES = ["txt", "docx", "pdf"]

st.set_page_config(page_title="Plagiarism Detection: LCS vs Edit Distance", layout="wide")
st.title("Plagiarism Detection System")
st.caption("A comparative study of Longest Common Subsequence and Edit Distance "
           "| Rohit, SGT University")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Plagiarism threshold (%)", 10, 100, 60, 5,
                          help="A pair or sentence is flagged when its similarity reaches this value.")
    remove_stop = st.checkbox("Remove stop words", value=True)
    limit_on = st.checkbox("Limit words per document (faster)", value=False,
                           help="Off = the ENTIRE document is compared. Turn on only to speed up "
                                "very long documents; the words after the limit are then NOT scanned.")
    max_tokens = st.slider("Compare only the first N words", 200, 20000, 3000, 100) if limit_on else None
    st.markdown("---")
    st.markdown("**Pipeline**\n\n1. Input documents\n2. Preprocess\n3. Encode tokens\n"
                "4. LCS (DP)\n5. Edit Distance (DP)\n6. Score and flag")


# ---------------- helpers ----------------
def read_uploads(files):
    docs, errors = {}, []
    for f in files:
        try:
            text = read_bytes(f.getvalue(), f.name)
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            continue
        name, k = f.name, 2
        while name in docs:                       # keep duplicate file names apart
            name, k = f"{f.name} ({k})", k + 1
        docs[name] = text
    return docs, errors


def load_samples():
    return {n: (DATA / n).read_text() for n in SAMPLE_FILES}


def shade(v):
    """Heat-map colour: white (0%) -> red (100%)."""
    if v is None or pd.isna(v):
        return "background-color: #555; color: #ddd"
    g = int(255 - (max(0.0, min(100.0, float(v))) / 100) * 155)
    return f"background-color: rgb(255,{g},{g}); color: black"


def matrix_frame(names, rows, key):
    mat = score_matrix(names, rows, key)
    df = pd.DataFrame([[float("nan") if v is None else v for v in r] for r in mat],
                      index=names, columns=names)
    styler = df.style.format("{:.1f}", na_rep="-")
    return (getattr(styler, "map", None) or styler.applymap)(shade)


def csv_button(df, label, filename, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=key)


# ---------------- tabs ----------------
tab_cmp, tab_web, tab_eval, tab_bench, tab_about = st.tabs(
    ["Compare documents", "Check one file against the web", "Evaluation", "Benchmark", "About"])

# =========================================================
# Tab 1: compare many documents
# =========================================================
with tab_cmp:
    st.markdown("Upload **two or more** documents (`.txt`, `.docx`, `.pdf`, mixed is fine). "
                "Every pair is compared with both algorithms.")
    files = st.file_uploader("Upload documents", type=TYPES, accept_multiple_files=True, key="multi_up")
    use_samples = st.checkbox("Use the 5 built-in sample documents instead")

    docs, errs = ({}, [])
    if use_samples:
        docs = load_samples()
    elif files:
        docs, errs = read_uploads(files)
    for e in errs:
        st.error(f"Could not read {e}")

    if len(docs) < 2:
        st.info("Add at least two documents to start.")
    else:
        n = len(docs)
        st.write(f"**{n} documents ready**, which makes **{n * (n - 1) // 2} pairs** to compare. "
                 "The entire text of every document is scanned.")
        lens = [len(preprocess(t, remove_stop)) for t in docs.values()]
        if max_tokens:
            lens = [min(x, max_tokens) for x in lens]
        cells = sum(lens[i] * lens[j] for i in range(n) for j in range(i + 1, n))
        est = cells * 0.5e-6
        st.caption(f"Rough time estimate: {est:.0f} seconds ({sum(lens):,} words in total; "
                   "the time grows with the product of the two document lengths).")
        if est > 180:
            st.warning("This may take several minutes. Very large pairs automatically use a "
                       "memory-saving version of the same algorithms (identical scores).")
        if st.button("Run comparison", type="primary"):
            bar = st.progress(0.0, text="Starting...")

            def cb(i, total, msg):
                bar.progress(i / total, text=f"Pair {i} of {total}: {msg}")

            res = compare_many(docs, remove_stop, max_tokens, cb)
            bar.empty()
            st.session_state["multi"] = {"res": res, "docs": docs,
                                         "remove_stop": remove_stop, "max_tokens": max_tokens}

    saved = st.session_state.get("multi")
    if saved:
        res, sdocs = saved["res"], saved["docs"]
        rows = res["rows"]
        st.markdown("---")
        st.subheader("Results")
        st.caption("These are the results of the last run. The threshold slider updates the flags instantly; "
                   "other settings need a new run.")
        if res["skipped"]:
            st.warning("Skipped (no readable words, possibly a scanned PDF): " + ", ".join(res["skipped"]))
        if res["truncated"]:
            st.warning(f"Only the first {saved['max_tokens']} words were compared for: "
                       + ", ".join(res["truncated"])
                       + ". Untick the word limit in the sidebar to scan everything.")
        if any(r.get("memory_saving") for r in rows):
            st.caption("Large pairs used the memory-saving version of the algorithms (identical scores, "
                       "same O(m x n) time). Their timings are not directly comparable with small pairs.")
        if not rows:
            st.info("Fewer than two documents had readable text.")
        else:
            lcs_flags = sum(is_flagged(r["lcs_score"], threshold) for r in rows)
            ed_flags = sum(is_flagged(r["edit_score"], threshold) for r in rows)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documents", len(res["names"]))
            c2.metric("Pairs compared", len(rows))
            c3.metric("Flagged by LCS", lcs_flags)
            c4.metric("Flagged by Edit Distance", ed_flags)

            table = pd.DataFrame([{
                "Document A": r["doc_a"], "Document B": r["doc_b"],
                "LCS %": round(r["lcs_score"], 1), "Edit Distance %": round(r["edit_score"], 1),
                "LCS verdict": "FLAGGED" if is_flagged(r["lcs_score"], threshold) else "ok",
                "Edit verdict": "FLAGGED" if is_flagged(r["edit_score"], threshold) else "ok",
                "LCS ms": round(r["lcs_ms"], 3), "Edit ms": round(r["edit_ms"], 3),
            } for r in rows]).sort_values("LCS %", ascending=False)
            st.dataframe(table, use_container_width=True, hide_index=True)
            csv_button(table, "Download results (CSV)", "pair_results.csv", "dl_multi")

            if len(res["names"]) > 2:
                m1, m2 = st.columns(2)
                m1.markdown("**LCS similarity matrix (%)**")
                m1.dataframe(matrix_frame(res["names"], rows, "lcs_score"), use_container_width=True)
                m2.markdown("**Edit Distance similarity matrix (%)**")
                m2.dataframe(matrix_frame(res["names"], rows, "edit_score"), use_container_width=True)

            st.markdown("### Inspect a pair")
            ordered = sorted(rows, key=lambda r: -r["lcs_score"])
            labels = [f"{r['doc_a']}  vs  {r['doc_b']}" for r in ordered]
            pick = ordered[labels.index(st.selectbox("Choose a pair", labels))]

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("#### Longest Common Subsequence")
                st.metric("Similarity", f"{pick['lcs_score']:.2f}%")
                st.write(f"LCS length: **{pick['lcs_length']}** words | time: **{pick['lcs_ms']:.3f} ms**")
                (st.error if is_flagged(pick["lcs_score"], threshold) else st.success)(
                    verdict(pick["lcs_score"], threshold))
            with d2:
                st.markdown("#### Edit Distance (Levenshtein)")
                st.metric("Similarity", f"{pick['edit_score']:.2f}%")
                st.write(f"Edit distance: **{pick['edit_distance']}** operations | "
                         f"time: **{pick['edit_ms']:.3f} ms**")
                (st.error if is_flagged(pick["edit_score"], threshold) else st.success)(
                    verdict(pick["edit_score"], threshold))

            g1, g2 = st.columns(2)
            g1.markdown("**Similarity score (%)**")
            g1.bar_chart(pd.DataFrame({"Similarity (%)": [pick["lcs_score"], pick["edit_score"]]},
                                      index=["LCS", "Edit Distance"]))
            g2.markdown("**Execution time (ms)**")
            g2.bar_chart(pd.DataFrame({"Time (ms)": [pick["lcs_ms"], pick["edit_ms"]]},
                                      index=["LCS", "Edit Distance"]))

            st.markdown("**Matching words highlighted** (the longest common subsequence)")
            ha, hb = highlighted_pair(sdocs[pick["doc_a"]], sdocs[pick["doc_b"]],
                                      saved["remove_stop"], saved["max_tokens"])
            if ha is None:
                st.info("This pair is too large to highlight (it would need too much memory). "
                        "The scores above still cover the entire documents.")
            else:
                h1, h2 = st.columns(2)
                h1.caption(pick["doc_a"])
                h1.markdown(ha, unsafe_allow_html=True)
                h2.caption(pick["doc_b"])
                h2.markdown(hb, unsafe_allow_html=True)

# =========================================================
# Tab 2: one file against the web
# =========================================================
with tab_web:
    st.markdown("Upload **one** document (or paste text). Each sentence is searched on the web, the "
                "matching pages are downloaded, and **LCS and Edit Distance** measure how close your "
                "sentence is to what was found.")
    st.caption("It can only find sources that a search engine returns. It needs an internet connection.")

    w_file = st.file_uploader("Upload one document", type=TYPES, key="web_up")
    w_text = st.text_area("...or paste text here", height=140, key="web_text")

    with st.expander("Search settings"):
        engine = st.radio("Search engine", ["DuckDuckGo (no key needed)", "Google Custom Search (needs key)"])
        g_key = g_cx = ""
        if engine.startswith("Google"):
            g_key = st.text_input("Google API key", type="password")
            g_cx = st.text_input("Search engine ID (cx)")
        whole = st.checkbox("Scan the entire document", value=True,
                            help="Every sentence is searched. Untick to check only an evenly spread sample.")
        max_chunks = None if whole else st.slider("Sentences to check", 5, 100, 20)
        per_query = st.slider("Pages to read per sentence", 2, 8, 4)

    text, err = "", ""
    if w_file is not None:
        try:
            text = read_bytes(w_file.getvalue(), w_file.name)
        except Exception as e:
            err = f"Could not read {w_file.name}: {e}"
    else:
        text = w_text
    if text.strip():
        n_all = len(web_check.make_chunks(text, remove_stop))
        n_run = min(n_all, max_chunks) if max_chunks else n_all
        st.caption(f"This document splits into **{n_all} chunks**; **{n_run}** will be searched. "
                   f"Rough time: {n_run * 3 / 60:.1f} to {n_run * 5 / 60:.1f} minutes "
                   "(about 3-5 seconds per chunk, depending on your connection).")

    if st.button("Check against the web", type="primary"):
        if err:
            st.error(err)
        elif not text.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            try:
                if engine.startswith("Google"):
                    if not (g_key and g_cx):
                        raise ValueError("Enter both the Google API key and the search engine ID.")
                    search = web_check.make_google_search(g_key, g_cx)
                else:
                    search = web_check.search_duckduckgo
                bar = st.progress(0.0, text="Starting...")

                def wcb(i, total, msg):
                    bar.progress(i / total, text=f"Sentence {i} of {total}: {msg}...")

                out = web_check.check_against_web(
                    text, threshold, search=search, max_chunks=max_chunks,
                    results_per_query=per_query, remove_stop_words=remove_stop, progress=wcb)
                bar.empty()
                st.session_state["web"] = out
            except ImportError as e:
                st.error(f"A required library is missing ({e}). Run: pip install -r requirements.txt")
            except Exception as e:
                st.error(f"The check could not run: {e}")

    out = st.session_state.get("web")
    if out:
        out = web_check.rescore(out, threshold)      # new threshold, no new searching
        meta = out["meta"]
        st.markdown("---")
        st.subheader("Web check results")
        if not out["findings"]:
            st.warning("No sentences long enough to check. Try a longer text.")
        else:
            if out["errors"] and out["pages_read"] == 0:
                st.error("No web pages could be read, so these results mean nothing. Check your "
                         "internet connection or search settings. First problem: " + out["errors"][0])
            st.write(f"**Scanned {meta['checked_chunks']} of {meta['total_chunks']} chunks "
                     f"({meta['coverage_percent']:.0f}% of the document's words).**")
            if meta["stopped_early"]:
                st.error("The search engine stopped responding (most likely a rate limit), so the check "
                         "stopped early and the rest of the document was NOT scanned. Wait a few minutes "
                         "and run it again, or switch to Google Custom Search.")
            elif meta["checked_chunks"] < meta["total_chunks"]:
                st.warning("Only a sample of the document was checked. Tick 'Scan the entire document' "
                           "in the search settings to check everything.")
            c1, c2 = st.columns(2)
            c1.metric("Plagiarism by LCS", f"{out['lcs_percent']:.1f}%",
                      f"{out['lcs_band']} | {out['lcs_flagged']} sentences flagged", delta_color="off")
            c2.metric("Plagiarism by Edit Distance", f"{out['edit_percent']:.1f}%",
                      f"{out['edit_band']} | {out['edit_flagged']} sentences flagged", delta_color="off")
            st.caption(f"Share of the checked words that sit in flagged sentences. Bands: under 20% low, "
                       f"20-40% moderate, over 40% high. Pages read: {out['pages_read']}. "
                       f"DP time spent: LCS {out['lcs_ms']:.1f} ms, Edit Distance {out['edit_ms']:.1f} ms.")

            frame = pd.DataFrame([{
                "Sentence": f["sentence"],
                "LCS %": round(f["lcs_score"], 1), "Edit %": round(f["edit_score"], 1),
                "Flagged": "YES" if max(f["lcs_score"], f["edit_score"]) >= threshold else "no",
                "Best source": f["lcs_url"] if f["lcs_score"] >= f["edit_score"] else f["edit_url"],
                "Matching text on that page": f["lcs_match"] if f["lcs_score"] >= f["edit_score"]
                else f["edit_match"],
            } for f in out["findings"]])
            st.markdown("**Flagged sentences**")
            flagged = frame[frame["Flagged"] == "YES"]
            if flagged.empty:
                st.success("No sentence reached the threshold.")
            else:
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            with st.expander("All checked sentences"):
                st.dataframe(frame, use_container_width=True, hide_index=True)
            csv_button(frame, "Download report (CSV)", "web_check_report.csv", "dl_web")
            if out["errors"]:
                with st.expander(f"{len(out['errors'])} notes (search or download problems)"):
                    for e in out["errors"][:20]:
                        st.write("- " + e)

# =========================================================
# Tab 3: evaluation (objectives 4, 5, 6)
# =========================================================
with tab_eval:
    st.markdown("Both algorithms score the **same 42 labelled document pairs** with the same preprocessing "
                "and the same threshold. We then measure detection quality and execution time and draw "
                "a conclusion.")
    st.caption("The pairs are generated from original paragraphs by simulating copying (word replacement, "
               "reordering, partial copying). It is a controlled test, not a real-world benchmark.")

    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Evaluating...")
        st.session_state["eval"] = run_evaluation(
            remove_stop, 3, lambda i, t, m: bar.progress(i / t, text=f"Pair {i} of {t}: {m}"))
        bar.empty()

    erows = st.session_state.get("eval")
    if erows:
        ml, me = metrics(erows, "lcs", threshold), metrics(erows, "edit", threshold)
        pos = sum(r["label"] for r in erows)
        st.write(f"**{len(erows)} pairs**: {pos} plagiarised, {len(erows) - pos} not plagiarised "
                 f"(threshold {threshold}%).")

        perf = pd.DataFrame({
            "LCS": [ml["accuracy"], ml["precision"], ml["recall"], ml["f1"], ml["avg_ms"],
                    ml["TP"], ml["FP"], ml["TN"], ml["FN"]],
            "Edit Distance": [me["accuracy"], me["precision"], me["recall"], me["f1"], me["avg_ms"],
                              me["TP"], me["FP"], me["TN"], me["FN"]],
        }, index=["Accuracy", "Precision", "Recall", "F1 score", "Avg time per pair (ms)",
                  "True positives", "False positives", "True negatives", "False negatives"])
        st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)

        q1, q2 = st.columns(2)
        q1.markdown("**Detection quality**")
        q1.bar_chart(pd.DataFrame({"LCS": [ml["accuracy"], ml["precision"], ml["recall"], ml["f1"]],
                                   "Edit Distance": [me["accuracy"], me["precision"], me["recall"], me["f1"]]},
                                  index=["Accuracy", "Precision", "Recall", "F1"]))
        q2.markdown("**Average time per pair (ms)**")
        q2.bar_chart(pd.DataFrame({"Time (ms)": [ml["avg_ms"], me["avg_ms"]]},
                                  index=["LCS", "Edit Distance"]))

        st.markdown("### Conclusion")
        for line in conclusion(erows, threshold):
            st.write("- " + line)

        st.markdown("### Average score by kind of copying")
        cats = pd.DataFrame(category_summary(erows))
        st.dataframe(cats.style.format({"avg LCS %": "{:.1f}", "avg Edit %": "{:.1f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Notice that the Edit Distance score is never higher than the LCS score for the same "
                   "pair, so at the same threshold LCS flags everything Edit Distance flags, and sometimes more.")

        st.markdown("### Effect of the threshold (F1 score)")
        sw = pd.DataFrame(sweep(erows)).set_index("threshold")
        st.line_chart(sw)
        best = sw.idxmax()
        st.write(f"Best threshold for LCS: **{best['LCS F1']}%** | "
                 f"for Edit Distance: **{best['Edit Distance F1']}%**")

        detail = pd.DataFrame(erows)
        csv_button(detail, "Download per-pair results (CSV)", "evaluation_pairs.csv", "dl_eval")

# =========================================================
# Tab 4: benchmark
# =========================================================
with tab_bench:
    st.markdown("Both algorithms fill an **m x n** table, so running time grows roughly with the "
                "**square** of the document length, which is O(m x n).")
    top = st.slider("Largest document size to test (words)", 200, 1600, 800, 200)
    if st.button("Run benchmark"):
        sizes = [s for s in (50, 100, 200, 400, 800, 1200, 1600) if s <= top]
        with st.spinner("Timing both algorithms..."):
            st.session_state["bench"] = pd.DataFrame(run_benchmark(sizes)).set_index("tokens")
    if "bench" in st.session_state:
        st.line_chart(st.session_state["bench"])
        st.dataframe(st.session_state["bench"], use_container_width=True)

# =========================================================
# Tab 5: about
# =========================================================
with tab_about:
    st.markdown("""
### What this project is about
Digital text is easy to copy and manual checking does not scale. This project builds a plagiarism
detector from two classical **dynamic programming** algorithms and studies how they compare:

- **Longest Common Subsequence (LCS)** measures how much content two documents share, in the same order.
- **Edit Distance (Levenshtein)** counts the insertions, deletions and substitutions needed to turn one document into the other.

Both are run on the **same** documents, preprocessing and threshold, and the project records
**similarity score** and **execution time** for each. That controlled comparison is the research gap
identified in the literature review.

### Research objectives and where to find them
""")
    st.table(pd.DataFrame([
        ["1. Design", "A detector using LCS and Edit Distance", "Compare documents / Check against the web"],
        ["2. Preprocess", "Consistent tokenization, lowercasing, cleaning", "preprocess.py, used by every mode"],
        ["3. Calculate", "Normalized similarity scores from both algorithms", "Scores in every results table"],
        ["4. Compare", "Detection performance on the same pairs", "Evaluation tab (accuracy, precision, recall, F1)"],
        ["5. Measure", "Execution time and cost", "ms per pair, Evaluation and Benchmark tabs"],
        ["6. Conclude", "Better balance of detection and efficiency", "Conclusion in the Evaluation tab"],
    ], columns=["Objective", "Goal", "Where in the app"]))
    st.markdown("""
### Limits to be aware of
- Both algorithms are order-sensitive, so heavy reordering or true paraphrasing lowers the scores.
- The web check only finds pages a search engine returns; it is not a private database like commercial checkers.
- Scanned PDFs have no text to read (OCR would be needed).
""")
