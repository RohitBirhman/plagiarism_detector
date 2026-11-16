"""Correctness checks. Run: python test_algorithms.py"""
import random
from pathlib import Path

from lcs import lcs, lcs_alignment, lcs_length_two_row
from edit_distance import edit_distance, edit_distance_two_row
from preprocess import preprocess, preprocess_with_spans
from similarity import lcs_similarity, edit_similarity, band
from compare import compare_texts, compare_tokens
from multi import compare_many, score_matrix
from highlight import highlighted_pair
from web_check import check_against_web, make_chunks, summarize, rescore
from evaluation import build_dataset, run_evaluation, metrics, conclusion, CATEGORIES

# --- the two algorithms (textbook examples) ---
assert lcs(list("ABCBDAB"), list("BDCABA"))[0] == 4
assert edit_distance(list("kitten"), list("sitting")) == 3
assert edit_distance([], list("abc")) == 3
assert lcs([], ["a"])[0] == 0
pairs = lcs_alignment(list("ABCBDAB"), list("BDCABA"))
assert len(pairs) == 4 and all("ABCBDAB"[i] == "BDCABA"[j] for i, j in pairs)

# --- similarity ---
a = ["x", "y", "z"]
assert lcs_similarity(lcs(a, a)[0], 3, 3) == 100.0
assert edit_similarity(edit_distance(a, a), 3, 3) == 100.0
assert edit_similarity(edit_distance(a, ["p", "q", "r"]), 3, 3) == 0.0
assert (band(5), band(25), band(60)) == ("Low", "Moderate", "High")

# --- property: Edit-Distance similarity can never exceed LCS similarity ---
rng = random.Random(1)
vocab = list("abcdefgh")
for _ in range(300):
    x = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
    y = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
    ls = lcs_similarity(lcs(x, y)[0], len(x), len(y))
    es = edit_similarity(edit_distance(x, y), len(x), len(y))
    assert es <= ls + 1e-9, (x, y, es, ls)

# --- memory-saving versions give identical results (so big documents can be scanned fully) ---
for _ in range(300):
    x = [rng.choice(vocab) for _ in range(rng.randint(0, 15))]
    y = [rng.choice(vocab) for _ in range(rng.randint(0, 15))]
    assert lcs_length_two_row(x, y) == lcs(x, y)[0]
    assert edit_distance_two_row(x, y) == edit_distance(x, y)
big_a = [rng.choice(vocab) for _ in range(120)]
big_b = [rng.choice(vocab) for _ in range(90)]
full = compare_tokens(big_a, big_b, repeats=1)
lite = compare_tokens(big_a, big_b, repeats=1, full_table_limit=0)
assert lite["memory_saving"] and not full["memory_saving"]
assert (full["lcs_score"], full["edit_score"]) == (lite["lcs_score"], lite["edit_score"])

# --- preprocessing ---
assert preprocess("The Cat, sat!") == ["cat", "sat"]
toks, spans = preprocess_with_spans("The Cat, sat!")
assert [("The Cat, sat!")[s:e] for s, e in spans] == ["Cat", "sat"]

# --- compare + empty-input guard ---
r = compare_texts("alpha beta gamma delta", "alpha beta gamma delta", 60, measure_memory=True)
assert r["lcs_score"] == 100 and r["edit_score"] == 100 and r["lcs_kb"] > 0
try:
    compare_texts("the a an", "alpha beta")
    raise SystemExit("empty document should raise")
except ValueError:
    pass

# --- multi-file ---
docs = {"a.txt": "red green blue yellow orange purple",
        "b.txt": "red green blue yellow orange purple",
        "c.txt": "cat dog horse sheep goat",
        "empty.txt": "the a an"}
res = compare_many(docs)
assert len(res["rows"]) == 3 and res["skipped"] == ["empty.txt"]
top = max(res["rows"], key=lambda r: r["lcs_score"])
assert {top["doc_a"], top["doc_b"]} == {"a.txt", "b.txt"} and top["lcs_score"] == 100
m = score_matrix(res["names"], res["rows"], "lcs_score")
assert m[0][1] == m[1][0] == 100 and m[0][0] is None

# --- highlight ---
ha, hb = highlighted_pair("The quick brown fox", "A quick red fox")
assert "<mark>quick</mark>" in ha and "<mark>fox</mark>" in hb and "brown" in ha

# --- web check with a fake search engine and fake pages (no internet needed) ---
essay = ("Dynamic programming solves complex problems by breaking them into smaller overlapping subproblems. "
         "My own sentence about monsoon rain falling over the northern plains every single year.")
page = ("Menu Home About. Dynamic programming solves complex problems by breaking them into "
        "smaller overlapping subproblems. Contact us today for more details about shipping.")
out = check_against_web(essay, 60, search=lambda q, n: ["http://example.com/p"],
                        fetch=lambda u: page, delay=0)
assert out["findings"][0]["lcs_score"] == 100 and out["findings"][0]["lcs_url"] == "http://example.com/p"
assert out["findings"][1]["lcs_score"] < 60
assert out["lcs_flagged"] == 1 and 0 < out["lcs_percent"] < 100
assert summarize(out["findings"], 101)["lcs_flagged"] == 0     # threshold re-applied without searching
# whole document is covered: short sentences are merged, none dropped, long ones split
tricky = "Hi there. Yes. This sentence is long enough to stand alone for sure. Ok. Bye now. " + "word " * 150 + "."
cov = make_chunks(tricky)
assert all(len(c.split()) <= 70 for c in cov)
assert sum(len(preprocess(c)) for c in cov) == len(preprocess(tricky))
assert len(make_chunks(" ".join(["This is a reasonably long test sentence number %d here." % i
                                 for i in range(40)]), max_chunks=10)) == 10
dead = check_against_web(essay, 60, search=lambda q, n: 1 / 0, fetch=lambda u: "", delay=0)
assert dead["errors"] and dead["lcs_flagged"] == 0

# the whole document is scanned by default, and coverage is reported
long_doc = " ".join("Unique sentence number %d talks about subject%d and topic%d in detail." % (i, i, i)
                    for i in range(25))
full_scan = check_against_web(long_doc, 60, search=lambda q, n: [], fetch=lambda u: "", delay=0)
assert full_scan["meta"]["checked_chunks"] == full_scan["meta"]["total_chunks"] == 25
assert full_scan["meta"]["coverage_percent"] == 100.0 and not full_scan["meta"]["stopped_early"]
# search engine dying: stop early and say so, instead of pretending everything was checked
broken = check_against_web(long_doc, 60, search=lambda q, n: 1 / 0, fetch=lambda u: "",
                           delay=0, retry_delay=0)
assert broken["meta"]["stopped_early"] and broken["meta"]["checked_chunks"] == 2
assert broken["meta"]["coverage_percent"] < 20
assert rescore(out, 101)["lcs_flagged"] == 0 and rescore(out, 101)["meta"]["total_chunks"] == 2

# --- evaluation ---
data = build_dataset()
assert len(data) == 6 * len(CATEGORIES)
rows = run_evaluation(repeats=1)
for algo in ("lcs", "edit"):
    mt = metrics(rows, algo, 60)
    assert mt["TP"] + mt["FP"] + mt["TN"] + mt["FN"] == len(rows)
assert any(l.startswith("Decision rule") for l in conclusion(rows, 60))

# --- Word / PDF readers (skipped if the libraries are missing) ---
try:
    from file_reader import read_file
    d = Path(__file__).parent / "data"
    t_pdf, t_docx = read_file(str(d / "original.pdf")), read_file(str(d / "lightly_edited.docx"))
    assert "dynamic programming" in t_pdf.lower() and "dynamic programming" in t_docx.lower()
except ImportError:
    print("(skipped Word/PDF test: python-docx or pypdf not installed)")

print("All tests passed")
