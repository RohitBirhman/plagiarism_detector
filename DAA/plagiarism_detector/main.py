"""Command-line entry point.

Usage:
    python main.py --demo                    # run the built-in sample pairs
    python main.py a.pdf b.docx c.txt        # compare every pair of the given files
    python main.py essay.docx --web          # check ONE file against the web
    python main.py --eval                    # accuracy and speed on the labelled dataset
    add -t 70 to change the threshold (%)
"""
import argparse
from pathlib import Path

from compare import compare_texts
from file_reader import read_file
from multi import compare_many
from similarity import verdict

DATA = Path(__file__).parent / "data"

DEMO_PAIRS = [
    ("original.txt", "copied.txt", "Exact copy"),
    ("original.txt", "lightly_edited.txt", "Lightly edited (synonyms)"),
    ("original.txt", "reordered.txt", "Sentences reordered"),
    ("original.txt", "unrelated.txt", "Unrelated"),
]


def show(title: str, r: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Tokens: {r['tokens_a']} vs {r['tokens_b']}")
    print(f"LCS           : length={r['lcs_length']:<4} similarity={r['lcs_score']:6.2f}%  "
          f"time={r['lcs_ms']:.3f} ms  -> {r['lcs_verdict']}")
    print(f"Edit Distance : distance={r['edit_distance']:<3} similarity={r['edit_score']:6.2f}%  "
          f"time={r['edit_ms']:.3f} ms  -> {r['edit_verdict']}")


def run_many(files, threshold):
    docs = {Path(f).name: read_file(f) for f in files}
    res = compare_many(docs)
    print(f"\n{len(res['names'])} documents, {len(res['rows'])} pairs (threshold {threshold:.0f}%)\n")
    print(f"{'Document A':<26}{'Document B':<26}{'LCS %':>8}{'Edit %':>8}   Verdict (LCS / Edit)")
    for r in sorted(res["rows"], key=lambda r: -r["lcs_score"]):
        print(f"{r['doc_a'][:25]:<26}{r['doc_b'][:25]:<26}{r['lcs_score']:>8.1f}{r['edit_score']:>8.1f}   "
              f"{verdict(r['lcs_score'], threshold)[:10]} / {verdict(r['edit_score'], threshold)[:10]}")


def run_web(path, threshold, max_sentences=None):
    from web_check import check_against_web
    print("Searching the web for the whole document, please wait (about 3-5 seconds per chunk)...")
    res = check_against_web(read_file(path), threshold, max_chunks=max_sentences,
                            progress=lambda i, n, msg: print(f"  [{i}/{n}] {msg}"))
    meta = res["meta"]
    print(f"\nScanned {meta['checked_chunks']} of {meta['total_chunks']} chunks "
          f"({meta['coverage_percent']:.0f}% of the words)")
    if meta["stopped_early"]:
        print("WARNING: the search engine stopped responding; the rest was NOT scanned.")
    print(f"LCS  : {res['lcs_percent']:.1f}% of checked text flagged ({res['lcs_band']})")
    print(f"Edit : {res['edit_percent']:.1f}% of checked text flagged ({res['edit_band']})")
    for f in res["findings"]:
        if max(f["lcs_score"], f["edit_score"]) >= threshold:
            print(f"\n- \"{f['sentence'][:90]}...\"\n  LCS {f['lcs_score']:.0f}% / Edit {f['edit_score']:.0f}%"
                  f"  source: {f['lcs_url'] or f['edit_url']}")
    for e in res["errors"][:3]:
        print("note:", e)


def run_eval(threshold):
    from evaluation import run_evaluation, metrics, conclusion
    rows = run_evaluation()
    print(f"\nLabelled dataset: {len(rows)} pairs, threshold {threshold:.0f}%\n")
    print(f"{'':<15}{'Acc':>6}{'Prec':>7}{'Recall':>8}{'F1':>6}{'ms/pair':>9}")
    for name, key in (("LCS", "lcs"), ("Edit Distance", "edit")):
        m = metrics(rows, key, threshold)
        print(f"{name:<15}{m['accuracy']:>6.2f}{m['precision']:>7.2f}{m['recall']:>8.2f}"
              f"{m['f1']:>6.2f}{m['avg_ms']:>9.3f}")
    print("\nConclusion:")
    for line in conclusion(rows, threshold):
        print(" -", line)


def main() -> None:
    p = argparse.ArgumentParser(description="Plagiarism detection: LCS vs Edit Distance")
    p.add_argument("files", nargs="*", help="files to compare (.txt, .docx, .pdf)")
    p.add_argument("--demo", action="store_true", help="run built-in sample pairs")
    p.add_argument("--web", action="store_true", help="check a single file against the web")
    p.add_argument("--eval", action="store_true", help="evaluate on the labelled dataset")
    p.add_argument("--max-sentences", type=int, default=None, help="web mode: only check N chunks")
    p.add_argument("-t", "--threshold", type=float, default=60.0, help="flag threshold in %%")
    args = p.parse_args()

    if args.demo:
        for a, b, title in DEMO_PAIRS:
            r = compare_texts((DATA / a).read_text(), (DATA / b).read_text(), args.threshold)
            show(f"{title}  [{a} vs {b}]", r)
    elif args.eval:
        run_eval(args.threshold)
    elif args.web and len(args.files) == 1:
        run_web(args.files[0], args.threshold, args.max_sentences)
    elif len(args.files) == 2:
        ta, tb = (read_file(f) for f in args.files)
        show(f"{args.files[0]} vs {args.files[1]}", compare_texts(ta, tb, args.threshold))
    elif len(args.files) > 2:
        run_many(args.files, args.threshold)
    else:
        p.error("give 2+ files, or ONE file with --web, or use --demo / --eval")


if __name__ == "__main__":
    main()
