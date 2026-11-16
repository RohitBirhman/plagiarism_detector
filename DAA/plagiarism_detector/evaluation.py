"""Objectives 4, 5 and 6: Compare detection performance, Measure cost, Conclude.

A labelled dataset of document pairs (1 = plagiarised, 0 = not) is generated from
original paragraphs. Both algorithms score the SAME pairs with the SAME preprocessing,
then we measure accuracy / precision / recall / F1 and execution time.
The data is synthetic (copying is simulated by editing words), so it is a controlled
test bench, not a claim about real-world performance.
"""
import random
import re

from compare import compare_tokens
from preprocess import preprocess

# (paragraph A, different paragraph B on the SAME topic) -- all original text
TOPICS = [
    ("Sorting algorithms arrange items in a defined order so that searching becomes faster. "
     "Merge sort divides the list into halves, sorts each half and merges them back together. "
     "Its running time grows in proportion to n log n, which makes it dependable for large inputs.",
     "Quick sort chooses a pivot element and partitions the remaining items around it. "
     "The partitions are then sorted recursively until every part is trivially small. "
     "On average it is very fast, although a poor pivot choice can slow it badly."),
    ("A graph models connections between objects using vertices and edges. "
     "Breadth first search visits all neighbours of a vertex before moving deeper into the graph. "
     "It finds the shortest path in an unweighted graph and uses a queue to track pending vertices.",
     "Depth first search follows one branch as far as possible before backing up. "
     "A stack, or recursion, remembers where to continue after a dead end. "
     "It is useful for detecting cycles and for ordering tasks that depend on each other."),
    ("A relational database stores information in tables made of rows and columns. "
     "Each table has a primary key that uniquely identifies every record. "
     "Queries written in structured query language retrieve and combine data from several tables.",
     "Normalization reduces repeated data by splitting tables into smaller related ones. "
     "Fewer duplicates mean updates are easier and mistakes are less likely. "
     "Too much splitting, however, can make queries slower because more joins are needed."),
    ("Computer networks let devices exchange data through agreed rules called protocols. "
     "The Internet Protocol gives every device an address so packets can find their destination. "
     "Routers examine these addresses and forward packets along the best available path.",
     "The Transmission Control Protocol makes communication reliable by numbering every segment. "
     "Lost segments are sent again and out of order segments are rearranged on arrival. "
     "This extra care adds delay but protects the accuracy of the received data."),
    ("An operating system manages hardware and provides services to application programs. "
     "The scheduler decides which process uses the processor at each moment. "
     "Fair scheduling keeps every program responsive and prevents one task from monopolising the machine.",
     "Virtual memory lets a program use more memory than the machine physically holds. "
     "Pages that are rarely used move to disk and return when needed. "
     "This illusion simplifies programming but heavy swapping can make the whole system crawl."),
    ("Machine learning builds models that improve from examples instead of fixed rules. "
     "A training set teaches the model, while a separate test set measures how well it generalizes. "
     "Overfitting happens when a model memorizes the training data and performs poorly on new cases.",
     "Decision trees classify data by asking a sequence of simple questions. "
     "Each branch narrows the possibilities until a final label is reached. "
     "They are easy to explain, though a single deep tree may fit noise in the data."),
]

FILLER = ["system", "method", "result", "approach", "value", "process", "example",
          "number", "form", "level", "part", "case", "way", "point", "kind", "type",
          "group", "order", "area", "idea"]

# category -> ground-truth label (1 = plagiarised)
CATEGORIES = {
    "exact_copy": 1, "light_edit": 1, "moderate_edit": 1, "reordered": 1,
    "partial_copy": 1, "unrelated": 0, "same_topic_different": 0,
}

DESCRIPTIONS = {
    "exact_copy": "identical text",
    "light_edit": "about 10% of words replaced",
    "moderate_edit": "about 25% of words replaced",
    "reordered": "same sentences, first one moved to the end",
    "partial_copy": "two copied sentences plus one unrelated sentence",
    "unrelated": "paragraph from a different topic",
    "same_topic_different": "different paragraph on the same topic",
}


def _sentences(p: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", p.strip())


def _replace_words(text: str, ratio: float, rng: random.Random) -> str:
    return " ".join(rng.choice(FILLER) if rng.random() < ratio else w for w in text.split())


def build_dataset(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    n = len(TOPICS)
    pairs = []
    for i, (a, b) in enumerate(TOPICS):
        s = _sentences(a)
        other = TOPICS[(i + 3) % n]
        variants = {
            "exact_copy": a,
            "light_edit": _replace_words(a, 0.10, rng),
            "moderate_edit": _replace_words(a, 0.25, rng),
            "reordered": " ".join(s[1:] + s[:1]),
            "partial_copy": " ".join(s[:2] + [_sentences(other[0])[-1]]),
            "unrelated": other[0],
            "same_topic_different": b,
        }
        for cat, text_b in variants.items():
            pairs.append({"category": cat, "label": CATEGORIES[cat], "text_a": a, "text_b": text_b})
    return pairs


def run_evaluation(remove_stop_words: bool = True, repeats: int = 3, progress=None) -> list[dict]:
    data = build_dataset()
    rows = []
    for k, p in enumerate(data, start=1):
        r = compare_tokens(preprocess(p["text_a"], remove_stop_words),
                           preprocess(p["text_b"], remove_stop_words), repeats=repeats)
        rows.append({"category": p["category"], "label": p["label"],
                     "lcs_score": r["lcs_score"], "edit_score": r["edit_score"],
                     "lcs_ms": r["lcs_ms"], "edit_ms": r["edit_ms"]})
        if progress:
            progress(k, len(data), p["category"])
    return rows


def metrics(rows: list[dict], algo: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        pred = r[f"{algo}_score"] >= threshold
        if pred and r["label"]:
            tp += 1
        elif pred and not r["label"]:
            fp += 1
        elif not pred and r["label"]:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": (tp + tn) / len(rows), "precision": prec, "recall": rec, "f1": f1,
            "avg_ms": sum(r[f"{algo}_ms"] for r in rows) / len(rows)}


def sweep(rows: list[dict], thresholds=range(20, 100, 5)) -> list[dict]:
    return [{"threshold": t,
             "LCS F1": metrics(rows, "lcs", t)["f1"],
             "Edit Distance F1": metrics(rows, "edit", t)["f1"]} for t in thresholds]


def category_summary(rows: list[dict]) -> list[dict]:
    out = []
    for cat in CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        out.append({"category": cat, "what it is": DESCRIPTIONS[cat],
                    "ground truth": "plagiarised" if CATEGORIES[cat] else "not plagiarised",
                    "avg LCS %": sum(r["lcs_score"] for r in sub) / len(sub),
                    "avg Edit %": sum(r["edit_score"] for r in sub) / len(sub)})
    return out


def conclusion(rows: list[dict], threshold: float) -> list[str]:
    """Objective 6: pick the better balance of detection quality and speed (rule is explicit)."""
    ml, me = metrics(rows, "lcs", threshold), metrics(rows, "edit", threshold)
    lines = [f"At a {threshold:.0f}% threshold, LCS reached F1 = {ml['f1']:.2f} "
             f"(recall {ml['recall']:.2f}, precision {ml['precision']:.2f}) and Edit Distance reached "
             f"F1 = {me['f1']:.2f} (recall {me['recall']:.2f}, precision {me['precision']:.2f})."]
    fast, slow = ("LCS", "Edit Distance") if ml["avg_ms"] <= me["avg_ms"] else ("Edit Distance", "LCS")
    t_fast = min(ml["avg_ms"], me["avg_ms"])
    t_slow = max(ml["avg_ms"], me["avg_ms"])
    ratio = t_slow / t_fast if t_fast else float("inf")
    lines.append(f"{fast} was faster, taking {t_fast:.3f} ms per pair against {t_slow:.3f} ms "
                 f"for {slow} ({ratio:.2f}x).")
    if abs(ml["f1"] - me["f1"]) < 0.05:
        better = fast
        lines.append(f"Detection quality is within 0.05 F1, so the faster algorithm, {better}, "
                     f"gives the better balance.")
    else:
        better = "LCS" if ml["f1"] > me["f1"] else "Edit Distance"
        lines.append(f"{better} detects noticeably better (F1 gap of {abs(ml['f1'] - me['f1']):.2f}), "
                     f"so it gives the better balance despite any difference in speed.")
    lines.append("Decision rule: if the F1 scores differ by less than 0.05, prefer the faster "
                 "algorithm; otherwise prefer the one with the higher F1.")
    return lines
