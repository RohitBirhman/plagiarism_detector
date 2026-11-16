# Plagiarism Detection System: LCS vs Edit Distance

DAA project by Rohit, SGT University.
A comparative study of **Longest Common Subsequence** and **Edit Distance (Levenshtein)**:
both algorithms run on the same documents, preprocessing and threshold, and we record
similarity score and execution time.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py            # web demo (best for presenting)
```
Command line alternatives:
```bash
python main.py --demo                  # built-in sample pairs
python main.py a.pdf b.docx c.txt      # compare every pair of any number of files
python main.py essay.docx --web        # check the ENTIRE file against the web (needs internet)
python main.py --eval                  # accuracy / speed on the labelled dataset
python benchmark.py                    # runtime vs document length
python test_algorithms.py              # correctness tests
```

## Modes

| Mode | What it does |
|---|---|
| Compare documents | The entire text of every document is compared (no word limit unless you set one). Upload 2 or more .txt/.docx/.pdf files. Every pair is scored by both algorithms; pairwise table, heat-map matrices, matching words highlighted, CSV export. |
| Check one file against the web | Splits the whole document into chunks (short sentences merged, long ones split, so no word is skipped), searches each on the web, downloads the result pages and scores your sentence against them with LCS and Edit Distance. Overall plagiarism % plus the source URL of every flagged sentence. |
| Evaluation | 42 labelled pairs (exact copy, light/moderate edit, reordered, partial copy, unrelated, same topic). Accuracy, precision, recall, F1, time per pair, threshold sweep and an automatic conclusion. |
| Benchmark | Runtime against document length (shows O(m x n) growth). |

## How a flag is decided

1. Score = similarity percentage (0-100) from each algorithm.
2. If score >= threshold (slider, default 60%), the pair (or sentence) is flagged.
3. LCS and Edit Distance are judged separately.
4. Web mode: overall % = share of checked words that sit in flagged sentences.
   Bands: under 20% low, 20-40% moderate, over 40% high.

Formulas: LCS similarity = LCS length / max(m, n) x 100.
Edit similarity = (1 - distance / max(m, n)) x 100. Edit similarity can never exceed LCS similarity.

## Files

| File | Purpose |
|---|---|
| file_reader.py | read .txt, .docx and .pdf into plain text |
| preprocess.py | tokenize, lowercase, strip punctuation and stop words |
| lcs.py / edit_distance.py | the two DP algorithms, O(m x n) |
| similarity.py | normalize to %, flags, bands |
| compare.py / multi.py | compare one pair / every pair |
| web_check.py | single-file web search mode |
| evaluation.py | labelled dataset, metrics, conclusion |
| highlight.py | highlight matching words |
| benchmark.py | timing across sizes |
| main.py / app.py | command line and Streamlit interfaces |
| data/ | sample documents |

## Whole-document scanning

- Comparison mode has no word limit by default. Pairs larger than 6 million table cells automatically use a two-row version of the same algorithms (identical scores, same O(m x n) time, much less memory).
- Web mode searches every chunk by default and reports coverage. If the search engine rate-limits and stops responding, the app says so instead of pretending the document was fully checked.

## Limitations

- Order-sensitive algorithms: heavy reordering and real paraphrasing lower the scores.
- Web mode only finds pages that the search engine returns (not a private database like Turnitin) and needs internet.
- Scanned PDFs need OCR (not included). Old .doc files are not supported, only .docx.
- The evaluation dataset is synthetic, so it is a controlled test bench, not a real-world benchmark.
