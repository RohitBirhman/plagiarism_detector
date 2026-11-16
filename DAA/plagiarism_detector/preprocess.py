"""Steps 1-3 of the methodology: clean the text and turn it into a token sequence.

Tokenize -> lowercase -> strip punctuation -> (optionally) drop stop words.
`preprocess_with_spans` also remembers where each token sat in the original text,
which lets the app highlight matching words in the documents.
"""
import re

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "i", "his", "her", "their", "our", "your", "my", "so", "than", "then", "there",
    "which", "who", "whom", "will", "would", "can", "could", "has", "have", "had",
    "do", "does", "did", "not", "no", "also", "into", "about", "over", "such",
}

# a "word" = a run of letters/digits (underscores and punctuation are separators)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def preprocess_with_spans(text: str, remove_stop_words: bool = True):
    """Return (tokens, spans); spans[i] = (start, end) of tokens[i] in `text`."""
    tokens, spans = [], []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group().lower()
        if remove_stop_words and tok in STOP_WORDS:
            continue
        tokens.append(tok)
        spans.append(m.span())
    return tokens, spans


def preprocess(text: str, remove_stop_words: bool = True) -> list[str]:
    return preprocess_with_spans(text, remove_stop_words)[0]
