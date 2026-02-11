"""
Text Metrics - Compute writing quality and LLM-detection metrics.

Metrics:
    flesch_kincaid    [-100, 120]  Readability score. High=easy (5th grade), low=hard (academic).
    gunning_fog       [0, 20+]     Years of education needed. High=complex, low=simple.
    coleman_liau      [-5, 20+]    Grade level via char counts. High=advanced, low=elementary.
    difficult_word_ratio [0, 1]    Fraction of hard vocabulary. High=sophisticated, low=basic.
    mtld              [0, 200+]    Lexical diversity. High=varied vocabulary, low=repetitive.
    hdd               [0, 1]       Lexical diversity (stable). High=diverse, low=repetitive.
    slop_score        [0, 1000+]   LLM-typical phrases. High=AI-like, low=human-like.
    repetition_rate   [0, 1]       Fraction of repeated trigrams. High=repetitive, low=varied.
    sentence_length_mean [0, 50+]  Avg words per sentence. High=complex, low=simple.
    sentence_length_variance [0, 20+] Std dev of sentence lengths. High=varied, low=uniform.
    clause_density    [0, 5+]      Avg commas+semicolons per sentence. High=complex, low=simple.

Usage:
    from text_metrics import compute_metrics
    result = compute_metrics("Your text here")
    result = compute_metrics(text, metrics=["slop_score", "mtld"])
    results = compute_metrics(["text1", "text2", ...])
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import requests
import textstat
from lexicalrichness import LexicalRichness
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_SLOP_LISTS = None
_SLOP_URLS = {
    "words": "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/slop_list.json",
    "bigrams": "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/slop_list_bigrams.json",
    "trigrams": "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/slop_list_trigrams.json",
}


def _load_slop_lists():
    global _SLOP_LISTS
    if _SLOP_LISTS is not None:
        return _SLOP_LISTS

    path = DATA_DIR / "slop_lists.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        print("Downloading slop lists from EQ-bench GitHub...")
        data = {}
        for key, url in _SLOP_URLS.items():
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data[key] = [item[0] for item in resp.json()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    _SLOP_LISTS = {k: set(w.lower() for w in v) for k, v in data.items()}
    return _SLOP_LISTS


def _hdd(text: str) -> float:
    lex = LexicalRichness(text)
    return lex.hdd(draws=42) if lex.words >= 42 else 0.0


def _mtld(text: str, threshold: float = 0.72) -> float:
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if len(tokens) < 10:
        return 0.0

    def one_dir(toks):
        factor_count, factor_start, types = 0, 0, set()
        for i, tok in enumerate(toks):
            types.add(tok)
            if len(types) / (i - factor_start + 1) <= threshold:
                factor_count += 1
                factor_start, types = i + 1, set()
        remaining = len(toks) - factor_start
        if remaining > 0:
            factor_count += (1 - len(types) / remaining) / (1 - threshold)
        return len(toks) / factor_count if factor_count > 0 else float(len(toks))

    return (one_dir(tokens) + one_dir(tokens[::-1])) / 2


def _repetition_rate(text: str, n: int = 3) -> float:
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if len(tokens) < n:
        return 0.0
    ng = list(ngrams(tokens, n))
    counts = Counter(ng)
    return sum(c - 1 for c in counts.values() if c > 1) / len(ng) if ng else 0.0


def _slop_score(text: str) -> float:
    slop = _load_slop_lists()
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if not tokens:
        return 0.0
    word_hits = sum(1 for t in tokens if t in slop["words"])
    bigram_hits = sum(
        1
        for i in range(len(tokens) - 1)
        if f"{tokens[i]} {tokens[i + 1]}" in slop["bigrams"]
    )
    trigram_hits = sum(
        1
        for i in range(len(tokens) - 2)
        if f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" in slop["trigrams"]
    )
    return (word_hits + 2 * bigram_hits + 8 * trigram_hits) / len(tokens) * 1000


def _sentence_stats(text: str) -> tuple[list[int], int]:
    sentences = sent_tokenize(text)
    lengths = [len([w for w in word_tokenize(s) if w.isalpha()]) for s in sentences]
    clause_markers = text.count(",") + text.count(";")
    return lengths, clause_markers


def _sentence_length_mean(text: str) -> float:
    lengths, _ = _sentence_stats(text)
    return sum(lengths) / len(lengths) if lengths else 0.0


def _sentence_length_variance(text: str) -> float:
    lengths, _ = _sentence_stats(text)
    return statistics.stdev(lengths) if len(lengths) >= 2 else 0.0


def _clause_density(text: str) -> float:
    lengths, clause_markers = _sentence_stats(text)
    return clause_markers / len(lengths) if lengths else 0.0


def _difficult_word_ratio(text: str) -> float:
    word_count = textstat.lexicon_count(text, removepunct=True)
    return textstat.difficult_words(text) / word_count if word_count > 0 else 0.0


ALL_METRICS = [
    "flesch_kincaid",
    "gunning_fog",
    "coleman_liau",
    "difficult_word_ratio",
    "mtld",
    "hdd",
    "slop_score",
    "repetition_rate",
    "sentence_length_mean",
    "sentence_length_variance",
    "clause_density",
]

_METRIC_FUNCS = {
    "flesch_kincaid": textstat.flesch_reading_ease,
    "gunning_fog": textstat.gunning_fog,
    "coleman_liau": textstat.coleman_liau_index,
    "difficult_word_ratio": _difficult_word_ratio,
    "mtld": _mtld,
    "hdd": _hdd,
    "slop_score": _slop_score,
    "repetition_rate": _repetition_rate,
    "sentence_length_mean": _sentence_length_mean,
    "sentence_length_variance": _sentence_length_variance,
    "clause_density": _clause_density,
}


def compute_metrics(
    texts: str | list[str],
    metrics: list[str] | None = None,
) -> dict[str, float] | list[dict[str, float]]:
    """Compute text metrics for one or more texts."""
    if metrics is None:
        metrics = ALL_METRICS

    def compute_one(text: str) -> dict[str, float]:
        return {m: _METRIC_FUNCS[m](text) for m in metrics}

    if isinstance(texts, str):
        return compute_one(texts)
    return [compute_one(t) for t in texts]


if __name__ == "__main__":
    test_texts = {
        "sloppy_fiction": """
        The sun dipped below the horizon, casting long shadows across the ancient cobblestones.
        Sarah felt a chill run down her spine as she made her way through the narrow alleyway.
        Her heart was pounding in her chest. She couldn't help but feel a sense of unease.
        """,
        "academic_dense": """
        The epistemological ramifications of quantum indeterminacy necessitate a fundamental
        reconceptualization of causality within contemporary theoretical frameworks. Furthermore,
        the ontological status of superposition states remains philosophically contentious; scholars
        diverge substantially regarding whether such phenomena constitute genuine metaphysical
        pluralities or merely epistemic uncertainties reflecting observational limitations.
        """,
        "simple_repetitive": """
        The dog ran. The dog jumped. The dog played. The dog was happy. The dog ran again.
        The cat sat. The cat watched. The cat waited. The cat was quiet. The cat sat still.
        The bird flew. The bird sang. The bird landed. The bird was small. The bird flew away.
        The fish swam. The fish splashed. The fish hid. The fish was fast. The fish swam deep.
        """,
    }

    for name, text in test_texts.items():
        print(f"=== {name} ===")
        text = " ".join(text.split())
        result = compute_metrics(text)
        for k, v in sorted(result.items()):
            print(f"  {k}: {v:.4f}")
        print()
