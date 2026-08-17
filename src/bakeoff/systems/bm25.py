"""Okapi BM25, implemented directly.

BM25 is the baseline the whole exercise turns on: if an expensive embedding
stack cannot beat a 1994 term-weighting scheme on a given corpus, that is the
single most useful thing the benchmark can tell you. It is written out here
rather than imported so the scoring is inspectable and so the core harness
keeps its no-heavy-dependency property.

Scoring follows the standard formulation:

    score(q, d) = sum over terms t in q of
        idf(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

with the BM25+ style idf that floors at zero, avoiding the negative weights
plain BM25 assigns to terms appearing in more than half the corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from bakeoff.systems.base import Document, Hit, Retriever

# Lowercased alphanumeric runs. Deliberately simple: a fancier analyser would
# help absolute scores but would make BM25 and the dense systems differ in
# preprocessing as well as in method, which muddies the comparison.
_TOKEN = re.compile(r"[a-z0-9]+")

# A short, standard stop list. Kept explicit rather than pulled from NLTK to
# avoid a dependency and a download for twenty-five words.
STOPWORDS = frozenset(
    """a an and are as at be by for from has he in is it its of on that the
    to was were will with this these those or not but if then than""".split()
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


class BM25Retriever(Retriever):
    """Classic sparse baseline. No model, no network, no cost beyond CPU."""

    name = "bm25"

    def __init__(
        self,
        name: str | None = None,
        *,
        k1: float = 0.9,
        b: float = 0.4,
        remove_stopwords: bool = True,
    ) -> None:
        super().__init__(name)
        # k1=0.9, b=0.4 are the values BEIR uses for its Anserini baseline.
        # Defaulting to them means the numbers here are comparable with
        # published BEIR tables rather than to a privately tuned variant.
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        self._doc_ids: list[str] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length = 0.0
        # k1 * (1 - b + b * |d| / avgdl) per document. Constant given the
        # corpus, so it is computed once at index time rather than inside the
        # scoring loop, where it would be recomputed for every posting.
        self._length_norms: list[float] = []
        # term -> list of (document index, term frequency)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._idf: dict[str, float] = {}

    def index(self, corpus: list[Document]) -> None:
        self._doc_ids = [doc.doc_id for doc in corpus]
        self._doc_lengths = []
        self._postings = defaultdict(list)

        document_frequency: Counter[str] = Counter()
        for position, doc in enumerate(corpus):
            tokens = tokenize(doc.text, remove_stopwords=self.remove_stopwords)
            self._doc_lengths.append(len(tokens))
            frequencies = Counter(tokens)
            for term, frequency in frequencies.items():
                self._postings[term].append((position, frequency))
            document_frequency.update(frequencies.keys())

        total_docs = len(corpus)
        self._avg_doc_length = sum(self._doc_lengths) / total_docs if total_docs else 0.0

        self._length_norms = [
            self.k1 * (1 - self.b + self.b * (length / self._avg_doc_length))
            if self._avg_doc_length
            else self.k1 * (1 - self.b)
            for length in self._doc_lengths
        ]

        # Robertson/Sparck-Jones idf with the customary +0.5 smoothing, floored
        # at zero so ubiquitous terms contribute nothing instead of subtracting.
        self._idf = {
            term: max(0.0, math.log(1 + (total_docs - df + 0.5) / (df + 0.5)))
            for term, df in document_frequency.items()
        }

        self.usage.queries = 0

    def search(self, query: str, k: int) -> list[Hit]:
        query_terms = tokenize(query, remove_stopwords=self.remove_stopwords)
        scores: dict[int, float] = defaultdict(float)

        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            if idf <= 0:
                continue

            for position, frequency in postings:
                denominator = frequency + self._length_norms[position]
                if denominator:
                    scores[position] += idf * (frequency * (self.k1 + 1)) / denominator

        self.usage.queries += 1

        # Sort by descending score, breaking ties on document id so the ranking
        # is deterministic across runs and platforms.
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._doc_ids[item[0]]),
        )[:k]
        return [Hit(self._doc_ids[position], score) for position, score in ranked]

    def describe(self) -> dict[str, str]:
        return {
            "type": "BM25Retriever",
            "k1": str(self.k1),
            "b": str(self.b),
            "remove_stopwords": str(self.remove_stopwords),
        }
