"""Embedding backends.

Everything that turns text into a vector lives behind one small interface, so
`DenseRetriever` is written once and works with a local sentence-transformer, a
metered API, or the deterministic stub used by the offline smoke test.

Each backend reports its own token usage. API backends read the exact figure
from the provider response where one is returned and fall back to the character
heuristic otherwise, flagging that they did so -- a cost table that cannot tell
measured from estimated is not worth much.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
from abc import ABC, abstractmethod

import numpy as np

from bakeoff.cost import estimate_tokens
from bakeoff.systems.base import NotInstalled


class Embedder(ABC):
    """Turns text into unit-norm vectors."""

    #: Price registry key, or None for backends with no per-token invoice.
    price_key: str | None = None

    def __init__(self) -> None:
        self.tokens = 0
        self.tokens_estimated = False

    @abstractmethod
    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of L2-normalised vectors.

        `is_query` exists because several instruction-tuned models require
        different prefixes for queries and documents; ignoring it costs real
        accuracy on e5-family models.
        """

    def describe(self) -> dict[str, str]:
        return {"embedder": type(self).__name__}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Zero vectors stay zero rather than becoming NaN.
    return matrix / np.where(norms == 0, 1.0, norms)


class HashingEmbedder(Embedder):
    """A deterministic, dependency-free stand-in for a real embedding model.

    This is a hashed bag-of-words projection, not a semantic model, and it is
    not meant to compete on quality. It exists so the whole pipeline -- index,
    search, score, cost, report -- can be exercised in CI with no model
    download, no network and no API key. Any result it produces is labelled as
    a smoke test and must never be presented as a benchmark number.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        from bakeoff.systems.bm25 import tokenize

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # Blake2b keeps this stable across processes and Python
                # versions, unlike the salted builtin hash().
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "big") % self.dim
                # Sign from a spare bit, the usual hashing-trick trick for
                # keeping collisions from systematically inflating scores.
                sign = 1.0 if digest[0] & 1 else -1.0
                vectors[row, bucket] += sign

        self.tokens += estimate_tokens(texts)
        self.tokens_estimated = True
        return l2_normalize(vectors)

    def describe(self) -> dict[str, str]:
        return {"embedder": "HashingEmbedder", "dim": str(self.dim), "semantic": "false"}


class SentenceTransformerEmbedder(Embedder):
    """Local open-weight models: e5, bge, gte, and anything else on the Hub.

    Has no invoice, so its cost comes from measured wall-clock priced at the
    machine's hourly rate (see `bakeoff.cost`).
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise NotInstalled(
                "sentence-transformers is not installed; run `pip install -e '.[dense]'`"
            ) from exc

        self.model_name = model_name
        # e5 and bge models are trained with asymmetric prefixes and lose
        # several nDCG points if they are dropped. Making them explicit here
        # keeps that visible in the config rather than buried in a wrapper.
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.document_prefix
        prepared = [f"{prefix}{text}" for text in texts] if prefix else texts

        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.tokens += estimate_tokens(prepared)
        self.tokens_estimated = True
        return vectors.astype(np.float32)

    def describe(self) -> dict[str, str]:
        return {
            "embedder": "SentenceTransformerEmbedder",
            "model": self.model_name,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
        }


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings, billed per token with exact usage from the response."""

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 128) -> None:
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise NotInstalled("openai is not installed; run `pip install -e '.[api]'`") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise NotInstalled("OPENAI_API_KEY is not set")

        self.model_name = model_name
        self.batch_size = batch_size
        self.price_key = f"openai/{model_name}"
        self.client = OpenAI()

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
            # Exact billed tokens, straight from the provider.
            self.tokens += response.usage.prompt_tokens

        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def describe(self) -> dict[str, str]:
        return {"embedder": "OpenAIEmbedder", "model": self.model_name}


class CohereEmbedder(Embedder):
    """Cohere embeddings. Requires the input type that matches the call."""

    def __init__(self, model_name: str = "embed-english-v3.0", batch_size: int = 96) -> None:
        super().__init__()
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise NotInstalled("cohere is not installed; run `pip install -e '.[api]'`") from exc
        if not os.environ.get("COHERE_API_KEY"):
            raise NotInstalled("COHERE_API_KEY is not set")

        self.model_name = model_name
        # Cohere caps a single embed call at 96 texts.
        self.batch_size = min(batch_size, 96)
        self.price_key = f"cohere/{model_name}"
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        input_type = "search_query" if is_query else "search_document"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type=input_type,
            )
            vectors.extend(response.embeddings)

        # Cohere's embed response does not carry a token count, so this figure
        # is estimated and the report says so.
        self.tokens += estimate_tokens(texts)
        self.tokens_estimated = True
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def describe(self) -> dict[str, str]:
        return {"embedder": "CohereEmbedder", "model": self.model_name}


class VoyageEmbedder(Embedder):
    """Voyage embeddings, billed per token with exact usage from the response."""

    def __init__(self, model_name: str = "voyage-3", batch_size: int = 128) -> None:
        super().__init__()
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover
            raise NotInstalled("voyageai is not installed; run `pip install -e '.[api]'`") from exc
        if not os.environ.get("VOYAGE_API_KEY"):
            raise NotInstalled("VOYAGE_API_KEY is not set")

        self.model_name = model_name
        self.batch_size = batch_size
        self.price_key = f"voyage/{model_name}"
        self.client = voyageai.Client()

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        input_type = "query" if is_query else "document"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embed(batch, model=self.model_name, input_type=input_type)
            vectors.extend(response.embeddings)
            self.tokens += response.total_tokens

        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def describe(self) -> dict[str, str]:
        return {"embedder": "VoyageEmbedder", "model": self.model_name}


class CachedEmbedder(Embedder):
    """Wraps an embedder with an on-disk cache for *document* embeddings.

    Two things this buys, both of which matter on a real run:

    * A config names the same model more than once -- `beir-v1.yaml` uses
      `e5-base` standalone, as a hybrid component, and as a rerank base -- and
      without a cache the corpus is embedded once per mention. On a 57k-document
      subset that is hours of duplicated GPU time, or three times the API bill.
    * A run that dies partway can be restarted without re-embedding what it
      already finished.

    **Queries are never cached.** A cache hit returns in microseconds, so caching
    the query side would turn the latency column into a measurement of disk
    speed. Only the `is_query=False` path touches the cache.

    Token counts are stored alongside the vectors, so a cached run still reports
    what the index *cost to build* rather than the zero it happened to spend
    this time. A cost table that reads $0 because of a warm cache is worse than
    no cost table.
    """

    #: Bumped when the on-disk format changes, to invalidate old entries.
    CACHE_VERSION = 1

    def __init__(self, inner: Embedder, cache_dir: str | pathlib.Path) -> None:
        super().__init__()
        self.inner = inner
        self.price_key = inner.price_key
        self.cache_dir = pathlib.Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if is_query:
            return self._encode_uncached(texts, is_query=True)

        path = self.cache_dir / f"{self._key(texts)}.npz"
        if path.exists():
            try:
                payload = np.load(path, allow_pickle=False)
                vectors = payload["vectors"]
                self.tokens += int(payload["tokens"])
                self.tokens_estimated |= bool(payload["tokens_estimated"])
                self.hits += 1
                return vectors
            except (OSError, ValueError, KeyError):
                # A truncated or half-written entry -- from an interrupted run --
                # is a cache miss, not a crash.
                path.unlink(missing_ok=True)

        self.misses += 1
        used_before = self.tokens
        vectors = self._encode_uncached(texts, is_query=False)
        self._store(path, vectors, self.tokens - used_before)
        return vectors

    def _encode_uncached(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        tokens_before = self.inner.tokens
        vectors = self.inner.encode(texts, is_query=is_query)
        self.tokens += self.inner.tokens - tokens_before
        self.tokens_estimated |= self.inner.tokens_estimated
        return vectors

    def _store(self, path: pathlib.Path, vectors: np.ndarray, tokens: int) -> None:
        # Write to a temporary name and rename, so an interrupted write cannot
        # leave a half-file that a later run would read as valid.
        temporary = path.with_name(path.name + ".partial")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Written through an open handle rather than by path: np.savez
            # appends ".npz" to any filename lacking it, which would leave the
            # temporary file under a name the rename below never finds.
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    vectors=vectors,
                    tokens=np.asarray(tokens),
                    tokens_estimated=np.asarray(self.tokens_estimated),
                )
            temporary.replace(path)
        except OSError:
            # A failed cache write must never fail the benchmark -- including
            # when the cleanup itself cannot run, which is what happens if the
            # cache path is unusable rather than merely full.
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _key(self, texts: list[str]) -> str:
        """Content hash of the model identity and the exact texts being embedded.

        Both halves are needed: the same corpus under two models must not
        collide, and neither must two different corpora under one model. The
        text count and total length go in as a cheap guard against a
        pathological hash collision changing the vector count.
        """
        digest = hashlib.sha256()
        descriptor = json.dumps(self.inner.describe(), sort_keys=True)
        digest.update(f"v{self.CACHE_VERSION}|{descriptor}|{len(texts)}".encode())
        for text in texts:
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()[:32]

    def describe(self) -> dict[str, str]:
        return {**self.inner.describe(), "cached": "true"}


#: Config `embedder.kind` values mapped to their classes.
EMBEDDERS: dict[str, type[Embedder]] = {
    "hashing": HashingEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
    "cohere": CohereEmbedder,
    "voyage": VoyageEmbedder,
}


def build_embedder(
    kind: str,
    cache_dir: str | pathlib.Path | None = None,
    **kwargs: object,
) -> Embedder:
    if kind not in EMBEDDERS:
        known = ", ".join(sorted(EMBEDDERS))
        raise ValueError(f"unknown embedder kind {kind!r}; known kinds are: {known}")

    embedder = EMBEDDERS[kind](**kwargs)  # type: ignore[arg-type]
    return CachedEmbedder(embedder, cache_dir) if cache_dir else embedder
