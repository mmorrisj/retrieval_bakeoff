"""Corpus loading.

Two sources, deliberately:

* **BEIR subsets** are the real benchmark. They are the standard, so numbers
  produced here can be checked against published tables -- which is the point.
  Loading them needs network access and the `beir` extra.
* **A tiny built-in corpus** ships in the repo so the whole pipeline runs in CI
  with no network, no downloads and no keys. It is 24 documents and exists to
  prove the harness works, never to rank anything.

BEIR ships each dataset as a corpus file, a queries file, and a qrels TSV. This
module normalises all three into the same shape the runner expects, so adding a
new dataset means adding a name to a config list.
"""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
from dataclasses import dataclass

from bakeoff.systems.base import Document

# BEIR subsets small enough to run on a laptop. Sizes are approximate corpus
# document counts, listed so a config author can predict the runtime.
BEIR_DATASETS = {
    "scifact": 5_183,
    "nfcorpus": 3_633,
    "arguana": 8_674,
    "scidocs": 25_657,
    "fiqa": 57_638,
    "trec-covid": 171_332,
}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    """A corpus, its queries, and the relevance judgements linking them."""

    name: str
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        # A query with no judgements cannot be scored, and silently averaging
        # zeros over it would drag every system down equally while looking like
        # a quality difference. Drop them and say how many.
        judged = {qid for qid, rels in self.qrels.items() if rels}
        self.dropped_queries = len(self.queries) - len(self.queries.keys() & judged)
        self.queries = {qid: text for qid, text in self.queries.items() if qid in judged}
        self.qrels = {qid: rels for qid, rels in self.qrels.items() if qid in self.queries}

    @property
    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "queries": len(self.queries),
            "dropped_unjudged_queries": self.dropped_queries,
            "judgements": sum(len(rels) for rels in self.qrels.values()),
        }


def load_smoke_corpus() -> Dataset:
    """The tiny offline dataset bundled with the repo."""
    path = pathlib.Path(__file__).parent / "data" / "smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    return Dataset(
        name="smoke",
        documents=[Document(doc["id"], doc["text"]) for doc in payload["documents"]],
        queries={query["id"]: query["text"] for query in payload["queries"]},
        qrels={
            qid: {doc_id: int(grade) for doc_id, grade in rels.items()}
            for qid, rels in payload["qrels"].items()
        },
    )


def load_beir(name: str, data_dir: pathlib.Path, *, split: str = "test") -> Dataset:
    """Load a BEIR dataset, downloading and unpacking it on first use."""
    if name not in BEIR_DATASETS:
        known = ", ".join(sorted(BEIR_DATASETS))
        raise ValueError(f"unknown BEIR dataset {name!r}; supported datasets are: {known}")

    dataset_dir = data_dir / name
    if not dataset_dir.exists():
        _download_beir(name, data_dir)

    documents = list(_read_beir_corpus(dataset_dir / "corpus.jsonl"))
    queries = _read_beir_queries(dataset_dir / "queries.jsonl")
    qrels = _read_beir_qrels(dataset_dir / "qrels" / f"{split}.tsv")

    # BEIR ships every query for every split in one file; keep only the ones
    # this split actually judges.
    queries = {qid: text for qid, text in queries.items() if qid in qrels}

    return Dataset(name=name, documents=documents, queries=queries, qrels=qrels)


def _download_beir(name: str, data_dir: pathlib.Path) -> None:
    import io
    import zipfile

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "requests is not installed; run `pip install -e '.[beir]'` to load BEIR datasets"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    url = BEIR_URL.format(name=name)

    response = requests.get(url, timeout=600, stream=True)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(data_dir)


def _open_maybe_gzip(path: pathlib.Path):
    """BEIR ships some corpora gzipped and some not."""
    if path.exists():
        return path.open("r", encoding="utf-8")
    gzipped = path.with_suffix(path.suffix + ".gz")
    if gzipped.exists():
        return gzip.open(gzipped, "rt", encoding="utf-8")
    raise FileNotFoundError(f"neither {path} nor {gzipped} exists")


def _read_beir_corpus(path: pathlib.Path):
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            record = json.loads(line)
            # BEIR splits document text across a title and a body; concatenating
            # them is what the official evaluation does, and dropping the title
            # costs several points on title-heavy corpora such as trec-covid.
            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            text = f"{title}\n{body}".strip() if title else body
            yield Document(str(record["_id"]), text)


def _read_beir_queries(path: pathlib.Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            record = json.loads(line)
            queries[str(record["_id"])] = (record.get("text") or "").strip()
    return queries


def _read_beir_qrels(path: pathlib.Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        # The header row is "query-id corpus-id score"; older dumps omit it.
        if header and header[0].strip().lower() not in {"query-id", "query_id"}:
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            qrels.setdefault(query_id, {})[doc_id] = int(float(score))
    return qrels


def load_dataset(spec: dict, data_dir: pathlib.Path) -> Dataset:
    """Dispatch a config `dataset` block to the right loader."""
    source = spec.get("source", "beir")
    if source == "smoke":
        return load_smoke_corpus()
    if source == "beir":
        return load_beir(spec["name"], data_dir, split=spec.get("split", "test"))
    raise ValueError(f"unknown dataset source {source!r}; expected 'beir' or 'smoke'")
