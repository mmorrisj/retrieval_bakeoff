# Retrieval Bake-Off

**Which retrieval strategy should you actually ship?** Most comparisons answer
that with nDCG alone. This one measures quality, tail latency, and dollars per
thousand queries in the same table, because those are the three numbers that
decide the architecture — and the cheapest option wins more often than the
literature suggests.

Twelve systems — BM25, open-weight dense models, three hosted embedding APIs,
RRF hybrids, and cross-encoder reranking at two candidate depths — over four
BEIR subsets chosen to disagree with each other. One command reproduces
everything.

```bash
make demo    # offline smoke run, no network, no API keys, ~5 seconds
make bench   # the real benchmark
```

---

## Results

<!-- BEGIN GENERATED TABLE -->

**The full BEIR run has not been published yet.** This table is generated from
`results/latest/results.json` by `make bench`, never written by hand, so
whatever appears here can be reproduced by running the same config. Until that
run lands, the section below shows the harness working end to end on the
bundled offline corpus.

<!-- END GENERATED TABLE -->

### Harness verification (offline smoke corpus)

This is `make demo`: 24 documents, 12 queries, no network. Its purpose is to
prove that indexing, search, scoring, costing, and reporting work together —
**it is not a benchmark result.** The "dense" system here is a hashing stub,
not a semantic model, and 24 documents cannot rank anything.

| System | ndcg@10 | recall@10 | mrr@10 | p50 ms | p95 ms | $ / 1k queries |
|---|---|---|---|---|---|---|
| `hybrid-rrf` | 0.7910 | 0.8333 | 0.8611 | 0.1 | 0.1 | $0 |
| `bm25` | 0.7525 | 0.7639 | 0.8333 | 0.0 | 0.0 | $0 |
| `hashing-dense` | 0.7419 | 0.8333 | 0.7569 | 0.1 | 0.1 | $0 |

The one thing worth reading into it: fusion beats both of its components, which
is the behaviour RRF is supposed to have.

---

## What it measures, and why those things

**Quality — nDCG@10, Recall@100, MRR@10.** Recall@100 is the one to watch for a
RAG pipeline. Anything the first stage misses cannot be recovered by a reranker
or by the generator downstream, so first-stage recall is a hard ceiling on the
whole system. nDCG@10 is what gets published; recall is what breaks you.

**Latency — p50 and p95, measured at batch size 1.** Embedding all evaluation
queries in one batch and dividing is faster, flatters every system, and measures
throughput rather than latency. Nobody issues a batch of 300 queries. Warm-up
calls are discarded so the tail reflects steady state rather than lazy model
loading and TLS setup.

**Cost — indexing and querying priced separately.** Embedding a corpus is a
one-off capital cost; embedding a query and reranking its candidates is paid on
every request. Blending them into one number flatters large-corpus, low-traffic
systems and punishes the reverse. Local models are priced too: they have no
invoice but they have a machine under them, and measured wall-clock times an
hourly rate makes "free" open models comparable with metered APIs.

Every price carries the date it was checked and a source link, and the report
warns when an entry goes stale rather than quietly quoting old pricing.

## Design decisions worth arguing with

**Flat vector search, not HNSW.** An approximate index would be faster on a
large corpus, but it adds a recall/latency knob that differs per library and
would confound the comparison — a dense row could lose to BM25 because of
somebody's `ef_search` setting rather than because of its embedding model. Flat
search is exact, so quality differences are attributable to the model. The
consequence is that dense latency here scales linearly with corpus size and is
a **ceiling**, not what a tuned production index would give you.

**RRF for hybrid, not score blending.** BM25 and cosine similarity live on
incomparable scales, so any weighted-sum blend needs a normalisation step that
has to be re-tuned per corpus. Reciprocal rank fusion needs no tuning and is
what most production hybrid stacks actually ship.

**BM25 at BEIR's own `k1=0.9, b=0.4`.** Using the published baseline's
parameters means the numbers here can be checked against published BEIR tables
instead of against a privately tuned variant.

**No averaging across datasets.** There is no single best retriever, only a best
one for a given corpus shape. The four subsets are chosen precisely because they
disagree — `scifact` has high lexical overlap and favours BM25, `nfcorpus` and
`fiqa` have vocabulary mismatch and favour dense, `arguana` is adversarial for
both. Collapsing them into one mean destroys the finding.

**Unjudged documents count as zero, not as skipped.** And a query with no
relevant documents scores 0, not 1.0. Both choices are stricter than some
implementations and are asserted in `tests/test_metrics.py`.

## Install and run

```bash
make install       # core harness: numpy and PyYAML only
make demo          # offline smoke run
make test          # 80 tests, no network required

make install-all   # adds local models, BEIR loaders, API clients
make bench         # the real thing
```

The core deliberately depends on numpy and PyYAML alone. Everything needing a
model download or an API key is an optional extra, so tests and the smoke run
work on a fresh clone with no credentials.

API keys are read from the environment and are all optional — a system whose
key is absent is recorded as **skipped**, not failed, so the run completes with
whatever credentials you happen to have:

```bash
export OPENAI_API_KEY=...    # optional
export COHERE_API_KEY=...    # optional
export VOYAGE_API_KEY=...    # optional
```

A full `configs/beir-v1.yaml` run costs single-digit US dollars at the prices
in `src/bakeoff/cost.py`, dominated by embedding the `fiqa` corpus several times
over. Cap it while iterating:

```bash
python -m bakeoff run configs/beir-v1.yaml --out results/quick --max-queries 50
```

## Adding a system

Systems are declared in YAML and built by a factory, so a new configuration is
a config change rather than a code change:

```yaml
- name: bm25-rerank-100
  kind: rerank
  candidate_k: 100
  base:
    name: rerank100-bm25
    kind: bm25
```

A genuinely new *kind* of retriever means implementing two methods —
`index(corpus)` and `search(query, k)` — and registering it in
`src/bakeoff/systems/__init__.py`. Systems report their own billable usage;
the runner does not try to infer cost from the outside, because that breaks as
soon as a system batches, caches, or composes.

## Limitations

Stated plainly, because a benchmark that hides these is worse than none:

- **Latency is machine-dependent.** Compare rows against each other within one
  run. Comparing a p95 here against one from another machine is meaningless.
- **Flat search overstates dense latency** on large corpora, as above.
- **BEIR subsets are not your corpus.** They are standard and therefore
  checkable, which is a different property from being representative. The
  ranking on your documents can differ; the harness is built so you can run it
  on them.
- **Prices drift.** Every entry is dated and the report warns past 90 days, but
  a stale table is still a stale table.
- **Token counts are estimated for providers that return no usage figure**
  (Cohere embeddings, all local models). Those rows are flagged in the report.
- **Single run per configuration.** Latency percentiles from one pass have real
  variance; treat small p95 differences as noise.
- **English only.** Every model and subset here is English, and the BM25
  analyser is a simple alphanumeric tokeniser with a short stop list.

## Repository layout

```
src/bakeoff/
  metrics.py       nDCG, recall, MRR -- implemented directly, tested by hand
  cost.py          dated price registry, usage accounting, cost model
  timing.py        latency percentiles with warm-up discipline
  corpus.py        BEIR loaders plus the bundled offline corpus
  runner.py        orchestration; a failing system is recorded, not fatal
  report.py        generates every table from results.json
  systems/         BM25, dense, hybrid, rerank, and the embedding backends
configs/
  smoke.yaml       offline, no dependencies -- runs in CI
  beir-v1.yaml     the real benchmark
```

## License

MIT. Built by [Aion Innovations](https://aionbuilt.com).
