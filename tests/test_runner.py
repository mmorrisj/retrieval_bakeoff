"""End-to-end runner, config validation, timing, and report rendering."""

import json

import pytest

from bakeoff.config import RunConfig
from bakeoff.corpus import load_smoke_corpus
from bakeoff.report import README_BEGIN, README_END, render_markdown, write_readme_table
from bakeoff.runner import run, write_results
from bakeoff.timing import LatencyRecorder, percentile

SMOKE_SYSTEMS = [
    {"kind": "bm25", "name": "bm25"},
    {"kind": "dense", "name": "dense", "embedder": {"kind": "hashing", "dim": 128}},
]


def make_config(**overrides):
    defaults = dict(
        name="test",
        datasets=[{"source": "smoke"}],
        systems=SMOKE_SYSTEMS,
        metrics=["ndcg@10"],
        top_k=10,
        warmup_queries=0,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


# --- corpus ---------------------------------------------------------------


def test_smoke_corpus_is_internally_consistent():
    dataset = load_smoke_corpus()
    doc_ids = {doc.doc_id for doc in dataset.documents}

    assert dataset.stats["documents"] == 24
    assert dataset.stats["queries"] == 12

    for query_id, relevance in dataset.qrels.items():
        assert query_id in dataset.queries, f"{query_id} is judged but has no query text"
        for doc_id in relevance:
            assert doc_id in doc_ids, f"{query_id} judges unknown document {doc_id}"


def test_unjudged_queries_are_dropped_and_counted():
    dataset = load_smoke_corpus()
    dataset.queries["extra"] = "a query nobody judged"
    dataset.qrels["extra"] = {}
    dataset.__post_init__()

    assert "extra" not in dataset.queries
    assert dataset.dropped_queries == 1


# --- config ---------------------------------------------------------------


def test_config_rejects_a_cutoff_above_top_k():
    with pytest.raises(ValueError, match="below the largest metric cutoff"):
        make_config(metrics=["recall@100"], top_k=10).validate()


def test_config_rejects_duplicate_system_names():
    duplicated = [{"kind": "bm25", "name": "same"}, {"kind": "bm25", "name": "same"}]
    with pytest.raises(ValueError, match="unique"):
        make_config(systems=duplicated).validate()


def test_config_rejects_empty_sections():
    with pytest.raises(ValueError, match="no datasets"):
        make_config(datasets=[]).validate()
    with pytest.raises(ValueError, match="no systems"):
        make_config(systems=[]).validate()


def test_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: yaml-test\n"
        "datasets:\n  - source: smoke\n"
        "systems:\n  - kind: bm25\n    name: bm25\n"
        "metrics:\n  - ndcg@10\n"
        "top_k: 10\n",
        encoding="utf-8",
    )
    config = RunConfig.from_file(path)
    assert config.name == "yaml-test"
    assert config.top_k == 10


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: x\ndatasets:\n  - source: smoke\nsystems:\n  - kind: bm25\ntypo_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_file(path)


# --- timing ---------------------------------------------------------------


def test_percentile_interpolates_between_ranks():
    # numpy.percentile([1,2,3,4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_recorder_excludes_warmup_samples():
    recorder = LatencyRecorder(warmup=2)
    for _ in range(5):
        with recorder.measure():
            pass

    assert len(recorder.warmup_ms) == 2
    assert recorder.stats().count == 3


def test_recorder_falls_back_to_warmup_when_nothing_else_ran():
    recorder = LatencyRecorder(warmup=10)
    with recorder.measure():
        pass
    # The low count is the signal that this figure is weak.
    assert recorder.stats().count == 1


def test_recorder_total_seconds_includes_warmup():
    recorder = LatencyRecorder(warmup=1)
    for _ in range(3):
        with recorder.measure():
            pass
    assert recorder.total_seconds >= 0


# --- runner ---------------------------------------------------------------


def test_run_produces_a_row_per_system():
    payload = run(make_config(), verbose=False)

    assert len(payload["results"]) == 2
    assert {row["system"] for row in payload["results"]} == {"bm25", "dense"}
    assert all(row["status"] == "ok" for row in payload["results"])


def test_run_scores_are_plausible_on_the_smoke_corpus():
    payload = run(make_config(), verbose=False)
    scores = {row["system"]: row["metrics"]["ndcg@10"] for row in payload["results"]}

    # The smoke queries are lexically close to their answers, so BM25 should be
    # far better than chance. This guards against a silent scoring regression.
    assert scores["bm25"] > 0.5


def test_run_records_the_environment_and_config():
    payload = run(make_config(), verbose=False)
    assert payload["environment"]["python"]
    assert payload["config"]["metrics"] == ["ndcg@10"]


def test_a_broken_system_is_recorded_without_aborting_the_run():
    systems = [{"kind": "bm25", "name": "good"}, {"kind": "nonsense", "name": "bad"}]
    payload = run(make_config(systems=systems), verbose=False)

    statuses = {row["system"]: row["status"] for row in payload["results"]}
    assert statuses == {"good": "ok", "bad": "failed"}


def test_max_queries_limits_the_evaluation_set():
    payload = run(make_config(max_queries=3), verbose=False)
    assert payload["results"][0]["latency"]["count"] == 3


def test_write_results_is_valid_json(tmp_path):
    payload = run(make_config(), verbose=False)
    path = write_results(payload, tmp_path / "out")

    assert path.exists()
    assert json.loads(path.read_text())["run"] == "test"


# --- report ---------------------------------------------------------------


def test_markdown_report_lists_systems_best_first():
    payload = run(make_config(), verbose=False)
    markdown = render_markdown(payload)

    assert "| System | ndcg@10 |" in markdown

    best = max(payload["results"], key=lambda row: row["metrics"]["ndcg@10"])["system"]
    data_rows = [line for line in markdown.splitlines() if line.startswith("| `")]
    assert data_rows, "report rendered no data rows"
    assert data_rows[0].startswith(f"| `{best}` |")


def test_markdown_report_names_skipped_systems():
    payload = run(make_config(), verbose=False)
    payload["results"].append(
        {
            "system": "openai",
            "dataset": "smoke",
            "status": "skipped",
            "metrics": {},
            "latency": {},
            "cost": {},
            "index_seconds": 0.0,
            "config": {},
            "detail": "OPENAI_API_KEY is not set",
        }
    )
    markdown = render_markdown(payload)
    assert "Skipped:" in markdown
    assert "OPENAI_API_KEY is not set" in markdown


def test_write_readme_table_replaces_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n\n{README_BEGIN}\nstale table\n{README_END}\n\nafter\n")

    payload = run(make_config(), verbose=False)
    assert write_readme_table(payload, readme) is True

    text = readme.read_text()
    assert "stale table" not in text
    assert "before" in text and "after" in text
    assert text.count(README_BEGIN) == 1


def test_write_readme_table_is_a_noop_without_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    payload = run(make_config(), verbose=False)

    assert write_readme_table(payload, readme) is False
    assert readme.read_text() == "no markers here\n"
