import json
from types import SimpleNamespace

import pytest

import scripts.run_dominant_trait_test as runner
from inspect_eval.trait import dominant_trait_pilot as pilot
from inspect_eval.trait import dominant_trait_test as heldout


def _record(record_id, document_id, trait, split="test", context="Document context"):
    return {
        "id": record_id,
        "text": f"Snippet {record_id}",
        "dominant_trait": trait,
        "source": {
            "document_id": document_id,
            "upstream_split": split,
        },
        "metadata": {"context1000": context},
    }


def _paired_rows(records):
    snippet = []
    context = []
    for row in records:
        target = row["dominant_trait"]
        common = {
            "sample_id": row["id"],
            "document_id": row["source"]["document_id"],
            "target": target,
            "invalid": False,
        }
        snippet.append({**common, "predicted": target})
        context.append({**common, "predicted": target})
    return {"snippet_only": snippet, "context1000": context}


def test_test_loader_requires_test_split_and_never_filters_missing_context(tmp_path):
    path = tmp_path / "test.jsonl"
    records = [
        _record("first", "doc-1", pilot.TRAITS[0]),
        _record("missing-context", "doc-2", pilot.TRAITS[1], context="  "),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing or blank context1000"):
        heldout.load_test_data(path)

    records[1]["metadata"]["context1000"] = "Valid context"
    records[1]["source"]["upstream_split"] = "train"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not from the test split"):
        heldout.load_test_data(path)


def test_test_manifest_verification_uses_test_output_hash(tmp_path):
    dataset = tmp_path / "test.jsonl"
    row = _record("test-one", "document-one", pilot.TRAITS[0])
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "context_column_used": "context1000",
        "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
        "output_sha256": {"test": pilot.sha256_file(dataset)},
        "test_records": 1,
        "train_test_document_overlap": 0,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    verified, rows = heldout.verify_test_dataset(dataset)

    assert verified == manifest
    assert rows == [row]


def test_training_majority_baseline_excludes_pilot_items(tmp_path):
    train_path = tmp_path / "train.jsonl"
    records = [
        _record(
            f"train-{trait}-{index}",
            f"doc-{trait}-{index}",
            trait,
            split="train",
        )
        for trait in pilot.TRAITS
        for index in range(10)
    ]
    records.extend([
        _record("train-majority-1", "doc-majority-1", pilot.TRAITS[1], split="train"),
        _record("train-majority-2", "doc-majority-2", pilot.TRAITS[1], split="train"),
        _record("train-majority-3", "doc-majority-3", pilot.TRAITS[1], split="train"),
    ])
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({
        "context_column_used": "context1000",
        "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
        "output_sha256": {"train": pilot.sha256_file(train_path)},
        "train_records": len(records),
        "train_test_document_overlap": 0,
    }), encoding="utf-8")

    assert heldout.training_majority_baseline(
        train_path
    ) == ("overriding_suspicion", 3, 3)


def test_full_test_report_includes_metrics_invalid_column_and_document_bootstrap():
    records = [
        _record(f"item-{index}", f"doc-{index // 2}", trait)
        for index, trait in enumerate(pilot.TRAITS)
    ]
    arm_rows = _paired_rows(records)
    arm_rows["context1000"][0]["predicted"] = None
    arm_rows["context1000"][0]["invalid"] = True

    report = heldout.build_test_report(
        arm_rows,
        records,
        baseline_trait=pilot.TRAITS[1],
        repetitions=25,
        seed=7,
    )

    assert report["evaluation_status"] == "descriptive_held_out_test"
    assert report["test_sample_count"] == 6
    assert report["source_document_count"] == 3
    assert report["arms"]["snippet_only"]["macro_f1"] == 1.0
    assert report["arms"]["context1000"]["invalid_output_count"] == 1
    assert report["arms"]["context1000"]["confusion_matrix"][pilot.TRAITS[0]][
        "invalid_output"
    ] == 1
    assert report["majority_baseline"]["predicted_trait"] == pilot.TRAITS[1]
    bootstrap = report["context_comparison"]["paired_document_bootstrap"]
    assert bootstrap["resampling_unit"] == "source.document_id"
    assert bootstrap["document_count"] == 3
    assert bootstrap["repetitions"] == 25
    with pytest.raises(ValueError, match="exactly all official test IDs"):
        heldout.build_test_report(
            {
                "snippet_only": arm_rows["snippet_only"][:-1],
                "context1000": arm_rows["context1000"][:-1],
            },
            records,
            baseline_trait=pilot.TRAITS[1],
            repetitions=5,
        )


def test_runner_runs_both_paired_arms_and_persists_report(tmp_path, monkeypatch):
    records = [
        _record(f"item-{index}", f"doc-{index // 2}", trait)
        for index, trait in enumerate(pilot.TRAITS)
    ]
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text("test data fixture", encoding="utf-8")
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    calls = []

    def fake_verify_test_dataset(path):
        assert path == dataset_path
        return (
            {
                "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
                "train_test_document_overlap": 0,
                "context_column_used": "context1000",
            },
            records,
        )

    def fake_training_baseline(train_path):
        assert train_path.name == "train.jsonl"
        return "overriding_suspicion", 697, 1244

    def fake_build_task(rows, arm):
        assert rows == records
        assert arm in pilot.ARMS
        return SimpleNamespace(dataset=records, arm=arm)

    monkeypatch.setattr(runner, "verify_test_dataset", fake_verify_test_dataset)
    monkeypatch.setattr(runner, "training_majority_baseline", fake_training_baseline)
    monkeypatch.setattr(runner, "build_arm_task", fake_build_task)

    def fake_eval(task, **kwargs):
        assert (results_dir / "run_manifest.json").is_file()
        assert kwargs["model_args"] == {
            "revision": pilot.EXPECTED_MODEL_REVISION,
            "dtype": "auto",
            "do_sample": False,
        }
        assert kwargs["temperature"] is None
        calls.append(kwargs)
        samples = []
        for record in task.dataset:
            target = record["dominant_trait"]
            completion = target
            samples.append(SimpleNamespace(
                id=record["id"],
                output=SimpleNamespace(completion=completion),
                scores={
                    "dominant_trait_pilot_scorer": SimpleNamespace(
                        value=json.dumps({
                            "sample_id": record["id"],
                            "document_id": record["source"]["document_id"],
                            "predicted": target,
                            "target": target,
                            "invalid": False,
                        }),
                        explanation=completion,
                    )
                },
            ))
        return [SimpleNamespace(status="success", samples=samples, location="mock.eval")]

    monkeypatch.setattr(runner, "inspect_eval", fake_eval)
    args = SimpleNamespace(
        dataset=dataset_path,
        train_dataset=tmp_path / "train.jsonl",
        model=pilot.EXPECTED_MODEL_ID,
        model_revision=pilot.EXPECTED_MODEL_REVISION,
        dtype="auto",
        seed=42,
        bootstrap_repetitions=25,
        max_tasks=1,
        max_samples=1,
        run_environment="mocked test",
        results_dir=results_dir,
        log_dir=log_dir,
    )

    report = runner.run(args)

    assert len(calls) == 2
    assert report["test_sample_count"] == len(records)
    assert report["arms"]["snippet_only"]["exact_accuracy"] == 1.0
    assert report["arms"]["context1000"]["macro_f1"] == 1.0
    assert report["run"]["inference_count"] == len(records) * 2
    assert json.loads((results_dir / "report.json").read_text(encoding="utf-8"))[
        "evaluation_status"
    ] == "descriptive_held_out_test"
