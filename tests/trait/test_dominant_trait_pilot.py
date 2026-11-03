import json
from types import SimpleNamespace

import pytest

import scripts.run_dominant_trait_pilot as runner
from inspect_eval.trait import dominant_trait_pilot as pilot
from scripts.import_conspired import TRAITS as IMPORT_TRAITS


def _record(record_id, document_id, trait, context="Document context"):
    return {
        "id": record_id,
        "text": f"Snippet {record_id}",
        "dominant_trait": trait,
        "source": {
            "document_id": document_id,
            "upstream_split": "train",
        },
        "metadata": {"context1000": context},
    }


def _balanced_rows(items_per_class=10):
    return [
        _record(
            f"{trait}-{index}",
            f"document-{trait}-{index}",
            trait,
        )
        for trait in pilot.TRAITS
        for index in range(items_per_class)
    ]


def test_trait_order_matches_the_excel_one_hot_importer():
    assert pilot.TRAITS == tuple(trait for trait, _ in IMPORT_TRAITS)


def test_import_manifest_requires_selected_context_and_matching_hash(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("fixture\n", encoding="utf-8")
    manifest = {
        "context_column_used": "context1000",
        "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
        "output_sha256": {"train": pilot.sha256_file(dataset)},
        "train_test_document_overlap": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert pilot.verify_import_manifest(dataset, manifest_path) == manifest
    manifest["context_column_used"] = "none"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="context_column_used"):
        pilot.verify_import_manifest(dataset, manifest_path)


def test_loader_excludes_missing_context_and_rejects_non_train_data(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(_record("usable", "doc-1", pilot.TRAITS[0])) + "\n"
        + json.dumps(_record("no-context", "doc-2", pilot.TRAITS[1], context=None))
        + "\n",
        encoding="utf-8",
    )

    rows, excluded = pilot.load_pilot_data(path)

    assert [row["id"] for row in rows] == ["usable"]
    assert excluded == [{
        "id": "no-context",
        "document_id": "doc-2",
        "reason": "missing_context1000",
    }]
    invalid_split = _record("test-row", "doc-test", pilot.TRAITS[0])
    invalid_split["source"]["upstream_split"] = "test"
    path.write_text(json.dumps(invalid_split), encoding="utf-8")
    with pytest.raises(ValueError, match="not from the training split"):
        pilot.load_pilot_data(path)


def test_balanced_sample_is_deterministic_balanced_and_prefers_documents():
    rows = _balanced_rows(items_per_class=12)

    first = pilot.select_balanced_sample(rows, items_per_class=10, seed=42)
    second = pilot.select_balanced_sample(rows, items_per_class=10, seed=42)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == 60
    assert {trait: sum(row["dominant_trait"] == trait for row in first) for trait in pilot.TRAITS} == {
        trait: 10 for trait in pilot.TRAITS
    }
    assert len({row["source"]["document_id"] for row in first}) == 60
    with pytest.raises(ValueError, match="Need 10 samples"):
        pilot.select_balanced_sample(_balanced_rows(items_per_class=9))


def test_parser_and_sample_input_keep_single_label_and_context_modes():
    row = _record("one", "doc-1", pilot.TRAITS[2])

    assert pilot.parse_dominant_prediction("nefarious_intent") == "nefarious_intent"
    assert pilot.parse_dominant_prediction('"nefarious_intent"') == "nefarious_intent"
    with pytest.raises(ValueError, match="one canonical"):
        pilot.parse_dominant_prediction('["nefarious_intent"]')
    with pytest.raises(ValueError, match="one canonical"):
        pilot.parse_dominant_prediction('"not_a_trait"')
    assert pilot.build_sample_input(row, include_context=False) == (
        "Annotated snippet:\nSnippet one"
    )
    assert "Context (context1000):\nDocument context" in pilot.build_sample_input(
        row, include_context=True
    )


def test_metric_summary_counts_invalid_predictions_as_incorrect():
    rows = [
        {
            "sample_id": "correct",
            "document_id": "doc-1",
            "predicted": pilot.TRAITS[0],
            "target": pilot.TRAITS[0],
            "invalid": False,
        },
        {
            "sample_id": "invalid",
            "document_id": "doc-2",
            "predicted": None,
            "target": pilot.TRAITS[1],
            "invalid": True,
        },
    ]

    summary = pilot.metric_summary(rows)

    assert summary["sample_count"] == 2
    assert summary["valid_single_label_count"] == 1
    assert summary["valid_single_label_rate"] == 0.5
    assert summary["exact_accuracy"] == 0.5
    assert summary["per_trait"][pilot.TRAITS[1]]["incorrect"] == 1
    assert sum(sum(row.values()) for row in summary["confusion_matrix"].values()) == 1


def test_report_pairs_context_arms_and_checks_sample_identity():
    rows = _balanced_rows(items_per_class=1)
    sample_ids = [row["id"] for row in rows]
    snippet_rows = [
        {
            "sample_id": row["id"],
            "document_id": row["source"]["document_id"],
            "predicted": row["dominant_trait"],
            "target": row["dominant_trait"],
            "invalid": False,
        }
        for row in rows
    ]
    context_rows = [dict(row) for row in snippet_rows]
    context_rows[0]["predicted"] = pilot.TRAITS[1]

    report = pilot.build_pilot_report(
        {"snippet_only": snippet_rows, "context1000": context_rows},
        sample_ids,
        baseline_trait=pilot.TRAITS[0],
    )

    assert report["evaluation_status"] == "diagnostic_pilot_only"
    assert report["context_comparison"]["paired_outcomes"]["snippet_only_correct_context_wrong"] == 1
    assert report["context_comparison"]["paired_outcomes"]["prediction_changed"] == 1
    assert report["majority_baseline"]["exact_accuracy"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="exactly the planned sample IDs"):
        pilot.build_pilot_report(
            {"snippet_only": snippet_rows, "context1000": context_rows[:-1]},
            sample_ids,
            baseline_trait=pilot.TRAITS[0],
        )


def test_runner_uses_manifest_before_evaluation_and_writes_paired_report(tmp_path, monkeypatch):
    rows = _balanced_rows(items_per_class=12)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    import_manifest = {
        "context_column_used": "context1000",
        "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
        "output_sha256": {"train": pilot.sha256_file(dataset_path)},
        "train_test_document_overlap": 0,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(import_manifest),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    calls = []

    def fake_eval(task, **kwargs):
        assert (results_dir / "sample_manifest.json").is_file()
        assert kwargs["model_args"] == {
            "revision": pilot.EXPECTED_MODEL_REVISION,
            "dtype": "auto",
            "do_sample": False,
        }
        assert kwargs["temperature"] is None
        calls.append(kwargs)
        samples = []
        for sample in task.dataset:
            target = json.loads(sample.target)
            samples.append(SimpleNamespace(
                id=sample.id,
                scores={
                    "dominant_trait_pilot_scorer": SimpleNamespace(
                        value=json.dumps({
                            "sample_id": sample.id,
                            "document_id": sample.metadata["item"]["source"]["document_id"],
                            "predicted": None,
                            "target": target,
                            "invalid": True,
                        }),
                        explanation=target,
                    )
                },
                output=SimpleNamespace(completion=target),
            ))
        return [SimpleNamespace(status="success", samples=samples, location="mock.eval")]

    monkeypatch.setattr(runner, "inspect_eval", fake_eval)
    args = SimpleNamespace(
        dataset=dataset_path,
        model=pilot.EXPECTED_MODEL_ID,
        model_revision=pilot.EXPECTED_MODEL_REVISION,
        dtype="auto",
        seed=42,
        items_per_class=10,
        max_tasks=1,
        max_samples=1,
        run_environment="mocked test",
        results_dir=results_dir,
        log_dir=log_dir,
        prepare_only=True,
    )

    assert runner.run(args) is None
    assert (results_dir / "sample_manifest.json").is_file()
    args.prepare_only = False
    report = runner.run(args)

    assert len(calls) == 2
    assert (results_dir / "sample_manifest.json").is_file()
    assert (results_dir / "report.json").is_file()
    assert report["run"]["sample_count"] == 60
    assert report["arms"]["snippet_only"]["exact_accuracy"] == 1.0
    assert report["arms"]["context1000"]["exact_accuracy"] == 1.0


def test_runner_rescores_saved_raw_completions_without_inference(tmp_path, monkeypatch):
    rows = _balanced_rows(items_per_class=12)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    import_manifest = {
        "context_column_used": "context1000",
        "source_sha256": pilot.EXPECTED_SOURCE_SHA256,
        "output_sha256": {"train": pilot.sha256_file(dataset_path)},
        "train_test_document_overlap": 0,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(import_manifest),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    args = SimpleNamespace(
        dataset=dataset_path,
        model=pilot.EXPECTED_MODEL_ID,
        model_revision=pilot.EXPECTED_MODEL_REVISION,
        dtype="auto",
        seed=42,
        items_per_class=10,
        max_tasks=1,
        max_samples=1,
        run_environment="mocked test",
        results_dir=results_dir,
        log_dir=log_dir,
        prepare_only=True,
        rescore_existing_logs=False,
    )
    assert runner.run(args) is None

    original_report = results_dir / "report.json"
    original_report.write_text('{"decision":"invalid-parser-run"}', encoding="utf-8")
    selected = pilot.select_balanced_sample(rows, items_per_class=10, seed=42)
    saved_logs = {}
    for arm in pilot.ARMS:
        arm_log_dir = log_dir / arm
        arm_log_dir.mkdir(parents=True)
        log_path = arm_log_dir / "saved.eval"
        log_path.write_bytes(b"mock inspect log")
        samples = []
        for row in selected:
            trait = row["dominant_trait"]
            samples.append(SimpleNamespace(
                id=row["id"],
                output=SimpleNamespace(completion=trait),
                scores={
                    "dominant_trait_pilot_scorer": SimpleNamespace(
                        value=json.dumps({
                            "sample_id": row["id"],
                            "document_id": row["source"]["document_id"],
                            "predicted": None,
                            "target": trait,
                            "invalid": True,
                        }),
                        explanation=trait,
                    )
                },
            ))
        saved_logs[str(log_path)] = SimpleNamespace(
            status="success",
            samples=samples,
            location=str(log_path),
        )

    monkeypatch.setattr(
        runner,
        "read_eval_log",
        lambda path: saved_logs[str(path)],
    )
    monkeypatch.setattr(
        runner,
        "inspect_eval",
        lambda *args, **kwargs: pytest.fail(
            f"Rescoring must not run inference: {args}, {kwargs}"
        ),
    )
    args.rescore_existing_logs = True

    report = runner.run(args)

    assert report["run"]["rescored_from_existing_logs"] is True
    assert report["arms"]["snippet_only"]["valid_single_label_rate"] == 1.0
    assert report["arms"]["context1000"]["exact_accuracy"] == 1.0
    assert json.loads(original_report.read_text(encoding="utf-8")) == {
        "decision": "invalid-parser-run"
    }
    assert (results_dir / "report_reparsed.json").is_file()
