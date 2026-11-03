import asyncio
import json
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Target

import scripts.run_trait_prompt_experiment as experiment_runner
from inspect_eval.trait.prompt_context_experiment import (
    ARMS,
    SCORED_TRAITS,
    _parse_six_trait_prediction,
    build_experiment_report,
    build_fold_plan,
    build_sample_input,
    build_system_prompt,
    load_experiment_data,
    metric_summary,
    paired_document_bootstrap,
    trait_prompt_experiment_scorer,
)


def _record(record_id, document_id, present=(), context="Document context"):
    return {
        "id": record_id,
        "text": f"Snippet {record_id}",
        "trait_annotations": {
            trait: "present" if trait in present else "absent"
            for trait in SCORED_TRAITS
        },
        "source": {"document_id": document_id},
        "metadata": {"context1000": context},
    }


def _score_row(record, predicted=(), invalid=False):
    return {
        "sample_id": record["id"],
        "document_id": record["source"]["document_id"],
        "predicted": list(predicted),
        "target": [
            trait for trait in SCORED_TRAITS
            if record["trait_annotations"][trait] == "present"
        ],
        "known": list(SCORED_TRAITS),
        "invalid": invalid,
    }


def test_loader_excludes_missing_context_consistently_and_checks_hash(tmp_path):
    eligible = _record("with-context", "doc-1", present=SCORED_TRAITS)
    missing = _record("without-context", "doc-2", context=None)
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(eligible) + "\n" + json.dumps(missing) + "\n",
        encoding="utf-8",
    )

    rows, excluded = load_experiment_data(path, expected_sha256=None)
    assert [row["id"] for row in rows] == ["with-context"]
    assert excluded == [{
        "id": "without-context",
        "document_id": "doc-2",
        "reason": "missing_context1000",
    }]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_experiment_data(path, expected_sha256="wrong")


def test_fold_plan_keeps_documents_together_and_uses_training_only_examples():
    rows = [
        _record(f"sample-{index}", f"doc-{index}", present=SCORED_TRAITS)
        for index in range(10)
    ]

    plan = build_fold_plan(rows, folds=5, seed=42)
    again = build_fold_plan(rows, folds=5, seed=42)
    assert plan == again
    assert plan["sample_count"] == 10
    assert plan["document_count"] == 10
    assert sorted(plan["document_to_fold"]) == [f"doc-{index}" for index in range(10)]

    validation_ids = [
        sample_id
        for fold in plan["folds"]
        for sample_id in fold["validation_sample_ids"]
    ]
    assert sorted(validation_ids) == sorted(row["id"] for row in rows)
    fold_sizes = [len(fold["validation_sample_ids"]) for fold in plan["folds"]]
    assert max(fold_sizes) - min(fold_sizes) <= 1
    for fold in plan["folds"]:
        validation_documents = set(fold["validation_document_ids"])
        demonstrations = fold["demonstrations"]
        assert len(demonstrations) == len(SCORED_TRAITS)
        assert len({demo["document_id"] for demo in demonstrations}) == len(demonstrations)
        assert all(demo["document_id"] not in validation_documents for demo in demonstrations)
        assert all(demo["id"] not in fold["validation_sample_ids"] for demo in demonstrations)


def test_prompt_and_sample_inputs_use_only_six_traits_and_optional_nested_context():
    row = _record("one", "doc", present=("nefarious_intent",))
    snippet = build_sample_input(row, include_context=False)
    with_context = build_sample_input(row, include_context=True)
    prompt = build_system_prompt([{
        "text": "Training snippet",
        "traits": ["nefarious_intent"],
    }])

    assert snippet == "Annotated text:\nSnippet one"
    assert "Document context" not in snippet
    assert "Context (context1000):\nDocument context" in with_context
    assert "Training snippet" in prompt
    assert "something_must_be_wrong" not in prompt
    assert _parse_six_trait_prediction('["nefarious_intent"]') == ["nefarious_intent"]
    with pytest.raises(ValueError, match="outside this experiment"):
        _parse_six_trait_prediction('["something_must_be_wrong"]')


def test_invalid_outputs_stay_in_metric_denominators_and_miss_exact_match():
    positive = _record("positive", "doc-1", present=("nefarious_intent",))
    negative = _record("negative", "doc-2")
    rows = [
        _score_row(positive, predicted=("nefarious_intent",)),
        _score_row(negative, invalid=True),
    ]

    summary = metric_summary(rows)
    assert summary["sample_count"] == 2
    assert summary["invalid_output_rate"] == 0.5
    assert summary["exact_match"] == 0.5
    assert summary["per_trait"]["nefarious_intent"]["positive_support"] == 1
    assert summary["candidate_detection"]["f1"] == 1.0
    assert summary["empty_prediction_rate"] == 0.5


def test_scorer_marks_unscored_trait_as_invalid_and_scores_no_prediction():
    row = _record("sample", "document")
    state = SimpleNamespace(
        metadata={"item": row},
        sample_id="sample",
        output=SimpleNamespace(completion='["something_must_be_wrong"]'),
    )

    score = asyncio.run(trait_prompt_experiment_scorer()(state, Target("[]")))
    result = json.loads(score.value)
    assert result["invalid"] is True
    assert result["predicted"] == []
    assert result["known"] == list(SCORED_TRAITS)


def test_paired_document_bootstrap_reports_all_factorial_contrasts():
    positive = _record("positive", "doc-1", present=SCORED_TRAITS)
    negative = _record("negative", "doc-2")
    rows = [
        _score_row(positive, predicted=SCORED_TRAITS),
        _score_row(negative),
    ]
    arms = {arm: rows for arm in ARMS}

    interval = paired_document_bootstrap(arms, repetitions=20, seed=7)
    report = build_experiment_report(arms, repetitions=20, seed=7)

    assert interval["resampling_unit"] == "source.document_id"
    assert set(interval["arm_macro_f1"]) == set(ARMS)
    assert set(interval["contrasts"]) == {
        "combined_minus_snippet_only",
        "combined_minus_all_present",
        "context_main_effect",
        "demonstrations_main_effect",
        "context_demonstrations_interaction",
    }
    assert interval["contrasts"]["context_main_effect"]["estimate"] == 0.0
    assert report["arms"]["context_plus_demos"]["macro_f1"] == 1.0
    assert report["baselines"]["all_six_present"]["sample_count"] == 2


def test_runner_writes_manifest_before_mocked_inspect_calls(tmp_path, monkeypatch):
    rows = [
        _record(f"sample-{index}", f"doc-{index}", present=SCORED_TRAITS)
        for index in range(10)
    ]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text("mock dataset", encoding="utf-8")
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    calls = []

    def fake_eval(task, **kwargs):
        assert (results_dir / "fold_manifest.json").is_file()
        manifest_path = results_dir / "fold_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["model_dtype"] == "auto"
        assert manifest["decoding"] == "greedy"
        assert kwargs["model_args"] == {
            "revision": "revision-1",
            "dtype": "auto",
            "do_sample": False,
        }
        assert kwargs["temperature"] is None
        assert kwargs["epochs"] == 1
        calls.append(kwargs)
        samples = []
        for sample in task.dataset:
            samples.append(SimpleNamespace(
                id=sample.id,
                scores={
                    "trait_prompt_experiment_scorer": SimpleNamespace(
                        value=json.dumps({
                            "sample_id": sample.id,
                            "document_id": sample.metadata["item"]["source"]["document_id"],
                            "predicted": json.loads(sample.target),
                            "target": json.loads(sample.target),
                            "known": list(SCORED_TRAITS),
                            "invalid": False,
                        })
                    )
                },
            ))
        return [SimpleNamespace(status="success", samples=samples, location="mock.eval")]

    monkeypatch.setattr(experiment_runner, "load_experiment_data", lambda path: (rows, []))
    monkeypatch.setattr(experiment_runner, "inspect_eval", fake_eval)
    args = SimpleNamespace(
        dataset=dataset_path,
        model="hf/Qwen/Qwen2.5-3B-Instruct",
        model_revision="revision-1",
        dtype="auto",
        folds=5,
        seed=42,
        temperature=0,
        max_tasks=1,
        run_environment="mocked test",
        bootstrap_repetitions=10,
        results_dir=results_dir,
        log_dir=log_dir,
    )

    report = experiment_runner.run(args)

    assert len(calls) == 5 * len(ARMS)
    assert (results_dir / "fold_manifest.json").is_file()
    assert (results_dir / "report.json").is_file()
    assert report["run"]["sample_count"] == 10


@pytest.mark.parametrize(
    ("smoke", "pilot_samples", "expected_count", "expected_status"),
    [
        (True, None, 1, "smoke_only_not_a_performance_estimate"),
        (False, 2, 2, "pilot_only_not_a_performance_estimate"),
    ],
)
def test_smoke_and_pilot_runner_use_paired_items_without_performance_decisions(
    tmp_path, monkeypatch, smoke, pilot_samples, expected_count, expected_status
):
    rows = [
        _record(f"sample-{index}", f"doc-{index}", present=SCORED_TRAITS)
        for index in range(10)
    ]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text("mock dataset", encoding="utf-8")
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    calls = []

    def fake_eval(task, **kwargs):
        samples = list(task.dataset)
        assert len(samples) == expected_count
        calls.append((kwargs, [sample.id for sample in samples]))
        return [SimpleNamespace(
            status="success",
            location="mock.eval",
            samples=[SimpleNamespace(
                id=sample.id,
                scores={
                    "trait_prompt_experiment_scorer": SimpleNamespace(
                        value=json.dumps({
                            "sample_id": sample.id,
                            "document_id": sample.metadata["item"]["source"]["document_id"],
                            "predicted": [],
                            "target": json.loads(sample.target),
                            "known": list(SCORED_TRAITS),
                            "invalid": False,
                        })
                    )
                },
            ) for sample in samples],
        )]

    monkeypatch.setattr(experiment_runner, "load_experiment_data", lambda path: (rows, []))
    monkeypatch.setattr(experiment_runner, "inspect_eval", fake_eval)
    args = SimpleNamespace(
        dataset=dataset_path,
        model="hf/Qwen/Qwen2.5-3B-Instruct",
        model_revision="revision-1",
        dtype="auto",
        folds=5,
        seed=42,
        temperature=0,
        max_tasks=1,
        max_samples=1,
        run_environment="mocked test",
        bootstrap_repetitions=10,
        results_dir=results_dir,
        log_dir=log_dir,
        smoke=smoke,
        pilot_samples=pilot_samples,
    )

    report = experiment_runner.run(args)

    assert len(calls) == len(ARMS)
    assert all(sample_ids == calls[0][1] for _, sample_ids in calls)
    assert all(kwargs["temperature"] is None for kwargs, _ in calls)
    assert all(kwargs["model_args"]["do_sample"] is False for kwargs, _ in calls)
    assert all(kwargs["max_samples"] == 1 for kwargs, _ in calls)
    assert report["evaluation_status"] == expected_status
    assert report["primary_comparison"]["decision"] == "not_evaluated_pilot"
    assert report["run"]["evaluated_sample_count"] == expected_count
    assert report["run"]["pilot_sample_ids"] == calls[0][1]
    assert report["run"]["smoke_sample_ids"] == (calls[0][1] if smoke else [])
