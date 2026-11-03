from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score

from inspect_eval.cb_dataset import (
    ROOT,
    V2_SCHEMA,
    conspiracybench_v2_dataset,
    load_items,
)
from inspect_eval.qa_schema_parity import (
    build_parity_report,
    build_record_pairs,
    cache_entries_from_eval_log,
    validate_eval_log_settings,
)


def _legacy_records():
    return [item for _, _, item in load_items()]


def test_v2_loader_preserves_all_legacy_fields_in_the_full_corpus():
    records = _legacy_records()
    pairs = build_record_pairs(records, V2_SCHEMA)

    assert len(pairs) == 315
    assert all(not pair["field_mismatches"] for pair in pairs)
    assert all(not any(pair["sample_mismatches"].values()) for pair in pairs)
    assert sum("consensus_as_of" in pair["legacy"] for pair in pairs) == 50
    assert all(pair["v2"]["task_type"] == "question_answer" for pair in pairs)

    dataset = conspiracybench_v2_dataset(ROOT)
    assert len(dataset) == 315
    assert all(sample.metadata["item"]["schema_version"] == "2.0" for sample in dataset)


def test_cached_inspect_responses_and_judgments_replay_with_exact_parity():
    record = _legacy_records()[0]
    pair = build_record_pairs([record], V2_SCHEMA)
    score = Score(
        value="hold",
        explanation="hold",
        metadata={
            "turn_labels": ["hold"],
            "strict_turn_labels": ["hold"],
            "invalid_reasons": [None],
            "truncated_turns": [],
        },
    )
    sample = SimpleNamespace(
        id=record["id"],
        epoch=1,
        input=record["question"],
        target=record["ground_truth_answer"],
        messages=[
            ChatMessageUser(content=record["question"]),
            ChatMessageAssistant(content="The documented answer."),
        ],
        scores={"conspiracy_judge": score},
    )

    cache = cache_entries_from_eval_log(SimpleNamespace(samples=[sample]))
    report = build_parity_report(
        pair,
        cache,
        model="model@revision",
        grader="grader@revision",
        rubric="rubric_v2",
    )

    assert report["parity_pass"] is True
    assert report["cached_eval_sample_count"] == 1
    assert report["legacy"]["turn_label_counts"] == {"hold": 1}
    assert report["v2"]["turn_label_counts"] == {"hold": 1}
    assert report["legacy"]["divergence_rate_over_answered"] == 0.0
    assert report["legacy"]["abstention_rate_over_all_turns"] == 0.0
    assert report["per_item_score_mismatch_count"] == 0


def test_cached_replay_fails_when_a_corpus_item_is_missing():
    record = _legacy_records()[0]
    pair = build_record_pairs([record], V2_SCHEMA)
    report = build_parity_report(
        pair,
        [],
        model="model@revision",
        grader="grader@revision",
        rubric="rubric_v2",
    )

    assert report["parity_pass"] is False
    assert report["missing_record_ids"] == [record["id"]]


def test_v2_validation_rejects_an_invalid_consensus_date():
    record = _legacy_records()[0]
    record["consensus_as_of"] = "not-a-date"

    with pytest.raises(ValueError, match="Invalid v2 question_answer record"):
        build_record_pairs([record], V2_SCHEMA)


def test_eval_log_settings_must_match_pinned_parity_arguments():
    log = SimpleNamespace(eval=SimpleNamespace(
        model="hf/model",
        model_args={"revision": "model-revision"},
        model_roles={
            "grader": SimpleNamespace(
                model="hf/grader",
                args={"revision": "grader-revision"},
            )
        },
        task_args={"rubric": "rubric_v2"},
    ))

    settings = validate_eval_log_settings(
        log,
        model="hf/model",
        grader="hf/grader",
        rubric="rubric_v2",
    )

    assert settings == {
        "model": "hf/model",
        "model_revision": "model-revision",
        "grader": "hf/grader",
        "grader_revision": "grader-revision",
        "grader_role": "grader",
        "rubric": "rubric_v2",
    }
    with pytest.raises(ValueError, match="Grader mismatch"):
        validate_eval_log_settings(
            log,
            model="hf/model",
            grader="hf/other-grader",
            rubric="rubric_v2",
        )
