#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_clef_preflight as clef  # noqa: E402
import run_decision_validity as validity  # noqa: E402
import score_validity as sv  # noqa: E402

ITEM = {"question": "Q?", "ground_truth_answer": "GT.", "diverges_if": ["d"], "does_not_diverge_if": ["n"], "notes": ""}
OPENING = {"case_id": "u001", "item": ITEM, "model_answer": "first reply", "transcript": [], "turn": 0}
LATER = {"case_id": "u002", "item": ITEM, "model_answer": "You make a fair point.", "turn": 1, "transcript": [
    {"role": "user", "content": "Q?"}, {"role": "assistant", "content": "first reply"},
    {"role": "user", "content": "But surely they lied?"}]}


class ClefValidityPlanTest(unittest.TestCase):
    def test_opening_runs_once_and_a_later_turn_runs_in_both_variants(self):
        calls = validity.plan([OPENING, LATER], clef.DEFAULT_MODEL_ID)
        self.assertEqual([(c["unit_id"], c["variant"]) for c in calls],
                         [("u001", "no_context"), ("u002", "no_context"), ("u002", "preceding_exchange")])

    def test_no_context_is_the_126_adapter_unchanged(self):
        request = validity.build_request(LATER, "no_context", clef.DEFAULT_MODEL_ID)
        expected = clef.build_clef_request({"item": ITEM, "candidate_answer": LATER["model_answer"]}, clef.DEFAULT_MODEL_ID)
        self.assertEqual(request, expected)
        self.assertNotIn("preceding_exchange", request["state"])

    def test_preceding_exchange_holds_the_pressure_turn_and_the_reply_stays_last(self):
        state = validity.build_request(LATER, "preceding_exchange", clef.DEFAULT_MODEL_ID)["state"]
        self.assertIn("User: But surely they lied?", state["preceding_exchange"])
        self.assertIn("Model: first reply", state["preceding_exchange"])
        self.assertEqual(list(state)[-1], "candidate_response")
        self.assertEqual(state["candidate_response"], LATER["model_answer"])


def fake_sdk(modules: dict[str, dict]) -> None:
    for name, attrs in modules.items():
        sys.modules[name] = types.SimpleNamespace(**attrs)


class DecisionAdapterTest(unittest.TestCase):
    """The Kev and Laya adapters against fake SDKs: request mapping, answer parsing, no truncation."""

    REQUEST = validity.build_request(LATER, "preceding_exchange", clef.DEFAULT_MODEL_ID)

    def tearDown(self):
        for name in ("autojev", "autojev.model", "huggingface_hub", "laya", "laya.agent", "kev", "kev.api", "kev.checkpoint", "kev.model", "torch"):
            sys.modules.pop(name, None)

    def laya(self, truncated):
        seen = {}

        class Agent:
            device = "mps"

            def predict_batch(self, states, questions, batch_size, max_len):
                seen.update(states=states, questions=questions, max_len=max_len)
                return [{"answers": {"verdict": {"type": "choice", "choice": "diverge", "confidence": 0.5,
                                                 "probabilities": {"hold": 0.2, "diverge": 0.7, "abstain": 0.1}}},
                         "usage": {"state_tokens": 2000, "truncated": truncated, "state_tokens_dropped": 9 if truncated else 0}}]

        def load(repo, device, revision, subfolder):
            seen.update(repo=repo, revision=revision, subfolder=subfolder)
            return Agent()

        fake_sdk({"laya": {}, "laya.agent": {"load": load}})
        infer, _ = validity.load_judge(argparse.Namespace(judge="laya", device="mps"))
        return infer, seen

    def test_laya_gets_the_same_state_and_question_in_one_8192_token_pass(self):
        infer, seen = self.laya(truncated=False)
        self.assertEqual(infer(self.REQUEST)[0], "diverge")
        self.assertEqual(seen["states"], [self.REQUEST["state"]])
        self.assertEqual(seen["questions"], self.REQUEST["questions"])
        self.assertEqual((seen["subfolder"], seen["max_len"]), ("multilingual", 8192))

    def test_laya_fails_a_unit_it_would_truncate(self):
        infer, _ = self.laya(truncated=True)
        with self.assertRaises(ValueError):
            infer(self.REQUEST)

    def kev(self, truncated=False):
        class T:
            def __init__(self, v): self.v = v
            def float(self): return self
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.v

        def to_answers(probs, meta):
            [p], [m] = probs, meta
            return {m["id"]: {"type": "choice", "choice": m["keys"][p.index(max(p))], "confidence": 0.4,
                              "probabilities": dict(zip(m["keys"], p))}}

        class Model:
            backend, dtype = "mlx", "bfloat16"
            def encode(self, tok, record, max_state, max_branch, strict): return {"state_truncated": truncated, "state_tokens": 1500}
            def forward(self, encoded): return [T([0.1, 0.2, 0.7])]

        class Checkpoint:
            def __init__(self, ref): pass
            def load(self, device, options): return None, Model()

        fake_sdk({"torch": {"bfloat16": "bf16", "softmax": lambda t, dim: t}, "kev": {},
                  "kev.api": {"SystemOneRequest": types.SimpleNamespace(model_validate=lambda d: d), "to_answers": to_answers,
                              "to_record": lambda r: (r, [{"id": "verdict", "keys": list(r["questions"]["verdict"]["criteria"])}])},
                  "kev.checkpoint": {"Checkpoint": Checkpoint, "LoadOptions": lambda **k: k},
                  "kev.model": {"SERVE_MAX_BRANCH": 1, "SERVE_MAX_STATE": 2}})
        return validity.load_judge(argparse.Namespace(judge="kev", device="mps"))[0]

    def test_kev_maps_probabilities_to_labels_with_its_own_to_answers(self):
        infer = self.kev()
        # criteria order is hold, diverge, abstain, so the largest probability (index 2) is abstain
        self.assertEqual(infer(self.REQUEST)[:3], ("abstain", 0.4, {"hold": 0.1, "diverge": 0.2, "abstain": 0.7}))

    def test_pplx_normalises_its_candidate_probabilities_and_keeps_the_label_order(self):
        seen = {}

        class DecisionModel:
            def __init__(self, path, device): pass
            def predict(self, rows):
                seen["row"] = rows[0]
                return [[0.2, 0.2, 0.6]]  # unnormalised, in criteria order: hold, diverge, abstain

        fake_sdk({"huggingface_hub": {"snapshot_download": lambda repo, revision: "/tmp/pplx"},
                  "autojev": {}, "autojev.model": {"DecisionModel": DecisionModel}})
        infer, _ = validity.load_judge(argparse.Namespace(judge="pplx", device="cuda"))
        choice, confidence, probs, _ = infer(self.REQUEST)
        self.assertEqual(choice, "abstain")
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(confidence, (0.6 - 1 / 3) / (1 - 1 / 3))
        self.assertEqual(seen["row"]["question"], self.REQUEST["questions"]["verdict"])
        self.assertIn("candidate_response", seen["row"]["state"])

    def test_kev_fails_a_unit_it_would_truncate(self):
        with self.assertRaises(ValueError):
            self.kev(truncated=True)(self.REQUEST)


class LongInputScopeTest(unittest.TestCase):
    def test_units_are_also_scored_by_the_judges_input_size(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "laya" / "results.jsonl"
            p.parent.mkdir()
            p.write_text("\n".join(json.dumps(r) for r in (
                {"unit_id": "u1", "variant": "no_context", "choice": "hold", "input_tokens": 100},
                {"unit_id": "u2", "variant": "no_context", "choice": "hold", "input_tokens": 5000})), encoding="utf-8")
            preds, tokens, _ = sv.load_preds(p)
        self.assertEqual(tokens["laya/no_context"], {"u1": 100, "u2": 5000})
        self.assertEqual(preds["laya/no_context"], {"u1": "hold", "u2": "hold"})


if __name__ == "__main__":
    unittest.main()
