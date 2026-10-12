"""Prompt Guard experiment contracts; no model downloads or judge calls."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from eval.judge_injection.prompt_guard import PromptGuardScanner
from eval.judge_injection import run_prompt_guard as experiment


class FakeTokenizer:
    def __init__(self):
        self.chunks = []

    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return [1 if c == '!' else 0 for c in text]

    def prepare_for_model(self, chunk, **kwargs):
        assert kwargs['prepend_batch_axis'] is True
        assert kwargs['truncation'] is False
        self.chunks.append(chunk)
        return {'input_ids': torch.tensor([chunk])}


class PromptGuardTest(unittest.TestCase):
    def scanner(self, model=None):
        scanner = PromptGuardScanner()
        scanner.torch = torch
        scanner.window_size = 10
        scanner.stride = 5
        scanner.malicious_index = 1
        scanner.tokenizer = FakeTokenizer()
        scanner.model = model or (lambda input_ids: SimpleNamespace(
            logits=torch.tensor([[-6., 6.] if input_ids.any() else [6., -6.]])))
        return scanner

    def test_full_input_and_overlap_find_long_tail_without_window_cap(self):
        scanner = self.scanner()
        result = scanner.scan('a'*100+'!')
        self.assertEqual(result['action'], 'BLOCK')
        self.assertEqual(result['token_count'], 101)
        self.assertGreater(result['window_count'], 8)
        self.assertIn(1, scanner.tokenizer.chunks[-1])
        self.assertTrue(result['config']['full_input'])
        self.assertEqual(result['input_length'], 101)

    def test_empty_input_is_scanned_once(self):
        result = self.scanner().scan('')
        self.assertEqual(result['window_count'], 1)
        self.assertEqual(result['action'], 'PASS')

    def test_inference_and_nonfinite_errors_surface(self):
        with self.assertRaisesRegex(RuntimeError, 'INFERENCE SENTINEL'):
            self.scanner(mock.Mock(side_effect=RuntimeError('INFERENCE SENTINEL'))).scan('text')
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.scanner(lambda **kwargs: SimpleNamespace(logits=torch.tensor([[float('nan'), 1.]]))).scan('text')

    def test_pinned_generic_labels_use_verified_meta_index(self):
        tokenizer = mock.Mock(model_max_length=512)
        tokenizer.num_special_tokens_to_add.return_value = 2
        model = mock.Mock()
        model.config = SimpleNamespace(id2label={0: 'LABEL_0', 1: 'LABEL_1'}, num_labels=2, max_position_embeddings=512)
        model.to.return_value = model
        model.eval.return_value = model
        with mock.patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), mock.patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=model):
            scanner = PromptGuardScanner()
            scanner.initialize()
            self.assertEqual(scanner.malicious_index, 1)
            self.assertEqual(scanner.window_size, 510)
            model.config.id2label = {0: 'UNEXPECTED', 1: 'LABEL_1'}
            with self.assertRaisesRegex(ValueError, 'Unexpected Prompt Guard labels'):
                PromptGuardScanner().initialize()

    def test_loading_error_surfaces(self):
        with mock.patch('transformers.AutoTokenizer.from_pretrained', side_effect=OSError('LOAD SENTINEL')):
            with self.assertRaisesRegex(OSError, 'LOAD SENTINEL'):
                PromptGuardScanner().initialize()

    def test_or_results_preserve_raw_default_actions_and_no_derived_latency(self):
        rows = [dict(case_id='x', injection_label='malicious', attack_family='direct_overrides',
                     pair_subtype='ordinary', action='PASS', score=.4, wall_duration_ms=2)]
        baseline = [dict(rows[0], action='BLOCK')]
        result = experiment.evaluate(rows, baseline, rows, .5)
        self.assertEqual(result['prompt_guard']['attacks_caught'], 0)
        self.assertEqual(result['baseline_prompt_guard']['attacks_caught'], 1)
        self.assertIsNone(result['baseline_prompt_guard']['warm_p50_ms'])
        self.assertEqual(rows[0]['action'], 'PASS')
        baseline[0]['error'] = 'BASELINE FAILED'
        result = experiment.evaluate(rows, baseline, rows, .5)
        self.assertEqual(len(result['baseline_prompt_guard']['scan_failures']), 1)

    def test_calibration_uses_baseline_actions_without_model_scores(self):
        rows = [dict(case_id='attack', injection_label='malicious', attack_family='direct_overrides',
                     action='BLOCK', score=.9, wall_duration_ms=1),
                dict(case_id='benign', injection_label='benign', attack_family=None,
                     action='PASS', score=.1, wall_duration_ms=1)]
        reference = [{k: v for k, v in r.items() if k != 'score'} for r in rows]
        _, selected = experiment.threshold_curve(rows, reference, reference)
        self.assertEqual(selected['baseline_prompt_guard']['metrics']['attacks_caught'], 1)
        self.assertTrue(all('score' not in r for r in reference))

    def test_calibration_rejects_failures(self):
        with self.assertRaisesRegex(ValueError, 'failed or missing'):
            experiment.threshold_curve([dict(score=None)], [], [])

    def test_complete_mock_run_keeps_140_cases_and_three_timing_passes(self):
        class FakeScanner:
            effective_config = {'window_tokens': 510, 'stride_tokens': 255}
            torch = SimpleNamespace(get_num_threads=lambda: 2)
            def initialize(self):
                pass
            def scan(self, text):
                return dict(action='PASS', score=.1)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiment, 'PromptGuardScanner', return_value=FakeScanner()), mock.patch.object(experiment, 'PromptShieldScanner', side_effect=lambda **kwargs: FakeScanner()), mock.patch('builtins.print'):
                result = experiment.run(Path(directory)/'attempt')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['completed_scans'], 1260)
            self.assertEqual(result['default_results']['prompt_guard']['attack_count'], 70)
            self.assertIsNone(result['default_results']['rules_prompt_guard']['warm_p95_ms'])
            self.assertEqual(result['latency']['prompt_guard']['successful_scans'], 420)
            self.assertEqual(result['latency']['prompt_guard']['action_disagreements_across_passes'], [])
            self.assertEqual(result['judge_calls'], 0)

    def test_initialization_failure_is_saved_without_case_scans(self):
        scanner = mock.Mock()
        scanner.initialize.side_effect = RuntimeError('ACCESS SENTINEL')
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiment, 'PromptGuardScanner', return_value=scanner):
                output = Path(directory)/'attempt'
                result = experiment.run(output)
            self.assertEqual(result['status'], 'initialization_error')
            self.assertEqual(result['completed_scans'], 0)
            self.assertEqual(result['judge_calls'], 0)
            self.assertTrue((output/'summary.json').exists())
            scanner.scan.assert_not_called()
            with self.assertRaises(FileExistsError):
                experiment.run(output)


if __name__ == '__main__':
    unittest.main()
