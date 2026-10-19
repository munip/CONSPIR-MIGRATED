"""Detector comparison contracts without inference or judge calls."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest import mock

from completed_experiments.judge_injection import compare_detectors as comparison


class ComparisonTest(unittest.TestCase):
    def test_frozen_counts_and_balanced_attack_families(self):
        original, _ = comparison.validate_fixtures()
        self.assertEqual(len(original), 140)
        counts = comparison.Counter(c['attack_family'] for c in original if c['injection_label'] == 'malicious')
        self.assertEqual(len(counts), 7)
        self.assertEqual(set(counts.values()), {10})
        self.assertEqual(sum(c['injection_label'] == 'benign' and c['pair_subtype'] == 'hard_negative' for c in original), 21)

    def test_fixture_hash_mismatch_is_rejected(self):
        with mock.patch.object(comparison.RELEASE_PATH.__class__, 'read_bytes', return_value=b'changed'):
            with self.assertRaisesRegex(ValueError, 'Frozen fixture hash mismatch'):
                comparison.validate_fixtures()

    def test_original_quotation_controls_and_acceptance(self):
        row = dict(case_id='quoted', injection_label='benign', pair_subtype='hard_negative',
                   attack_family=None, action='BLOCK', wall_duration_ms=1)
        result = comparison.summarize([row])
        self.assertEqual(result['quotation_false_positive_case_ids'], ['quoted'])
        result.update(attacks_caught=70)
        self.assertTrue(comparison.acceptance(result))
        result['false_positive_case_ids'] = ['a', 'b', 'c']
        self.assertFalse(comparison.acceptance(result))

    def test_run_scans_only_frozen_cases_and_resume_skips_completed_profiles(self):
        original, _ = comparison.validate_fixtures()
        expected_ids = [c['case_id'] for c in original]
        def saved_rows(scanner, cases, profile):
            self.assertEqual([c['case_id'] for c in cases], expected_ids)
            return [dict(case_id=c['case_id'], injection_label=c['injection_label'],
                         attack_family=c['attack_family'], pair_subtype=c['pair_subtype'],
                         action='BLOCK' if profile.startswith(('rules', 'combined')) and c['injection_label'] == 'malicious' else 'PASS',
                         wall_duration_ms=1,
                         local_scores={'cb_semantic_classifier': .9 if c['injection_label'] == 'malicious' else .1})
                    for c in cases]
        scanner = mock.Mock(effective_config={'profile': 'mock'})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comparison'
            with mock.patch.object(comparison, 'PromptShieldScanner', return_value=scanner) as factory, mock.patch.object(comparison, 'scan_cases', side_effect=saved_rows) as scan:
                result = comparison.run(output)
                self.assertEqual(scan.call_count, 4)
                self.assertEqual(factory.call_count, 4)
                self.assertEqual(result['judge_calls'], 0)
                rows = json.loads((output/'scans.json').read_text())
                self.assertTrue(all(set(v) == {'original'} for v in rows.values()))
                self.assertEqual(sum(len(v['original']) for v in rows.values()), 560)
                comparison.run(output, resume=True)
                self.assertEqual(factory.call_count, 4)

    def test_threshold_uses_original_scores_and_preserves_native_scan(self):
        scores = [.9, .6, .7]
        labels = ['malicious', 'malicious', 'benign']
        baseline = [{'case_id': str(i), 'action': 'PASS'} for i in range(3)]
        rows = [{'case_id': str(i), 'action': 'BLOCK', 'attack_family': 'test',
                 'injection_label': label, 'wall_duration_ms': 1,
                 'local_scores': {'cb_semantic_classifier': score},
                 'config': {'ml_threshold': .5}}
                for i, (score, label) in enumerate(zip(scores, labels))]
        threshold, evaluated = comparison.select_threshold(rows, baseline)
        self.assertEqual(threshold, .6)
        self.assertEqual(comparison.summarize(evaluated)['attacks_caught'], 2)
        self.assertEqual(rows[0]['config']['ml_threshold'], .5)
        self.assertNotIn('evaluated_action', rows[0])
        rows[1]['error'] = 'scan failed'
        rows[1]['local_scores'] = {}
        with self.assertRaises(RuntimeError):
            comparison.select_threshold(rows, baseline)

    def test_scan_failures_are_reported_and_no_judge_is_imported(self):
        scanner = mock.Mock()
        scanner.scan.side_effect = RuntimeError('SCAN SENTINEL')
        rows = comparison.scan_cases(scanner, [{'case_id': 'x', 'model_answer': 'text',
                                      'injection_label': 'malicious', 'attack_family': 'test'}], 'test')
        summary = comparison.summarize(rows)
        self.assertEqual(summary['scan_failures'], [{'case_id': 'x', 'error': 'SCAN SENTINEL'}])
        self.assertEqual(summary['attacks_caught'], 0)
        self.assertIsNone(summary['warm_p50_ms'])


if __name__ == '__main__':
    unittest.main()
