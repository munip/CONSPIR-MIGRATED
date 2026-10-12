"""Experiment 2: frozen-set Prompt Guard comparison, without judge calls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'inspect_eval'))
from eval.judge_injection.compare_detectors import dependency_versions, percentile, summarize, validate_fixtures
from eval.judge_injection.load_dataset import project_scan_input
from eval.judge_injection.prompt_guard import MODEL_ID, MODEL_REVISION, PromptGuardScanner
from promptshield import PromptShieldScanner

EXPERIMENT = Path(__file__).with_name('experiments')/'002-prompt-guard-2-86m'


def detected(row):
    return row.get('action') in ('BLOCK', 'FLAG')


def evaluate(model_rows, baseline_rows, rules_rows, threshold):
    baseline = {r['case_id']: r for r in baseline_rows}
    rules = {r['case_id']: r for r in rules_rows}
    output = {}
    for profile in ('prompt_guard', 'baseline_prompt_guard', 'rules_prompt_guard'):
        rows = []
        for row in model_rows:
            value = dict(row)
            references = [] if profile == 'prompt_guard' else [
                (baseline if profile == 'baseline_prompt_guard' else rules)[row['case_id']]]
            value['error'] = row.get('error') or next((r.get('error') for r in references if r.get('error')), None)
            value['evaluated_action'] = ('BLOCK' if row.get('score', -1) >= threshold or any(detected(r) for r in references) else 'PASS') if not value['error'] else None
            rows.append(value)
        output[profile] = summarize(rows)
        # Recombined profiles have no independently measured end-to-end latency.
        if profile != 'prompt_guard':
            output[profile].update(warm_p50_ms=None, warm_p95_ms=None)
    return output


def threshold_curve(model_rows, baseline_rows, rules_rows):
    if any(r.get('error') for r in model_rows + baseline_rows + rules_rows) or any(r.get('score') is None for r in model_rows):
        raise ValueError('Cannot calibrate with failed or missing scans')
    import math
    scores = [r['score'] for r in model_rows]
    candidates = sorted({.5, 1., *scores, *[math.nextafter(s, 1.) for s in scores if s < 1.]})
    curve = []
    for threshold in candidates:
        metrics = evaluate(model_rows, baseline_rows, rules_rows, threshold)
        for profile, values in metrics.items():
            curve.append(dict(profile=profile, threshold=threshold,
                              attacks_caught=values['attacks_caught'],
                              benign_false_positives=len(values['false_positive_case_ids'])))
    selected = {}
    for profile in ('prompt_guard', 'baseline_prompt_guard', 'rules_prompt_guard'):
        points = [v for v in curve if v['profile'] == profile]
        best = max(points, key=lambda v: (v['benign_false_positives'] <= 2,
                                          v['attacks_caught'], -v['benign_false_positives'], v['threshold']))
        selected[profile] = dict(threshold=best['threshold'],
                                budget_feasible=best['benign_false_positives'] <= 2,
                                metrics=evaluate(model_rows, baseline_rows, rules_rows, best['threshold'])[profile])
    return curve, selected


def save(output, summary, rows):
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'scans.json.gz').write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0))


def run(output):
    cases, fixture = validate_fixtures()
    protocol = json.loads((EXPERIMENT/'protocol.json').read_text())
    if protocol['fixture_sha256'] != fixture['release_sha256']:
        raise ValueError('Protocol fixture mismatch')
    if protocol['model_revision'] != MODEL_REVISION or protocol['model_id'] != MODEL_ID:
        raise ValueError('Protocol model mismatch')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(experiment=2, status='initializing', started_at=datetime.now(timezone.utc).isoformat(),
                   protocol=protocol, protocol_sha256=hashlib.sha256((EXPERIMENT/'protocol.json').read_bytes()).hexdigest(),
                   fixture=fixture, judge_calls=0, completed_scans=0, initialization={},
                   runtime=dict(python=sys.version, platform=platform.platform(), dependencies=dependency_versions()))
    summary['source_sha256'] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in ('eval/judge_injection/run_prompt_guard.py', 'eval/judge_injection/prompt_guard.py', 'eval/judge_injection/load_dataset.py', 'inspect_eval/promptshield.py', 'inspect_eval/promptshield_detectors.py')}
    summary['runtime']['environment'] = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}
    rows = []
    save(output, summary, rows)
    scanners = {'prompt_guard': PromptGuardScanner(), 'baseline': PromptShieldScanner(),
                'rules': PromptShieldScanner(profile='rules')}
    # Load the gated model first, so access failure does not waste baseline scans.
    for name, scanner in scanners.items():
        started = time.perf_counter()
        try:
            scanner.initialize()
            if name == 'prompt_guard':
                config = scanner.effective_config
                if (config['window_tokens'], config['stride_tokens']) != (protocol['window_tokens'], protocol['stride_tokens']):
                    raise ValueError('Resolved token windows differ from frozen protocol')
                summary['runtime']['torch_num_threads'] = scanner.torch.get_num_threads()
                if summary['runtime']['torch_num_threads'] != protocol['omp_num_threads']:
                    raise ValueError('CPU thread count differs from frozen protocol; set OMP_NUM_THREADS=2')
            load_ms = (time.perf_counter()-started)*1000
            warm_started = time.perf_counter()
            scanner.scan('Ordinary factual answer for warm-up.')
            summary['initialization'][name] = dict(load_ms=load_ms, warmup_ms=(time.perf_counter()-warm_started)*1000,
                                                    config=scanner.effective_config)
        except Exception as exc:
            summary['status'] = 'initialization_error'
            summary['initialization'][name] = dict(elapsed_ms=(time.perf_counter()-started)*1000,
                                                    error_type=type(exc).__name__, error=str(exc))
            summary['finished_at'] = datetime.now(timezone.utc).isoformat()
            save(output, summary, rows)
            return summary
    summary['status'] = 'scanning'
    for pass_index, seed in enumerate(protocol['timing_seeds'], 1):
        order = list(cases)
        rng = random.Random(seed)
        rng.shuffle(order)
        profiles = list(scanners)
        rng.shuffle(profiles)
        for name in profiles:
            for index, case in enumerate(order):
                text = project_scan_input(case)
                row = {k: case[k] for k in ('case_id', 'injection_label', 'attack_family', 'pair_subtype')}
                row.update(profile=name, timing_pass=pass_index, position=index,
                           input_hash=hashlib.sha256(text.encode('utf-8')).hexdigest())
                started = time.perf_counter()
                try:
                    row.update(scanners[name].scan(text))
                except Exception as exc:
                    row.update(action=None, error_type=type(exc).__name__, error=str(exc))
                row['wall_duration_ms'] = (time.perf_counter()-started)*1000
                rows.append(row)
                if (index+1) % 20 == 0:
                    print(f'pass {pass_index}, {name}: {index+1}/{len(order)}', flush=True)
            summary['completed_scans'] = len(rows)
            save(output, summary, rows)
    first = {p: [r for r in rows if r['profile'] == p and r['timing_pass'] == 1] for p in scanners}
    summary['default_threshold'] = .5
    summary['default_results'] = evaluate(first['prompt_guard'], first['baseline'], first['rules'], .5)
    for name in ('baseline', 'rules'):
        summary['default_results'][name] = summarize(first[name])
    summary['latency'] = {}
    for name in scanners:
        passes = [[r for r in rows if r['profile'] == name and r['timing_pass'] == i] for i in range(1, len(protocol['timing_seeds'])+1)]
        successful = [r['wall_duration_ms'] for group in passes for r in group if not r.get('error')]
        summary['latency'][name] = dict(warm_p50_ms=percentile(successful, .5), warm_p95_ms=percentile(successful, .95),
                                        successful_scans=len(successful), per_pass=[summarize(group) for group in passes])
        reference = {r['case_id']: r['action'] for r in passes[0]}
        summary['latency'][name]['action_disagreements_across_passes'] = sorted({r['case_id'] for group in passes[1:] for r in group if r['action'] != reference[r['case_id']]})
    try:
        curve, selected = threshold_curve(first['prompt_guard'], first['baseline'], first['rules'])
        summary['selected_in_sample'] = selected
        with (output/'thresholds.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]), lineterminator='\n')
            writer.writeheader();writer.writerows(curve)
    except ValueError as exc:
        summary['calibration_error'] = str(exc)
    summary['scan_failures'] = [{k: r[k] for k in ('case_id', 'profile', 'timing_pass', 'error')} for r in rows if r.get('error')]
    summary['status'] = 'completed_with_scan_errors' if summary['scan_failures'] else 'completed'
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(output, summary, rows)
    return summary


def analyze_saved(attempt):
    """Repair calibration from saved scores; preserve native summary and scans."""
    summary = json.loads((attempt/'summary.json').read_text())
    rows = json.loads(gzip.decompress((attempt/'scans.json.gz').read_bytes()))
    first = {p: [r for r in rows if r['profile'] == p and r['timing_pass'] == 1]
             for p in ('prompt_guard', 'baseline', 'rules')}
    if summary['status'] != 'completed' or any(len(v) != 140 for v in first.values()):
        raise ValueError('Saved analysis requires a completed 140-case comparison')
    curve, selected = threshold_curve(first['prompt_guard'], first['baseline'], first['rules'])
    summary['selected_in_sample'] = selected
    summary['calibration_repair'] = dict(original_error=summary.pop('calibration_error', None),
                                         method='Saved first-pass scores; baseline/rules require actions, not classifier scores. No extra inference.',
                                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (attempt/'thresholds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(curve)
    (attempt/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path)
    group.add_argument('--analyze-attempt', type=Path)
    args = parser.parse_args()
    result = analyze_saved(args.analyze_attempt) if args.analyze_attempt else run(args.output_dir)
    print(json.dumps(dict(status=result['status'], completed_scans=result['completed_scans']), indent=2))
    sys.exit(0 if result['status'] == 'completed' else 1)
