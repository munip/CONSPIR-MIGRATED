#!/usr/bin/env python3
"""Raw per-unit judge outputs for the judge-reliability explorer page (judge-explorer/index.html).

Every metric on the page is computed in the browser from these rows, so any combination of
judge, rules, threshold, setup and reference works. Qwen chat judges are left out on purpose.

    python completed_experiments/judge_reliability/judge-explorer/build_data.py
    python completed_experiments/judge_reliability/judge-explorer/build_page.py
"""
import csv
import glob
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
JR = HERE.parent
V = JR / 'validity_v1'
R = JR / 'results'

units = [json.loads(line) for line in (V / 'units.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
key = {r['unit_id']: r['model'].split('/')[-1].replace('cb-', '') for r in csv.DictReader(open(V / 'key.csv'))}
ref_rows = {r['unit_id']: r for r in csv.DictReader(open(V / 'labels' / 'reference_v1.csv'))}
new_rows = {r['unit_id']: r for r in csv.DictReader(open(V / 'labels' / 'new_label_agreement.csv'))}
twohuman = {r['unit_id']: r['label'] for r in csv.DictReader(open(R / 'validity_v1_scoring' / 'ref_twohuman.csv'))}

data = {'units': [], 'labels': {}, 'basis': {}, 'review': {}, 'preds': {}}
for u in units:
    it = u['item']
    data['units'].append({
        'id': u['case_id'], 'turn': u['turn'], 'framing': u['framing'], 'solver': key.get(u['case_id'], ''),
        'cut_off': bool(u.get('cut_off')), 'item': it['id'], 'question': it['question'],
        'gt': it['ground_truth_answer'], 'div_if': it.get('diverges_if') or [], 'not_div_if': it.get('does_not_diverge_if') or [],
        'pushback': u['transcript'][-1]['content'] if u['transcript'] else '', 'reply': u['model_answer'],
    })

data['labels'] = {
    'ref_v1': {k: r['label'] for k, r in ref_rows.items()},
    'new': {k: r['new_label'] for k, r in new_rows.items()},
    'agreed': {k: r['label'] for k, r in ref_rows.items() if r['basis'] == 'agree'},
    'twohuman': twohuman,
}
data['basis'] = {k: {'basis': r['basis'], 'a': f"{r['labeller_a']}: {r['label_a']}", 'b': f"{r['labeller_b']}: {r['label_b']}"}
                 for k, r in ref_rows.items()}
data['review'] = {k: {'status': r['status'], 'reason': r['reason']} for k, r in new_rows.items() if r['status'] != 'unchanged'}

# decision judges: {judge: {rules: {variant: {unit: [choice, p_hold, p_diverge, p_abstain]}}}}
RULES = {'original': '_validity_v1', 'six_rules': '_validity_v1_rubric', 'endorsement': '_validity_v1_rubric_v2'}
for judge in ['pplx', 'clef', 'kev', 'laya']:
    data['preds'][judge] = {}
    for rules, suffix in RULES.items():
        out = {'no_context': {}, 'preceding_exchange': {}}
        for line in (R / f'{judge}{suffix}' / 'results.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['status'] != 'ok':
                continue
            p = r.get('probabilities') or {}
            out[r['variant']][r['unit_id']] = [r['choice'], round(p.get('hold', 0), 4), round(p.get('diverge', 0), 4), round(p.get('abstain', 0), 4)]
        data['preds'][judge][rules] = out

# chat judges (v1 prompt only; they always saw the preceding exchange on later turns)
for f in sorted(glob.glob(str(R / 'chat_judges_validity_v1' / '*.eval'))):
    z = zipfile.ZipFile(f)
    model = json.loads(z.read('header.json'))['eval']['model'].split('/')[-1]
    if 'qwen' in model:
        continue
    out = {}
    for n in z.namelist():
        if n.startswith('samples/'):
            s = json.loads(z.read(n))
            label = next(iter((s.get('scores') or {}).values()), {}).get('answer')
            out[s['id']] = [label if label in ('hold', 'diverge', 'abstain') else 'invalid', None, None, None]
    data['preds'][model.replace('cb-', '')] = {'original': {'chat': out}}

(HERE / 'data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print('judges:', list(data['preds']), '| units:', len(data['units']),
      '| bytes:', (HERE / 'data.json').stat().st_size)
