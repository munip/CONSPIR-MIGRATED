"""Every number in the judge-reliability report, computed from the repo's published files."""
import csv, json, glob, zipfile, collections, re
_JR = __import__('pathlib').Path(__file__).resolve().parents[2]
V = str(_JR / 'validity_v1') + '/'
R = str(_JR / 'results') + '/'
L = ['hold', 'diverge', 'abstain']
OUT = str(__import__('pathlib').Path(__file__).resolve().parent / 'data.json')

units = {u['case_id']: u for u in map(json.loads, open(V + 'units.jsonl'))}
ref_rows = {r['unit_id']: r for r in csv.DictReader(open(V + 'labels/reference_v1.csv'))}
ref = {u: r['label'] for u, r in ref_rows.items()}
new = {r['unit_id']: r['new_label'] for r in csv.DictReader(open(V + 'labels/new_label_agreement.csv'))}
key = {r['unit_id']: r['model'] for r in csv.DictReader(open(V + 'key.csv'))}


def decision(path, variant='no_context'):
    out, probs = {}, {}
    for line in open(path):
        r = json.loads(line)
        if r['variant'] == variant and r['status'] == 'ok':
            out[r['unit_id']] = r['choice']; probs[r['unit_id']] = r['probabilities']
    return out, probs


def chat(name):
    f = next(f for f in glob.glob(R + 'chat_judges_validity_v1/*.eval')
             if json.loads(zipfile.ZipFile(f).read('header.json'))['eval']['model'].endswith(name))
    z = zipfile.ZipFile(f); out = {}
    for n in z.namelist():
        if n.startswith('samples/'):
            s = json.loads(z.read(n))
            out[s['id']] = next(iter((s.get('scores') or {}).values()), {}).get('answer')
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in L) / n / n
    return (po - pe) / (1 - pe)


def stats(pred, reference, units_=None):
    us = [u for u in (units_ or reference) if u in pred]
    pairs = [(pred[u], reference[u]) for u in us]
    called = [b for a, b in pairs if a == 'diverge']
    nd = sum(b == 'diverge' for _, b in pairs); nh = sum(b == 'hold' for _, b in pairs)
    return {'n': len(us), 'kappa': round(kappa(pairs), 3),
            'dprec': round(sum(b == 'diverge' for b in called) / len(called), 3) if called else None,
            'drec': round(sum(a == 'diverge' for a, b in pairs if b == 'diverge') / nd, 3) if nd else None,
            'hrec': round(sum(a == 'hold' for a, b in pairs if b == 'hold') / nh, 3) if nh else None,
            'drate': round(len(called) / len(us), 3),
            'acc': round(sum(a == b for a, b in pairs) / len(us), 3)}


J = {}
P = {}
for j, name in [('pplx', 'pplx-decider'), ('clef', 'Clef'), ('kev', 'Kev-4B'), ('laya', 'Laya')]:
    J[name], P[name] = decision(R + f'{j}_validity_v1/results.jsonl')
for j, name in [('cb-gpt-oss-20b', 'gpt-oss-20b'), ('cb-gemma3-4b', 'Gemma 3 4B'), ('cb-llama3.2-3b', 'Llama 3.2 3B')]:
    J[name] = chat(j)
ORDER = ['pplx-decider', 'Clef', 'gpt-oss-20b', 'Kev-4B', 'Gemma 3 4B', 'Llama 3.2 3B', 'Laya']
USABLE = ['pplx-decider', 'Clef', 'gpt-oss-20b', 'Kev-4B']

data = {'reference': dict(collections.Counter(ref.values())), 'new_reference': dict(collections.Counter(new.values())),
        'human_kappa': 0.497, 'human_agreement': 0.813}
data['judges'] = [{'name': n, **stats(J[n], ref)} for n in ORDER]

agree = [u for u, r in ref_rows.items() if r['basis'] == 'agree']
disputed = [u for u, r in ref_rows.items() if r['basis'] != 'agree']
data['agreed_vs_disputed'] = [{'name': n, 'agreed': stats(J[n], ref, agree), 'disputed': stats(J[n], ref, disputed)}
                              for n in ORDER]
data['n_agreed'], data['n_disputed'] = len(agree), len(disputed)
data['basis_counts'] = {
    'Both labellers agreed': len(agree),
    'Adjudicated (#355)': sum(r['basis'].startswith('adjudicated') for r in ref_rows.values()),
    'Human vs human, resolved by rule': sum(r['basis'].startswith('disagree: likelytobelaura') for r in ref_rows.values()),
    'Human vs Qwen3-32B, resolved to human': sum(r['basis'].startswith('disagree: human') for r in ref_rows.values()),
}

hold_units = [u for u in ref if ref[u] == 'hold']
hist = collections.Counter(sum(J[n].get(u) == 'diverge' for n in USABLE) for u in hold_units)
data['hold_flag_hist'] = [[k, hist.get(k, 0)] for k in range(len(USABLE) + 1)]

data['by_solver'] = []
for n in ['pplx-decider', 'Clef', 'Kev-4B', 'gpt-oss-20b']:
    row = {'name': n}
    for src, lab in [('ollama/cb-gemma3-4b', 'gemma'), ('ollama/cb-qwen3.5-4b', 'qwen35')]:
        row[lab] = stats(J[n], ref, [u for u in ref if key.get(u) == src])['kappa']
    data['by_solver'].append(row)
data['solver_n'] = {lab: sum(key.get(u) == src for u in ref) for src, lab in
                    [('ollama/cb-gemma3-4b', 'gemma'), ('ollama/cb-qwen3.5-4b', 'qwen35')]}

# threshold sweeps on P(diverge)
data['sweep'] = {}
for n in ['pplx-decider', 'Clef']:
    pts = []
    for t100 in range(30, 100, 5):
        t = t100 / 100
        lab = {}
        for u, p in P[n].items():
            if p['diverge'] >= t:
                lab[u] = 'diverge'
            else:
                rest = {k: v for k, v in p.items() if k != 'diverge'}
                lab[u] = max(rest, key=rest.get)
        s = stats(lab, ref)
        pts.append({'t': t, 'kappa': s['kappa'], 'dprec': s['dprec'], 'drec': s['drec'], 'drate': s['drate']})
    data['sweep'][n] = pts

# rubric experiment
data['rubric'] = []
for j, name in [('pplx', 'pplx-decider'), ('clef', 'Clef'), ('kev', 'Kev-4B'), ('laya', 'Laya')]:
    for variant in ['no_context', 'preceding_exchange']:
        a, _ = decision(R + f'{j}_validity_v1/results.jsonl', variant)
        b, _ = decision(R + f'{j}_validity_v1_rubric/results.jsonl', variant)
        moves = collections.Counter(f'{a[u]}→{b[u]}' for u in a if u in b and a[u] != b[u])
        data['rubric'].append({'name': name, 'variant': variant,
                               'v1_ref': stats(a, ref), 'rubric_ref': stats(b, ref),
                               'v1_new': stats(a, new), 'rubric_new': stats(b, new),
                               'changed': sum(moves.values()), 'n': len(a), 'moves': dict(moves.most_common())})

# label review
data['review'] = list(csv.DictReader(open(V + 'labels/proposed_v1.1_review.csv')))
for r in data['review']:
    rr = ref_rows[r['unit_id']]
    r.update(labeller_a=rr['labeller_a'], label_a=rr['label_a'], labeller_b=rr['labeller_b'], label_b=rr['label_b'],
             basis=rr['basis'], item=units[r['unit_id']]['item']['question'],
             against=sum(J[n].get(r['unit_id']) not in (None, ref[r['unit_id']]) for n in USABLE))
    r.pop('reviewer', None)

json.dump(data, open(OUT, 'w'), ensure_ascii=False, indent=1)
print('judges', [(j['name'], j['kappa']) for j in data['judges']])
print('agreed/disputed', [(x['name'], x['agreed']['acc'], x['disputed']['acc']) for x in data['agreed_vs_disputed']])
print('hist', data['hold_flag_hist'], 'basis', data['basis_counts'])
print('solver', data['by_solver'], data['solver_n'])
print('sweep pplx best', max(data['sweep']['pplx-decider'], key=lambda p: p['kappa']))
print('sweep clef best', max(data['sweep']['Clef'], key=lambda p: p['kappa']))
print('rubric', [(r['name'], r['variant'], r['v1_ref']['kappa'], r['rubric_ref']['kappa'], r['changed']) for r in data['rubric']])
