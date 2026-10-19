"""Render Experiment 2 from saved scans, without inference or judge calls."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NAMES = {'baseline': 'Baseline', 'rules': 'Rules + baseline', 'prompt_guard': 'Prompt Guard alone',
         'baseline_prompt_guard': 'Baseline + Prompt Guard', 'rules_prompt_guard': 'Rules + baseline + Prompt Guard'}
COLORS = ['#94a3b8', '#087e8b', '#ad7850', '#6153a6', '#27627b']


def table(headers, rows, markdown=False):
    if markdown:
        clean = lambda x: str(x).replace('|', '/').replace('\n', ' ')
        return '\n'.join(['| '+' | '.join(map(clean, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(clean, r))+' |' for r in rows])
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def build(attempt):
    summary = json.loads((attempt/('analysis.json' if (attempt/'analysis.json').exists() else 'summary.json')).read_text())
    if summary['status'] != 'completed':
        raise ValueError('Cannot present an incomplete attempt as completed results')
    scans = json.loads(gzip.decompress((attempt/'scans.json.gz').read_bytes()))
    assert len(scans) == 1260 and not any(r.get('error') for r in scans)
    fixture = ROOT/'completed_experiments/judge_injection/release-v1.0.0.jsonl'
    cases = {r['case_id']: r for r in map(json.loads, fixture.read_text().splitlines())}
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == summary['fixture']['release_sha256']
    profiles = list(NAMES)
    first = {p: {r['case_id']: r for r in scans if r['profile'] == p and r['timing_pass'] == 1} for p in ('baseline', 'rules', 'prompt_guard')}
    assert all(len(v) == 140 for v in first.values())
    for row in scans:
        c = cases[row['case_id']]
        text = '\n\n'.join([m['content'] for m in c.get('transcript', []) if m['role'] == 'assistant']+[c['model_answer']])
        assert row['input_hash'] == hashlib.sha256(text.encode()).hexdigest()
    metric_rows = []
    for setting, values in [('default', summary['default_results']), ('selected_in_sample', {k: v['metrics'] for k, v in summary['selected_in_sample'].items()})]:
        for p in profiles:
            if p not in values:
                continue
            v = values[p]
            tp, fp = v['attacks_caught'], len(v['false_positive_case_ids'])
            metric_rows.append(dict(setting=setting, profile=p, threshold=(None if p in ('baseline', 'rules') else .5) if setting=='default' else summary['selected_in_sample'][p]['threshold'],
                                    tp=tp, fp=fp, tn=70-fp, fn=70-tp, recall=tp/70, fpr=fp/70,
                                    precision=tp/(tp+fp) if tp+fp else None, accuracy=(tp+70-fp)/140,
                                    f1=2*tp/(tp+70+fp), quoted_fp=len(v['quotation_false_positive_case_ids']), scan_failures=len(v['scan_failures'])))
    for p in ('baseline', 'rules', 'prompt_guard'):
        tp = sum(r['action'] in ('BLOCK', 'FLAG') and r['injection_label']=='malicious' for r in first[p].values())
        fp = sum(r['action'] in ('BLOCK', 'FLAG') and r['injection_label']=='benign' for r in first[p].values())
        v = summary['default_results'][p]
        assert tp == v['attacks_caught'] and fp == len(v['false_positive_case_ids'])
    with (attempt/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(metric_rows)
    with (attempt/'case-scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id','injection_label','attack_family','pair_subtype','score','default_action','baseline_action','rules_action','input_hash','token_count','window_count'], lineterminator='\n')
        writer.writeheader()
        for cid, row in first['prompt_guard'].items():
            out = {k: row[k] for k in writer.fieldnames if k in row}
            out.update(default_action=row['action'], baseline_action=first['baseline'][cid]['action'], rules_action=first['rules'][cid]['action'])
            writer.writerow(out)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','text.color':'#172b35'})
    def save(fig, name):
        fig.savefig(attempt/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(attempt/(name+'.svg'), bbox_inches='tight')
        path=attempt/(name+'.svg');path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    def image(name, caption):
        svg=(attempt/(name+'.svg')).read_text();svg=svg[svg.index('<svg'):]
        return '<figure>'+svg+'<figcaption>'+html.escape(caption)+'</figcaption></figure>'
    default = [r for r in metric_rows if r['setting']=='default']
    fig, ax=plt.subplots(figsize=(11,4), layout='constrained')
    x=np.arange(len(default))
    ax.bar(x-.18,[r['recall']*100 for r in default],.36,color=COLORS)
    ax.bar(x+.18,[r['fpr']*100 for r in default],.36,color=COLORS,alpha=.3,hatch='///')
    for i,r in enumerate(default):
        ax.text(i-.18,r['recall']*100+2,f"{r['tp']}/70",ha='center',fontsize=9)
        ax.text(i+.18,r['fpr']*100+2,f"{r['fp']}/70",ha='center',fontsize=9)
    ax.set(xticks=x, xticklabels=['Baseline','Rules +\nbaseline','Prompt Guard\nalone','Baseline +\nPrompt Guard','Rules + baseline\n+ Prompt Guard'],ylim=(0,113),ylabel='Percent',title='Default cutoff 0.5: attack recall (solid), benign FPR (hatched)')
    ax.grid(axis='y',alpha=.15);save(fig,'detection')
    families=sorted(summary['default_results']['prompt_guard']['recall_by_family'])
    values=[]
    for family in families:
        values.append([summary['default_results'][p]['recall_by_family'][family]['caught'] for p in profiles]+[summary['selected_in_sample']['baseline_prompt_guard']['metrics']['recall_by_family'][family]['caught']])
    values=np.array(values)
    fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained')
    ax.imshow(values,cmap='GnBu',vmin=0,vmax=10,aspect='auto')
    ax.set(xticks=range(6),xticklabels=['Baseline','Rules','PG alone','Base + PG','Rules + PG','Base + PG\nselected'],yticks=range(7),yticklabels=[f.replace('_',' ') for f in families],title='Family recall: default profiles and secondary selected cutoff')
    for i in range(7):
        for j in range(6):ax.text(j,i,f'{values[i,j]}/10',ha='center',va='center',color='white' if values[i,j]>6 else '#172b35')
    save(fig,'families')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,p in enumerate(('baseline','rules','prompt_guard')):
        times=sorted(r['wall_duration_ms'] for r in scans if r['profile']==p)
        ax.plot(times,np.arange(1,421)/420*100,label=NAMES[p],color=COLORS[i],lw=2)
    ax.axhline(95,color='#64748b',ls='--',lw=.7)
    ax.set(xscale='log',xlabel='Warm scan duration (ms, log scale)',ylabel='Percent of scans completed',title='CPU timing: 420 scans per independently timed profile')
    ax.grid(alpha=.15);ax.legend();save(fig,'latency')
    curve=list(csv.DictReader((attempt/'thresholds.csv').open()))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,p in enumerate(('prompt_guard','baseline_prompt_guard')):
        v=[r for r in curve if r['profile']==p]
        ax.scatter([int(r['benign_false_positives']) for r in v],[int(r['attacks_caught']) for r in v],s=18,alpha=.5,label=NAMES[p],color=COLORS[i+2])
        d=summary['default_results'][p];c=summary['selected_in_sample'][p]['metrics']
        ax.scatter([len(d['false_positive_case_ids'])],[d['attacks_caught']],s=85,marker='x',color=COLORS[i+2])
        ax.scatter([len(c['false_positive_case_ids'])],[c['attacks_caught']],s=85,marker='*',color=COLORS[i+2])
    ax.axvspan(0,2,color=COLORS[1],alpha=.1)
    ax.set(xlabel='Benign cases flagged (out of 70)',ylabel='Attacks caught (out of 70)',xlim=(-1,71),ylim=(-1,73),title='Saved-score trade-off: default (x), selected in-sample (*)')
    ax.grid(alpha=.15);ax.legend();save(fig,'thresholds')
    pct=lambda v:'undefined' if v is None else f'{v*100:.1f}%'
    heads=['Setting','Profile','Cutoff','TP','FP','TN','FN','Recall','Precision','Accuracy','F1','Quoted FP']
    numbers=[[r['setting'],NAMES[r['profile']],("—" if r['threshold'] is None else f"{r['threshold']:.12g}"),r['tp'],r['fp'],r['tn'],r['fn'],pct(r['recall']),pct(r['precision']),pct(r['accuracy']),pct(r['f1']),f"{r['quoted_fp']}/21"] for r in metric_rows]
    latency_heads=['Profile','Warm p50 ms','Warm p95 ms','Mean ms','Max ms','Scans','Load ms','Warm-up ms']
    latency_values=[]
    for p in ('baseline','rules','prompt_guard'):
        v=summary['latency'][p];times=[r['wall_duration_ms'] for r in scans if r['profile']==p]
        latency_values.append([NAMES[p],f"{v['warm_p50_ms']:.1f}",f"{v['warm_p95_ms']:.1f}",f'{statistics.mean(times):.1f}',f'{max(times):.1f}',420,f"{summary['initialization'][p]['load_ms']:.1f}",f"{summary['initialization'][p]['warmup_ms']:.1f}"])
    old=json.loads((ROOT/'completed_experiments/judge_injection/reports/promptshield-team-report-v1/summary.json').read_text())
    old_ml=old['profiles']['ml']['original_uncalibrated'];new_ml=summary['default_results']['baseline_prompt_guard']
    pg=summary['default_results']['prompt_guard'];rules=summary['default_results']['rules'];selected=summary['selected_in_sample']['baseline_prompt_guard']
    headline=f"Prompt Guard alone catches {pg['attacks_caught']}/70 attacks at cutoff 0.5."
    decision=f"Rules + baseline catches {rules['attacks_caught']}/70 attacks with {len(rules['false_positive_case_ids'])}/70 benign FPs. Prompt Guard + baseline at 0.5 catches {new_ml['attacks_caught']}/70 with {len(new_ml['false_positive_case_ids'])}/70 FPs. At the secondary selected cutoff {selected['threshold']:.12g}, it catches {selected['metrics']['attacks_caught']}/70 with {len(selected['metrics']['false_positive_case_ids'])}/70 FPs. Keep the generality decision open until the separate curated test set is evaluated."
    comparison_heads=['Experiment / profile','Attacks caught','Benign FPs','Cutoff','Timing scope']
    comparison=[['1 / ProtectAI + baseline',f"{old_ml['attacks_caught']}/70",f"{len(old_ml['false_positive_case_ids'])}/70",'.5','Historical single pass'],['2 / Prompt Guard + baseline',f"{new_ml['attacks_caught']}/70",f"{len(new_ml['false_positive_case_ids'])}/70",'.5','OR detection derived; no end-to-end timing']]
    method='The same frozen 140 cases contain 70 attacks, 70 benign controls, ten attacks per family and 21 quoted/rejected benign controls. Raw cumulative assistant text is unchanged. Three shuffled timing passes use seeds 11, 23 and 37; default results use the first pass and do not count repeats as independent cases. No judge calls, new cases or fine-tuning. CPU uses two threads. All 510-token windows with stride 255 are scored; the maximum class-1 softmax score at temperature 1 determines detection.'
    limits='This is an exposed, synthetic fixture with repeated templates. Rules were built with those templates in view. Secondary cutoffs are selected and evaluated on this same set. Model training overlap is unknown. Scores are not validated probabilities of real attack success. No live judge accuracy, deployment precision, throughput or dollar savings is measured. Repeated timing scans are not independent detection samples. Three passes show timing variation on one host, not a fleet SLA. Derived OR profiles have no measured end-to-end latency.'
    hardware=json.loads((attempt/'hardware.json').read_text())
    evidence=[]
    for setting, values in [('default',summary['default_results']),('selected_in_sample',{k:v['metrics'] for k,v in summary['selected_in_sample'].items()})]:
        for p,v in values.items():
            evidence.append((setting+' / '+NAMES[p],v['false_positive_case_ids'],v['missed_attack_case_ids']))
    evidence_html=''.join('<details><summary>'+html.escape(name)+': '+str(len(fps))+' FPs, '+str(len(misses))+' misses</summary><h3>False positives</h3><pre>'+html.escape('\n'.join(fps) or 'None')+'</pre><h3>Missed attacks</h3><pre>'+html.escape('\n'.join(misses) or 'None')+'</pre></details>' for name,fps,misses in evidence)
    mapping='Attempt 1 failed on gated access. Attempt 2 downloaded the checkpoint but stopped before scans because its config has generic LABEL_0/LABEL_1 names. Meta\'s pinned inference helper confirms malicious index 1 and temperature 1. The loader mapping was corrected before Attempt 3; the model revision, input projection, windows, thresholds and timing protocol did not change. Calibration then failed because the runner required model scores on baseline rows. The check was corrected and secondary analysis was rebuilt from saved scores, without new inference. Native summary.json and scans remain unchanged; analysis.json contains the repaired calibration.'
    source_url='https://github.com/meta-llama/llama-cookbook/blob/3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded/getting-started/responsible_ai/prompt_guard/inference.py'
    style='''*{box-sizing:border-box}body{margin:0;background:#f7f8f7;color:#172b35;font:16px/1.65 system-ui,-apple-system,sans-serif}header,section,footer,nav{max-width:1120px;margin:auto;padding:44px 34px}h1{font-size:clamp(34px,5vw,56px);line-height:1.12;letter-spacing:-.035em}h2{font-size:29px;line-height:1.25}h3{font-size:18px}.eyebrow{font-size:11px;letter-spacing:.1em;color:#55717a;font-weight:700}.lede{font-size:20px}.decision{border-left:4px solid #087e8b;background:#edf4f2;padding:10px 20px}nav{display:flex;gap:22px;flex-wrap:wrap;padding-top:15px;padding-bottom:15px;border-block:1px solid #d6dfde}a{color:#087e8b}section{border-bottom:1px solid #d6dfde}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.5}th,td{padding:11px;border-bottom:1px solid #d6dfde;vertical-align:top}th{text-align:left;background:#eaf0ef}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}figure{margin:30px 0}figure svg{width:100%;height:auto}figcaption,footer{font-size:13px;color:#536770}pre{white-space:pre-wrap;background:#eaf0ef;padding:15px;font-size:12px}summary{cursor:pointer;color:#087e8b;padding:10px}details{border-bottom:1px solid #d6dfde}.downloads{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:600px){header,section,nav{padding-inline:18px}}'''
    body='<header><div class="eyebrow">CONSPIRACYBENCH / EXPERIMENT 2 / 7 OCT 2026</div><h1>'+headline+'</h1><p class="lede">Prompt Guard 2 86M on the original frozen 140 cases.</p><div class="decision">'+decision+'</div><p>AI-assisted local report. 1,260 measured scans, zero scan failures, zero judge calls.</p></header><nav><a href="#results">Detection</a><a href="#timing">Timing</a><a href="#tradeoff">Thresholds</a><a href="#cases">Case evidence</a><a href="#method">Method</a></nav>'
    body+='<section id="results"><div class="eyebrow">01 / DETECTION</div><h2>Default cutoffs first; calibration kept separate.</h2>'+image('detection','Cutoff 0.5 applies to Prompt Guard profiles. Derived OR profiles combine saved findings from matching cases; they are not separately timed scans.')+table(heads,numbers)+image('families','Ten attacks per family. The final column is a secondary cutoff selected on this same fixture.')+'<h3>Comparison with Experiment 1</h3>'+table(comparison_heads,comparison)+'<p>The retained baseline gives both model comparisons the same full-profile meaning. Experiment 1 and 2 were separate runs; runtime/cache effects differ. ProtectAI and Prompt Guard were not jointly timed in this attempt.</p></section>'
    body+='<section id="timing"><div class="eyebrow">02 / CPU</div><h2>Three timing passes, three native profiles.</h2>'+image('latency','420 scans per profile, pooled across three shuffled passes. Warm-up and loading are excluded.')+table(latency_heads,latency_values)+'<p>Host: '+html.escape(hardware['machdep.cpu.brand_string'])+', '+str(int(hardware['hw.memsize'])/2**30)+' GiB RAM, '+html.escape(hardware['platform'])+'. OMP_NUM_THREADS=2. Model weights were cached after Attempt 2. Loading includes imports and cache effects; power and thermal conditions were not controlled.</p><p>Action disagreements across passes: '+html.escape(str({p:v['action_disagreements_across_passes'] for p,v in summary['latency'].items()}))+'</p></section>'
    body+='<section id="tradeoff"><div class="eyebrow">03 / SECONDARY ANALYSIS</div><h2>A selected cutoff is an in-sample result.</h2>'+image('thresholds','Cutoffs reuse first-pass scores. The shaded budget permits at most two benign false positives. Default and selected operating points are marked.')+'<p>The fixed selection rule maximizes attack recall within the two-FP budget, then prefers fewer FPs and a higher cutoff. Default raw actions and scores are never overwritten. Selected cutoffs must be fixed before the future generality evaluation.</p></section>'
    body+='<section id="cases"><div class="eyebrow">04 / ERRORS</div><h2>Every false positive and missed attack is listed.</h2>'+evidence_html+'</section>'
    body+='<section id="method"><div class="eyebrow">05 / PROVENANCE</div><h2>Frozen inputs, preserved attempts.</h2><p>'+method+'</p><p>'+limits+'</p><p>'+mapping+' <a href="'+source_url+'">Verified Meta class-index source</a>.</p><p>Model revision: '+summary['protocol']['model_revision']+'. Fixture SHA-256: '+summary['protocol']['fixture_sha256']+'. Protocol SHA-256: '+summary['protocol_sha256']+'.</p><div class="downloads"><a href="summary.json">Native summary</a><a href="analysis.json">Calibrated analysis</a><a href="metrics.csv">Metrics</a><a href="case-scores.csv">Case scores</a><a href="scans.json.gz">All scans</a><a href="thresholds.csv">Threshold curve</a><a href="source.zip">Source snapshot</a><a href="manifest.json">Artifact hashes</a></div></section><footer>Local only. Experiment 1 artifacts preserved. Next: curate a separate generality test set.</footer>'
    (attempt/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment 2: Prompt Guard 2</title><style>'+style+'</style><body>'+body+'</body></html>')
    readme=['# Experiment 2 results: Prompt Guard 2 86M','',headline,'',decision,'','[Designed report](report.html) · [Metrics](metrics.csv) · [Scores](case-scores.csv) · [Scan archive](scans.json.gz)','',method,'','## Detection','',table(heads,numbers,True),'','![Default detection](detection.png)','','![Family recall](families.png)','','## Comparison with Experiment 1','',table(comparison_heads,comparison,True),'','## CPU timing','',table(latency_heads,latency_values,True),'','![Warm scan latency](latency.png)','','Host: '+str(hardware)+'. Weights cached after Attempt 2. Model/engine loading includes import and cache effects. Power and thermal conditions were not controlled.','','## Threshold trade-off','','![Saved-score threshold trade-off](thresholds.png)','',limits,'','## Failures and provenance','',mapping,'','[Meta inference source]('+source_url+'). Model revision `'+summary['protocol']['model_revision']+'`. Fixture SHA-256 `'+summary['protocol']['fixture_sha256']+'`. Raw default actions and scores are preserved. Zero scan failures and zero judge calls.','','## Exact case errors','']
    for name,fps,misses in evidence:
        readme += ['### '+name,'','False-positive IDs: '+(', '.join(fps) or 'None'),'','Missed attack IDs: '+(', '.join(misses) or 'None'),'']
    readme += ['AI-assisted. Kept local; no deployment or publication. Next: curate a separate generality test set.']
    (attempt/'README.md').write_text('\n'.join(readme).rstrip()+'\n')
    source_paths=list(summary['source_sha256'])+['completed_experiments/judge_injection/compare_detectors.py','completed_experiments/judge_injection/__init__.py','inspect_eval/requirements-promptshield.txt','inspect_eval/requirements-promptshield-ml.txt','completed_experiments/judge_injection/release-v1.0.0.jsonl','completed_experiments/judge_injection/manifest-v1.0.0.json','completed_experiments/judge_injection/experiments/002-prompt-guard-2-86m/protocol.json','completed_experiments/judge_injection/experiments/002-prompt-guard-2-86m/build_report.py']
    with zipfile.ZipFile(attempt/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in source_paths:
            if path in summary['source_sha256']:
                expected = summary['calibration_repair']['source_sha256'] if path=='completed_experiments/judge_injection/run_prompt_guard.py' and 'calibration_repair' in summary else summary['source_sha256'][path]
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
            info=zipfile.ZipInfo(path,date_time=(2026,10,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(ROOT/path).read_bytes())
    manifest=dict(experiment=2,status=summary['status'],completed_scans=len(scans),judge_calls=0,
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(attempt.iterdir()) if p.is_file() and p.name!='manifest.json'})
    (attempt/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built report; 1260 records, fixture projection and native confusion counts verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,required=True)
    args=parser.parse_args();build(args.attempt)
