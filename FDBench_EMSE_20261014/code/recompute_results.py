#!/usr/bin/env python3
"""Recompute the submitted FDBench results from archived observations.

Uses only the Python standard library. No model calls or new ratings.
Wilson intervals assume independent Bernoulli units; they are descriptive
precision references and do not adjust for task or source-page clustering.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_jsonl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_csv(p, rows):
    rows = list(rows)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wilson(k, n):
    z = 1.959963984540054
    a = z*z/n
    mid = (k/n + a/2)/(1+a)
    half = z*math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/(1+a)
    return 100*(mid-half), 100*(mid+half)


def agreement(mat):
    n, m = len(mat), len(mat[0])
    counts = Counter(v for row in mat for v in row)
    p = sum(sum(v*(v-1) for v in Counter(row).values())/(m*(m-1)) for row in mat)/n
    pe = sum((v/(n*m))**2 for v in counts.values())
    # These questionnaire dimensions are binary even when all ratings are 1.
    ac_pe = sum((v/(n*m))*(1-v/(n*m)) for v in counts.values())
    return dict(pairwise_agreement_pct=100*p,
                unanimous_agreement_pct=100*sum(len(set(row))==1 for row in mat)/n,
                fleiss_kappa=None if pe==1 else (p-pe)/(1-pe),
                gwet_ac1=(p-ac_pe)/(1-ac_pe))


def ordinal_alpha(mat):
    # Krippendorff ordinal distance uses cumulative marginal frequencies.
    freq = Counter(v for row in mat for v in row)
    cats = sorted(freq)
    total = sum(freq.values())
    coincidence = Counter()
    for row in mat:
        c = Counter(row)
        for a in cats:
            for b in cats:
                coincidence[a,b] += (c[a]*c[b]-(c[a] if a==b else 0))/(len(row)-1)
    dist = {(a,b): (sum(freq[k] for k in cats if min(a,b)<=k<=max(a,b))
                    -(freq[a]+freq[b])/2)**2 if a!=b else 0
            for a in cats for b in cats}
    observed = sum(coincidence[a,b]*dist[a,b] for a in cats for b in cats)/total
    expected = sum(freq[a]*freq[b]*dist[a,b] for a in cats for b in cats)/(total*(total-1))
    return 1-observed/expected


def icc(mat):
    n, k = len(mat), len(mat[0])
    grand = sum(sum(row) for row in mat)/(n*k)
    row_means = [sum(row)/k for row in mat]
    col_means = [sum(row[j] for row in mat)/n for j in range(k)]
    ms_rows = k*sum((v-grand)**2 for v in row_means)/(n-1)
    ms_cols = n*sum((v-grand)**2 for v in col_means)/(k-1)
    ms_error = sum((mat[i][j]-row_means[i]-col_means[j]+grand)**2
                   for i in range(n) for j in range(k))/((n-1)*(k-1))
    return {'icc_2_1': (ms_rows-ms_error)/(ms_rows+(k-1)*ms_error+k*(ms_cols-ms_error)/n),
            'icc_2_3': (ms_rows-ms_error)/(ms_rows+(ms_cols-ms_error)/n)}


def criterion(c):
    tp, tn, fp, fn = (c[x] for x in ['TP','TN','FP','FN'])
    n = tp+tn+fp+fn
    pe = ((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/(n*n)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return dict(n=n, TP=tp, TN=tn, FP=fp, FN=fn,
                sensitivity_pct=100*sensitivity, specificity_pct=100*specificity,
                PPV_pct=100*tp/(tp+fp), NPV_pct=100*tn/(tn+fn),
                balanced_accuracy_pct=50*(sensitivity+specificity),
                cohen_kappa=((tp+tn)/n-pe)/(1-pe))


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(root/'data/tasks/tasks.jsonl')
    akey = read_csv(root/'human_evaluation/construction_case_key.csv')
    bkey = read_csv(root/'human_evaluation/repair_candidate_key.csv')
    a = [{x['case_id']:x for x in read_csv(root/f'human_evaluation/ratings/A_R{i}.csv')} for i in range(1,4)]
    b = [{x['candidate_id']:x for x in read_csv(root/f'human_evaluation/ratings/B_R{i}.csv')} for i in range(1,4)]
    assert len(tasks)==len(akey)==106 and len(bkey)==160
    assert len({x['task_id'] for x in bkey})==40
    assert all(len(x)==106 for x in a) and all(len(x)==160 for x in b)
    for r in a:
        assert all(x[q] in ['0','1'] for x in r.values()
                   for q in ['defect_visible','single_primary_defect','page_intact','css_like_defect'])
    for r in b:
        assert all(x['defect_resolution'] in ['0','1','2'] and x['new_regression'] in ['0','1']
                   and x['target_similarity_1_5'] in ['1','2','3','4','5'] for x in r.values())
    labels = []
    valid = {}
    table2 = []
    for x in akey:
        row = dict(x)
        for q in ['defect_visible','single_primary_defect','page_intact','css_like_defect']:
            for i in range(3):
                row[f'R{i+1}_{q}'] = int(a[i][x['case_id']][q])
            row['majority_'+q] = int(sum(row[f'R{i+1}_{q}'] for i in range(3))>=2)
        row['construction_valid'] = int(all(row['majority_'+q] for q in ['defect_visible','single_primary_defect','page_intact']))
        valid[x['task_id']] = bool(row['construction_valid'])
        labels.append(row)
    assert sum(valid.values())==88
    write_csv(out/'construction_labels.csv', labels)
    for q in ['defect_visible','single_primary_defect','page_intact','css_like_defect']:
        mat = [[int(r[x['case_id']][q]) for r in a] for x in akey]
        count = sum(sum(row)>=2 for row in mat)
        table2.append(dict(dimension=q, majority_yes=count, n=106, proportion_pct=100*count/106, **agreement(mat)))
    write_csv(out/'table_02_construction_agreement.csv', table2)
    categories = []
    for cat in ['text','layout','boxmodel','stacking']:
        ts = [x for x in tasks if x['category']==cat]
        n, k = len(ts), sum(valid[x['task_id']] for x in ts)
        lo, hi = wilson(k,n)
        categories.append(dict(category=cat, valid=k, total=n, invalid=n-k,
                               proportion_pct=100*k/n, wilson_lower_pct=lo, wilson_upper_pct=hi))
    write_csv(out/'table_03_category_validity.csv', categories)
    write_csv(out/'figure_02a_category_counts.csv', categories)
    joined = []
    cm = {k:Counter({x:0 for x in ['TP','TN','FP','FN']}) for k in ['all','valid','invalid']}
    for x in bkey:
        row = dict(x)
        votes = []
        for i,r in enumerate(b,1):
            score = r[x['candidate_id']]
            for q in ['defect_resolution','new_regression','target_similarity_1_5']:
                row[f'R{i}_{q}'] = int(score[q])
            success = score['defect_resolution']=='2' and score['new_regression']=='0'
            row[f'R{i}_human_fix'] = int(success)
            votes.append(success)
        human = sum(votes)>=2
        auto = x['automatic_fix'].lower() in ['1','true']
        cell = ('TP' if auto else 'FN') if human else ('FP' if auto else 'TN')
        row.update(human_fix=int(human), construction_valid=int(valid[x['task_id']]), confusion_cell=cell)
        joined.append(row)
        cm['all'][cell]+=1
        cm['valid' if valid[x['task_id']] else 'invalid'][cell]+=1
    expected = {'all':dict(TP=41,TN=104,FP=0,FN=15), 'valid':dict(TP=34,TN=90,FP=0,FN=4), 'invalid':dict(TP=7,TN=14,FP=0,FN=11)}
    assert dict(cm)==expected, dict(cm)
    write_csv(out/'repair_candidates_joined.csv', joined)
    metrics = [dict(scope=k, **criterion(v)) for k,v in cm.items()]
    write_csv(out/'table_04_criterion_performance.csv', metrics)
    write_csv(out/'figure_02b_criterion_metrics.csv', [x for x in metrics if x['scope']!='invalid'])
    confusion = []
    for scope in ['all','valid']:
        for cell,hv,av in [('TP','positive','positive'),('FN','positive','negative'),('FP','negative','positive'),('TN','negative','negative')]:
            confusion.append(dict(scope=scope, human=hv, automatic=av, count=cm[scope][cell]))
    write_csv(out/'figure_02cd_confusion_matrices.csv', confusion)
    precision=[]
    for scope in ['all','valid']:
        c=cm[scope]
        for name,k,n in [('sensitivity',c['TP'],c['TP']+c['FN']),('specificity',c['TN'],c['TN']+c['FP']),('PPV',c['TP'],c['TP']+c['FP'])]:
            lo,hi=wilson(k,n)
            precision.append(dict(scope=scope,metric=name,numerator=k,denominator=n,
                                  estimate_pct=100*k/n,independent_wilson_lower_pct=lo,independent_wilson_upper_pct=hi))
    write_csv(out/'table_05_precision_reference.csv', precision)
    resolution=[[int(r[x['candidate_id']]['defect_resolution']) for r in b] for x in bkey]
    regression=[[int(r[x['candidate_id']]['new_regression']) for r in b] for x in bkey]
    similarity=[[int(r[x['candidate_id']]['target_similarity_1_5']) for r in b] for x in bkey]
    reliability = dict(krippendorff_alpha_ordinal=ordinal_alpha(resolution),
                       regression=agreement(regression), **icc(similarity))
    assert round(reliability['krippendorff_alpha_ordinal'],3)==.746, reliability
    assert round(reliability['icc_2_1'],3)==.674 and round(reliability['icc_2_3'],3)==.861, reliability
    assert round(reliability['regression']['fleiss_kappa'],3)==.903
    assert round(reliability['regression']['gwet_ac1'],3)==.913
    hist=read_jsonl(root/'historical_systems/metrics.jsonl')
    table6=[]
    for arm,pct in zip('ABCDEF',[14.2,14.2,15.1,20.8,15.1,26.4]):
        rows=[x for x in hist if x['arm']==arm]
        assert len(rows)==106 and len({x['task_id'] for x in rows})==106
        count=sum(int(x['fix']) for x in rows)
        assert round(100*count/106,1)==pct
        table6.append(dict(configuration=arm,successful=count,n=106,automatic_fix_pct=100*count/106))
    write_csv(out/'table_06_historical_usage.csv', table6)
    summary=dict(task_count=106,source_pages=len({x['source_page'] for x in tasks}),construction_valid=88,
                 sampled_tasks=40,sampled_valid_tasks=len({x['task_id'] for x in bkey if valid[x['task_id']]}),
                 repair_candidates=160,confusion_matrices={k:dict(v) for k,v in cm.items()},repair_rating_reliability=reliability,
                 independent_wilson_warning='Precision reference only; task and source-page clustering is not adjusted.',
                 result='PASS')
    assert summary['source_pages']==63 and summary['sampled_valid_tasks']==32
    (out/'recomputed_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,default=Path('recomputed_results'))
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.out),ensure_ascii=False,indent=2))
