#!/usr/bin/env python3
"""Verify packaged FDBench files and the original repair-candidate evidence hashes."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def run(root):
    checked=0
    for label in ['core','knowledge']:
        manifest=root/f'provenance/{label}_file_manifest.json'
        if not manifest.exists():
            if label=='core':raise ValueError('Core manifest is missing')
            continue
        for x in json.loads(manifest.read_text(encoding='utf-8')):
            p=root/x['path']
            assert p.is_file(), f"Missing {x['path']}"
            assert p.stat().st_size==x['bytes'], f"Wrong size {x['path']}"
            assert file_sha(p)==x['sha256'], f"Wrong checksum {x['path']}"
            checked+=1
        for x in rows(root/f'provenance/{label}_file_lineage.csv'):
            assert file_sha(root/x['package_relative_path'])==x['package_sha256']
            if x['transformation']=='none; original bytes preserved':
                assert x['source_sha256']==x['package_sha256']
    tasks=lines(root/'data/tasks/tasks.jsonl')
    assert [x['task_id'] for x in tasks]==[f'T{i:04d}' for i in range(106)]
    sample=json.loads((root/'human_evaluation/sample_40_tasks.json').read_text())
    assert sorted(random.Random(20260907).sample([x['task_id'] for x in tasks],40))==sample['sampled_task_ids']
    candidates=rows(root/'human_evaluation/repair_candidate_key.csv')
    raw={}
    for exp in ['E6','E8']:
        observations=lines(root/f'human_evaluation/model_outputs/{exp}_sampled_outputs.jsonl')
        assert len(observations)==80
        for x in observations:
            raw[(exp,x['task_id'],x['condition'],int(x.get('replicate',1)),x['raw_response_sha256'])]=x
    for x in candidates:
        key=(x['source_experiment'],x['task_id'],x['condition'],int(x['replicate']),x['raw_response_sha256'])
        observation=raw[key]
        assert hashlib.sha256(observation['raw_response'].encode('utf-8')).hexdigest()==x['raw_response_sha256']
        assert hashlib.sha256(observation['fixed_html_final'].encode('utf-8')).hexdigest()==x['fixed_html_sha256']
        assert observation['prompt_sha256']==x['prompt_sha256']
        assert file_sha(root/f"human_evaluation/repaired_html/{x['candidate_id']}.html")==x['fixed_html_sha256']
        assert file_sha(root/f"human_evaluation/stimuli/B_metric_validity/images/{x['candidate_id']}_repaired.png")==x['render_sha256']
    for module,key in [('A_construction','case_id'),('B_metric_validity','candidate_id')]:
        manifest=lines(root/f'human_evaluation/stimuli/{module}/manifest.jsonl')
        for x in manifest:
            for k,path in x.items():
                if k.endswith('_image'):assert (root/'human_evaluation/stimuli'/path).is_file()
    if (root/'data/kb/docs.jsonl').exists():
        docs=lines(root/'data/kb/docs.jsonl');chunks=lines(root/'data/kb/chunks.jsonl')
        assert len(docs)==642 and len(chunks)==9200
        assert len({x['doc_id'] for x in chunks})==626
        assert all(x['subset']!='KB-3' for x in chunks)
        for d in rows(root/'data/kb/document_manifest.csv'):
            assert file_sha(root/d['image'])==d['image_sha256']
            if d['source_file']:assert file_sha(root/d['source_file'])==d['source_sha256']
    result={'status':'PASS','files_checked':checked,'repair_candidates_verified':len(candidates),
            'knowledge_library_included':(root/'data/kb/docs.jsonl').exists()}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    print(json.dumps(run(a.root),indent=2))
