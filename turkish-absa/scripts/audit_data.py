"""Offline audit of prepared public JSONL splits and label quality."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from absa.data import read_jsonl, text_hash, write_jsonl
from absa.quality import canonical_text


def collect(work, include_synthetic=False):
    files={}
    dirs = [work/'data'/'splits']
    if include_synthetic:
        dirs.append(work/'data'/'synth')
    for d in dirs:
        if d.exists():
            for f in sorted(d.glob('*.jsonl')):
                if f.stem in ('train_full', 'train_clean', 'val_full', 'curated_stress_test'):
                    continue
                files[f.stem]=read_jsonl(f)
    return files


def audit(files,review_n=100,seed=42):
    report={"splits":{},"duplicates":{},"normalized_duplicates":{},"invalid_spans":[],"conflicts":[],"warnings":[]}
    for name,docs in files.items():
        labels=Counter(a.get('polarity') for d in docs for a in d.get('aspects',[]))
        sources=Counter(d.get('source','unknown') for d in docs)
        flags=Counter(flag for d in docs for flag in d.get('flags',[]))
        report['splits'][name]={"rows":len(docs),"sources":dict(sources),
                                "polarities":dict(labels),"flags":dict(flags),
                                "implicit_gold":sum(1 for d in docs if d.get('span_supervision',True)
                                                    for a in d.get('aspects',[]) if a.get('span') is None),
                                "category_only":sum(1 for d in docs if not d.get('span_supervision',True))}
        for d in docs:
            seen={}
            for a in d.get('aspects',[]):
                surf=a.get('span');start=a.get('start');end=a.get('end')
                if surf is not None and not(isinstance(start,int) and isinstance(end,int)
                                            and d['text'][start:end]==surf):
                    report['invalid_spans'].append({"split":name,"id":d.get('id'),"span":surf})
                k=(str(a.get('aspect','')).casefold(),surf or '<implicit>')
                if k in seen and seen[k]!=a.get('polarity'):
                    report['conflicts'].append({"split":name,"id":d.get('id'),"aspect":k})
                seen[k]=a.get('polarity')
    train_rows=files.get('train_public',[])+files.get('local_train',[])+files.get('gemini_train',[])
    train={text_hash(d['text']) for d in train_rows}
    normalized_train={canonical_text(d['text']) for d in train_rows}
    for k,v in files.items():
        if k in ['train_public','local_train','gemini_train']: continue
        overlap=len(train & {text_hash(d['text']) for d in v})
        report['duplicates']['train_vs_'+k]=overlap
        report['normalized_duplicates']['train_vs_'+k]=len(normalized_train & {canonical_text(d['text']) for d in v})
        if overlap:report['warnings'].append(f'Potential train leakage into {k}: {overlap} exact matches')
    report['warnings'].append('Cross-split exact hashing is not semantic deduplication; evaluate semantic near-duplicates separately.')
    rand=random.Random(seed)
    synth=files.get('gemini_train',[])+files.get('local_train',[])
    if synth:
        report['warnings'].append('Generated labels are not native-annotated gold.')
    rand.shuffle(synth)
    review=[{"id":d.get('id'),"text":d.get('text'),"suggested_aspects":d.get('aspects'),
             "reviewer_aspects":None,"acceptable":None,"error_type":None,
             "flags":d.get('flags',[]),"source":d.get('source')} for d in synth[:review_n]]
    return report,review


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--review_n',type=int,default=100)
    p.add_argument('--include_synthetic',action='store_true')
    a=p.parse_args()
    files=collect(a.work,a.include_synthetic)
    report,review=audit(files,a.review_n)
    out=a.work/'reports'
    out.mkdir(parents=True,exist_ok=True)
    (out/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write_jsonl(out/'review_queue.jsonl',review)
    print(json.dumps({'sources':list(files),'invalid_spans':len(report['invalid_spans']),
                      'conflicts':len(report['conflicts']), 'warnings':report['warnings']},ensure_ascii=False,indent=2))
    print('Saved',out/'data_audit.json',out/'review_queue.jsonl')


if __name__=='__main__':main()
