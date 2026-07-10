#!/usr/bin/env python3
import argparse, csv, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.train_jax as tj


def read_json(path):
    with tj._open_file(path, 'r') as f:
        return json.loads(f.read())

def read_text(path):
    with tj._open_file(path, 'r') as f:
        return f.read()

def parse_kv_line(line):
    values = {}
    for part in line.strip().split()[1:]:
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        values[key] = value
    return values

def parse_training_log(path):
    summary = {}
    last_eval = {}
    for line in read_text(path).splitlines():
        if line.startswith('[eval]'):
            last_eval = parse_kv_line(line)
        elif line.startswith('[summary]'):
            summary = parse_kv_line(line)
    values = dict(last_eval)
    values.update(summary)
    return values

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True, help='Run root, e.g. gs://.../downstream_runs/baseline_400M')
    ap.add_argument('--out', default='downstream_summary.csv')
    args=ap.parse_args()
    rows=[]
    for root in args.root:
        # New downstream runs keep train/eval/final summary in one training log.
        try:
            logs = (
                tj._list_files(root, '*/*/training_log_*.txt')
                + tj._list_files(root, '*/training_log_*.txt')
                + tj._list_files(root, 'training_log_*.txt')
            )
        except Exception:
            logs = []
        for f in sorted(set(logs)):
            try:
                vals = parse_training_log(f)
                parts=f.rstrip('/').split('/')
                # .../<task>/<run>/training_log_*.txt or .../<run>/training_log_*.txt
                task=parts[-3] if len(parts)>=3 else ''
                run=parts[-2] if len(parts)>=2 else ''
                rows.append({
                    'root': root,
                    'task': vals.get('task', task),
                    'run': run,
                    'step': vals.get('step', ''),
                    'accuracy': vals.get('best_acc', vals.get('acc', '')),
                    'total': vals.get('total', ''),
                    'path': f,
                })
            except Exception as e:
                print(f'WARN failed {f}: {e}', file=sys.stderr)
        # Backward-compatible fallback for old runs that wrote best_eval.json.
        try:
            files = tj._list_files(root, '*/*/best_eval.json') + tj._list_files(root, '*/best_eval.json')
        except Exception:
            files = []
        seen_runs = {(row['root'], row['task'], row['run']) for row in rows}
        for f in files:
            try:
                ev=read_json(f)
                parts=f.rstrip('/').split('/')
                task=parts[-3] if len(parts)>=3 else ''
                run=parts[-2] if len(parts)>=2 else ''
                if (root, task, run) in seen_runs:
                    continue
                rows.append({'root':root,'task':task,'run':run,'step':ev.get('step',''),'accuracy':ev.get('accuracy',''),'total':ev.get('total',''),'path':f})
            except Exception as e:
                print(f'WARN failed {f}: {e}', file=sys.stderr)
    with open(args.out,'w',newline='') as out:
        w=csv.DictWriter(out, fieldnames=['root','task','run','step','accuracy','total','path'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {args.out} rows={len(rows)}')
if __name__=='__main__': main()
