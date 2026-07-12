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
    results_by_task = {}
    task_order = []
    current_task = ''
    last_eval_by_task = {}
    for line in read_text(path).splitlines():
        if line.startswith('Downstream fine-tune:'):
            current_task = parse_kv_line(line).get('task', current_task)
        elif line.startswith('[eval]'):
            if current_task:
                last_eval_by_task[current_task] = parse_kv_line(line)
        elif line.startswith('[summary]'):
            summary = parse_kv_line(line)
            task = summary.get('task', current_task)
            values = dict(last_eval_by_task.get(task, {}))
            values.update(summary)
            if task not in results_by_task:
                task_order.append(task)
            results_by_task[task] = values
    return [results_by_task[task] for task in task_order]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True, help='Run root, e.g. gs://.../downstream_runs/baseline_400M')
    ap.add_argument('--out', default='downstream_summary.csv')
    args=ap.parse_args()
    rows=[]
    for root in args.root:
        # A sequence run appends every task to one <run>/train.log. Keep the
        # timestamped patterns for older per-task runs.
        try:
            logs = (
                tj._list_files(root, '*/*/train.log')
                + tj._list_files(root, '*/train.log')
                + tj._list_files(root, 'train.log')
                + tj._list_files(root, '*/*/training_log_*.txt')
                + tj._list_files(root, '*/training_log_*.txt')
                + tj._list_files(root, 'training_log_*.txt')
            )
        except Exception:
            logs = []
        for f in sorted(set(logs)):
            try:
                parts=f.rstrip('/').split('/')
                # .../<task>/<run>/training_log_*.txt or .../<run>/train.log
                legacy_task=parts[-3] if len(parts)>=3 else ''
                run=parts[-2] if len(parts)>=2 else ''
                for vals in parse_training_log(f):
                    rows.append({
                        'root': root,
                        'task': vals.get('task', legacy_task),
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
