"""
Export training-dynamics CSVs from a train.py run: one separate two-column
(iter, value) CSV per metric -- train_loss, val_loss, train_acc, val_acc --
ready to plot directly (e.g. pandas.read_csv(...).plot(x='iter')).

Reads either of two sources:
  - the JSONL log train.py writes when you pass --log_file=<path> (one JSON
    object per eval step: iter, train_loss, val_loss, lr, mfu, and
    train_acc/val_acc when eval_accuracy is on). This is the recommended
    source since it comes straight from train.py's own numbers.
  - a plain-text capture of train.py's stdout (e.g. via
    `python train.py ... 2>&1 | tee out-small/train.log`), parsed with a
    regex against the exact line train.py prints:
    "step <iter>: train loss <x>, val loss <y>[, train acc <a>, val acc <b>]"
    -- for runs already captured before --log_file existed.

Example:
    python train.py config/small.py --log_file=out-small/train_log.jsonl
    python export_training_dynamics_csv.py \\
        --log_file=out-small/train_log.jsonl --csv_out_dir=out-small/dynamics_csv
"""
import csv
import json
import os
import re

# -----------------------------------------------------------------------------
log_file = ''                    # REQUIRED: train.py's --log_file JSONL, or a captured stdout text log
csv_out_dir = 'training_dynamics' # directory to write the per-metric CSVs into
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

if not log_file:
    raise SystemExit(
        "Please pass --log_file=<path>. Either a JSONL log written by "
        "train.py's --log_file=<path> option, or a text file with train.py's "
        "captured stdout (e.g. `python train.py ... 2>&1 | tee train.log`)."
    )

STDOUT_LINE_RE = re.compile(
    r"step (?P<iter>\d+): train loss (?P<train_loss>[\d.]+), "
    r"val loss (?P<val_loss>[\d.]+)"
    r"(?:, train acc (?P<train_acc>[\d.]+), val acc (?P<val_acc>[\d.]+))?"
)


def _looks_like_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                return line.startswith('{')
    return False


def _parse_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_stdout_text(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = STDOUT_LINE_RE.search(line)
            if not m:
                continue
            row = {
                'iter': int(m['iter']),
                'train_loss': float(m['train_loss']),
                'val_loss': float(m['val_loss']),
            }
            if m['train_acc'] is not None:
                row['train_acc'] = float(m['train_acc'])
                row['val_acc'] = float(m['val_acc'])
            rows.append(row)
    return rows


rows = _parse_jsonl(log_file) if _looks_like_jsonl(log_file) else _parse_stdout_text(log_file)
if not rows:
    raise SystemExit(f"No training-dynamics rows found in {log_file!r}. Is this really a train.py log?")
print(f"Parsed {len(rows)} logged eval step(s) from {log_file!r}")

os.makedirs(csv_out_dir, exist_ok=True)
for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
    present = [(r['iter'], r[metric]) for r in rows if metric in r]
    if not present:
        continue
    path = os.path.join(csv_out_dir, f'{metric}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', metric])
        writer.writerows(present)
    print(f"Wrote {len(present)} row(s) to {path}")
