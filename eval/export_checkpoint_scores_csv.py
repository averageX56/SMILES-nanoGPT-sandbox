"""
Compare exact-match accuracy across multiple trained checkpoints -- e.g.
different model sizes (out-small/out-medium/out-large), different tasks, or
several separately-trained architecture variants -- and write one separate
two-column (checkpoint, accuracy) CSV per gen_param_overrides entry, ready
to plot directly.

Not tied to KV-retrieval: works with any generator registered in
generator/__init__.py (see task_inference.py's docstring for how
gen_param_overrides and --dataset generalize across tasks).

``checkpoints`` (a {label: out_dir} dict) is structured, multi-valued config,
so -- exactly like a training config/*.py's gen_params dict -- set it with a
small config file rather than a CLI flag (a dict literal on the command line
needs careful quoting: unquoted curly braces are brace-expansion syntax to
bash and will silently split into the wrong number of arguments).

Example config file (checkpoints_to_compare.py):
    checkpoints = {
        'small': 'out-small',
        'medium': 'out-medium',
        'large': 'out-large',
    }

Example usage (KV retrieval -- an OOD sweep over n_pairs; note the whole
--gen_param_overrides value is double-quoted, same reason as in
task_inference.py: unquoted, bash strips the dict keys' quote characters and
brace-expands a multi-key dict into separate broken arguments):
    python export_checkpoint_scores_csv.py checkpoints_to_compare.py \\
        --gen_param_overrides="[{'n_pairs':48},{'n_pairs':96}]"

Example usage (a different task, same script -- addition):
    python export_checkpoint_scores_csv.py checkpoints_to_compare.py \\
        --dataset=addition --gen_param_overrides="[{'n_digits':10},{'n_digits':20}]"
"""
import csv
import os
from contextlib import nullcontext

import torch

from task_eval_utils import build_generator, evaluate_accuracy, load_checkpoint

# -----------------------------------------------------------------------------
checkpoints = {}       # REQUIRED: {label: out_dir, ...}; set via a config file (see module docstring)
model = ''             # override the architecture in models/ for every checkpoint (empty = use each checkpoint's own)
dataset = ''           # override the generator/task for every checkpoint (empty = use each checkpoint's own)
gen_param_overrides = []   # list of dicts, each merged over each checkpoint's own gen_params; one CSV per dict.
                           # empty = a single CSV using exactly each checkpoint's own trained gen_params.
n_eval = 500           # freshly sampled examples per (checkpoint, gen_param_overrides entry) cell
csv_out_dir = 'checkpoint_scores'  # directory to write the per-gen_param_overrides-entry CSVs into
seed = 1337
device = 'cuda'        # examples: 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

if not checkpoints:
    raise SystemExit(
        "Please provide checkpoints = {label: out_dir, ...} via a config "
        "file, e.g. a checkpoints_to_compare.py containing\n"
        "    checkpoints = {'small': 'out-small', 'medium': 'out-medium'}\n"
        "then: python export_checkpoint_scores_csv.py checkpoints_to_compare.py\n"
        "(A quoted CLI dict literal also works: "
        "--checkpoints=\"{'small': 'out-small'}\" -- but unquoted curly "
        "braces are brace-expansion syntax to bash and will silently break.)"
    )

torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"Loading {len(checkpoints)} checkpoint(s): {list(checkpoints)}")
loaded = {}
for label, ckpt_out_dir in checkpoints.items():
    net, checkpoint = load_checkpoint(ckpt_out_dir, device=device, model_override=(model or None))
    loaded[label] = (net, checkpoint)
    print(f"  {label!r} <- {ckpt_out_dir!r}: iter={checkpoint.get('iter_num', '?')}, "
          f"dataset={checkpoint.get('config', {}).get('dataset', '?')!r}, "
          f"gen_params={checkpoint.get('config', {}).get('gen_params', {})}")

overrides = gen_param_overrides or [{}]
print(f"gen_param overrides to test: {overrides}")

os.makedirs(csv_out_dir, exist_ok=True)

for override in overrides:
    label_suffix = '_'.join(f'{k}{v}' for k, v in override.items()) or 'trained'
    rows = []
    for label, (net, checkpoint) in loaded.items():
        block_size = checkpoint['model_args']['block_size']
        try:
            gen = build_generator(checkpoint, dataset_override=(dataset or None), gen_param_overrides=override, seed=seed)
            result = evaluate_accuracy(net, gen, n_eval=n_eval, block_size=block_size, device=device, ctx=ctx)
        except Exception as e:
            print(f"  {label_suffix} {label:>10}: skipped ({type(e).__name__}: {e})")
            continue
        print(f"  {label_suffix} {label:>10}: accuracy={result['accuracy']:.4f} (n={result['n_eval']})")
        rows.append((label, result['accuracy']))

    if not rows:
        continue
    path = os.path.join(csv_out_dir, f'scores_{label_suffix}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['checkpoint', 'accuracy'])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {path}")
