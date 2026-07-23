"""
Test inference for any registered task: load a trained checkpoint and
measure exact-match accuracy (the same metric train.py reports as val_acc)
over one or more gen_params overrides -- e.g. an OOD sweep over whatever
difficulty knob the task exposes.

Works with any generator registered in generator/__init__.py, selected with
--dataset (same flag name and semantics as train.py) or, if omitted,
whichever generator the checkpoint itself was trained on. Accuracy is
computed generically via generator.collate()'s answer-masked (x, y) plus a
full forward pass, so it handles single-token answers (kv_retrieval,
indexing, function_composition) and multi-token ones (addition, sorting,
dyck) the same way, with no per-task special-casing baked into this script.

gen_param_overrides is a list of dicts, each merged over the checkpoint's
own recorded gen_params; one row of the summary/examples output is produced
per dict. This is the generic replacement for a task-specific "context
size" flag: every task calls its own difficulty knob something different
(see each generator/*.py's docstring), so rather than hardcoding one task's
parameter name, you pass whatever dict your task actually takes.

IMPORTANT -- always double-quote the *entire* --gen_param_overrides value.
Dict keys need real quote characters to be valid Python-literal syntax, but
bash strips unquoted quote characters from a word before python ever sees
it; and if one override dict has more than one key, the comma between them
is bash brace-expansion syntax and will silently split it into multiple
broken arguments. Both failure modes disappear once the whole value is
wrapped in one pair of double quotes, as in every example below.

Example (KV retrieval -- an OOD sweep over n_pairs):
    python task_inference.py --out_dir=out-small
    python task_inference.py --out_dir=out-small \\
        --gen_param_overrides="[{'n_pairs':96},{'n_pairs':192},{'n_pairs':384}]" \\
        --summary_csv=kv_inference_summary.csv \\
        --examples_csv=kv_inference_examples.csv

Example (a different task, same script -- addition, OOD over digit count):
    python task_inference.py --out_dir=out-addition --dataset=addition \\
        --gen_param_overrides="[{'n_digits':10},{'n_digits':20}]"

Example (dyck -- an override with two keys at once, which is exactly the
case unquoted brace-expansion would corrupt):
    python task_inference.py --out_dir=out-dyck --dataset=dyck \\
        --gen_param_overrides="[{'prefix_len':40,'max_depth':8}]"
"""
import csv
from contextlib import nullcontext

import torch

from task_eval_utils import build_generator, evaluate_accuracy, load_checkpoint

# -----------------------------------------------------------------------------
out_dir = 'out-small'      # directory holding the trained checkpoint (ckpt.pt)
model = ''                 # override the architecture in models/ (empty = use the checkpoint's own)
dataset = ''               # override the generator/task in generator/ (empty = use the checkpoint's own)
gen_param_overrides = []   # list of dicts, each merged over the checkpoint's own gen_params; one row per dict.
                           # e.g. [{'n_pairs':96},{'n_pairs':192}] for kv_retrieval, [{'n_digits':10}] for addition.
                           # empty = a single row using exactly the checkpoint's own trained gen_params.
n_eval = 500               # freshly sampled examples per gen_param_overrides row, for the accuracy estimate
n_show = 20                # how many individual examples to write to examples_csv per row, for manual inspection
summary_csv = ''           # optional: path for one row per gen_param_overrides entry -> accuracy
examples_csv = ''          # optional: path for individual prompt/target/prediction rows (n_show per row)
seed = 1337
device = 'cuda'            # examples: 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

net, checkpoint = load_checkpoint(out_dir, device=device, model_override=(model or None))
block_size = checkpoint['model_args']['block_size']
trained_dataset = checkpoint.get('config', {}).get('dataset', '?')
trained_gen_params = checkpoint.get('config', {}).get('gen_params', {})
print(
    f"Loaded checkpoint from {out_dir!r}: iter={checkpoint.get('iter_num', '?')}, "
    f"dataset={trained_dataset!r}, gen_params={trained_gen_params}"
)

overrides = gen_param_overrides or [{}]
print(f"gen_param overrides to test: {overrides}")

summary_rows = []   # (override_label, accuracy, n_eval)
example_rows = []   # (override_label, example_index, prompt, true_answer, pred_answer, correct)

for override in overrides:
    label = ','.join(f'{k}={v}' for k, v in override.items()) or 'trained'
    try:
        gen = build_generator(checkpoint, dataset_override=(dataset or None), gen_param_overrides=override, seed=seed)
        result = evaluate_accuracy(net, gen, n_eval=n_eval, block_size=block_size, device=device, ctx=ctx)
    except Exception as e:
        # Any task-specific failure -- an override the generator's own
        # constructor rejects, a context size collate() can't fit into this
        # checkpoint's block_size, etc. -- skips this row rather than
        # aborting the whole sweep.
        print(f"  {label}: skipped ({type(e).__name__}: {e})")
        continue
    acc = result['accuracy']
    print(f"  {label}: exact-match accuracy = {acc:.4f}  (n={result['n_eval']}, loss={result['loss']:.4f})")
    summary_rows.append((label, acc, result['n_eval']))

    if examples_csv:
        show_gen = build_generator(checkpoint, dataset_override=(dataset or None), gen_param_overrides=override, seed=seed + 1)
        show_result = evaluate_accuracy(net, show_gen, n_eval=min(n_show, n_eval), block_size=block_size,
                                         device=device, ctx=ctx)
        for i in range(show_result['n_eval']):
            item = show_result['items'][i]
            row_mask = show_result['mask'][i]
            true_answer = show_result['targets'][i][row_mask].tolist()
            pred_answer = show_result['preds'][i][row_mask].tolist()
            example_rows.append((
                label, i,
                item.prompt.tolist(), true_answer, pred_answer,
                pred_answer == true_answer,
            ))

if summary_csv:
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gen_params', 'accuracy', 'n_eval'])
        writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} row(s) to {summary_csv}")

if examples_csv:
    with open(examples_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gen_params', 'example_index', 'prompt', 'true_answer', 'pred_answer', 'correct'])
        writer.writerows(example_rows)
    print(f"Wrote {len(example_rows)} row(s) to {examples_csv}")
