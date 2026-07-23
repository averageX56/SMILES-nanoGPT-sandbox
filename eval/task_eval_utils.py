"""
Shared helpers for evaluating a trained checkpoint on any task registered in
generator/__init__.py -- not tied to KV-retrieval specifically. Used by
task_inference.py and export_checkpoint_scores_csv.py so there is one
definition of "how do we load a checkpoint, rebuild its task, and score it"
instead of several copies.

Covers:
    - load_checkpoint()    mirrors sample.py / inference.py's loading
    - build_generator()    rebuilds whichever generator a checkpoint used
                            (or an explicit --dataset override), with an
                            arbitrary gen_params override merged in -- this
                            is the hook for OOD sweeps, whatever the task's
                            own difficulty knob is called (n_pairs, n_digits,
                            n_items, prefix_len/max_depth, n_steps, ...)
    - evaluate_accuracy()  exact-match accuracy, same metric train.py
                           reports as val_acc -- via generator.collate()'s
                           answer-masked (x, y), so it handles single-token
                           and multi-token answers alike without needing to
                           know anything about the task's token layout
"""
import os
import sys

# This file lives in eval/, but generator/ and models/ live at the repo
# root. Running a script directly (`python eval/task_inference.py`) puts
# only eval/ on sys.path, not the repo root -- so without this, the imports
# below fail with ModuleNotFoundError. Prepending the repo root here (once,
# before those imports run) fixes it for every script that imports this
# module, without switching the whole project over to `python -m` invocation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from generator import get_generator
from models import get_model


def load_checkpoint(out_dir, device='cpu', model_override=None):
    """Load ckpt.pt from out_dir and rebuild the model it was trained with.

    Mirrors the loading logic in sample.py / inference.py exactly (same
    architecture lookup, same `_orig_mod.` prefix stripping), so a checkpoint
    behaves identically here as it would under those scripts.

    :param model_override: force a specific models/<name>.py instead of the
        one recorded in the checkpoint's config (falls back to 'base' if
        the checkpoint has neither).
    :returns: (model, checkpoint_dict)
    """
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path!r}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = model_override or checkpoint.get('config', {}).get('model', 'base')
    ModelConfig, Model = get_model(model_name)
    modelconf = ModelConfig(**checkpoint['model_args'])
    model = Model(modelconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def build_generator(checkpoint, dataset_override=None, gen_param_overrides=None, seed=1337):
    """Rebuild the generator a checkpoint was trained with, or an explicit
    ``dataset_override`` (a name from generator/__init__.py's GENERATORS
    registry). ``gen_param_overrides`` is merged over the checkpoint's own
    recorded gen_params -- this is the generic hook for evaluating at a
    different difficulty/context size (an OOD sweep), whatever the task's
    own knob happens to be called (e.g. {'n_pairs': 96} for kv_retrieval,
    {'n_digits': 10} for addition, {'n_items': 32} for sorting/indexing,
    {'prefix_len': 40, 'max_depth': 8} for dyck, {'n_steps': 6} for
    function_composition -- see each generator/*.py's own docstring).

    Uses a fresh ``seed`` so eval batches are reproducible but independent
    of training/val data.
    """
    dataset_name = dataset_override or checkpoint.get('config', {}).get('dataset')
    if not dataset_name:
        raise ValueError(
            "Checkpoint has no recorded config['dataset'] and no dataset "
            "override was given -- pass dataset_override explicitly."
        )
    gen_params = dict(checkpoint.get('config', {}).get('gen_params', {}))
    gen_params.update(gen_param_overrides or {})
    gen_params['seed'] = seed
    return get_generator(dataset_name, gen_params)


@torch.no_grad()
def evaluate_accuracy(model, generator, n_eval, block_size, device, ctx):
    """Exact-match accuracy of ``generator``'s task on ``model``, over
    ``n_eval`` freshly sampled examples.

    Uses ``generator.collate()`` (every task gets this for free from
    generator/base.py) to pack prompt+answer into an answer-masked (x, y),
    then a single forward pass with targets so the model returns logits at
    every position (Model.forward only computes the last-position logits
    when targets=None). Comparing argmax(logits) to y wherever y != -1 and
    requiring every such position correct is exactly train.py's own
    exact-match accuracy metric (see estimate_loss() there) -- it works for
    single-token answers (kv_retrieval, indexing, function_composition) and
    multi-token ones (addition, sorting, dyck) identically, with no
    per-task special-casing.

    ``collate`` itself raises ValueError if a packed example is longer than
    ``block_size + 1`` -- letting that propagate is the generic version of
    "this context size doesn't fit this checkpoint's block_size", for any
    task, without this function needing to know the task's prompt layout.
    """
    items = generator.sample(n_eval)
    x_np, y_np = generator.collate(items, block_size)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    with ctx:
        logits, loss = model(x, y)
    mask = y != -1
    pred = logits.argmax(dim=-1)
    row_correct = ((pred == y) | ~mask).all(dim=1)
    return {
        'accuracy': float(row_correct.float().mean()),
        'loss': float(loss),
        'n_eval': len(items),
        'preds': pred.cpu().numpy(),
        'targets': y.cpu().numpy(),
        'mask': mask.cpu().numpy(),
        'items': items,
    }
