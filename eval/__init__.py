"""
Checkpoint evaluation and CSV-export tooling: task_inference.py (test
inference on any registered task), export_checkpoint_scores_csv.py,
export_training_dynamics_csv.py, export_pdf_metrics_csv.py, and the shared
task_eval_utils.py they build on.

Run these directly from the repo root, e.g.:

    python eval/task_inference.py --out_dir=out-small

task_eval_utils.py adds the repo root to sys.path itself (see its own
docstring) so it can import the top-level generator/ and models/ packages
regardless of running from this subfolder -- no need for `python -m`.
"""
