import time
from src.train_loop_crf import parse_arguments, train

import os

# `aim` (https://aimstack.io/) is an optional experiment tracker used by the
# original project for live metric dashboards. It is NOT required to train or
# evaluate the model -- if it isn't installed, fall back to a no-op stand-in so
# `run['hparams'] = ...` / `run.track(...)` calls inside train() are harmless.
# This is the same fallback already used by analysis/experiments/train_nested_cell.py
# and analysis/experiments/run_experiment.py.
try:
    from aim import Run
    run = Run()
except Exception:
    class _NoOpRun(dict):
        def track(self, *args, **kwargs):
            pass
    run = _NoOpRun()

start_time = time.time()
train(parse_arguments(), run=run)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
