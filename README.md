# Peptide/propeptide segmentation from frozen protein-language-model embeddings

This repository trains and evaluates a sequence-labeling model that finds
cleaved peptide and propeptide segments inside full-length protein sequences.
It is an anonymized, minimal, self-contained release for double-blind review:
code only, no experiment outputs, no real data payload, no author-identifying
information.

## What the code does

The model is a per-residue tagger over a frozen protein-language-model (pLM)
embedding:

```
frozen pLM embedding  -->  (optional input adapter)  -->  CNN + BiLSTM encoder
                        -->  linear emissions  -->  extended-state linear-chain CRF
```

- **Encoder** (`src/models/lstm_cnn.py`): a 1D convolution stack followed by a
  bidirectional LSTM over the (frozen, precomputed) per-residue embedding.
- **Extended-state CRF decoder** (`src/models/multi_tag_crf.py`,
  `src/models/crf_models.py`): rather than a plain 2/3-label CRF, each
  peptide/propeptide is represented as a *skip-chain* of up to 50 states
  (`CRFBaseModel.get_crf_constraints`), so a single "peptide" or "propeptide"
  label expands into a length-aware start -> inside -> end state path. This
  lets the CRF's transition structure encode a minimum (5 aa) and maximum
  (50 aa) segment length directly in its allowed transitions, instead of only
  in the loss.
- **Input adapter** (`ProjectedLSTMCNN` / `LSTMCNNCRFProjector*` in
  `src/models/crf_models.py`): a small trainable projection (LayerNorm +
  linear, optionally multi-scale or gated with a second embedding branch such
  as 3Di structure tokens) placed in front of the encoder, so the frozen
  embedding can be re-projected/adapted without unfreezing the pLM itself.
- **Boundary head** (`LSTMCNNCRFBoundaryBondLoss`, `LSTMCNNCRFGated3DiBoundary`
  in `src/models/crf_models.py`): a small auxiliary head that adds
  position-specific corrections to the CRF's start/inside/end emissions
  (and, in the bond-loss variant, an auxiliary soft boundary-proximity loss),
  intended to sharpen cleavage-site localization beyond what the base CRF
  emissions alone provide.

`src/train_loop_crf.py:get_model()` enumerates every model variant
(`--model` flag) that combines these pieces; `lstmcnncrf` is the base
CNN-BiLSTM-CRF with none of the adapter/boundary additions, used as the
default and as the target of the smoke test below.

This is a derivative of the openly available
[DeepPeptide](https://github.com/fteufel/DeepPeptide) codebase, extended with
additional pLM embedding backends (ESM2, ESM-C, ProstT5/3Di), the adapter and
boundary-head model variants above, and the nested cross-validation tooling
under `analysis/`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate     # or conda
python3 -m pip install -r requirements.txt
```

`requirements.txt` is a pip-freeze of the original (Linux + CUDA) training
environment; see the note at its top for what was actually verified for this
release (CPU, newer package versions) and why the CUDA wheel pins were
removed. Install `torch` via https://pytorch.org/get-started/locally/ first,
picking the CPU or CUDA build for your machine.

`aim` (experiment tracking) is optional -- `run.py` and the nested-CV driver
scripts fall back to a no-op stand-in if it isn't installed.

## Expected data layout

See [`data/README.md`](data/README.md) for the exact schema of
`labeled_sequences.csv`, `graphpart_assignments.csv`, and the precomputed
embeddings directory that `src/utils/dataset.py` expects. No real data is
included in this release -- only the small preprocessing scripts
(`prepare_data/`, `data/preprocessing/`) and that format description.

## Precomputing embeddings

`src/utils/embedding_generators/` has one script per supported frozen pLM
(ESM2, ESM-C 600M/6B, ProstT5 3Di, and combinations), each taking a FASTA file
and an output directory and writing one `<md5(sequence)>.pt` file per
sequence. These scripts have their own, heavier dependencies (`esm`,
`fair-esm`, `transformers`, network access to fetch pretrained weights or an
API token for hosted inference) that are not required just to train/evaluate
against already-precomputed embeddings.

## Running a single training run

```bash
PYTHONPATH=. python3 run.py \
  --embeddings_dir data/embeddings \
  --data_file data/labeled_sequences.csv \
  --partitioning_file data/graphpart_assignments.csv \
  --out_dir runs/train_run
```

This trains on folds `[0,1,2]`, validates on fold `[3]`, and reports a final
test on fold `[4]` (`src/train_loop_crf.py:train()`'s defaults), writing
`config.json`, periodic checkpoints, `valid_metrics.json`, `test_metrics.json`
and `model.pt` (best-validation checkpoint) to `--out_dir`. See
`parse_arguments()` in `src/train_loop_crf.py` for every architecture/training
flag (there are many -- one set per model variant described above).
[`configs/esm2_base.json`](configs/esm2_base.json) is a complete,
scrubbed config for the base `lstmcnncrf` model (this is the exact config this
release's own smoke-training run produced, with paths generalized) -- use it
as a starting point for `--set key=value` overrides, or as the `--base` for
the nested-CV driver below.

To evaluate a saved checkpoint separately (rather than only the automatic
end-of-training test pass above):

```bash
PYTHONPATH=. python3 -m src.infer_and_score \
  --embeddings_dir data/embeddings \
  --data_file data/labeled_sequences.csv \
  --partitioning_file data/graphpart_assignments.csv \
  --checkpoint runs/train_run/model.pt \
  --out_dir runs/train_run/infer_test \
  --split test --num_workers 0
```

(`--num_workers 0` avoids multiprocessing DataLoader worker startup on
CPU-only / small-dataset runs; the default of 8 is tuned for a real GPU node
with a large dataset.) `infer.py` at the repo root does the same for every
`runs/<name>/` subdirectory at once and additionally computes residue-level
MCC/AUC on top of the peptide/propeptide-level metrics.

## Running the nested 5x4 cross-validation protocol

The reported numbers use 5-fold nested CV: for each of the 5 outer folds,
4 models are trained (one per remaining "inner" fold used for early
stopping/validation), each evaluated on the untouched outer fold; the 4 inner
estimates are averaged per outer fold, and mean +/- std is reported **across
the 5 outer folds** (the actual independent-sample axis -- the 4 inner cells
per outer fold share the same held-out test set).

```bash
# one (outer, inner) cell:
PYTHONPATH=. python3 analysis/experiments/train_nested_cell.py \
  --base configs/esm2_base.json \
  --emb data/embeddings \
  --split data/graphpart_assignments.csv \
  --data data/labeled_sequences.csv \
  --out my_5cv_run --outer 0 --inner 1 --n_folds 5

# all 20 cells (5 outer x 4 inner), reboot-safe/resumable:
echo '{"name": "my_5cv_run", "base": "configs/esm2_base.json", "emb": "data/embeddings", "split": "data/graphpart_assignments.csv", "set": {}}' > analysis/experiments/queue.jsonl
PYTHONPATH=. python3 analysis/experiments/nested_cv_driver.py \
  analysis/experiments/queue.jsonl analysis/experiments/status.json
```

This writes `runs/<name>/outer{o}_inner{k}/cell_result.json` per cell and
`runs/<name>/nested_cv_summary.json` once all 20 cells are done. Both scripts
are verified to run end-to-end against the synthetic dataset below (see
"Smoke test").

## The experiment grid

All results in the paper come from one grid of eight configurations, plus a ninth run that
reproduces the original 2022-release setting on the same protocol:

|                        | boundary head | input adapter | config |
|------------------------|:-------------:|:-------------:|--------|
| ESM-2, base            | --            | --            | `configs/esm2_base.json` |
| ESM-2, + boundary head | yes           | --            | `configs/esm2_boundary.json` |
| ESM-2, + adapter       | --            | yes           | `configs/esm2_adapter.json` |
| ESM-2, + both          | yes           | yes           | `configs/esm2_full.json` |
| ESM-C 6B, base         | --            | --            | `configs/esmc6b_base.json` |
| ESM-C 6B, + boundary   | yes           | --            | `configs/esmc6b_boundary.json` |
| ESM-C 6B, + adapter    | --            | yes           | `configs/esmc6b_adapter.json` |
| ESM-C 6B, + both       | yes           | yes           | `configs/esmc6b_full.json` |
| ESM-2, base, 2022 data | --            | --            | `configs/esm2_base_2022.json` |

The last row is the reproduction run: the same base architecture and the same nested
cross-validation, on the 2022 UniProtKB/Swiss-Prot release and its original homology split
rather than the 2026 rebuild, so that our numbers can be put next to the ones the original
work published.

Every configuration trains in fp32 (`"amp": false`). Mixed precision is supported by the
training loop but is not used for any reported result, so that the comparison between
configurations is not confounded by numerics.

Each configuration is 20 nested-CV cells, so the grid is 180 trainings:

```bash
bash analysis/experiments/run_full_grid.sh          # sequential
bash analysis/experiments/run_full_grid.sh 4        # 4 cells at a time
```

Cells are resumable -- a cell with an existing `cell_result.json` is skipped --
so the script can be re-run after an interruption. The queue it reads is
`analysis/experiments/nested_cv_queue.jsonl`; edit that to run a subset.

### How the two switches map onto the code

The configs select three model classes, which is easy to misread as three
different architectures. They are not. What each switch does:

- **Boundary head**: `boundary_state_scale` is `1.0` (on) or `0.0` (off). At
  `0.0` the head's contribution is skipped by an early return in
  `_add_boundary_state_emissions`, so it adds *exactly* nothing to the
  emissions and receives no gradient. `configs/esm2_adapter.json` and
  `configs/esm2_full.json` differ in this one number and in nothing else; both
  build a model with an identical parameter count.
- **Input adapter**: model class `lstmcnncrf_gated3di_boundary` with
  `gated_seq_only: true`. Under `seq_only` the structural (3Di) branch is
  *not constructed at all* -- see the `if not seq_only:` guard in
  `GatedResidualConvSplitProjector` -- and the projector reduces to
  LayerNorm + reprojection, which is the adapter described in the paper. The
  `3di` in the class name is historical; no structural channel is used
  anywhere in this grid.
- **`lstmcnncrf_boundary_bond_loss`** is the boundary head without the adapter.
  It also builds an auxiliary bond head, but every config here sets
  `bond_loss_lambda: 0.0`, so that head contributes `0.0 * bond_loss` to the
  objective (no gradient) and is unused at inference. It is inert.

Two consequences worth knowing when reading checkpoints:

1. Parameter counts differ between the on/off variants of a switch even where
   the architecture is functionally identical, because inert heads are still
   constructed. This is expected.
2. Configs carry keys their model class ignores -- `configs/esmc6b_base.json`
   has `boundary_state_scale` and `bond_loss_lambda`, but `lstmcnncrf` reads
   neither. Read the model class, not the key list, to know what was built.

### One inconsistency in the published runs

Every config here has `"amp": false`, and so did every run behind the reported
numbers. An earlier version of the ESM-2 baseline was trained with bf16 autocast,
before mixed precision was switched off for the rest of the grid, and the
baseline was retrained in fp32 rather than left as the odd one out. The
difference between the two is measured rather than assumed: under the same
protocol and the same corrected matcher, the bf16 baseline scores
0.5725 +/- 0.0253 F1 at +-3 against 0.5761 +/- 0.0294 for the fp32 baseline the
paper reports, a gap of 0.0036 with fp32 the higher of the two.

The reason to care is that the autocast region covered the whole forward pass,
CRF included (see `train_loop_crf.py`), so under `amp: true` the CRF's log-space
forward algorithm ran in bfloat16. bf16 carries the same exponent range as fp32,
which is what that algorithm mainly depends on, but only about three decimal
digits of mantissa. `run_dataloader` also carries a nan-guard that skips the
optimizer step on a non-finite loss, so numerical trouble in this region is not
purely hypothetical. Mixed precision remains available in the training loop; it
is simply not used for anything reported.

## Recomputing the paper's numbers without a GPU

`results/` ships the scores every table and every figure in the paper is built
from: for each of the nine configurations, the per-tolerance aggregate, the
per-cell records under both the original and the corrected matcher, and the two
tolerance-free suites. No model weights, no embeddings, ~3 MB in total.

```
python3 results/reproduce_paper_numbers.py
```

prints Table 1 in full, the paired-by-fold effects and their sign counts, the
interaction contrast, the gap to the base at every tolerance, the tolerance-free
IoU sweep, the paired placement comparison and what each block does to the
segment count, what the corrected matcher changes per run, a rebuild of Table 1
from the individual cells, the per-fold base scores behind the
fold-exchangeability argument, and the 2022-release reproduction. Sections run
individually with `--table1`, `--effects`, `--tolerance`, `--overlap`,
`--sites`, `--matcher`, `--cells`, `--folds`, `--reproduction`, `--uncertainty`
and `--breakdown`. The last two print the paired effects with their bootstrap
intervals, and the same effects split by segment type, annotation evidence and
segment length, from `results/suites/review_response_cv.json`.

Each cell directory carries `tolerance_metrics.json`, the corrected matcher's
counts on that cell's own test fold at every tolerance from exact to +-3.
`--cells` averages the four inner cells of an outer fold and then the five outer
folds, and checks the result against the aggregate each run publishes: over the
180 cells of the nine runs the worst disagreement is 5e-07. Every number in
Table 1 therefore rests on per-cell records that ship with this repository,
rather than on an aggregate the reader has to take on trust.

The set is deliberately wider than any single write-up quotes, so a reader
holding a different version of this work can still find the number in front of
them here.

Two things about `results/` are worth knowing before the numbers are used,
and the script prints each of them where it matters.

Every cell of every run ran to the end of its training schedule and records the
epoch its validation curve selected, so no number here rests on a checkpoint
whose selection could still have moved. `--epochs` performs that check and
prints what it finds.

Both readings of every cell come from the segments that cell saved:
`tolerance_metrics.json` under the corrected matcher and `original_metrics.json`
under the upstream one, which puts every run in the matcher comparison. Each
cell also recorded an original-matcher F1 during training, and `--matcher`
checks the reconstruction against it: all 180 cells agree to 1.2e-03 F1 at worst
and to 3.1e-07 at the median, so what separates the two readings is inference
nondeterminism rather than disagreement between two deterministic functions of
the same segments.

And an aggregate spread is always taken across the five outer folds rather than
across all twenty cells, since four models sharing an outer fold share a test
set.

Training from scratch is a different matter: one cell is about 10 GPU-hours, a
configuration 200, and the full grid roughly 1,600. The commands are above.

## Rebuilding the span dumps and the tolerance-free tables

The tolerance-free analyses (the IoU sweep, the placement of each end of a
segment, the scores by segment length and by segment type, the paired effects
with intervals, the split by annotation evidence) are computed from *span
dumps*, not from model weights. For every nested-CV cell,
`runs/<run>/outer<o>_inner<k>/segments.json.gz` holds the decoded true and
predicted segments of that cell's test fold. The dumps are not shipped, because
they come from the trained checkpoints, which are not shipped either. A reader
who trains the grid writes them with one inference pass per cell, and every step
after that runs on a CPU in seconds to a minute:

```bash
# 1. the dumps, plus the per-cell scores at every tolerance (needs the checkpoints)
python3 analysis/metrics/src/tolerance_sweep_cv.py --models 5cv_esm2_full --device 0
# 2. everything that reads the dumps (CPU)
python3 analysis/metrics/src/segment_quality_cv.py    # IoU sweep, paired placement
python3 analysis/metrics/src/boundary_error_cv.py     # which end of a segment moves
python3 analysis/metrics/src/bylength_cv.py           # F1 by segment length
python3 analysis/metrics/src/review_response_cv.py    # paired effects with intervals; by type, evidence, length
# 3. the LaTeX fragments of the two appendices built from that output
python3 analysis/metrics/src/generators/paper/tex_review_tables.py
```

`review_response_cv.py` also reads the processed 2026 data: `labeled_sequences.csv`
(the organism column gives the genus used for resampling), the GraphPart
assignment files, and a per-feature table `processed_data/features_master.parquet`
with the columns `accession`, `label` (`PEPTIDE` or `PROPEP`), `start`, `end`,
`evidence_codes` (a list of ECO identifiers) and `is_valid_for_training`, which
the dataset build writes from the UniProt pull and which carries the evidence
code of each annotated segment. The locations are set with `--runs-dir` and
`--data-dir`. The output for the paper's runs ships as
`results/suites/review_response_cv.json`, so the numbers can be inspected
without any of this.

## Smoke test / trying it without real data

`tests/synthetic/` contains a tiny, fully synthetic stand-in for the three
real-data inputs above (~40 fake sequences, random 1280-d embeddings, 5-fold
partitioning), generated by `tests/synthetic/generate_synthetic_data.py` and
already materialized in this directory so the pipeline can be exercised
immediately. See that script's docstring for exactly how it constructs
label sequences that are valid under the project's own CRF grammar. Regenerate
it any time with:

```bash
python3 tests/synthetic/generate_synthetic_data.py
```

This release was smoke-tested end-to-end on CPU using exactly this synthetic
data -- see "Verification" below for the commands and real output.

## Configuration fidelity

Every file in `configs/` was checked key by key against the configuration that
produced the corresponding run reported in the paper; all eight agree on every
key except the input/output paths. `esm2_adapter.json` and `esm2_full.json`,
for instance, differ only in `boundary_state_scale` (0.0 vs 1.0), which is
exactly what turns the boundary head off and on.

## Known issues found while preparing this release

These are pre-existing properties/bugs of the research code, not introduced
by anonymization. They are reported here rather than silently patched, except
where noted as a portability fix.

1. **Metric-matching bug in `src/utils/manuscript_metrics.py:get_counts_for_protein`.**
   The outer loop over true segments and the inner loop over predicted
   segments both reuse the loop variable `idx` from `DataFrame.iterrows()`, so
   a match's "matched" flag can be written to the wrong true-segment row. This
   can deflate recall/F1 (more so for proteins with more predicted than true
   segments). It affects both the metrics printed during training/eval and
   the numbers `src/infer_and_score.py` / `infer.py` report.
   `analysis/experiments/rescore_nested_cv_corrected.py` and
   `analysis/errors/src/error_analysis.py` implement and cross-check a
   corrected matcher against this bug; both require real trained checkpoints
   (`runs/<name>/outer*_inner*/model.pt`) that are not part of this release
   and so cannot be exercised by the smoke test -- they are included only as
   documentation of a methodological detail. **Left unfixed** in
   `manuscript_metrics.py` itself, deliberately, since changing scoring
   behavior is a research decision outside the scope of this anonymization
   pass.

2. **`_sample_from_overlapping_peptides` (`src/utils/dataset.py`) assumes every
   protein has at least one peptide or propeptide segment.** A protein with
   zero segments makes `zip(*[])` raise `ValueError: not enough values to
   unpack`. This is never triggered by the original curated dataset
   (empirically, 0 of 8,449 real rows in `data/uniprot_2022/labeled_sequences.csv`
   have both `coordinates` and `propeptide_coordinates` empty), which is why it
   went unnoticed -- but it was triggered immediately by an earlier,
   more naive version of the synthetic-data generator in this release, which
   did generate zero-segment proteins. The generator was changed to respect
   the same invariant the real dataset has (every protein gets >=1 segment);
   `src/utils/dataset.py` itself was **left unfixed**, since patching a
   dataset-loading edge case in the shipped research code is a substantive
   change beyond anonymization/packaging.

3. **CPU portability (fixed).** `src/train_loop_crf.py:train()` /
   `train_homo_loo()` and `src/infer_and_score.py:main()` originally hard
   -required CUDA (`assert torch.cuda.is_available()`), making them impossible
   to run or smoke-test on a CPU-only machine. Both now fall back to CPU when
   CUDA is unavailable, mirroring the CPU fallback `infer.py`'s own
   `resolve_device()` already implements. No CLI flags changed.

4. **`aim` as a hard dependency (fixed).** `run.py` imported and instantiated
   `aim.Run()` unconditionally. It now falls back to a no-op stand-in if
   `aim` is not installed, matching the fallback already used by
   `analysis/experiments/train_nested_cell.py` / `run_experiment.py`.

5. **Repo-root auto-detection via `.git` (fixed).**
   `analysis/experiments/rescore_nested_cv_corrected.py` and
   `analysis/errors/src/error_analysis.py` located the project root by walking
   up to the nearest ancestor containing a `.git` directory, which raises
   `StopIteration` whenever the code is used outside a git checkout (an
   extracted archive, a vendored copy). Both now look for the nearest ancestor
   containing `run.py`, the project's actual entry point.

6. **Hardcoded absolute paths (fixed, no behavior change for a fresh checkout).**
   A handful of scripts (`src/train_loop_crf.py`, `infer.py`,
   `prepare_data/*.py`) carried absolute defaults from the machine they were
   developed on. These are now relative, repo-local defaults; every affected
   flag remains fully overridable from the command line as before.

7. **Missing runtime keys in `configs/*.json` (fixed).** The configuration
   files here carry architecture and hyper-parameters only, while
   `src/train_loop_crf.py` reads `args.device` and `args.port` directly, so
   `analysis/experiments/train_nested_cell.py` raised `AttributeError:
   'Namespace' object has no attribute 'device'` on every shipped config. The
   runner now supplies both defaults itself.

8. **`--set` in `analysis/experiments/train_nested_cell.py` silently drops
   overrides when the flag is repeated (not fixed).** The argument is declared
   with `nargs="*"`, so a second `--set` *replaces* the first rather than
   appending to it:

   ```bash
   # WRONG -- epochs=2 is silently discarded, the base config's value is used
   ... --set epochs=2 --set batch_size=4
   # RIGHT -- one flag, several key=value pairs
   ... --set epochs=2 batch_size=4
   ```

   Nothing warns about this, so a mistyped invocation produces a run with the
   wrong hyper-parameters and a `config.json` that looks deliberate. Left as-is
   because changing it to `action="append"` would alter the CLI contract of a
   script that produced already-reported results; the correct usage is
   documented here instead. Verified by running one cell both ways.


## Anonymization

This is a code-only release prepared for double-blind review. It carries no
author, affiliation, or host-machine information, and no experiment outputs or
real data. The tree was re-scanned after every edit and again before
publication.

## License

The upstream DeepPeptide project is BSD-3-Clause licensed. Its `LICENSE` file
names an individual copyright holder, so it is omitted here and will be
restored at camera-ready.

## Verification (this release)

Environment: Python 3.10 with `torch==2.10.0` (CPU-only build,
`torch.cuda.is_available()` is `False`), `pandas==2.3.3`, `numpy==2.2.6`,
`tqdm==4.67.1`. `aim` was not installed and was not needed, given fix (4)
above.

```bash
PYTHONPATH=. python3 run.py \
  --embeddings_dir tests/synthetic/embeddings \
  --data_file tests/synthetic/labeled_sequences.csv \
  --partitioning_file tests/synthetic/graphpart_assignments.csv \
  --epochs 2 --batch_size 4 \
  --out_dir runs/smoke_test --checkpoints_dir runs/smoke_test/checkpoints
```

Result: exit code 0. Loaded 24 train / 8 valid / 8 test synthetic sequences.
Training loss decreased monotonically across the 2 logged epochs
(`train_loss=47.538` -> `42.986`; `valid_loss=37.851` -> `33.198`), and the
final end-of-training test pass ran to completion, writing `test_metrics.json`.
`f1_all`/`f1 peptides`/`f1 propeptides` were all `0.0` at epoch 2 -- expected
and unsurprising: the embeddings are random noise, so there is no real signal
for the model to learn from in 2 epochs on 24 sequences; this run demonstrates
that the training/eval *plumbing* works end to end, not that the architecture
learns anything from this data. `src/infer_and_score.py` was separately run
against the resulting checkpoint (`--num_workers 0`) and reproduced metrics
computation end to end. One nested-CV cell
(`analysis/experiments/train_nested_cell.py --outer 0 --inner 1`) was run
against the same synthetic data with each of the eight files in `configs/`,
each producing a `cell_result.json`, confirming the nested-CV path is not
merely documented but actually runs.
