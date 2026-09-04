# Expected data layout

This directory does not ship any real data (the underlying UniProt-derived
dataset is large and is not part of this minimal, anonymized release). It only
ships the small preprocessing scripts under `data/preprocessing/` and this
description of the exact file formats `src/utils/dataset.py` expects, so the
pipeline can be reproduced against your own data pull, or exercised
immediately against the tiny synthetic dataset in `tests/synthetic/`
(see the top-level README).

At training time you need three things, whose paths are passed to `run.py` /
`src/train_loop_crf.py` as `--data_file`, `--partitioning_file` and
`--embeddings_dir`:

## 1. `labeled_sequences.csv` (`--data_file`)

A CSV read with `pandas.read_csv(..., index_col='protein_id')`. Required
columns:

| column                    | type   | meaning                                                                 |
|---------------------------|--------|--------------------------------------------------------------------------|
| `protein_id`               | str    | Unique accession (e.g. a UniProt AC). Used as the row index.            |
| `sequence`                  | str    | Amino-acid sequence, upper-case, one letter per residue.                 |
| `organism`                  | str    | Free-text organism string (not used for modeling, kept for bookkeeping). |
| `coordinates`               | str    | Peptide segments, 1-based inclusive, e.g. `(97-101),(104-108)`. Empty string if none. |
| `propeptide_coordinates`    | str    | Propeptide segments, same coordinate format. Empty string if none.      |
| `is_peptide` (optional)     | str    | Per-residue 0/1 bit string, same length as `sequence` (legacy/simple label mode; not read by the CRF dataset classes). |
| `is_propeptide` (optional)  | str    | Same, for propeptides.                                                  |

Coordinate parsing (`src/utils/crf_label_utils.py:parse_coordinate_string`)
splits on `,`, then on `-`, stripping the parentheses -- so
`"(97-101),(104-108)"` becomes `[(97, 101), (104, 108)]`.

Every row must have at least one peptide OR propeptide segment (empirically
true of the original curated dataset: 0 of 8,449 real rows have both empty).
See "Known issues" in the top-level README for what happens if this
invariant is violated.

Peptide/propeptide segments follow an extended-state CRF grammar with
`min_len=5` and `max_len=50` residues per segment
(`src/models/crf_models.py:CRFBaseModel`). A segment shorter than 5 or longer
than 50 residues cannot be represented by the state grammar; the loader either
warns and drops it, or (if segments overlap) the dataset class randomly picks
one member per overlap group at each epoch
(`PrecomputedCSVForOverlapCRFDataset._sample_from_overlapping_peptides`).

## 2. `graphpart_assignments.csv` (`--partitioning_file`)

The output format of [GraphPart](https://github.com/graph-part/graph-part),
read with `pandas.read_csv(..., index_col='AC')` and joined onto
`labeled_sequences.csv` by index. Required columns:

| column                 | meaning                                                            |
|-------------------------|---------------------------------------------------------------------|
| `AC`                    | Accession, must match `protein_id` in `labeled_sequences.csv`.      |
| `cluster`                | Fold assignment (0-4 for the standard 5-fold protocol). Any protein whose `AC` is missing from this file is excluded from training/eval entirely. |
| `priority`, `label-val`, `between_connectivity` | GraphPart bookkeeping columns, not read by the training/eval code but kept for schema fidelity with real GraphPart output. |

Default partitioning in `src/train_loop_crf.py`: train = folds `[0,1,2]`,
valid = fold `[3]`, test = fold `[4]`. The nested 5x4 CV protocol
(`analysis/experiments/`) instead rotates all `(outer, inner)` fold pairs --
see the top-level README.

## 3. Precomputed embeddings directory (`--embeddings_dir`)

One file per sequence: `<md5(sequence)>.pt`, loaded with
`torch.load(path, map_location='cpu', weights_only=False)`. Each file holds a
single tensor (or numpy array; the loader accepts either) of shape
`(seq_len, embedding_dim)`, dtype float32, giving one embedding vector per
residue in the same order as `sequence`. `embedding_dim` must match the
`--embedding_dim` argument passed to training (1280 for the ESM2-650M
default; other pLMs use a different width -- see `src/utils/embedding_generators/`).

The hash is computed with `hashlib.md5(sequence.encode()).hexdigest()`
(`src/utils/dataset.py:make_hashes`) -- this makes the embeddings directory
dataset-agnostic: identical sequences reused across dataset revisions share
one embedding file. As a fallback, the loader also tries `<protein_id>.pt` if
the hash-named file is missing.

## Generating your own embeddings

See `src/utils/embedding_generators/` for one script per supported frozen
protein language model (ESM2, ESM-C 600M/6B, ProstT5 3Di, and combinations).
Each takes a FASTA file and an output directory and writes one `.pt` file per
sequence in the format above. `prepare_data/README.md` documents the earlier
flanking-region-based data preparation steps.

## Trying it without real data

`tests/synthetic/generate_synthetic_data.py` generates a tiny fully-synthetic
version of all three inputs above (~40 fake sequences, random embeddings,
5-fold partitioning) so the whole pipeline can be smoke-tested without any
real data or a real protein language model. See the top-level README.
