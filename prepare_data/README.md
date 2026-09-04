# This folder contains code preparing the training and test datasets.

## 00_flanking_regions_table.py
This code exracts flanking regions of the peptides and propeptides.

## 01_fill_in_missing_flanking_residues.py
This code replaces all empty positions of the flanking regions with a specific symbol (not equivalent to the AA symbols).

## esm2_2d_embeddings_concat_and_means.py
The code generate embeddings from the flanking regions using ESM-2 model.

## esm3_2d_embeddings_concat_and_means.py
The code generate embeddings from the flanking regions using ESM-3 model.

## get_clusters.py
This code splits embeddings into clusters.

## filter_fasta_and_get_folds.py
The code performs cross-validation.

# old 
The folder with some useful old versions of the code.
