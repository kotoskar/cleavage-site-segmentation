"""Generate ONLY the 20-d 3Di one-hot per residue (ProstT5 AA->3Di), one .pt per sequence,
filename = md5(sequence). This is the structure channel used by the esmc6b_3di concat
(2560 ESM-C 6B (+) 20 3Di = 2580). Decoupled from the concat so we can backfill 3Di for
new sequences and reuse the already-computed ESM-C 6B embeddings separately.

Reuses the exact ProstT5 + alignment helpers from make_embeddings_esm2_3di_concat.py
(the EOS/length handling in _3di_to_onehot is easy to get subtly wrong), runs only the
Phase-1 loop. Skips sequences whose .pt already exists (resumable).

Run from repo root, PYTHONPATH=.:
  env/bin/python src/utils/embedding_generators/make_3di_only.py <fasta> <out_dir> [--device cuda]
"""
import argparse, gc, os, pathlib, sys
import torch
from tqdm.auto import tqdm
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.getcwd())
from src.utils.embedding_generators.make_embeddings_esm2_3di_concat import (
    hash_aa_string, iter_fasta, _prep_aa_for_prostt5, _3di_to_onehot, _sorted_batches,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta_file", type=pathlib.Path)
    ap.add_argument("output_dir", type=pathlib.Path)
    ap.add_argument("--prostt5_model", default="Rostlab/ProstT5_fp16")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--batch_max_residues", type=int, default=1536)
    ap.add_argument("--batch_max_seqs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    # gather pending (skip existing, skip too-long like the concat script)
    pending, skipped_cached, skipped_long = [], 0, 0
    for name, seq in iter_fasta(a.fasta_file):
        h = hash_aa_string(seq)
        if (a.output_dir / f"{h}.pt").is_file():
            skipped_cached += 1; continue
        if a.max_tokens > 0 and (len(seq) + 2) > a.max_tokens:
            skipped_long += 1; continue
        pending.append((name, seq))
    print(f"[3di] pending={len(pending)} cached={skipped_cached} too_long(> {a.max_tokens})={skipped_long}", flush=True)
    if not pending:
        print("[3di] nothing to do"); return

    tok = T5Tokenizer.from_pretrained(a.prostt5_model, do_lower_case=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.prostt5_model).to(a.device).eval()

    mism = 0
    pbar = tqdm(total=len(pending), desc="3Di one-hot (ProstT5)", dynamic_ncols=True)
    with torch.inference_mode():
        for batch in _sorted_batches(pending, a.batch_max_residues, a.batch_max_seqs):
            seqs = [s for _, s in batch]
            hashes = [hash_aa_string(s) for s in seqs]
            inp = tok([_prep_aa_for_prostt5(s) for s in seqs], return_tensors="pt",
                      add_special_tokens=True, padding=True)
            inp = {k: v.to(a.device, non_blocking=True) for k, v in inp.items()}
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if a.device == "cuda" \
                else __import__("contextlib").nullcontext()
            with ctx:
                gen = model.generate(**inp, max_new_tokens=max(len(s) for s in seqs),
                                     num_beams=a.num_beams, do_sample=False, use_cache=True)
            preds = tok.batch_decode(gen, skip_special_tokens=True)
            for seq, h, pred in zip(seqs, hashes, preds):
                pred = "".join(pred.split()).lower()
                if len(pred) != len(seq):
                    mism += 1
                onehot = _3di_to_onehot(pred, length=len(seq))   # (L, 20)
                torch.save(onehot.cpu(), a.output_dir / f"{h}.pt")
            pbar.update(len(batch)); pbar.set_postfix(mismatch=mism)
    pbar.close()
    del model, tok; gc.collect()
    if a.device == "cuda":
        torch.cuda.empty_cache()
    print(f"[3di] DONE: wrote into {a.output_dir} (length-mismatch fallbacks: {mism})", flush=True)


if __name__ == "__main__":
    main()
