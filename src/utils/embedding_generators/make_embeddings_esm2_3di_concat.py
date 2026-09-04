from hashlib import md5
import os
import re
import gc
import argparse
import pathlib
from typing import Iterator, Tuple, Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, AutoModelForSeq2SeqLM

THREE_DI_ALPHABET = "acdefghiklmnpqrstvwy"
THREE_DI_TO_ID: Dict[str, int] = {ch: i for i, ch in enumerate(THREE_DI_ALPHABET)}


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


def iter_fasta(path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    label = None
    seq_parts = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    yield label, "".join(seq_parts)
                label = line[1:].strip() or "seq"
                seq_parts = []
            else:
                seq_parts.append(line.replace(" ", "").upper())
    if label is not None:
        yield label, "".join(seq_parts)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prep_aa_for_prostt5(seq: str) -> str:
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    return "<AA2fold> " + " ".join(list(seq))


def _3di_to_onehot(three_di: str, length: int) -> torch.Tensor:
    three_di = "".join(ch for ch in three_di.lower() if ch in THREE_DI_TO_ID)
    if len(three_di) > length:
        three_di = three_di[:length]
    elif len(three_di) < length:
        pad_ch = three_di[-1] if len(three_di) > 0 else "d"
        three_di = three_di + (pad_ch * (length - len(three_di)))
    ids = torch.tensor([THREE_DI_TO_ID[ch] for ch in three_di], dtype=torch.long)
    out = torch.zeros(length, len(THREE_DI_ALPHABET), dtype=torch.float32)
    out[torch.arange(length), ids] = 1.0
    return out


def _trim_esm_to_residues(rep: torch.Tensor, seq_len: int) -> torch.Tensor:
    if rep.size(0) == seq_len + 2:
        return rep[1:-1, :]
    return rep[:seq_len, :]


def _sorted_batches(items: List[Tuple[str, str]], batch_max_residues: int, batch_max_seqs: int):
    items = sorted(items, key=lambda x: len(x[1]))
    batch: List[Tuple[str, str]] = []
    batch_res = 0
    for item in items:
        seq_len = len(item[1])
        if batch and (len(batch) >= batch_max_seqs or batch_res + seq_len > batch_max_residues):
            yield batch
            batch = []
            batch_res = 0
        batch.append(item)
        batch_res += seq_len
    if batch:
        yield batch


def generate_concat_esm2_3di_embeddings(
    fasta_file: pathlib.Path,
    output_dir: pathlib.Path,
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    esm_repr_layer: int = 33,
    esm_max_tokens: int = 0,
    prostt5_model_name: str = "Rostlab/ProstT5_fp16",
    prostt5_num_beams: int = 1,
    prostt5_max_tokens: int = 1024,
    prostt5_batch_max_residues: int = 1536,
    prostt5_batch_max_seqs: int = 8,
    keep_tmp_3di: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    items = list(iter_fasta(fasta_file))
    if not items:
        return

    device = _pick_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tmp_3di_dir = pathlib.Path(output_dir) / "__tmp_3di_onehot__"
    os.makedirs(tmp_3di_dir, exist_ok=True)

    # Phase 1: ProstT5 batched AA -> 3Di one-hot
    prost_tokenizer = T5Tokenizer.from_pretrained(prostt5_model_name, do_lower_case=False, use_fast=False)
    prost_model = AutoModelForSeq2SeqLM.from_pretrained(prostt5_model_name)
    prost_model.eval().to(device)
    if device == "cpu":
        prost_model = prost_model.float()
    elif device == "cuda":
        prost_model = prost_model.half()

    pending = []
    skipped_cached = 0
    skipped_long = 0
    for label, seq in items:
        seq_hash = hash_aa_string(seq)
        final_path = pathlib.Path(output_dir) / f"{seq_hash}.pt"
        tmp_3di_path = tmp_3di_dir / f"{seq_hash}.pt"
        if final_path.is_file() or tmp_3di_path.is_file():
            skipped_cached += 1
            continue
        # account for special token(s) conservatively
        if prostt5_max_tokens > 0 and (len(seq) + 2) > prostt5_max_tokens:
            skipped_long += 1
            continue
        pending.append((label, seq))

    pbar = tqdm(total=len(pending), desc="Phase 1/2: ProstT5 3Di (batched)", dynamic_ncols=True)
    mismatch_count = 0
    with torch.inference_mode():
        for batch in _sorted_batches(pending, prostt5_batch_max_residues, prostt5_batch_max_seqs):
            seqs = [seq for _, seq in batch]
            seq_hashes = [hash_aa_string(seq) for seq in seqs]
            model_inputs = [_prep_aa_for_prostt5(seq) for seq in seqs]
            batch_max_len = max(len(seq) for seq in seqs)

            tok = prost_tokenizer(
                model_inputs,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            )
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

            if device == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                generated = prost_model.generate(
                    **tok,
                    max_new_tokens=batch_max_len,
                    num_beams=prostt5_num_beams,
                    do_sample=False,
                    use_cache=True,
                )
            preds = prost_tokenizer.batch_decode(generated, skip_special_tokens=True)

            for seq, seq_hash, pred in zip(seqs, seq_hashes, preds):
                pred = "".join(pred.split()).lower()
                if len(pred) != len(seq):
                    mismatch_count += 1
                onehot_3di = _3di_to_onehot(pred, length=len(seq))
                torch.save(onehot_3di.cpu(), tmp_3di_dir / f"{seq_hash}.pt")

            pbar.update(len(batch))
            pbar.set_postfix(mismatch=mismatch_count, cached=skipped_cached, long=skipped_long)

    del prost_model
    del prost_tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: keep simple; user can skip if ESM2 already exists elsewhere
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name, do_lower_case=False)
    esm_model = AutoModel.from_pretrained(esm_model_name)
    esm_model.eval().to(device)
    if esm_max_tokens <= 0:
        esm_max_tokens = int(getattr(esm_model.config, "max_position_embeddings", 0)) or 1026
    n_layers = int(getattr(esm_model.config, "num_hidden_layers", 0))
    if n_layers and not (0 <= esm_repr_layer <= n_layers):
        raise ValueError(
            f"esm_repr_layer={esm_repr_layer} is invalid for this model (num_hidden_layers={n_layers}). "
            f"Use 1..{n_layers} for transformer layers (33 is last for esm2_t33)."
        )

    skipped_long_esm = 0
    pbar = tqdm(items, desc="Phase 2/2: ESM2 + concat", dynamic_ncols=True)
    with torch.inference_mode():
        for _, seq in pbar:
            seq_hash = hash_aa_string(seq)
            out_path = pathlib.Path(output_dir) / f"{seq_hash}.pt"
            tmp_3di_path = tmp_3di_dir / f"{seq_hash}.pt"
            if out_path.is_file():
                continue
            if not tmp_3di_path.is_file():
                continue
            inputs = esm_tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            if inputs["input_ids"].size(1) > esm_max_tokens:
                skipped_long_esm += 1
                pbar.set_postfix(long=skipped_long_esm)
                continue
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            out = esm_model(**inputs, output_hidden_states=True, return_dict=True)
            rep = out.hidden_states[esm_repr_layer][0]
            esm_res = _trim_esm_to_residues(rep, len(seq)).float().cpu()
            three_di_onehot = torch.load(tmp_3di_path, map_location="cpu")
            concat = torch.cat([esm_res, three_di_onehot], dim=-1)
            torch.save(concat, out_path)

    if not keep_tmp_3di:
        for p in tmp_3di_dir.glob("*.pt"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_3di_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--esm_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--esm_repr_layer", type=int, default=33)
    parser.add_argument("--esm_max_tokens", type=int, default=0)
    parser.add_argument("--prostt5_model", default="Rostlab/ProstT5_fp16")
    parser.add_argument("--prostt5_num_beams", type=int, default=1)
    parser.add_argument("--prostt5_max_tokens", type=int, default=1024)
    parser.add_argument("--prostt5_batch_max_residues", type=int, default=1536)
    parser.add_argument("--prostt5_batch_max_seqs", type=int, default=8)
    parser.add_argument("--keep_tmp_3di", action="store_true")
    args = parser.parse_args()
    generate_concat_esm2_3di_embeddings(
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        esm_model_name=args.esm_model,
        esm_repr_layer=args.esm_repr_layer,
        esm_max_tokens=args.esm_max_tokens,
        prostt5_model_name=args.prostt5_model,
        prostt5_num_beams=args.prostt5_num_beams,
        prostt5_max_tokens=args.prostt5_max_tokens,
        prostt5_batch_max_residues=args.prostt5_batch_max_residues,
        prostt5_batch_max_seqs=args.prostt5_batch_max_seqs,
        keep_tmp_3di=args.keep_tmp_3di,
    )


if __name__ == "__main__":
    main()
