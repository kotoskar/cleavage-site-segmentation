"""
Generate ESM-C embeddings (per residue) and save one .pt file per sequence.
Filename is md5 hash of the amino-acid sequence.

Optionally appends cheap per-residue features (K=10) -> output [L, D+10].
For esmc_600m: D=1152 => D_total=1162.
"""

from hashlib import md5
import os
import argparse
import pathlib

import torch
from tqdm.auto import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


# ---------- Compatibility patch for broken esm installs ----------
def _ensure_load_local_model_exists() -> None:
    """
    Some inconsistent `esm` installs have ESMC.from_pretrained() importing
    `esm.pretrained.load_local_model`, but that function is missing.
    If LOCAL_MODEL_REGISTRY exists, we can reconstruct it.
    """
    import esm.pretrained as pretrained

    if hasattr(pretrained, "load_local_model"):
        return

    registry = getattr(pretrained, "LOCAL_MODEL_REGISTRY", None)
    if registry is None or not isinstance(registry, dict) or len(registry) == 0:
        raise RuntimeError(
            "Your installed `esm` package is inconsistent (missing load_local_model and no LOCAL_MODEL_REGISTRY).\n"
            "Fix: reinstall a consistent version that supports Python 3.10, e.g.:\n"
            "  pip uninstall -y esm && pip install --force-reinstall esm==3.1.6\n"
        )

    def load_local_model(model_name: str, device: torch.device = torch.device("cpu")):
        if model_name not in registry:
            raise ValueError(
                f"Model '{model_name}' not found in LOCAL_MODEL_REGISTRY. "
                f"Available keys (first 20): {list(registry.keys())[:20]}"
            )
        return registry[model_name](device)

    setattr(pretrained, "load_local_model", load_local_model)


# ---------- Cheap deterministic per-residue features (K=10) ----------

_HYDRO_KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
_CHARGE7 = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": 0.1}
_AROMATIC = {"F", "W", "Y"}
_PRO = {"P"}
_GLY = {"G"}
_POLAR = {"S", "T", "N", "Q", "C", "Y", "D", "E", "K", "R", "H"}


def _sliding_mean_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    L = int(x.numel())
    if L == 0:
        return x
    cs = torch.zeros(L + 1, dtype=x.dtype, device=x.device)
    cs[1:] = torch.cumsum(x, dim=0)
    idx = torch.arange(L, device=x.device)
    start = torch.clamp(idx - radius, min=0)
    end = torch.clamp(idx + radius + 1, max=L)
    s = cs[end] - cs[start]
    denom = (end - start).to(x.dtype)
    return s / denom


def _sliding_sum_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    L = int(x.numel())
    if L == 0:
        return x
    cs = torch.zeros(L + 1, dtype=x.dtype, device=x.device)
    cs[1:] = torch.cumsum(x, dim=0)
    idx = torch.arange(L, device=x.device)
    start = torch.clamp(idx - radius, min=0)
    end = torch.clamp(idx + radius + 1, max=L)
    return cs[end] - cs[start]


def compute_features(seq: str, device) -> torch.Tensor:
    """Returns float32 tensor [L, 10]."""
    L = len(seq)
    if L == 0:
        return torch.zeros((0, 10), dtype=torch.float32, device=device)

    hydro = torch.tensor([_HYDRO_KD.get(a, 0.0) for a in seq], dtype=torch.float32, device=device)
    charge = torch.tensor([_CHARGE7.get(a, 0.0) for a in seq], dtype=torch.float32, device=device)

    is_polar = torch.tensor([1.0 if a in _POLAR else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_arom = torch.tensor([1.0 if a in _AROMATIC else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_pro = torch.tensor([1.0 if a in _PRO else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_gly = torch.tensor([1.0 if a in _GLY else 0.0 for a in seq], dtype=torch.float32, device=device)

    if L == 1:
        pos = torch.zeros(L, dtype=torch.float32, device=device)
    else:
        pos = torch.arange(L, dtype=torch.float32, device=device) / float(L - 1)
    rev_pos = 1.0 - pos

    mean_hydro_w7 = _sliding_mean_1d(hydro, radius=3)  # window 7
    net_charge_w9 = _sliding_sum_1d(charge, radius=4)  # window 9

    return torch.stack(
        [pos, rev_pos, hydro, charge, is_polar, is_arom, is_pro, is_gly, mean_hydro_w7, net_charge_w9],
        dim=-1,
    )


def generate_esmc_embeddings(
    fasta_file: pathlib.Path,
    esm_embeddings_dir: pathlib.Path,
    model_name: str = "esmc_600m",
    repr_layer: int = 36,
    max_tokens: int = 2048,
    add_features: bool = True,
) -> None:
    _ensure_load_local_model_exists()

    esm_model = ESMC.from_pretrained(model_name)
    esm_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model = esm_model.to(device)

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0

    cfg_kwargs = {"sequence": True, "return_embeddings": True}
    ann = getattr(LogitsConfig, "__annotations__", {}) or {}
    if "return_hidden_states" in ann:
        cfg_kwargs["return_hidden_states"] = True
    cfg = LogitsConfig(**cfg_kwargs)

    def iter_fasta(path: pathlib.Path):
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

    items = iter_fasta(fasta_file)
    pbar = tqdm(items, desc="Embeddings", dynamic_ncols=True)

    base_D = None
    K = 10 if add_features else 0

    with torch.no_grad():
        for label, seq in pbar:
            out_path = os.path.join(str(esm_embeddings_dir), f"{hash_aa_string(seq)}.pt")

            if os.path.isfile(out_path):
                skipped_cached += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            if len(seq) > max_tokens:
                skipped_long += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            protein = ESMProtein(sequence=seq)
            protein_tensor = esm_model.encode(protein)
            if hasattr(protein_tensor, "to"):
                protein_tensor = protein_tensor.to(device)

            out = esm_model.logits(protein_tensor, cfg)

            hs = getattr(out, "hidden_states", None)
            if hs is not None and isinstance(hs, (list, tuple)) and len(hs) > 0:
                rep = hs[repr_layer] if 0 <= repr_layer < len(hs) else hs[-1]
            else:
                rep = out.embeddings

            if rep.dim() == 3:
                rep = rep[0]  # [T, D]

            if rep.size(0) == len(seq) + 2:
                res = rep[1:-1, :]
            else:
                res = rep[: len(seq), :]

            if base_D is None:
                base_D = int(res.size(-1))

            if add_features:
                feats = compute_features(seq, device=res.device)  # [L, 10]
                if feats.size(0) != res.size(0):
                    feats = feats[: res.size(0), :]
                res = torch.cat([res, feats.to(res.dtype)], dim=-1)

            torch.save(res.float().cpu(), out_path)

    if base_D is None:
        base_D = 0
    print(f"Done. Base D={base_D}. Added K={K}. Output D_total={base_D + K}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--repr_layer", type=int, default=36)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--no_features", action="store_true")
    args = parser.parse_args()

    generate_esmc_embeddings(
        fasta_file=args.fasta_file,
        esm_embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
        add_features=not args.no_features,
    )


if __name__ == "__main__":
    main()