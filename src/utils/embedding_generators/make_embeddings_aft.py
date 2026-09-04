from __future__ import annotations

"""
Generate per-residue AFToolkit embeddings while keeping the main script runnable
from the user's regular DeepPeptide environment.

This script:
1) reads FASTA
2) downloads AFDB PDBs by UniProt accession
3) writes a manifest CSV
4) invokes a worker script using a separate Python interpreter from the
   dedicated AFToolkit environment.

Output files are saved as md5(sequence).pt to match existing DeepPeptide
embedding pipelines.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple


UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-Z0-9]{10}$")
AFDB_ID_RE = re.compile(r"AF-([A-Z0-9]+)-F(\d+)", re.IGNORECASE)

AFDB_API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
AFDB_PDB_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"


def hash_aa_string(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


def iter_fasta(path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    label: Optional[str] = None
    seq_parts: List[str] = []
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


def parse_accession_from_label(label: str) -> Tuple[Optional[str], str]:
    m = AFDB_ID_RE.search(label)
    if m:
        return m.group(1), "A"

    pipe_parts = label.split("|")
    for i, tok in enumerate(pipe_parts):
        tok = tok.strip()
        if tok in {"sp", "tr"} and i + 1 < len(pipe_parts):
            acc = pipe_parts[i + 1].strip()
            if UNIPROT_RE.fullmatch(acc):
                return acc, "A"

    for tok in re.split(r"[\s|;,:()\[\]{}]+", label):
        tok = tok.strip()
        if UNIPROT_RE.fullmatch(tok):
            return tok, "A"

    return None, "A"


def read_mapping_csv(path: Optional[pathlib.Path]) -> Dict[str, Tuple[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mapping: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        accession = (row.get("accession") or "").strip()
        chain = (row.get("chain") or "A").strip() or "A"
        if not accession:
            continue
        label = (row.get("label") or "").strip()
        seq_hash = (row.get("seq_hash") or "").strip()
        if label:
            mapping[f"label::{label}"] = (accession, chain)
        if seq_hash:
            mapping[f"seq_hash::{seq_hash}"] = (accession, chain)
    return mapping


def _http_get_json(url: str, timeout: int = 60, retries: int = 3) -> object:
    headers = {"User-Agent": "deeppeptide-aft-embedding-generator/1.0"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_exc is not None
    raise last_exc


def _download_file(url: str, dst: pathlib.Path, timeout: int = 120, retries: int = 3) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    headers = {"User-Agent": "deeppeptide-aft-embedding-generator/1.0"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(dst)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt < retries:
                time.sleep(2.0 * attempt)
    assert last_exc is not None
    raise last_exc


def resolve_afdb_pdb_url_and_name(accession: str) -> Tuple[str, str]:
    api_url = AFDB_API_TEMPLATE.format(accession=accession)
    try:
        payload = _http_get_json(api_url)
        if isinstance(payload, list) and payload:
            rec = payload[0]
            pdb_url = rec.get("pdbUrl") or rec.get("pdb_url")
            entry_id = rec.get("entryId") or rec.get("entry_id") or f"AF-{accession}-F1"
            if pdb_url:
                return str(pdb_url), f"{entry_id}.pdb"
    except Exception:
        pass
    return AFDB_PDB_TEMPLATE.format(accession=accession), f"AF-{accession}-F1-model_v4.pdb"


def ensure_afdb_pdb(accession: str, afdb_dir: pathlib.Path, force_redownload: bool = False) -> pathlib.Path:
    afdb_dir.mkdir(parents=True, exist_ok=True)
    url, filename = resolve_afdb_pdb_url_and_name(accession)
    out_path = afdb_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0 and not force_redownload:
        return out_path
    _download_file(url, out_path)
    return out_path


def write_csv(path: pathlib.Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--afdb-dir", type=pathlib.Path, required=True)
    parser.add_argument("--aft-python", required=True, help="Path to python binary from aftoolkit_env")
    parser.add_argument("--worker-script", type=pathlib.Path, required=True)
    parser.add_argument("--mapping-csv", type=pathlib.Path, default=None)
    parser.add_argument("--features", nargs="+", default=["single", "pair", "lddt_logits", "plddt"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recycles", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.afdb_dir.mkdir(parents=True, exist_ok=True)

    mapping = read_mapping_csv(args.mapping_csv)

    manifest_rows: List[dict] = []
    dl_rows: List[dict] = []
    queued = cached = no_acc = dl_err = 0

    items = list(iter_fasta(args.fasta_file))
    if args.limit > 0:
        items = items[: args.limit]

    for label, seq in items:
        seq_hash = hash_aa_string(seq)
        out_path = output_dir / f"{seq_hash}.pt"
        if out_path.exists() and out_path.stat().st_size > 0:
            cached += 1
            continue

        acc_chain = mapping.get(f"seq_hash::{seq_hash}") or mapping.get(f"label::{label}")
        if acc_chain is None:
            accession, chain = parse_accession_from_label(label)
        else:
            accession, chain = acc_chain

        if not accession:
            no_acc += 1
            dl_rows.append({
                "label": label, "seq_hash": seq_hash, "accession": "", "chain": chain,
                "status": "no_accession", "detail": "Could not parse accession from FASTA header or mapping CSV"
            })
            continue

        try:
            pdb_path = ensure_afdb_pdb(accession, args.afdb_dir, force_redownload=args.force_redownload)
            dl_rows.append({
                "label": label, "seq_hash": seq_hash, "accession": accession, "chain": chain,
                "status": "ok", "detail": str(pdb_path)
            })
        except Exception as exc:  # noqa: BLE001
            dl_err += 1
            dl_rows.append({
                "label": label, "seq_hash": seq_hash, "accession": accession, "chain": chain,
                "status": "download_error", "detail": repr(exc)
            })
            continue

        manifest_rows.append({
            "label": label,
            "sequence": seq,
            "seq_hash": seq_hash,
            "accession": accession,
            "chain": chain,
            "pdb_path": str(pdb_path.resolve()),
            "out_path": str(out_path.resolve()),
        })
        queued += 1

    manifest_path = output_dir / "aft_manifest.csv"
    dl_status_path = output_dir / "aft_download_status.csv"
    write_csv(manifest_path, manifest_rows, ["label", "sequence", "seq_hash", "accession", "chain", "pdb_path", "out_path"])
    write_csv(dl_status_path, dl_rows, ["label", "seq_hash", "accession", "chain", "status", "detail"])

    print(f"Prepared manifest: queued={queued}, cached={cached}, no_accession={no_acc}, download_error={dl_err}")
    print(f"Manifest: {manifest_path}")
    print(f"Download status: {dl_status_path}")

    if queued == 0:
        print("Nothing to run in AFToolkit worker.")
        return

    cmd = [
        args.aft_python,
        str(args.worker_script),
        str(manifest_path),
        "--device", args.device,
        "--recycles", str(args.recycles),
        "--features", *args.features,
    ]
    print("Running worker:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
