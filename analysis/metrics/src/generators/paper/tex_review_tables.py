#!/usr/bin/env python3
"""LaTeX fragments for the appendices on the uncertainty and the breakdowns of the paired effects.

Reads analysis/metrics/review_response_cv.json (written by
analysis/metrics/src/review_response_cv.py) and writes, next to the paper,

  figures/rr_paired_table.tex     paired effects, all proteins and outside the screening folds
  figures/rr_matcher_table.tex    the same effects under the original and the corrected matcher
  figures/rr_bytype_table.tex     effects by segment type
  figures/rr_evidence_table.tex   recall by annotation evidence code
  figures/rr_bylength_table.tex   effects by true-segment length
  figures/rr_numbers.tex          \\newcommand macros for every number quoted in prose

so no number in those appendices is typed by hand.

Usage: python analysis/metrics/src/generators/paper/tex_review_tables.py [--paper-dir DIR] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "run.py").exists())
EMBS = ("ESM-2", "ESM-C 6B")
ADDS = (("head", "boundary head"), ("adapter", "adapter"), ("both", "both"))


def sg(x, nd=3):
    """Signed number for math mode."""
    return f"{x:+.{nd}f}"


def spread(sd):
    return f"{sd:.3f}"[1:]  # drop the leading zero, as in Table 1


def cell_ms(mean, sd):
    return rf"\ms{{{sg(mean)}}}{{{spread(sd)}}}"


def interval(lo, hi):
    return rf"$[{sg(lo)},\,{sg(hi)}]$"


def folds(k):
    return f"{k}/5"


def paired_rows(block, label):
    rows = [rf"\multicolumn{{8}}{{l}}{{\emph{{{label}}}}}\\"]
    for emb in EMBS:
        for key, name in ADDS:
            r = block[emb][key]
            t3, t0 = r["tau3_all"], r["tau0_all"]
            rows.append(
                f"{emb if key == 'head' else ''} & {name} & "
                f"{cell_ms(t3['mean'], t3['sd'])} & {interval(*t3['ci95_genus'])} & {folds(t3['pos_folds'])} & "
                f"{cell_ms(t0['mean'], t0['sd'])} & {interval(*t0['ci95_genus'])} & {folds(t0['pos_folds'])} \\\\")
        if emb == EMBS[0]:
            rows.append(r"\addlinespace")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(REPO / "analysis/metrics/latex_tables"), help="fragments go to <paper-dir>/figures")
    ap.add_argument("--json", default=str(REPO / "analysis/metrics/review_response_cv.json"), help="output of review_response_cv.py")
    ap.add_argument("--data-2026", default=str(REPO / "data/uniprot_2026"))
    ap.add_argument("--data-2022", default=str(REPO / "data/uniprot_2022"))
    args = ap.parse_args()
    out = Path(args.paper_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads(Path(args.json).read_text())

    # ---- paired effects: all proteins and outside the screening comparison ------------------
    head = [r"\begin{tabular}{llcccccc}", r"\toprule",
            r" & & \multicolumn{3}{c}{\textbf{at $\pm3$}} & \multicolumn{3}{c}{\textbf{exact match}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r" & \textbf{Additions} & \textbf{$\Delta$F1} & \textbf{95\% interval} & \textbf{folds} & "
            r"\textbf{$\Delta$F1} & \textbf{95\% interval} & \textbf{folds} \\", r"\midrule"]
    allb = {e: {k: r["paired"][e][k] for k, _ in ADDS} for e in EMBS}
    outb = {e: {k: r["untouched"][e][k] for k, _ in ADDS} for e in EMBS}
    body = paired_rows(allb, "All test proteins")
    body += [r"\midrule"]
    body += paired_rows(outb, "Test proteins outside the screening comparison")
    (out / "rr_paired_table.tex").write_text("\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- matcher ------------------------------------------------------------------------
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r" & \textbf{Additions} & \textbf{original matcher} & \textbf{corrected matcher} \\", r"\midrule"]
    for emb in EMBS:
        for key, name in ADDS:
            m = r["matcher"][emb][key]
            lines.append(f"{emb if key == 'head' else ''} & {name} & "
                         f"{cell_ms(m['original_mean'], m['original_sd'])} ({folds(m['original_pos'])}) & "
                         f"{cell_ms(m['corrected_mean'], m['corrected_sd'])} ({folds(m['corrected_pos'])}) \\\\")
        if emb == EMBS[0]:
            lines.append(r"\addlinespace")
    (out / "rr_matcher_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- by type ---------------------------------------------------------------------------
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{\textbf{at $\pm3$}} & \multicolumn{2}{c}{\textbf{exact match}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & \textbf{Additions} & \textbf{peptides} & \textbf{propeptides} & \textbf{peptides} & \textbf{propeptides} \\",
             r"\midrule"]
    for emb in EMBS:
        for key, name in ADDS:
            p = r["paired"][emb][key]
            cells = [p[f"tau{t}_{s}"] for t in (3, 0) for s in ("peptides", "propeptides")]
            lines.append(f"{emb if key == 'head' else ''} & {name} & " +
                         " & ".join(f"${sg(c['mean'])}$ ({c['pos_folds']})" for c in cells) + r" \\")
        if emb == EMBS[0]:
            lines.append(r"\addlinespace")
    (out / "rr_bytype_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- evidence ---------------------------------------------------------------------------
    classes = list(r["evidence"][EMBS[0]].keys())
    lines = [r"\begin{tabular}{llrcccc}", r"\toprule",
             r" & \textbf{Evidence code} & \textbf{segments per fold} & \textbf{base recall} & "
             r"\textbf{boundary head} & \textbf{adapter} & \textbf{both} \\", r"\midrule"]
    for emb in EMBS:
        for i, c in enumerate(classes):
            row = r["evidence"][emb][c]
            t0 = row["tau0"]
            lines.append(f"{emb if i == 0 else ''} & {c} & {row['n_groups_per_fold']:.0f} & ${t0['recall_base']:.3f}$ & " +
                         " & ".join(f"${sg(t0[k]['delta'])}$ ({t0[k]['pos_folds']})" for k, _ in ADDS) + r" \\")
        if emb == EMBS[0]:
            lines.append(r"\addlinespace")
    (out / "rr_evidence_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- by length --------------------------------------------------------------------------
    lines = [r"\begin{tabular}{lrcccc}", r"\toprule",
             r"\textbf{Length} & \textbf{segments per fold} & \textbf{base F1} & "
             r"\textbf{boundary head} & \textbf{adapter} & \textbf{both} \\", r"\midrule"]
    for emb in EMBS:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{emb}}}}}\\")
        for b, row in r["bylength"][emb].items():
            lines.append(f"{b} & {row['n_true_per_fold']:.0f} & ${row['base']:.3f}$ & " +
                         " & ".join(f"${sg(row[k]['delta'])}$ ({row[k]['pos_folds']})" for k, _ in ADDS) + r" \\")
        if emb == EMBS[0]:
            lines.append(r"\midrule")
    (out / "rr_bylength_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- prose numbers --------------------------------------------------------------------
    def mac(name, value):
        return rf"\newcommand{{\{name}}}{{{value}}}"

    def grp(n):
        return f"{n:,}".replace(",", "{,}")

    u2, uc = r["untouched"]["ESM-2"], r["untouched"]["ESM-C 6B"]
    ratios = []
    for sec in ("paired", "untouched"):
        for e in EMBS:
            for k, _ in ADDS:
                for t in ("tau3_all", "tau0_all"):
                    v = r[sec][e][k][t]
                    ratios.append((v["ci95_protein"][1] - v["ci95_protein"][0]) /
                                  (v["ci95_genus"][1] - v["ci95_genus"][0]))
    import math
    import pandas as pd
    d26 = Path(args.data_2026)
    cv_ids = set(pd.read_csv(d26 / "graphpart_assignments_5motif.esm2covered.csv").AC)
    f7 = pd.read_csv(d26 / "graphpart_assignments.esm2covered.csv")
    f7["cluster"] = f7["cluster"].astype(int)
    scr = set(f7.loc[f7.cluster.isin([2, 5]), "AC"])
    n_dev_out = len((cv_ids - scr) & set(f7.loc[f7.cluster == 1, "AC"]))
    ids22 = set(pd.read_csv(Path(args.data_2022) / "labeled_sequences.csv", usecols=["protein_id"]).protein_id)
    share22 = len(cv_ids & ids22) / len(cv_ids)
    lo_all = min(r[s][e][k][t]["ci95_genus"][0] for s in ("paired", "untouched") for e in EMBS
                 for k, _ in ADDS for t in ("tau3_all", "tau0_all"))
    m2, mc = r["matcher"]["ESM-2"]["both"], r["matcher"]["ESM-C 6B"]["both"]
    macros = [
        mac("rrOutBothTwo", sg(u2["both"]["tau3_all"]["mean"])),
        mac("rrOutBothC", sg(uc["both"]["tau3_all"]["mean"])),
        mac("rrNOutTwo", grp(u2["n_unseen"])), mac("rrNAllTwo", grp(u2["n_test_proteins"])),
        mac("rrNOutC", grp(uc["n_unseen"])), mac("rrNAllC", grp(uc["n_test_proteins"])),
        mac("rrRatioLo", f"{5 * math.floor(20 * min(ratios))}"), mac("rrRatioHi", f"{5 * math.ceil(20 * max(ratios))}"),
        mac("rrNDevOut", grp(round(n_dev_out, -1))), mac("rrShareTwoTwo", f"{100 * share22:.0f}"),
        mac("rrMinLower", f"{lo_all:+.3f}"),
        mac("rrOrigBothTwo", sg(m2["original_mean"])), mac("rrCorrBothTwo", sg(m2["corrected_mean"])),
        mac("rrOrigBothC", sg(mc["original_mean"])), mac("rrCorrBothC", sg(mc["corrected_mean"])),
        mac("rrMatcherMax", f"{max(abs(r['matcher'][e][k]['original_mean'] - r['matcher'][e][k]['corrected_mean']) for e in EMBS for k, _ in ADDS):.3f}"),
        mac("rrTypeAdCProThree", sg(r["paired"]["ESM-C 6B"]["adapter"]["tau3_propeptides"]["mean"])),
        mac("rrTypeHeadTwoPepExact", sg(r["paired"]["ESM-2"]["head"]["tau0_peptides"]["mean"])),
        mac("rrEvExpTwo", sg(r["evidence"]["ESM-2"]["experimental"]["tau0"]["both"]["delta"])),
        mac("rrEvExpC", sg(r["evidence"]["ESM-C 6B"]["experimental"]["tau0"]["both"]["delta"])),
        mac("rrEvOtherLo", f"{min(r['evidence'][e][c]['tau0']['both']['delta'] for e in EMBS for c in r['evidence'][e] if c != 'experimental'):+.3f}"),
        mac("rrEvOtherHi", f"{max(r['evidence'][e][c]['tau0']['both']['delta'] for e in EMBS for c in r['evidence'][e] if c != 'experimental'):+.3f}"),
        mac("rrBLPairMinTwo", sg(r["bylength"]["ESM-2"]["25-29"]["both"]["delta"])),
        mac("rrBLPairMaxTwo", sg(r["bylength"]["ESM-2"]["35-39"]["both"]["delta"])),
        mac("rrBLAdCa", sg(r["bylength"]["ESM-C 6B"]["25-29"]["adapter"]["delta"])),
        mac("rrBLAdCb", sg(r["bylength"]["ESM-C 6B"]["30-34"]["adapter"]["delta"])),
        mac("rrBLHeadTwo", sg(r["bylength"]["ESM-2"]["25-29"]["head"]["delta"])),
        mac("rrWilcoxTwo", f"{r['min_p_wilcoxon_n5']:.4f}"),
        mac("rrWilcoxOne", f"{r['min_p_wilcoxon_n5'] / 2:.3f}"),
    ]
    (out / "rr_numbers.tex").write_text("\n".join(macros) + "\n")
    print("wrote tables and numbers to", out)


if __name__ == "__main__":
    main()
