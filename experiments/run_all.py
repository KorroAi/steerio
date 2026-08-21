"""Orchestrateur : lance les 3 expériences et écrit un résumé markdown.

Usage :  python experiments/run_all.py [--exp1] [--exp2] [--exp3] [--models gpt2 qwen05 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (_HERE, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C

from exp1_steering_vs_prompting import main as exp1_main
from exp2_cross_model import main as exp2_main
from exp3_transfer import main as exp3_main


def latest(path_glob: str) -> str:
    import glob
    files = sorted(glob.glob(os.path.join(C.REPORT_DIR, path_glob)), key=os.path.getmtime)
    return files[-1] if files else ""


def summarize() -> str:
    """Construit un résumé markdown depuis les derniers rapports JSON."""
    exp1 = latest("exp1_steering_vs_prompting_*.json")
    exp2 = latest("exp2_cross_model_summary_*.json")
    exp3 = latest("exp3_transfer_*.json")
    lines = ["# steerio — Résultats d'expériences (chiffres réels)", ""]

    # ---- Exp 1
    if exp1:
        data = json.load(open(exp1, encoding="utf-8"))
        lines.append("## Exp 1 — Steering vs prompting")
        lines.append("")
        for mid, r in data.items():
            lines.append(f"### {mid} ({r['family']})")
            lines.append("")
            lines.append("| dimension | contrôle | mean | écart vs baseline | test signe (p_pos) |")
            lines.append("|---|---|---:|---:|---:|")
            for dim, dd in r["dimensions"].items():
                for cname, agg in dd["conditions"].items():
                    st = agg["sign_test_vs_baseline"]
                    lines.append(f"| {dim} | {cname} | {agg['mean']:+.3f} | {agg['effect_vs_baseline']:+.3f} | {st.get('p_pos', float('nan')):.2f} |")
                lines.append(f"| {dim} | **control power steer** | | {dd['control_power_steer']:+.3f} | |")
                lines.append(f"| {dim} | **control power prompt** | | {dd['control_power_prompt']:+.3f} | |")
            lines.append("")

    # ---- Exp 2
    if exp2:
        data = json.load(open(exp2, encoding="utf-8"))
        lines.append("## Exp 2 — Cross-modèles (corrélation échelle→score)")
        lines.append("")
        lines.append("| modèle | famille | sentiment corr | sentiment pente | tone corr | tone pente | refusal corr |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in data["models"]:
            def g(k, field):
                return (f"{row[k][field]:+.3f}" if row.get(k) else "—") if k in row else "—"
            lines.append(f"| {row['model'].split('/')[-1]} | {row['family']} | {g('sentiment','correlation')} | {g('sentiment','slope')} | {g('tone','correlation')} | {g('tone','slope')} | {g('refusal','correlation')} |")
        lines.append("")

    # ---- Exp 3
    if exp3:
        data = json.load(open(exp3, encoding="utf-8"))
        lines.append("## Exp 3 — Transfert de directions")
        lines.append("")
        lines.append("| design | source → cible | hidden | delta natif | delta transféré | coef transfert |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for d in data:
            nd, td = d.get("native", {}).get("delta"), d.get("transferred", {}).get("delta")
            coef = d.get("transfer_coefficient")
            lines.append(f"| {d['name']} | {d['source'].split('/')[-1]} → {d['target'].split('/')[-1]} | {d.get('source_hidden')}→{d.get('target_hidden')} | {nd:+.3f} | {td:+.3f} | {coef:+.3f} |")
        lines.append("")

    md_path = os.path.join(C.REPORT_DIR, "SUMMARY.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1", action="store_true")
    ap.add_argument("--exp2", action="store_true")
    ap.add_argument("--exp3", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="modèles pour exp1/exp2")
    args = ap.parse_args()
    t_start = time.time()
    if args.exp1:
        print("=== EXP 1 ==="); exp1_main(args.models)
    if args.exp2:
        print("=== EXP 2 ==="); exp2_main(args.models)
    if args.exp3:
        print("=== EXP 3 ==="); exp3_main()
    md = summarize()
    print(f"\nALL EXPERIMENTS DONE in {round(time.time()-t_start,1)}s -> {md}")


if __name__ == "__main__":
    main()
