#!/usr/bin/env python3
"""
Visualisiert Knöchel-Zeitreihen mit Schrittmarken und schätzt BPM aus Schrittintervallen.

Erwartete Inputs (Standardpfade, anpassbar per CLI):
- out/step_report.json   -> {"step_timestamps":[...]}  (oder .csv mit Spalte step_time_s)
- out/left_ankle.csv     -> t,x,y,z
- out/right_ankle.csv    -> t,x,y,z   (optional)

Outputs (Standard: out_viz/):
- out_viz/plot_left_ankle.png
- out_viz/plot_right_ankle.png      (falls rechte CSV vorhanden)
- out_viz/bpm_estimate.csv          (Intervalle & BPM pro Intervall)
- out_viz/summary.txt               (Kurzinfo: Schrittzahl, BPM-Median)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_step_timestamps(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for k in ("step_timestamps", "steps", "times"):
                if k in data:
                    return np.asarray(data[k], dtype=float)
        if isinstance(data, list):
            return np.asarray(data, dtype=float)
        raise ValueError(f"Unerwartetes JSON-Format in {path}")
    else:
        df = pd.read_csv(path)
        col = "step_time_s" if "step_time_s" in df.columns else df.columns[0]
        return df[col].to_numpy(dtype=float)

def load_joint_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    t = cols.get("t") or cols.get("time") or cols.get("timestamp")
    y = cols.get("y")
    if not t or not y:
        raise ValueError(f"{path} benötigt Spalten t/time/timestamp und y")
    x = cols.get("x"); z = cols.get("z")
    out = pd.DataFrame({
        "t": df[t].astype(float),
        "x": df[x].astype(float) if x else np.nan,
        "y": df[y].astype(float),
        "z": df[z].astype(float) if z else np.nan,
    }).sort_values("t").reset_index(drop=True)
    return out

def robust_bpm(step_ts: np.ndarray):
    if step_ts.size < 2:
        return None, pd.DataFrame(columns=["interval_s","bpm_from_interval"])
    intervals = np.diff(step_ts)
    med = np.median(intervals)
    keep = (intervals > 0) & (intervals > 0.25*med) & (intervals < 4*med)
    clean = intervals[keep] if keep.any() else intervals
    bpms = 60.0 / clean
    bpm_point = float(np.median(bpms)) if bpms.size else None
    return bpm_point, pd.DataFrame({"interval_s": clean, "bpm_from_interval": bpms})

def plot_ankle(df: pd.DataFrame, step_ts: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["y"])
    for st in step_ts:
        plt.axvline(st, linestyle="--", alpha=0.6)
    plt.title(title)
    plt.xlabel("Zeit (s)"); plt.ylabel("Vertikale Position (y)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="out/step_report.json",
                    help="Pfad zu step_report.json oder .csv")
    ap.add_argument("--left", default="out/left_ankle.csv",
                    help="CSV für linken Knöchel (t,x,y,z)")
    ap.add_argument("--right", default="out/right_ankle.csv",
                    help="CSV für rechten Knöchel (optional)")
    ap.add_argument("--outdir", default="out_viz", help="Ausgabeordner")
    args = ap.parse_args()

    steps_path = Path(args.steps)
    left_path  = Path(args.left)
    right_path = Path(args.right)
    outdir     = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Schrittzeiten laden
    if not steps_path.exists():
        # Fallback: CSV mit step times
        csv_fallback = steps_path.with_suffix(".csv")
        if csv_fallback.exists():
            steps_path = csv_fallback
        else:
            raise FileNotFoundError(f"Schrittdatei fehlt: {args.steps}")
    step_ts = load_step_timestamps(steps_path)

    # Left ankle laden & plotten
    if not left_path.exists():
        raise FileNotFoundError(f"Linke Knöchel-CSV fehlt: {args.left}")
    la = load_joint_csv(left_path)
    plot_ankle(la, step_ts, "Linker Knöchel – vertikale Position (y) mit Schrittmarken",
               outdir / "plot_left_ankle.png")

    # Right ankle (optional)
    if right_path.exists():
        ra = load_joint_csv(right_path)
        plot_ankle(ra, step_ts, "Rechter Knöchel – vertikale Position (y) mit Schrittmarken",
                   outdir / "plot_right_ankle.png")

    # BPM
    bpm_point, bpm_df = robust_bpm(step_ts)
    bpm_df.to_csv(outdir / "bpm_estimate.csv", index=False)

    # Summary
    summary = f"steps_count={step_ts.size}\n"
    summary += f"bpm_estimate_median={bpm_point if bpm_point is not None else 'n/a'}\n"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()