"""Figures and the results page, from the JSON written by `blurfft evaluate` and `blurfft benchmark`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["build_report"]

INK, INK_2, PAPER, SIGNAL, RULE = "#121314", "#555855", "#f4f4f1", "#e8401f", "#d4d4cf"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": RULE,
        "grid.linewidth": 0.6, "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    })


def _precision_figures(plt, evaluation: dict, performance: dict, out: Path) -> list[Path]:
    written = []
    results = evaluation.get("varied_contrast") or evaluation.get("standard")
    sweep = sorted(((v["mantissa_bits"], name, v) for name, v in results["precisions"].items()
                    if v["exponent_bits"] == 8 and not v["native"]), key=lambda t: -t[0])
    bits = [b for b, _, _ in sweep]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(bits, [v["agreement"] * 100 for *_, v in sweep], color=SIGNAL, marker="o", lw=2, label="same decision as float64")
    ax.plot(bits, [v["balanced_accuracy"] * 100 for *_, v in sweep], color=INK, marker="s", lw=1.5, label="balanced accuracy")
    ax.set_xlabel("mantissa bits in the FFT (8-bit exponent)")
    ax.set_ylabel("%")
    ax.invert_xaxis()
    ax.set_ylim(40, 101)
    for marker, label in ((10, "float16"), (7, "bfloat16"), (12, "FloatX\n(original)")):
        ax.axvline(marker, color=INK_2, lw=0.8, ls=":")
        ax.text(marker, 43, label, ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_title("Blur verdicts as the FFT's precision falls")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path = out / "precision_detection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    rows = [r for r in performance.get("accuracy", []) if r["exponent_bits"] in (8, 11) or r["precision"] in ("float16",)]
    if rows:
        rows = sorted(rows, key=lambda r: -r["mantissa_bits"])
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        e8 = [r for r in rows if r["exponent_bits"] == 8 or r["precision"] == "float32"]
        ax.semilogy([r["mantissa_bits"] for r in e8], [r["error"] for r in e8], color=SIGNAL, marker="o", lw=2,
                    label="this project (Bluestein, exact twiddles)")
        orig = [r for r in e8 if r.get("original_error")]
        ax.semilogy([r["mantissa_bits"] for r in orig], [r["original_error"] for r in orig], color=INK, marker="s", lw=1.5,
                    label="original (padded radix-2, twiddle recurrence)")
        ax.semilogy([r["mantissa_bits"] for r in e8], [r["unit_roundoff"] for r in e8], color=INK_2, ls="--", lw=1,
                    label="unit roundoff of the format")
        ax.set_xlabel("mantissa bits")
        ax.set_ylabel("relative error of the spectrum")
        ax.invert_xaxis()
        ax.set_title("Spectrum error tracks the format's precision")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        fig.tight_layout()
        path = out / "precision_fft_error.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def _methods_figure(plt, evaluation: dict, out: Path) -> Path:
    conditions = [c for c in ("standard", "varied_contrast") if c in evaluation]
    methods = list(evaluation[conditions[0]]["methods"])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(methods) + 1.2))
    y = np.arange(len(methods))
    height = 0.38
    colours = {"standard": INK, "varied_contrast": SIGNAL}
    labels = {"standard": "fixed contrast", "varied_contrast": "varied contrast"}
    for i, condition in enumerate(conditions):
        values = [evaluation[condition]["methods"][m]["auc"] for m in methods]
        ax.barh(y + (i - 0.5) * height, values, height=height, color=colours[condition], label=labels[condition])
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("ROC AUC on held-out photographs (1 is perfect)")
    ax.set_title("Blur detection, compared")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = out / "methods.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _speed_figure(plt, performance: dict, out: Path) -> Path:
    rows = [r for r in performance["speed"] if r["size"].startswith("720p")]
    labels = [f"{r['precision'].replace('numpy pocketfft (float64)', 'numpy (pocketfft)')}\n{r['threads']} thread{'s' if r['threads'] > 1 else ''}" for r in rows]
    values = [r["fft_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colours = [INK_2 if "numpy" in r["precision"] else (SIGNAL if r["threads"] > 1 else INK) for r in rows]
    ax.bar(np.arange(len(rows)), values, color=colours)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ms per 2-D FFT")
    ax.set_title("A 720p frame's spectrum")
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    path = out / "speed_720p.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _markdown(evaluation: dict, performance: dict, figures: list[Path], out: Path) -> Path:
    parts = ["# Results", "",
             "Generated by `blurfft evaluate` and `blurfft benchmark`, then `blurfft report`. "
             "Every number below comes from those runs; rerun them to reproduce or update this page.", ""]
    machine = performance.get("machine", {})
    if machine:
        parts += [f"Measured on {machine.get('cpu', 'unknown CPU')} ({machine.get('cores')} cores), "
                  f"{machine.get('system')} {machine.get('machine')}, Python {machine.get('python')}; "
                  f"hardware float16: {'yes' if machine.get('native_float16') else 'no'}.", ""]
    counts = (evaluation.get("varied_contrast") or evaluation["standard"])["counts"]
    parts += ["## Detection accuracy", "",
              f"{counts['samples']} images from {counts['sources']} public-domain photographs: {counts['sharp']} sharp, "
              f"{counts['blurred']} blurred, {counts['unlabelled']} in the borderline band (equivalent blur radius between "
              "0.5 and 1.5 px) that are left out of the sharp/blurred scores. Two-fold cross-validation by photograph: "
              "no test image shares a photograph with a training image.", ""]
    for condition, title in (("standard", "Fixed contrast"), ("varied_contrast", "Varied contrast (0.35 to 1)")):
        if condition not in evaluation:
            continue
        rows = [[m, f"{v['auc']:.3f}", f"{v['balanced_accuracy']:.3f}", f"{v['spearman_sigma']:.2f}"]
                for m, v in evaluation[condition]["methods"].items()]
        parts += [f"### {title}", "", _table(rows, ["Method", "ROC AUC", "Balanced accuracy", "Rank correlation with blur radius"]), ""]
    res = evaluation.get("varied_contrast") or evaluation["standard"]
    parts += ["## How much precision does blur detection need?", "",
              "The model is fitted once at float64; the FFT then runs at each precision. "
              "\"Same decision\" is the share of images whose sharp/blurred verdict matches float64.", ""]
    rows = []
    for name, v in sorted(res["precisions"].items(), key=lambda kv: (-kv[1]["mantissa_bits"], kv[0])):
        rows.append([name, str(v["mantissa_bits"]), "hardware" if v["native"] else "emulated", f"{v['auc']:.3f}",
                     f"{v['balanced_accuracy']:.3f}", f"{v['agreement'] * 100:.1f}%", f"{v['mean_abs_dp']:.4f}"])
    parts += [_table(rows, ["Format", "Mantissa bits", "Arithmetic", "ROC AUC", "Balanced accuracy", "Same decision", "Mean change in probability"]), ""]
    sev, mot = res["severity"], res["motion"]
    parts += ["## Severity and blur type", "",
              f"* Blur radius estimate: median relative error {sev['median_relative_sigma_error'] * 100:.0f}%, "
              f"rank correlation {sev['spearman_sigma_estimate']:.2f} with the true radius (blurred images).",
              f"* Motion blur against defocus: balanced accuracy {mot['balanced_accuracy']:.2f}; "
              f"median error in the motion direction {mot['median_angle_error_deg']:.1f} degrees.", ""]
    if performance:
        parts += ["## Speed", "", "Median of repeated runs. Analysis is the FFT plus the measures; fps is analyses per second.", ""]
        rows = [[r["size"], r["precision"], str(r["threads"]), f"{r['fft_ms']:.2f}",
                 f"{r['analysis_ms']:.2f}" if r["analysis_ms"] is not None else "-", f"{r['fps']:.0f}"] for r in performance["speed"]]
        parts += [_table(rows, ["Frame", "Precision", "Threads", "FFT (ms)", "Analysis (ms)", "fps"]), ""]
        parts += ["## The cost of an exact transform", "",
                  "Padding to powers of two (the original project's approach) is cheaper: one power-of-two transform "
                  "per row, where Bluestein runs two of at least twice the length. But it transforms a different, "
                  "larger image, so the spectrum changes with the padding. Bluestein buys the exact DFT of the true "
                  "image for about twice the time.", ""]
        rows = [[r["size"], f"{r['exact_ms']:.2f}", r["padded_shape"], f"{r['padded_ms']:.2f}", f"{r['extra_pixels'] * 100:.0f}%"]
                for r in performance["padding"]]
        parts += [_table(rows, ["Frame", "Exact, Bluestein (ms)", "Padded to", "Padded (ms)", "Extra pixels"]), ""]
        parts += ["## Memory", "", "The half-plane spectrum of a 1080p frame in each format, packed at the format's width.", ""]
        rows = [[r["precision"], f"{r['spectrum_mb']:.1f} MB"] for r in performance["speed"]
                if r["size"].startswith("1080p") and r.get("spectrum_mb") and r["threads"] == 1]
        parts += [_table(rows, ["Format", "Spectrum"]), ""]
        energy = performance.get("energy", {})
        if any(v is not None for v in energy.values()):
            rows = [[k, f"{v * 1000:.1f} mJ" if v is not None else "n/a"] for k, v in energy.items()]
            parts += ["## Energy", "", _table(rows, ["Precision", "Energy per 720p analysis"]), ""]
        else:
            parts += ["## Energy", "", "Not measured on this machine: macOS exposes power counters only to root. "
                      "On Linux, `blurfft benchmark` reads the RAPL counters when they are readable.", ""]
    if figures:
        parts += ["## Figures", ""] + [f"![{p.stem.replace('_', ' ')}](figures/{p.name})" for p in figures] + [""]
    path = out / "results.md"
    path.write_text("\n".join(parts))
    return path


def build_report(results: Path, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    evaluation = json.loads((results / "evaluation.json").read_text())
    performance_path = results / "performance.json"
    performance = json.loads(performance_path.read_text()) if performance_path.exists() else {}
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = _precision_figures(plt, evaluation, performance, figures_dir)
    figures.append(_methods_figure(plt, evaluation, figures_dir))
    if performance:
        figures.append(_speed_figure(plt, performance, figures_dir))
    return figures + [_markdown(evaluation, performance, figures, out)]
