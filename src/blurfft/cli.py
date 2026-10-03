"""Command line: blurfft analyse | map | compare | formats | evaluate | benchmark | report | gui."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .precision import STANDARD_FORMATS, SWEEP_FORMATS, parse_precision

RESULTS_DIR = Path("results")


def _detector(args):
    from .detector import BlurDetector

    return BlurDetector(precision=args.precision, threads=args.threads)


def cmd_analyse(args) -> int:
    from .imageio import find_images

    detector = _detector(args)
    reports = []
    for path in find_images(*args.paths):
        report = detector.analyse(path)
        reports.append({"file": str(path), **report.to_dict()})
        if not args.json:
            print(f"{path}: {report.summary()}")
    if args.json:
        for r in reports:
            r["metrics"].pop("radial_frequency", None)
            r["metrics"].pop("radial_power", None)
        print(json.dumps(reports, indent=2))
    return 0


def cmd_map(args) -> int:
    from .render import save_map_overlay

    detector = _detector(args)
    result = detector.map(args.path, tile=args.tile, stride=args.stride)
    out = Path(args.out or Path(args.path).with_suffix(".blurmap.png"))
    save_map_overlay(args.path, result, out)
    finite = result.probability[np.isfinite(result.probability)]
    share = float(np.mean(finite >= 0.5)) if finite.size else 0.0
    print(f"{args.path}: {share:.0%} of judged tiles look blurred ({result.probability.shape[1]} x {result.probability.shape[0]} "
          f"tiles of {result.tile} px). Map written to {out}")
    return 0


def cmd_compare(args) -> int:
    from .detector import BlurDetector
    from .imageio import load_gray

    gray = load_gray(args.path)
    names = [p.strip() for p in args.precisions.split(",") if p.strip()]
    print(f"{args.path} ({gray.shape[1]} x {gray.shape[0]})")
    print(f"{'precision':<10} {'bits':>4} {'verdict':<8} {'p(blur)':>8} {'radius':>7} {'FFT ms':>8}")
    for name in names:
        p = parse_precision(name)
        report = BlurDetector(p, threads=args.threads).analyse(gray)
        verdict = "blurred" if report.blurry else "sharp"
        radius = f"{report.sigma:.2f}" if report.blurry else "-"
        mode = "" if p.native else " (emulated)"
        print(f"{p.name:<10} {p.bits:>4} {verdict:<8} {report.probability:>8.3f} {radius:>7} {report.fft_ms:>8.2f}{mode}")
    return 0


def cmd_formats(args) -> int:
    print(f"{'name':<10} {'kind':<9} {'bits':>4} {'exp':>4} {'mant':>4} {'unit roundoff':>14}")
    for name in dict.fromkeys(STANDARD_FORMATS + SWEEP_FORMATS):
        p = parse_precision(name)
        print(f"{p.name:<10} {p.kind:<9} {p.bits:>4} {p.exponent_bits:>4} {p.mantissa_bits:>4} {p.unit_roundoff:>14.3e}")
    print("\nAny eXmY format works, for example e5m10 or e8m4; prefix a hardware name with 'emulated-' to emulate it.")
    return 0


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def cmd_evaluate(args) -> int:
    from .evaluate import fit_shipped_model, run_evaluation

    precisions = [p.strip() for p in args.precisions.split(",")] if args.precisions else list(dict.fromkeys(STANDARD_FORMATS + SWEEP_FORMATS))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition, vary in (("standard", False), ("varied_contrast", True)):
        if args.condition != "both" and args.condition != condition:
            continue
        print(f"== {condition.replace('_', ' ')}")
        started = time.time()
        ev = run_evaluation(precisions, crops_per_source=args.crops, seed=args.seed, vary_contrast=vary,
                            progress=lambda m: print("  " + m, flush=True))
        summary[condition] = ev.results
        print(f"  done in {time.time() - started:.0f} s")
        for method, v in ev.results["methods"].items():
            print(f"  {method:<38} AUC {v['auc']:.4f}  balanced accuracy {v['balanced_accuracy']:.4f}")
        if args.fit and vary:
            model = fit_shipped_model(ev, args.seed)
            model.provenance["condition"] = condition
            target = Path(args.model_out) if args.model_out else Path(__file__).with_name("model.json")
            model.save(target)
            print(f"  model saved to {target}")
        # Per-sample data for the report's figures.
        np.savez_compressed(
            out / f"evaluation_{condition}.npz",
            labels=ev.labels, sigmas=ev.sigmas, folds=ev.folds,
            kinds=np.array([s.blur.kind for s in ev.samples]),
            **{f"features_{k}": v for k, v in ev.features.items()},
            **{f"baseline_{k}": v for k, v in ev.baselines.items()},
        )
    (out / "evaluation.json").write_text(json.dumps(_to_jsonable(summary), indent=2) + "\n")
    print(f"results written to {out}")
    return 0


def cmd_benchmark(args) -> int:
    from .dataset import make_benchmark
    from .perf import run_performance

    images = [s.image for i, s in enumerate(make_benchmark(crops_per_source=1, seed=args.seed)) if i % 13 == 0]
    formats = list(dict.fromkeys(parse_precision(p).name for p in STANDARD_FORMATS + SWEEP_FORMATS))
    result = run_performance(images, ("float64", "float32", "float16"), formats,
                             repeats=args.repeats, progress=lambda m: print("  " + m, flush=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance.json").write_text(json.dumps(_to_jsonable(result), indent=2) + "\n")
    print(f"{'size':<22} {'precision':<26} {'threads':>7} {'FFT ms':>8} {'analysis ms':>12} {'fps':>7}")
    for row in result["speed"]:
        analysis = f"{row['analysis_ms']:.2f}" if row["analysis_ms"] is not None else "-"
        print(f"{row['size']:<22} {row['precision']:<26} {row['threads']:>7} {row['fft_ms']:>8.2f} {analysis:>12} {row['fps']:>7.1f}")
    print(f"results written to {out / 'performance.json'}")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    paths = build_report(Path(args.results), Path(args.out))
    for p in paths:
        print(f"wrote {p}")
    return 0


def cmd_gui(args) -> int:
    from .webapp import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blurfft", description="Image blur detection with approximate FFTs.")
    sub = parser.add_subparsers(dest="command", required=True)

    def precision_arg(p):
        p.add_argument("--precision", "-p", default="float64", help="float64, float32, float16, bfloat16, floatx or eXmY")
        p.add_argument("--threads", type=int, default=0, help="threads for the FFT (0: one per core)")

    a = sub.add_parser("analyse", aliases=["analyze"], help="detect blur in images, files, folders or globs")
    a.add_argument("paths", nargs="+")
    a.add_argument("--json", action="store_true", help="print the full reports as JSON")
    precision_arg(a)
    a.set_defaults(func=cmd_analyse)

    m = sub.add_parser("map", help="write a blur map for a partly blurred image")
    m.add_argument("path")
    m.add_argument("--out", "-o")
    m.add_argument("--tile", type=int)
    m.add_argument("--stride", type=int)
    precision_arg(m)
    m.set_defaults(func=cmd_map)

    c = sub.add_parser("compare", help="analyse one image at several precisions")
    c.add_argument("path")
    c.add_argument("--precisions", default="float64,float32,float16,bfloat16,floatx,e8m6,e8m4,e8m2")
    c.add_argument("--threads", type=int, default=0)
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("formats", help="list the precision formats")
    f.set_defaults(func=cmd_formats)

    e = sub.add_parser("evaluate", help="run the ground-truth benchmark (needs scikit-image)")
    e.add_argument("--crops", type=int, default=4, help="crops per source photograph")
    e.add_argument("--seed", type=int, default=2024)
    e.add_argument("--precisions", help="comma-separated; default: the standard formats and the e8mY sweep")
    e.add_argument("--condition", choices=["both", "standard", "varied_contrast"], default="both")
    e.add_argument("--fit", action="store_true", help="refit the model on the varied-contrast benchmark")
    e.add_argument("--model-out", help="where --fit writes the model (default: the installed package's model.json)")
    e.add_argument("--out", default=str(RESULTS_DIR))
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="measure speed, memory and accuracy")
    b.add_argument("--repeats", type=int, default=7)
    b.add_argument("--seed", type=int, default=2024)
    b.add_argument("--out", default=str(RESULTS_DIR))
    b.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("report", help="draw the figures and tables from saved results (needs matplotlib)")
    r.add_argument("--results", default=str(RESULTS_DIR))
    r.add_argument("--out", default="docs")
    r.set_defaults(func=cmd_report)

    g = sub.add_parser("gui", help="open the drag-and-drop interface in your browser")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(func=cmd_gui)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"blurfft: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
