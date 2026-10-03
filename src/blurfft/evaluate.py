"""Ground-truth evaluation: accuracy, precision trade-off and comparison with other methods.

Protocol. The benchmark's photographs are split into two groups; a model is
fitted on one group and tested on the other, then the other way round, so no
test image shares a photograph with any training image. Methods with a single
score are thresholded at the value that maximises balanced accuracy on the
training group. The reduced-precision runs reuse the model fitted at float64:
that is how a deployed detector would run, trained once and executed in
cheaper arithmetic.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .baselines import BASELINES
from .dataset import SOURCES, Sample, make_benchmark
from .detector import measure
from .model import BlurModel, feature_vector
from .precision import parse_precision

__all__ = ["FOLDS", "auc", "balanced_accuracy", "best_threshold", "spearman", "Evaluation", "run_evaluation"]

#: Two groups of photographs, each mixing people, objects, textures and science images.
FOLDS = (
    ("astronaut", "coffee", "coins", "grass", "rocket", "immunohistochemistry"),
    ("camera", "chelsea", "brick", "gravel", "text", "hubble_deep_field"),
)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (higher score = positive), with ties counted half."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    combined = np.concatenate([pos, neg])[order]
    # average ranks over ties
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    ordered = np.asarray(labels, int)[order]
    tpr = np.concatenate([[0], np.cumsum(ordered) / max(1, ordered.sum())])
    fpr = np.concatenate([[0], np.cumsum(1 - ordered) / max(1, (1 - ordered).sum())])
    return fpr, tpr


def balanced_accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    predicted, labels = np.asarray(predicted, bool), np.asarray(labels, int)
    tpr = np.mean(predicted[labels == 1]) if np.any(labels == 1) else float("nan")
    tnr = np.mean(~predicted[labels == 0]) if np.any(labels == 0) else float("nan")
    return float((tpr + tnr) / 2.0)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The threshold (score >= t means positive) with the best balanced accuracy."""
    candidates = np.unique(scores)
    mids = np.concatenate([[candidates[0] - 1], (candidates[:-1] + candidates[1:]) / 2, [candidates[-1] + 1]])
    best, best_t = -1.0, float(mids[0])
    for t in mids:
        acc = balanced_accuracy(scores >= t, labels)
        if acc > best:
            best, best_t = acc, float(t)
    return best_t


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def ranks(x):
        x = np.asarray(x, float)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        # average ties
        for value in np.unique(x):
            idx = np.where(x == value)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _parallel_map(fn: Callable, items: Sequence, workers: Optional[int] = None) -> list:
    """Maps in threads: the C++ core releases the GIL, so this uses every core."""
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return list(pool.map(fn, items))


@dataclass
class Evaluation:
    samples: list[Sample]
    labels: np.ndarray  # -1 where unlabelled
    sigmas: np.ndarray
    folds: np.ndarray  # 0 or 1 per sample
    features: dict  # precision name -> n x 2 features
    metrics: dict  # precision name -> list of measure dicts
    baselines: dict  # method -> scores (higher = sharper)
    results: dict

    @property
    def labelled(self) -> np.ndarray:
        return self.labels >= 0


def _fold_of(source: str) -> int:
    return 0 if source in FOLDS[0] else 1


def run_evaluation(
    precisions: Iterable[str] = ("float64",),
    crops_per_source: int = 4,
    seed: int = 2024,
    include_baselines: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    vary_contrast: bool = False,
) -> Evaluation:
    say = progress or (lambda message: None)
    samples = list(make_benchmark(list(SOURCES), crops_per_source=crops_per_source, seed=seed, vary_contrast=vary_contrast))
    labels = np.array([-1 if s.label is None else s.label for s in samples])
    sigmas = np.array([s.sigma for s in samples])
    folds = np.array([_fold_of(s.source) for s in samples])
    say(f"{len(samples)} samples: {int((labels == 0).sum())} sharp, {int((labels == 1).sum())} blurred, "
        f"{int((labels < 0).sum())} unlabelled")

    names = list(dict.fromkeys(parse_precision(p).name for p in precisions))  # bfloat16 is e8m7, and so on
    if "float64" not in names:
        names.insert(0, "float64")
    features, all_metrics = {}, {}
    for name in names:
        started = time.perf_counter()
        metrics = _parallel_map(lambda s, name=name: measure(s.image, name, threads=1), samples)
        all_metrics[name] = metrics
        features[name] = np.array([feature_vector(m) for m in metrics])
        say(f"  measured {name} in {time.perf_counter() - started:.1f} s")

    baselines = {}
    if include_baselines:
        for method, fn in BASELINES.items():
            started = time.perf_counter()
            baselines[method] = np.array(_parallel_map(lambda s, fn=fn: fn(s.image), samples))
            say(f"  baseline {method} in {time.perf_counter() - started:.1f} s")

    evaluation = Evaluation(samples, labels, sigmas, folds, features, all_metrics, baselines, {})
    evaluation.results = summarise(evaluation)
    return evaluation


def summarise(ev: Evaluation) -> dict:
    """Cross-validated results for every method and precision."""
    lab = ev.labelled
    kinds = np.array([s.blur.kind for s in ev.samples])
    angles = np.array([s.blur.angle for s in ev.samples])
    ref = ev.features["float64"]
    results: dict = {"methods": {}, "precisions": {}, "severity": {}, "motion": {}, "folds": [list(f) for f in FOLDS]}

    # Our detector, cross-validated, at every precision (models fitted at float64).
    per_precision = {name: {"auc": [], "balanced_accuracy": [], "agreement": [], "mean_abs_dp": []} for name in ev.features}
    probabilities = {name: np.full(len(ev.samples), np.nan) for name in ev.features}
    sigma_estimates = np.full(len(ev.samples), np.nan)
    motion_threshold = []
    for test_fold in (0, 1):
        train = (ev.folds != test_fold) & lab
        test = ev.folds == test_fold
        model = BlurModel.fit(ref[train], ev.labels[train], ev.sigmas[train])
        p_ref = model.probability(ref[test])
        for name, feats in ev.features.items():
            p = model.probability(feats[test])
            probabilities[name][test] = p
            t = test & lab
            pt = model.probability(feats[t])
            per_precision[name]["auc"].append(auc(pt, ev.labels[t]))
            per_precision[name]["balanced_accuracy"].append(balanced_accuracy(pt >= model.threshold, ev.labels[t]))
            per_precision[name]["agreement"].append(float(np.mean((p >= model.threshold) == (p_ref >= model.threshold))))
            per_precision[name]["mean_abs_dp"].append(float(np.mean(np.abs(p - p_ref))))
        sigma_estimates[test] = model.sigma(ref[test])
        # Motion vs defocus: anisotropy threshold chosen on the training fold's blurred images.
        an = np.array([m["anisotropy"] for m in ev.metrics["float64"]])
        blurred_train = train & (ev.labels == 1)
        motion_threshold.append(best_threshold(an[blurred_train], (kinds[blurred_train] == "motion").astype(int)))
    for name, values in per_precision.items():
        results["precisions"][name] = {k: float(np.nanmean(v)) for k, v in values.items()}
        p = parse_precision(name)
        results["precisions"][name].update({"exponent_bits": p.exponent_bits, "mantissa_bits": p.mantissa_bits, "native": p.native})

    results["methods"]["This project (spectral model)"] = {
        "auc": results["precisions"]["float64"]["auc"],
        "balanced_accuracy": results["precisions"]["float64"]["balanced_accuracy"],
        "spearman_sigma": spearman(probabilities["float64"], ev.sigmas),
    }
    single = {
        "High-frequency share (this project)": -np.array([m["high_frequency_ratio"] for m in ev.metrics["float64"]]),
        "Spectral slope (this project)": np.array([m["slope"] for m in ev.metrics["float64"]]),
    }
    for method, scores in ev.baselines.items():
        single[method] = -np.asarray(scores)  # baselines report sharpness; flip to blurriness
    for method, blurriness in single.items():
        aucs, accs = [], []
        for test_fold in (0, 1):
            train = (ev.folds != test_fold) & lab
            test = (ev.folds == test_fold) & lab
            t = best_threshold(blurriness[train], ev.labels[train])
            aucs.append(auc(blurriness[test], ev.labels[test]))
            accs.append(balanced_accuracy(blurriness[test] >= t, ev.labels[test]))
        results["methods"][method] = {
            "auc": float(np.mean(aucs)),
            "balanced_accuracy": float(np.mean(accs)),
            "spearman_sigma": spearman(blurriness, ev.sigmas),
        }

    blurred = ev.labels == 1
    rel = np.abs(sigma_estimates[blurred] - ev.sigmas[blurred]) / ev.sigmas[blurred]
    results["severity"] = {
        "spearman_probability_vs_sigma": spearman(probabilities["float64"], ev.sigmas),
        "spearman_sigma_estimate": spearman(sigma_estimates[blurred], ev.sigmas[blurred]),
        "median_relative_sigma_error": float(np.median(rel)),
    }

    an = np.array([m["anisotropy"] for m in ev.metrics["float64"]])
    orient = np.array([m["orientation_deg"] for m in ev.metrics["float64"]])
    accs, angle_errors = [], []
    for test_fold in (0, 1):
        test = (ev.folds == test_fold) & blurred
        thr = motion_threshold[test_fold]
        accs.append(balanced_accuracy(an[test] >= thr, (kinds[test] == "motion").astype(int)))
        m = test & (kinds == "motion") & (an >= thr)
        d = np.abs(orient[m] - angles[m]) % 180
        angle_errors.extend(np.minimum(d, 180 - d).tolist())
    results["motion"] = {
        "balanced_accuracy": float(np.mean(accs)),
        "median_angle_error_deg": float(np.median(angle_errors)) if angle_errors else float("nan"),
        "anisotropy_threshold": float(np.mean(motion_threshold)),
    }
    results["counts"] = {
        "samples": len(ev.samples),
        "sharp": int((ev.labels == 0).sum()),
        "blurred": int((ev.labels == 1).sum()),
        "unlabelled": int((ev.labels < 0).sum()),
        "sources": len(SOURCES),
    }
    ev.results = results
    return results


def fit_shipped_model(ev: Evaluation, seed: int) -> BlurModel:
    """The model shipped in model.json: fitted on every labelled sample at float64."""
    lab = ev.labelled
    model = BlurModel.fit(
        ev.features["float64"][lab], ev.labels[lab], ev.sigmas[lab],
        provenance={
            "benchmark_seed": seed,
            "samples": int(lab.sum()),
            "cross_validated_auc": ev.results["precisions"]["float64"]["auc"],
            "cross_validated_balanced_accuracy": ev.results["precisions"]["float64"]["balanced_accuracy"],
            "fitted_on": time.strftime("%Y-%m-%d"),
        },
    )
    model.motion_anisotropy = ev.results["motion"]["anisotropy_threshold"]
    return model

