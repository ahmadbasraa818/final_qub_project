"""The benchmark generator, the statistics and the baselines."""

import numpy as np
import pytest

from blurfft.baselines import BASELINES
from blurfft.dataset import BlurSpec, disk_kernel, gaussian_kernel, make_benchmark, motion_kernel
from blurfft.evaluate import auc, balanced_accuracy, best_threshold, spearman


def test_kernels_are_normalised_and_have_the_stated_spread():
    for kernel in (gaussian_kernel(2.0), disk_kernel(3.0), motion_kernel(9, 30)):
        assert kernel.sum() == pytest.approx(1.0)
    g = gaussian_kernel(2.0)
    r = np.arange(g.shape[0]) - g.shape[0] // 2
    assert np.sqrt((g.sum(axis=0) * r**2).sum()) == pytest.approx(2.0, rel=0.02)
    d = disk_kernel(4.0)
    r = np.arange(d.shape[0]) - d.shape[0] // 2
    assert np.sqrt((d.sum(axis=0) * r**2).sum()) == pytest.approx(BlurSpec("disk", 4.0).sigma, rel=0.05)


def test_labels_follow_sigma():
    assert BlurSpec("none").label == 0
    assert BlurSpec("gaussian", 0.5).label == 0
    assert BlurSpec("gaussian", 1.0).label is None
    assert BlurSpec("disk", 3.0).label == 1
    assert BlurSpec("motion", 7.0).label == 1


def test_benchmark_is_deterministic():
    a = [s.image for _, s in zip(range(5), make_benchmark(["camera"], crops_per_source=1, seed=7))]
    b = [s.image for _, s in zip(range(5), make_benchmark(["camera"], crops_per_source=1, seed=7))]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_statistics():
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert balanced_accuracy(np.array([True, False, False, False]), np.array([1, 1, 0, 0])) == 0.75
    scores, labels = np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])
    t = best_threshold(scores, labels)
    assert balanced_accuracy(scores >= t, labels) == 0.75
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


@pytest.mark.parametrize("method", list(BASELINES))
def test_baselines_score_sharp_above_blurred(method):
    samples = list(make_benchmark(["camera"], crops_per_source=1, seed=3))
    sharp = next(s for s in samples if s.blur.kind == "none")
    blurred = next(s for s in samples if s.blur.kind == "gaussian" and s.blur.amount == 3.0)
    fn = BASELINES[method]
    assert fn(sharp.image) > fn(blurred.image)
