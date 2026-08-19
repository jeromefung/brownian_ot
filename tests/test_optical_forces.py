import numpy as np
import pytest

from brownian_ot.particles import SphereCluster


def test_mismatched_indices_ratios():
    with pytest.raises(ValueError):
        SphereCluster(np.array([[0, 0, 1],
                                [0, 0, -1]]),
                      np.ones((6, 6)), np.zeros(3),
                      1e-6, np.ones(3) * 1.2,
                      np.ones(2))
