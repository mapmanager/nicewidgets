"""Fixtures for contrast widget tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a small 2D numpy array for testing."""
    return np.array([[10, 50, 100], [25, 75, 150], [0, 125, 200]], dtype=np.float32)
