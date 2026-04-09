"""Vision helper tests (synthetic images)."""

from __future__ import annotations

import numpy as np

from src.models import Region
from src import vision


def test_crop_region_bounds() -> None:
    img = np.zeros((100, 200), dtype=np.uint8)
    img[10:20, 30:80] = 255
    r = Region(left=30, top=10, width=50, height=10)
    c = vision.crop_region(img, r)
    assert c.shape == (10, 50)
    assert float(c.mean()) > 200


def test_template_match_perfect() -> None:
    scene = np.zeros((64, 64), dtype=np.uint8)
    scene[20:40, 20:40] = 200
    tmpl = scene[20:40, 20:40].copy()
    score, loc = vision.template_match_best(scene, tmpl)
    assert score > 0.99
    assert loc == (20, 20)


def test_motion_score_zero_on_identical() -> None:
    a = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    assert vision.motion_score(a, a) == 0.0


def test_motion_score_nonzero_on_shift() -> None:
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.zeros((32, 32), dtype=np.uint8)
    b[10, 10] = 255
    assert vision.motion_score(a, b) > 0.0


def test_parse_numeric_amount_handles_noisy_separators() -> None:
    value, conf = vision.parse_numeric_amount("WIN 1'234.50")
    assert value == 1234.5
    assert conf > 0
