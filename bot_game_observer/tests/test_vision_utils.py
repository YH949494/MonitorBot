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


def test_parse_numeric_amount_prefers_win_over_balance() -> None:
    value, _conf = vision.parse_numeric_amount("BALANCE 10000 WIN 2.00", hint="win")
    assert value == 2.0


def test_parse_numeric_amount_prefers_bet_over_others() -> None:
    value, _conf = vision.parse_numeric_amount("BALANCE 1234 BET 15 WIN 3", hint="bet")
    assert value == 15.0


def test_parse_numeric_amount_credit_with_ocr_noise() -> None:
    value, _conf = vision.parse_numeric_amount("CREDIT 99°984.00 BET 2.00", hint="credit")
    assert value == 99984.0


def test_parse_numeric_amount_without_hint_preserves_original_behavior() -> None:
    value, _conf = vision.parse_numeric_amount("BALANCE 1234 WIN 12")
    assert value == 12.0


def test_parse_numeric_amount_without_numeric_returns_none() -> None:
    value, conf = vision.parse_numeric_amount("BALANCE WIN")
    assert value is None
    assert conf == 0.0


def test_parse_numeric_amount_normalizes_ocr_separators() -> None:
    value, _conf = vision.parse_numeric_amount("CREDIT 99'984-00")
    assert value == 99984.0


def test_parse_numeric_amount_normalizes_comma_thousands() -> None:
    value, _conf = vision.parse_numeric_amount("CREDIT 1,250")
    assert value == 1250.0


def test_parse_numeric_amount_normalizes_large_comma_thousands() -> None:
    value, _conf = vision.parse_numeric_amount("CREDIT 99,984")
    assert value == 99984.0


def test_parse_numeric_amount_normalizes_dot_thousands() -> None:
    value, _conf = vision.parse_numeric_amount("CREDIT 1.250")
    assert value == 1250.0
