"""Tests for stage 3 (calibrate).

The regression test uses the v1 Florida overlap file (19,689 coincident cell-days,
2012-2021). The new code must reproduce the published v1 equation exactly:
    VIIRS_FRP = 0.316162 x MODIS_FRP + 24.701681
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shift.calibrate import build_pairs, fit_ols, load_v1_overlap, split_window, validate

FIXTURE = Path(__file__).parent / "fixtures" / "florida_v1_overlap.csv"
SCALES = [
    {"name": "cell_day", "time": "day", "space_deg": None},
    {"name": "area025_month", "time": "month", "space_deg": 0.25},
    {"name": "region_month", "time": "month", "space_deg": "all"},
]


@pytest.fixture(scope="module")
def v1_pairs():
    return load_v1_overlap(FIXTURE)


def test_reproduces_v1_equation_exactly(v1_pairs):
    model = fit_ols(v1_pairs)
    assert model["n_pairs"] == 19689
    assert model["slope"] == pytest.approx(0.316162, abs=5e-7)
    assert model["intercept"] == pytest.approx(24.701681, abs=5e-7)


def test_formula_is_stable_and_validates_out_of_sample(v1_pairs):
    """Fit 2012-2018, test on unseen 2019-2021 (results recorded 12 Oct 2026)."""
    train = split_window(v1_pairs, "2012-01-01", "2018-12-31")
    test = split_window(v1_pairs, "2019-01-01", "2021-12-31")
    model = fit_ols(train)
    assert model["slope"] == pytest.approx(0.3143, abs=5e-4)       # stable vs 0.3162 on all years
    val = validate(test, model, SCALES).set_index("scale")
    assert abs(val.loc["region_month", "total_error_pct"]) < 1.0
    assert val.loc["region_month", "r2_calibrated"] > 0.93
    assert val.loc["region_month", "r2_calibrated"] > val.loc["region_month", "r2_raw_modis"]
    # Error falls as the scale grows: the calendar scale is the one that matters.
    assert val.loc["cell_day", "r2_calibrated"] < val.loc["area025_month", "r2_calibrated"] < val.loc["region_month", "r2_calibrated"]


def test_build_pairs_joins_same_cell_and_day():
    day = pd.Timestamp("2015-04-01")
    cd = pd.DataFrame({
        "acq_date": [day, day, day, day, day + pd.Timedelta(days=1)],
        "cell_id": [10, 10, 10, 11, 10],
        "cell_lat": [27.025] * 5,
        "cell_lon": [-81.025] * 5,
        "satellite": ["Terra", "Aqua", "S-NPP", "S-NPP", "Terra"],
        "frp_sum_mw": [10.0, 5.0, 8.0, 3.0, 7.0],
        "n_detections": [1, 1, 3, 1, 1],
    })
    pairs = build_pairs(cd, ["Terra", "Aqua"], ["S-NPP"])
    assert len(pairs) == 1                               # only cell 10 on day 1 has both
    row = pairs.iloc[0]
    assert row["frp_modis"] == 15.0 and row["count_modis"] == 2   # Terra + Aqua summed
    assert row["frp_viirs"] == 8.0 and row["count_viirs"] == 3


def test_fit_needs_data():
    with pytest.raises(ValueError):
        fit_ols(pd.DataFrame({"acq_date": [], "frp_modis": [], "frp_viirs": []}))


def test_validate_perfect_model_scores_one():
    rng = np.random.default_rng(1)
    n = 200
    modis = rng.gamma(2, 20, n)
    pairs = pd.DataFrame({
        "acq_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "cell_lat": rng.uniform(25, 30, n), "cell_lon": rng.uniform(-85, -80, n),
        "frp_modis": modis, "count_modis": 1, "frp_viirs": 0.5 * modis + 3, "count_viirs": 2,
    })
    model = fit_ols(pairs)
    val = validate(pairs, model, SCALES)
    assert np.allclose(val["r2_calibrated"], 1.0) and np.allclose(val["total_error_pct"], 0.0, atol=1e-9)
