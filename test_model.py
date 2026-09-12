# Model quality tests - these run in CI before any image is built.
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "model/metadata.json").read_text())
FEATURES = META["features"]["categorical"] + META["features"]["numeric"]


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "model/flight_price_pipeline.joblib")


def _row(**over):
    base = {"from": "Sao Paulo (SP)", "to": "Rio de Janeiro (RJ)", "flightType": "economic",
            "agency": "CloudFy", "gender": "male", "company": "4You",
            "distance": 331.89, "time": 0.86, "year": 2023, "month": 6, "day": 15,
            "day_of_week": 3, "quarter": 2, "is_weekend": 0, "age": 35}
    base.update(over)
    return pd.DataFrame([base])[FEATURES]


def test_artifact_metadata_is_complete():
    assert META["test_metrics"]["R2"] > 0.95
    assert len(FEATURES) == 15


def test_prediction_is_a_positive_number(model):
    price = float(model.predict(_row())[0])
    assert 100 < price < 5000


def test_first_class_costs_more_than_economy(model):
    economy = float(model.predict(_row(flightType="economic"))[0])
    first = float(model.predict(_row(flightType="firstClass"))[0])
    assert first > economy


def test_longer_route_costs_more(model):
    short = float(model.predict(_row(distance=331.89, time=0.86))[0])
    long = float(model.predict(_row(to="Florianopolis (SC)", distance=808.85, time=2.10))[0])
    assert long > short


def test_unknown_category_does_not_crash(model):
    # OneHotEncoder(handle_unknown="ignore") must degrade gracefully, not raise
    price = float(model.predict(_row(company="A Company That Does Not Exist"))[0])
    assert price > 0
