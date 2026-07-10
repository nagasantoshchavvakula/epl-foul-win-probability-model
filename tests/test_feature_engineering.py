"""
test_feature_engineering.py
===========================

Unit tests for the FeatureEngineer class.

Run:

    pytest tests/test_feature_engineering.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


########################################################################
# Fixtures
########################################################################

@pytest.fixture
def sample_dataset():
    """
    Create a small sample dataset for feature engineering tests.
    """

    return pd.DataFrame(
        {
            "location": [
                "[60,40]",
                "[100,30]",
                "[20,60]",
            ],
            "type.name": [
                "Ball Receipt*",
                "Ball Recovery",
                "Ball Receipt*",
            ],
            "minute": [10, 20, 35],
            "second": [15, 30, 45],
            "match_id": [1, 1, 1],
            "possession": [1, 1, 2],
            "index": [1, 2, 3],
            "target": [1, 0, 1],
        }
    )


########################################################################
# Tests
########################################################################

def test_create_features_returns_dataframe(sample_dataset):
    """
    Feature engineering should return a DataFrame.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert isinstance(dataset, pd.DataFrame)


def test_location_features_created(sample_dataset):
    """
    x and y coordinates should be created.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "x" in dataset.columns
    assert "y" in dataset.columns


def test_distance_feature_created(sample_dataset):
    """
    Distance to goal feature should exist.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "distance_to_goal" in dataset.columns


def test_goal_angle_created(sample_dataset):
    """
    Goal angle feature should exist.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "goal_angle" in dataset.columns


def test_pitch_zone_created(sample_dataset):
    """
    Pitch zone feature should exist.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "pitch_zone" in dataset.columns


def test_event_encoding(sample_dataset):
    """
    Event type should be encoded correctly.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "event_type" in dataset.columns

    assert set(dataset["event_type"].unique()) == {0, 1}


def test_match_time_feature(sample_dataset):
    """
    Match time should be converted to seconds.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert "match_time_seconds" in dataset.columns

    assert dataset.loc[0, "match_time_seconds"] == 615


def test_distance_values(sample_dataset):
    """
    Distance to goal should be numeric and positive.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert np.all(dataset["distance_to_goal"] >= 0)


def test_pitch_zone_values(sample_dataset):
    """
    Pitch zone should contain valid categories.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    valid = {
        "Defensive",
        "Middle",
        "Attacking",
    }

    assert set(dataset["pitch_zone"]).issubset(valid)


def test_missing_required_columns():
    """
    Missing required columns should raise ValueError.
    """

    engineer = FeatureEngineer()

    dataframe = pd.DataFrame(
        {
            "A": [1, 2, 3]
        }
    )

    with pytest.raises(ValueError):

        engineer.create_features(dataframe)


def test_parse_location_string():
    """
    Parse string location.
    """

    x, y = FeatureEngineer._parse_location("[80,20]")

    assert x == 80

    assert y == 20


def test_parse_location_list():
    """
    Parse list location.
    """

    x, y = FeatureEngineer._parse_location([50, 30])

    assert x == 50

    assert y == 30


def test_parse_location_nan():
    """
    NaN location should return NaN values.
    """

    x, y = FeatureEngineer._parse_location(np.nan)

    assert np.isnan(x)

    assert np.isnan(y)


def test_save_dataset(sample_dataset, tmp_path):
    """
    Engineered dataset should be saved.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    output = tmp_path / "engineered_dataset.csv"

    dataset.to_csv(output, index=False)

    assert output.exists()


def test_feature_count(sample_dataset):
    """
    Engineered dataset should contain additional columns.
    """

    engineer = FeatureEngineer()

    dataset = engineer.create_features(sample_dataset)

    assert dataset.shape[1] > sample_dataset.shape[1]