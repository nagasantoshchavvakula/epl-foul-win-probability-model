"""
test_target_creation.py
=======================

Unit tests for the TargetCreator class.

Run:

    pytest tests/test_target_creation.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from src.target_creation import TargetCreator


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_events():
    """
    Create a small synthetic StatsBomb-like dataset.

    Possession 1

        Ball Receipt (Player 10)
        Pass
        Foul Won (Player 10)

    Possession 2

        Ball Recovery (Player 20)
        Pass

    Expected

        target = [1, 0]
    """

    return pd.DataFrame(
        {

            "match_id": [1,1,1,1,1],

            "possession":[1,1,1,2,2],

            "index":[1,2,3,4,5],

            "player.id":[
                10,
                10,
                10,
                20,
                20
            ],

            "type.name":[
                "Ball Receipt*",
                "Pass",
                "Foul Won",
                "Ball Recovery",
                "Pass"
            ]
        }
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_create_target_returns_dataframe(sample_events):
    """
    Should return a pandas DataFrame.
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    assert isinstance(dataset, pd.DataFrame)


def test_target_column_exists(sample_events):
    """
    Target column should be created.
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    assert "target" in dataset.columns


def test_candidate_events_only(sample_events):
    """
    Output should contain only Ball Receipt and Ball Recovery.
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    assert dataset["type.name"].isin(
        [
            "Ball Receipt*",
            "Ball Recovery",
        ]
    ).all()


def test_target_values(sample_events):
    """
    Verify expected target values.

    Ball Receipt -> target = 1

    Ball Recovery -> target = 0
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    assert dataset.iloc[0]["target"] == 1

    assert dataset.iloc[1]["target"] == 0


def test_number_of_candidate_events(sample_events):
    """
    Only two candidate events exist.
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    assert len(dataset) == 2


def test_missing_required_columns():
    """
    Missing columns should raise ValueError.
    """

    creator = TargetCreator()

    dataframe = pd.DataFrame(
        {
            "A":[1,2,3]
        }
    )

    with pytest.raises(ValueError):

        creator.create_target(dataframe)


def test_empty_dataframe():
    """
    Empty dataframe should raise ValueError.
    """

    creator = TargetCreator()

    dataframe = pd.DataFrame()

    with pytest.raises(ValueError):

        creator.create_target(dataframe)


def test_save_dataset(sample_events, tmp_path):
    """
    Dataset should be saved successfully.
    """

    creator = TargetCreator()

    dataset = creator.create_target(sample_events)

    output = tmp_path / "dataset.csv"

    dataset.to_csv(output, index=False)

    assert output.exists()


def test_no_positive_target():
    """
    No foul won should produce only zeros.
    """

    dataframe = pd.DataFrame(
        {

            "match_id": [1,1],

            "possession": [1,1],

            "index": [1,2],

            "player.id": [100,100],

            "type.name": [
                "Ball Receipt*",
                "Pass"
            ]
        }
    )

    creator = TargetCreator()

    dataset = creator.create_target(dataframe)

    assert dataset.iloc[0]["target"] == 0