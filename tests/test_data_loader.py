"""
test_data_loader.py
===================

Unit tests for the DataLoader class.

These tests create temporary CSV files so they do not depend on the
real EPL datasets being present. This makes them suitable for GitHub
Actions and other CI environments.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader


@pytest.fixture
def sample_data_dir(tmp_path):
    """
    Create a temporary data directory containing sample EPL datasets.
    """

    data_dir = tmp_path

    # Sample events dataset
    events = pd.DataFrame(
        {
            "match_id": [1, 1],
            "type.name": ["Pass", "Ball Receipt*"],
            "player.id": [10, 11],
        }
    )

    events.to_csv(
        data_dir / "epl_event_data_15.csv",
        index=False,
    )

    # Sample matches dataset
    matches = pd.DataFrame(
        {
            "match_id": [1],
            "home_team": ["Team A"],
            "away_team": ["Team B"],
        }
    )

    matches.to_csv(
        data_dir / "epl_matches_15.csv",
        index=False,
    )

    return data_dir


@pytest.fixture
def loader(sample_data_dir):
    """
    Create DataLoader using temporary datasets.
    """
    return DataLoader(sample_data_dir)


def test_loader_initialization(loader, sample_data_dir):
    """
    Test DataLoader initialization.
    """

    assert loader.data_dir == sample_data_dir


def test_load_events(loader):
    """
    Test loading events dataset.
    """

    events = loader.load_events()

    assert isinstance(events, pd.DataFrame)

    assert not events.empty

    assert "match_id" in events.columns

    assert "type.name" in events.columns

    assert len(events) == 2


def test_load_matches(loader):
    """
    Test loading matches dataset.
    """

    matches = loader.load_matches()

    assert isinstance(matches, pd.DataFrame)

    assert not matches.empty

    assert "match_id" in matches.columns

    assert len(matches) == 1


def test_load_all(loader):
    """
    Test loading both datasets.
    """

    events, matches = loader.load_all()

    assert len(events) == 2

    assert len(matches) == 1


def test_dataset_info(loader):
    """
    Test dataset info.
    """

    info = loader.get_dataset_info()

    assert info["events_exists"]

    assert info["matches_exists"]

    assert Path(info["events_file"]).exists()

    assert Path(info["matches_file"]).exists()


def test_invalid_directory():
    """
    Test invalid directory.
    """

    loader = DataLoader(Path("invalid_directory"))

    with pytest.raises(FileNotFoundError):
        loader.load_events()


def test_invalid_match_directory():
    """
    Test invalid match directory.
    """

    loader = DataLoader(Path("invalid_directory"))

    with pytest.raises(FileNotFoundError):
        loader.load_matches()

