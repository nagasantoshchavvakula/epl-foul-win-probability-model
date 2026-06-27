"""
test_data_loader.py
===================

Unit tests for the DataLoader class.

Author
------
Nagasantosh Chavvakula

Project
-------
EPL Foul Win Probability Model
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import RAW_DATA_DIR
from src.data_loader import DataLoader

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def loader():
    """
    Create a DataLoader instance using the default raw data directory.
    """
    return DataLoader()


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


def test_loader_initialization(loader):
    """
    Test DataLoader initialization.
    """

    assert loader is not None
    assert loader.data_dir == RAW_DATA_DIR


# ---------------------------------------------------------------------
# Event Dataset Tests
# ---------------------------------------------------------------------


def test_load_events(loader):
    """
    Test loading the EPL event dataset.
    """

    events = loader.load_events()

    assert isinstance(events, pd.DataFrame)
    assert not events.empty

    # Required columns
    assert "match_id" in events.columns
    assert "type.name" in events.columns

    # Dataset should contain many rows
    assert events.shape[0] > 1000


# ---------------------------------------------------------------------
# Match Dataset Tests
# ---------------------------------------------------------------------


def test_load_matches(loader):
    """
    Test loading the EPL match dataset.
    """

    matches = loader.load_matches()

    assert isinstance(matches, pd.DataFrame)
    assert not matches.empty

    assert "match_id" in matches.columns

    assert matches.shape[0] > 0


# ---------------------------------------------------------------------
# Load All Datasets
# ---------------------------------------------------------------------


def test_load_all(loader):
    """
    Test loading both datasets together.
    """

    events, matches = loader.load_all()

    assert isinstance(events, pd.DataFrame)
    assert isinstance(matches, pd.DataFrame)

    assert not events.empty
    assert not matches.empty


# ---------------------------------------------------------------------
# Dataset Info
# ---------------------------------------------------------------------


def test_dataset_info(loader):
    """
    Test dataset information dictionary.
    """

    info = loader.get_dataset_info()

    assert isinstance(info, dict)

    assert info["events_exists"] is True
    assert info["matches_exists"] is True

    assert Path(info["events_file"]).exists()
    assert Path(info["matches_file"]).exists()


# ---------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------


def test_invalid_directory():
    """
    Test loading from an invalid directory.
    """

    loader = DataLoader(Path("invalid_directory"))

    with pytest.raises(FileNotFoundError):

        loader.load_events()


def test_invalid_match_directory():
    """
    Test loading match dataset from an invalid directory.
    """

    loader = DataLoader(Path("invalid_directory"))

    with pytest.raises(FileNotFoundError):

        loader.load_matches()
