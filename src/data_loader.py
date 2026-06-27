"""
data_loader.py
==============

Provides functionality for loading the raw EPL datasets used throughout
the EPL Foul Win Probability Model project.

Datasets
--------
1. epl_event_data_15.csv
2. epl_matches_15.csv

Author
------
Nagasantosh Chavvakula

Project
-------
EPL Foul Win Probability Model
"""

from pathlib import Path
import logging

import pandas as pd

from src.config import RAW_DATA_DIR

# ---------------------------------------------------------------------
# Configure Logger
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads EPL raw datasets.

    Parameters
    ----------
    data_dir : Path | None, optional
        Path to the raw data directory. If None, the default path
        defined in config.py is used.

    Examples
    --------
    >>> loader = DataLoader()
    >>> events = loader.load_events()
    >>> matches = loader.load_matches()

    >>> events, matches = loader.load_all()
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Initialize the DataLoader.

        Parameters
        ----------
        data_dir : Path | None, optional
            Custom raw data directory.
        """

        self.data_dir = data_dir if data_dir is not None else RAW_DATA_DIR

    # -----------------------------------------------------------------

    def load_events(self) -> pd.DataFrame:
        """
        Load the EPL event dataset.

        Returns
        -------
        pd.DataFrame
            Event dataset.

        Raises
        ------
        FileNotFoundError
            If the events dataset cannot be found.
        """

        file_path = self.data_dir / "epl_event_data_15.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Events dataset not found:\n{file_path}")

        logger.info("Loading events dataset...")

        events = pd.read_csv(file_path)

        logger.info(
            "Events dataset loaded successfully "
            f"({events.shape[0]:,} rows × {events.shape[1]} columns)"
        )

        return events

    # -----------------------------------------------------------------

    def load_matches(self) -> pd.DataFrame:
        """
        Load the EPL matches dataset.

        Returns
        -------
        pd.DataFrame
            Match dataset.

        Raises
        ------
        FileNotFoundError
            If the matches dataset cannot be found.
        """

        file_path = self.data_dir / "epl_matches_15.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Matches dataset not found:\n{file_path}")

        logger.info("Loading matches dataset...")

        matches = pd.read_csv(file_path)

        logger.info(
            "Matches dataset loaded successfully "
            f"({matches.shape[0]:,} rows × {matches.shape[1]} columns)"
        )

        return matches

    # -----------------------------------------------------------------

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both datasets.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing:

            - Event dataset
            - Match dataset
        """

        logger.info("Loading all datasets...")

        events = self.load_events()
        matches = self.load_matches()

        logger.info("All datasets loaded successfully.")

        return events, matches

    # -----------------------------------------------------------------

    def get_dataset_info(self) -> dict:
        """
        Get information about the available datasets.

        Returns
        -------
        dict
            Dictionary containing dataset paths and availability.
        """

        event_path = self.data_dir / "epl_event_data_15.csv"
        match_path = self.data_dir / "epl_matches_15.csv"

        return {
            "data_directory": str(self.data_dir),
            "events_file": str(event_path),
            "matches_file": str(match_path),
            "events_exists": event_path.exists(),
            "matches_exists": match_path.exists(),
        }
