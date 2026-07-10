"""
feature_engineering.py
======================

Feature engineering pipeline for the EPL Foul Win Probability Model.

This module transforms the target dataset into a machine learning-ready
dataset by creating numerical and categorical features from the raw
StatsBomb event data.

Current Features
----------------
1. Event coordinates (x, y)
2. Coordinate validation
3. Event type encoding (Part 3A.2)
4. Distance to goal (Part 3A.2)
5. Pitch zones (Part 3A.2)
6. Possession features (Part 3A.3)

Author
------
Nagasantosh Chavvakula
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for the EPL Foul Win Probability Model.

    This class creates machine learning features from the
    candidate event dataset generated during the target creation phase.
    """

    REQUIRED_COLUMNS = [
        "location",
        "type.name",
        "minute",
        "second",
        "match_id",
        "possession",
        "index",
        "target",
    ]

    def __init__(self) -> None:
        """
        Initialize feature engineering pipeline.
        """

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    ####################################################################
    # Main Pipeline
    ####################################################################

    def create_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create engineered features.

        Parameters
        ----------
        dataset : pd.DataFrame
            Target dataset.

        Returns
        -------
        pd.DataFrame
            Feature engineered dataset.
        """

        logger.info("Starting feature engineering...")

        self._validate_columns(dataset)

        dataset = dataset.copy()
 
    ############################################################
    # Coordinate Features
    ############################################################

        dataset = self._extract_location_features(dataset)

    ############################################################
    # Spatial Features
    ############################################################

        dataset = self._create_distance_to_goal(dataset)

        dataset = self._create_goal_angle(dataset)

        dataset = self._create_pitch_zones(dataset)

    ############################################################
    # Event Features
    ############################################################

        dataset = self._encode_event_type(dataset)

    ############################################################
    # Temporal Features
    ############################################################

        dataset = self._create_match_time(dataset)

        logger.info("Feature engineering completed successfully.")

        logger.info(
            "Current dataset shape: %s",
            dataset.shape,
        )

        return dataset

    ####################################################################
    # Feature Creation
    ####################################################################

    def _extract_location_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract x and y coordinates from StatsBomb location field.

        Parameters
        ----------
        dataset : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info(
            "Extracting location coordinates..."
        )

        coordinates = dataset["location"].apply(
            self._parse_location
        )

        dataset["x"] = coordinates.apply(
            lambda value: value[0]
        )

        dataset["y"] = coordinates.apply(
            lambda value: value[1]
        )

        return dataset

    ####################################################################
    # Spatial Features
    ####################################################################

    def _create_distance_to_goal(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Euclidean distance from the event location
        to the opponent's goal.

        StatsBomb pitch dimensions:
            Length = 120
            Width  = 80

        Goal Centre = (120, 40)
        """

        logger.info("Creating distance to goal feature...")

        goal_x = 120
        goal_y = 40

        dataset["distance_to_goal"] = np.sqrt(
            (goal_x - dataset["x"]) ** 2
            +
            (goal_y - dataset["y"]) ** 2
        )

        return dataset

    def _create_goal_angle(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate shooting angle relative to
        the center of the goal.
        """

        logger.info("Creating goal angle feature...")

        goal_x = 120
        goal_y = 40

        dataset["goal_angle"] = np.degrees(
            np.arctan2(
                goal_y - dataset["y"],
                goal_x - dataset["x"],
            )
        )

        return dataset

    def _create_pitch_zones(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Divide the pitch into three vertical zones.

        Defensive
        Middle
        Attacking
        """

        logger.info("Creating pitch zones...")

        conditions = [

            dataset["x"] < 40,

            (dataset["x"] >= 40)
            &
            (dataset["x"] < 80),

            dataset["x"] >= 80,

        ]

        values = [

            "Defensive",

            "Middle",

            "Attacking",

        ]

        dataset["pitch_zone"] = np.select(
            conditions,
            values,
            default="Unknown",
        )

        return dataset

    ####################################################################
    # Event Features
    ####################################################################

    def _encode_event_type(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Encode event type.

        Ball Receipt* = 0

        Ball Recovery = 1
        """

        logger.info("Encoding event types...")

        dataset["event_type"] = dataset[
            "type.name"
        ].map(
            {
                "Ball Receipt*": 0,
                "Ball Recovery": 1,
            }
        )

        return dataset

    ####################################################################
    # Temporal Features
    ####################################################################

    def _create_match_time(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create elapsed match time in seconds.
        """

        logger.info("Creating match time feature...")

        dataset["match_time_seconds"] = (

            dataset["minute"] * 60

            +

            dataset["second"]

        )

        return dataset
    ####################################################################
    # Helper Functions
    ####################################################################

    @staticmethod
    def _parse_location(
        value,
    ) -> Tuple[float, float]:
        """
        Parse StatsBomb location column.

        Parameters
        ----------
        value : object

        Returns
        -------
        tuple
            (x, y)
        """

        if isinstance(value, (list, tuple, np.ndarray)):

            if len(value) >= 2:
                return float(value[0]), float(value[1])

            return np.nan, np.nan

        if pd.isna(value):
            return np.nan, np.nan

        if isinstance(value, str):

            try:

                value = value.strip("[]")

                x, y = value.split(",")

                return float(x), float(y)

            except Exception:

                return np.nan, np.nan

        return np.nan, np.nan

    ####################################################################
    # Validation
    ####################################################################

    def _validate_columns(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Validate required columns.

        Parameters
        ----------
        dataset : pd.DataFrame

        Raises
        ------
        ValueError
        """

        missing_columns = [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in dataset.columns

        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

    ####################################################################
    # Save Dataset
    ####################################################################

    def save_dataset(
        self,
        dataset: pd.DataFrame,
        filename: str = "engineered_dataset.csv",
    ) -> Path:
        """
        Save engineered dataset.

        Parameters
        ----------
        dataset : pd.DataFrame

        filename : str

        Returns
        -------
        pathlib.Path
        """

        output_path = (
            PROCESSED_DATA_DIR
            / filename
        )

        dataset.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Engineered dataset saved to %s",
            output_path,
        )

        return output_path