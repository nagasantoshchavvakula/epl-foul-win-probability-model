"""
target_creation.py
==================

Creates the supervised learning target for the EPL Foul Win
Probability Model.

Definition
----------
Candidate Events

    • Ball Receipt*
    • Ball Recovery

Target

    target = 1

        if the SAME player wins a foul later
        in the SAME possession.

    target = 0

        otherwise.

The resulting dataset is the foundation for feature engineering
and model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


class TargetCreator:
    """
    Creates the machine learning target variable.
    """

    CANDIDATE_EVENTS = [
        "Ball Receipt*",
        "Ball Recovery",
    ]

    TARGET_EVENT = "Foul Won"

    def __init__(self):

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_target(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create the target variable.

        Parameters
        ----------
        events : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Creating target variable...")

        required_columns = [
            "match_id",
            "possession",
            "index",
            "player.id",
            "type.name",
        ]

        self._validate_columns(
            events,
            required_columns,
        )

        events = (
            events
            .sort_values(
                [
                    "match_id",
                    "possession",
                    "index",
                ]
            )
            .reset_index(drop=True)
        )

        candidate_events = events[
            events["type.name"].isin(
                self.CANDIDATE_EVENTS
            )
        ].copy()

        logger.info(
            "Candidate events found: %d",
            len(candidate_events),
        )

        targets: List[int] = []

        grouped_events = events.groupby(
            [
                "match_id",
                "possession",
            ]
        )

        for _, row in candidate_events.iterrows():

            possession_events = grouped_events.get_group(
                (
                    row["match_id"],
                    row["possession"],
                )
            )

            future_events = possession_events[
                possession_events["index"]
                > row["index"]
            ]

            foul_won = future_events[
                (
                    future_events["type.name"]
                    == self.TARGET_EVENT
                )
                &
                (
                    future_events["player.id"]
                    == row["player.id"]
                )
            ]

            targets.append(
                int(not foul_won.empty)
            )

        candidate_events["target"] = targets

        logger.info(
            "Positive targets : %d",
            candidate_events["target"].sum(),
        )

        logger.info(
            "Negative targets : %d",
            len(candidate_events)
            - candidate_events["target"].sum(),
        )

        return candidate_events

    def save_dataset(
        self,
        dataset: pd.DataFrame,
        filename: str = "model_dataset.csv",
    ) -> Path:
        """
        Save processed dataset.

        Parameters
        ----------
        dataset : pd.DataFrame

        filename : str

        Returns
        -------
        Path
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
            "Dataset saved to %s",
            output_path,
        )

        return output_path

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        required_columns: List[str],
    ) -> None:
        """
        Validate required columns.
        """

        missing_columns = [
            col
            for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )