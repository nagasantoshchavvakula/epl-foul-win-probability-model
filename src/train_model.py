"""
train_model.py
==============

Train a baseline Random Forest model for the EPL Foul Win Probability Model.

This module loads the engineered dataset, selects numeric features,
trains the model, evaluates predictions, and saves the trained model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


class TrainModel:
    """
    Train a Random Forest baseline for EPL foul win probability.
    """

    EXCLUDE_COLUMNS = [
        "target",
        "id",
        "match_id",
        "index",
        "Unnamed: 0",
    ]

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        model_path: Path | str | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the training pipeline.

        Parameters
        ----------
        dataset_path : Path | str | None, optional
            Path to the engineered dataset. Defaults to
            data/processed/engineered_dataset.csv.

        model_path : Path | str | None, optional
            Path where the trained model will be saved.

        test_size : float
            Fraction of the dataset to reserve for testing.

        random_state : int
            Random seed for reproducibility.
        """

        self.dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else PROCESSED_DATA_DIR / "engineered_dataset.csv"
        )
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else MODELS_DIR / "random_forest_model.pkl"
        )
        self.test_size = test_size
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None

    ####################################################################
    # Main Pipeline
    ####################################################################

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the engineered dataset from disk.

        Returns
        -------
        pd.DataFrame
            Loaded engineered dataset.
        """

        logger.info("Loading dataset from %s", self.dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Engineered dataset not found: {self.dataset_path}"
            )

        dataset = pd.read_csv(self.dataset_path)

        logger.info(
            "Dataset loaded successfully (%d rows × %d columns)",
            dataset.shape[0],
            dataset.shape[1],
        )

        return dataset

    def prepare_data(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Select numeric features and extract target values.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset containing the engineered features and target.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature matrix and target vector.
        """

        logger.info("Preparing features and target variables...")

        if "target" not in dataset.columns:
            raise ValueError("Dataset does not contain a 'target' column.")

        numeric_dataset = dataset.select_dtypes(include=[np.number]).copy()

        if "target" not in numeric_dataset.columns:
            raise ValueError("Target column is not numeric.")

        feature_columns = [
            column
            for column in numeric_dataset.columns
            if column not in self.EXCLUDE_COLUMNS
        ]

        if not feature_columns:
            raise ValueError("No numeric features found for training.")

        X = numeric_dataset[feature_columns].copy()
        y = numeric_dataset["target"].copy()

        logger.info(
            "Selected %d numeric features for training.",
            X.shape[1],
        )

        logger.info("Feature names: %s", list(X.columns))

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing subsets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target vector.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            Training and testing splits.
        """

        logger.info("Splitting data into train and test sets...")

        if y.isna().any():
            raise ValueError("Target column contains missing values.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Training rows: %d, Testing rows: %d",
            X_train.shape[0],
            X_test.shape[0],
        )

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RandomForestClassifier:
        """
        Train the Random Forest model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.

        y_train : pd.Series
            Training target vector.

        Returns
        -------
        RandomForestClassifier
            Trained model.
        """

        logger.info("Training Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        self.model = model

        logger.info("Model training completed.")

        return model

    def predict(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate class predictions from a trained model.

        Parameters
        ----------
        model : RandomForestClassifier
            Trained classifier.

        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Array of predicted labels.
        """

        logger.info("Generating predictions...")

        return model.predict(X)

    def predict_proba(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate predicted probabilities for the positive class.

        Parameters
        ----------
        model : RandomForestClassifier
            Trained classifier.

        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability estimates for each class.
        """

        logger.info("Generating prediction probabilities...")

        return model.predict_proba(X)

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics for the model.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_pred : np.ndarray
            Predicted labels.

        y_proba : np.ndarray
            Predicted probabilities.

        Returns
        -------
        dict[str, Any]
            Dictionary of evaluation metrics.
        """

        logger.info("Evaluating model performance...")

        if y_proba.ndim != 2 or y_proba.shape[1] != 2:
            raise ValueError(
                "Prediction probabilities must contain two class columns."
            )

        proba_positive = y_proba[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba_positive),
            "classification_report": sk_classification_report(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "confusion_matrix": sk_confusion_matrix(y_true, y_pred),
        }

        logger.info("Evaluation metrics computed successfully.")

        return metrics

    def feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Create a feature importance DataFrame.

        Parameters
        ----------
        model : RandomForestClassifier
            Trained classifier.

        feature_names : list[str]
            Names of the features used for training.

        Returns
        -------
        pd.DataFrame
            Sorted feature importance values.
        """

        logger.info("Computing feature importance...")

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )

        importance = importance.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

        logger.info("Feature importance computed.")

        return importance

    def save_model(
        self,
        model: RandomForestClassifier,
    ) -> Path:
        """
        Save the trained model to disk.

        Parameters
        ----------
        model : RandomForestClassifier
            Trained classifier.

        Returns
        -------
        Path
            Path to the saved model file.
        """

        logger.info("Saving model to %s", self.model_path)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, self.model_path)

        logger.info("Model saved successfully.")

        return self.model_path

    def run_pipeline(self) -> dict[str, Any]:
        """
        Execute the full training pipeline end to end.

        Returns
        -------
        dict[str, Any]
            "Training artifacts, including metrics,"
            "model path and feature importance."
        """

        dataset = self.load_dataset()
        X, y = self.prepare_data(dataset)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        model = self.train(X_train, y_train)
        predictions = self.predict(model, X_test)
        probabilities = self.predict_proba(model, X_test)
        metrics = self.evaluate(y_test, predictions, probabilities)
        importance = self.feature_importance(model, list(X.columns))
        model_path = self.save_model(model)

        logger.info("Training pipeline completed successfully.")

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "model_path": model_path,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
        }
