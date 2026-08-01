"""
evaluate_model.py
=================

Model evaluation pipeline for the EPL Foul Win Probability Model.
This module loads the engineered dataset and trained model, computes
classification metrics, generates evaluation visuals, and exports
evaluation artifacts for reporting and portfolio presentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

from src.config import MODELS_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, PLOTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate a pre-trained Random Forest model on engineered EPL data.
    """

    EXCLUDE_COLUMNS = [
        "target",
        "match_id",
        "id",
        "index",
        "Unnamed: 0",
    ]

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        model_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        plots_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize the model evaluation pipeline.

        Parameters
        ----------
        dataset_path : Path | str | None, optional
            Path to the engineered dataset.

        model_path : Path | str | None, optional
            Path to the serialized model file.

        output_dir : Path | str | None, optional
            Directory for evaluation outputs.

        plots_dir : Path | str | None, optional
            Directory for plot outputs.
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
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else OUTPUT_DIR
        )
        self.plots_dir = (
            Path(plots_dir)
            if plots_dir is not None
            else PLOTS_DIR
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.dataset: pd.DataFrame | None = None
        self.model: RandomForestClassifier | Any = None
        self.feature_columns: list[str] = []

    ####################################################################
    # Loading and Preparation
    ####################################################################

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the engineered dataset from disk.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """

        logger.info("Loading engineered dataset from %s", self.dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Engineered dataset not found: {self.dataset_path}"
            )

        self.dataset = pd.read_csv(self.dataset_path)

        logger.info(
            "Dataset loaded successfully (%d rows × %d columns)",
            self.dataset.shape[0],
            self.dataset.shape[1],
        )

        return self.dataset

    def load_model(self) -> Any:
        """
        Load the trained Random Forest model.

        Returns
        -------
        Any
            Loaded model object.
        """

        logger.info("Loading trained model from %s", self.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        logger.info("Model loaded successfully.")

        return self.model

    def prepare_data(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the feature matrix and target vector for evaluation.

        Parameters
        ----------
        dataset : pd.DataFrame
            Loaded engineered dataset.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature matrix and target vector.
        """

        logger.info("Preparing data for evaluation...")

        if "target" not in dataset.columns:
            raise ValueError("Dataset must contain a 'target' column.")

        numeric_dataset = dataset.select_dtypes(include=[np.number]).copy()

        if "target" not in numeric_dataset.columns:
            raise ValueError("Target column must be numeric.")

        feature_columns = [
            column
            for column in numeric_dataset.columns
            if column not in self.EXCLUDE_COLUMNS
        ]

        if not feature_columns:
            raise ValueError("No numeric features available for evaluation.")

        self.feature_columns = feature_columns

        X = numeric_dataset[feature_columns].copy()
        y = numeric_dataset["target"].copy()

        logger.info(
            "Selected %d feature columns for evaluation.",
            X.shape[1],
        )

        logger.info("Feature columns: %s", self.feature_columns)

        return X, y

    ####################################################################
    # Prediction and Probability
    ####################################################################

    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate class predictions using the trained model.

        Parameters
        ----------
        model : Any
            Trained classification model.

        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """

        logger.info("Generating predictions for %d observations.", X.shape[0])

        predictions = model.predict(X)

        logger.info("Predictions generated.")

        return predictions

    def predict_probability(
        self,
        model: Any,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate class probability estimates using the trained model.

        Parameters
        ----------
        model : Any
            Trained classification model.

        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class probability matrix.
        """

        logger.info(
            "Generating predicted probabilities for %d observations.",
            X.shape[0]
            )

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Model does not support probability prediction."
            )

        probabilities = model.predict_proba(X)

        logger.info("Probability estimates generated.")

        return probabilities

    ####################################################################
    # Metrics and Reports
    ####################################################################

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics and supporting artifacts.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_pred : np.ndarray
            Predicted class labels.

        y_proba : np.ndarray
            Predicted class probabilities.

        Returns
        -------
        dict[str, Any]
            Dictionary containing evaluation metrics and report objects.
        """

        logger.info("Calculating evaluation metrics...")

        if y_true.nunique() < 2:
            raise ValueError("True targets must contain at least two classes.")

        y_scores = y_proba[:, 1]
        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    y_scores,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "matthews_correlation_coefficient": float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),
            "classification_report": self.classification_report_df(
                y_true,
                y_pred,
            ),
            "confusion_matrix": self.confusion_matrix_df(
                y_true,
                y_pred,
            ),
        }
        logger.info("Evaluation metrics calculated.")

        return metrics

    def classification_report_df(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """
        Convert the classification report into a pandas DataFrame.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_pred : np.ndarray
            Predicted class labels.

        Returns
        -------
        pd.DataFrame
            Classification report table.
        """

        logger.info("Building classification report DataFrame...")

        report = sk_classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
            .reset_index()
            .rename(columns={"index": "class"})
        )

        logger.info("Classification report DataFrame created.")

        return report_df

    def confusion_matrix_df(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """
        Convert the confusion matrix into a pandas DataFrame.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_pred : np.ndarray
            Predicted class labels.

        Returns
        -------
        pd.DataFrame
            Confusion matrix table.
        """

        logger.info("Building confusion matrix DataFrame...")

        matrix = sk_confusion_matrix(y_true, y_pred)
        classes = [f"Actual {label}" for label in np.unique(y_true)]
        columns = [f"Predicted {label}" for label in np.unique(y_pred)]

        matrix_df = pd.DataFrame(
            matrix,
            index=classes,
            columns=columns,
        )

        logger.info("Confusion matrix DataFrame created.")

        return matrix_df

    def feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Generate a sorted feature importance DataFrame.

        Parameters
        ----------
        model : Any
            Trained classifier.

        feature_names : list[str]
            Feature names used in the model.

        top_n : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Sorted feature importance values.
        """

        logger.info("Extracting feature importance values...")

        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                "Model does not expose feature_importances_."
            )

        importance_values = np.array(model.feature_importances_)

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )

        importance_df = (
            importance_df
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("Feature importance DataFrame created.")

        return importance_df.head(top_n)

    ####################################################################
    # Visualization
    ####################################################################

    def plot_confusion_matrix(
        self,
        confusion_matrix: pd.DataFrame,
        title: str = "Confusion Matrix",
    ) -> plt.Figure:
        """
        Plot the confusion matrix using matplotlib.

        Parameters
        ----------
        confusion_matrix : pd.DataFrame
            DataFrame representing the confusion matrix.

        title : str
            Plot title.

        Returns
        -------
        plt.Figure
            The generated figure.
        """

        logger.info("Plotting confusion matrix...")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(confusion_matrix.values, cmap="Blues", aspect="auto")

        ax.set_title(title)
        ax.set_xticks(np.arange(confusion_matrix.shape[1]))
        ax.set_yticks(np.arange(confusion_matrix.shape[0]))
        ax.set_xticklabels(confusion_matrix.columns)
        ax.set_yticklabels(confusion_matrix.index)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    int(confusion_matrix.iat[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        output_path = self.plots_dir / "confusion_matrix.png"
        fig.savefig(output_path, dpi=200)

        logger.info("Confusion matrix saved to %s", output_path)

        return fig

    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
    ) -> plt.Figure:
        """
        Plot the receiver operating characteristic curve.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_scores : np.ndarray
            Predicted positive-class probabilities.

        title : str
            Plot title.

        Returns
        -------
        plt.Figure
            The generated figure.
        """

        logger.info("Plotting ROC curve...")

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random chance"
        )
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        fig.tight_layout()

        output_path = self.plots_dir / "roc_curve.png"
        fig.savefig(output_path, dpi=200)

        logger.info("ROC curve saved to %s", output_path)

        return fig

    def plot_precision_recall(
        self,
        y_true: pd.Series,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
    ) -> plt.Figure:
        """
        Plot the precision-recall curve.

        Parameters
        ----------
        y_true : pd.Series
            True target values.

        y_scores : np.ndarray
            Predicted positive-class probabilities.

        title : str
            Plot title.

        Returns
        -------
        plt.Figure
            The generated figure.
        """

        logger.info("Plotting precision-recall curve...")

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        fig.tight_layout()

        output_path = self.plots_dir / "precision_recall_curve.png"
        fig.savefig(output_path, dpi=200)

        logger.info("Precision-recall curve saved to %s", output_path)

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
    ) -> tuple[plt.Figure, plt.Figure]:
        """
        Plot feature importance as a bar chart and horizontal bar chart.

        Parameters
        ----------
        importance_df : pd.DataFrame
            Feature importance DataFrame.

        Returns
        -------
        tuple[plt.Figure, plt.Figure]
            Bar chart and horizontal bar chart figures.
        """

        logger.info("Plotting feature importance charts...")

        top_features = importance_df.head(20)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(
            top_features["feature"],
            top_features["importance"],
            color="#1f77b4",
        )
        ax1.set_title("Top 20 Feature Importances")
        ax1.set_xlabel("Feature")
        ax1.set_ylabel("Importance")
        ax1.set_xticks(np.arange(len(top_features)))
        ax1.set_xticklabels(top_features["feature"], rotation=45, ha="right")
        ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        fig1.tight_layout()

        bar_path = self.plots_dir / "feature_importance_top20.png"
        fig1.savefig(bar_path, dpi=200)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.barh(
            top_features["feature"].iloc[::-1],
            top_features["importance"].iloc[::-1],
            color="#2ca02c",
        )
        ax2.set_title("Feature Importance Horizontal Bar Chart")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

        fig2.tight_layout()

        horizontal_path = self.plots_dir / "feature_importance_horizontal.png"
        fig2.savefig(horizontal_path, dpi=200)

        logger.info(
            "Feature importance charts saved to %s and %s",
            bar_path,
            horizontal_path
        )

        return fig1, fig2

    ####################################################################
    # Persistence
    ####################################################################

    def save_metrics(
        self,
        metrics: dict[str, Any],
    ) -> dict[str, Path]:
        """
        Save evaluation artifacts to disk.

        Parameters
        ----------
        metrics : dict[str, Any]
            Dictionary containing evaluation outputs.

        Returns
        -------
        dict[str, Path]
            Saved file paths.
        """

        logger.info("Saving evaluation artifacts to %s", self.output_dir)

        evaluation_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, str))
        }

        metrics_path = self.output_dir / "evaluation_metrics.csv"
        pd.DataFrame([evaluation_metrics]).to_csv(metrics_path, index=False)

        classification_path = self.output_dir / "classification_report.csv"
        metrics["classification_report"].to_csv(
            classification_path,
            index=False,
        )

        confusion_path = self.output_dir / "confusion_matrix.csv"
        metrics["confusion_matrix"].to_csv(
            confusion_path,
            index=True,
        )

        logger.info("Evaluation artifacts saved.")

        return {
            "metrics_path": metrics_path,
            "classification_report_path": classification_path,
            "confusion_matrix_path": confusion_path,
        }

    ####################################################################
    # Pipeline
    ####################################################################

    def run_pipeline(self) -> dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Returns
        -------
        dict[str, Any]
            Evaluation results and saved artifact paths.
        """

        dataset = self.load_dataset()
        model = self.load_model()
        X, y = self.prepare_data(dataset)

        predictions = self.predict(model, X)
        probabilities = self.predict_probability(model, X)
        scores = probabilities[:, 1]

        metrics = self.evaluate(y, predictions, probabilities)
        importance_df = self.feature_importance(model, self.feature_columns)

        self.plot_confusion_matrix(metrics["confusion_matrix"])
        self.plot_roc_curve(y, scores)
        self.plot_precision_recall(y, scores)
        self.plot_feature_importance(importance_df)

        saved_paths = self.save_metrics(metrics)

        result = {
            "metrics": metrics,
            "feature_importance": importance_df,
            "saved_paths": saved_paths,
            "output_dir": self.output_dir,
            "plots_dir": self.plots_dir,
            "feature_count": X.shape[1],
            "dataset_shape": dataset.shape,
            "prediction_count": len(predictions),
        }

        logger.info("Evaluation pipeline completed successfully.")

        return result
