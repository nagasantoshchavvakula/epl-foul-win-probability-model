"""
Unit tests for the ModelEvaluator class.

Run:

    pytest tests/test_evaluate_model.py -v
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluate_model import ModelEvaluator


########################################################################
# Fixtures
########################################################################

@pytest.fixture
def sample_dataset():
    """
    Create a simple dataset with numeric features and a binary target.
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "match_id": [100, 100, 100, 200, 200, 200],
            "index": [1, 2, 3, 4, 5, 6],
            "feature_a": [0.2, 0.4, 0.3, 0.8, 0.7, 0.5],
            "feature_b": [5.0, 4.0, 6.0, 7.0, 8.0, 9.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def evaluator(tmp_path):
    """
    Create a ModelEvaluator instance configured to use temporary output paths.
    """
    dataset_path = tmp_path / "engineered_dataset.csv"
    model_path = tmp_path / "random_forest_model.pkl"
    output_dir = tmp_path / "outputs"
    plots_dir = tmp_path / "plots"

    return ModelEvaluator(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir=output_dir,
        plots_dir=plots_dir,
    )


@pytest.fixture
def trained_model(sample_dataset, tmp_path):
    """
    Create and persist a simple RandomForestClassifier for evaluation.
    """
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_dataset[["feature_a", "feature_b"]]
    y = sample_dataset["target"]
    model.fit(X, y)

    model_path = tmp_path / "random_forest_model.pkl"
    joblib.dump(model, model_path)

    return model, model_path


########################################################################
# Tests
########################################################################

def test_load_dataset(sample_dataset, evaluator):
    sample_dataset.to_csv(evaluator.dataset_path, index=False)

    dataset = evaluator.load_dataset()

    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == sample_dataset.shape


def test_load_model(trained_model, evaluator):
    _, model_path = trained_model
    evaluator.model_path = model_path

    model = evaluator.load_model()

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_prepare_data(sample_dataset, evaluator):
    X, y = evaluator.prepare_data(sample_dataset)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "target" not in X.columns
    assert X.shape[0] == sample_dataset.shape[0]
    assert y.equals(sample_dataset["target"])
    assert X.shape[1] == 2


def test_predict_and_predict_probability(
    trained_model, sample_dataset, evaluator
):
    model, _ = trained_model
    X = sample_dataset[["feature_a", "feature_b"]]

    predictions = evaluator.predict(model, X)
    probabilities = evaluator.predict_probability(model, X)

    assert predictions.shape[0] == X.shape[0]
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (X.shape[0], 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_evaluate_metrics(trained_model, sample_dataset, evaluator):
    model, _ = trained_model
    X = sample_dataset[["feature_a", "feature_b"]]
    y = sample_dataset["target"]

    predictions = evaluator.predict(model, X)
    probabilities = evaluator.predict_probability(model, X)
    metrics = evaluator.evaluate(y, predictions, probabilities)

    assert isinstance(metrics, dict)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["classification_report"], pd.DataFrame)
    assert isinstance(metrics["confusion_matrix"], pd.DataFrame)


def test_classification_report_df(sample_dataset, evaluator):
    y_true = sample_dataset["target"]
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    report_df = evaluator.classification_report_df(y_true, y_pred)

    assert isinstance(report_df, pd.DataFrame)
    assert "class" in report_df.columns
    assert "precision" in report_df.columns
    assert "recall" in report_df.columns


def test_confusion_matrix_df(sample_dataset, evaluator):
    y_true = sample_dataset["target"]
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    matrix_df = evaluator.confusion_matrix_df(y_true, y_pred)

    assert isinstance(matrix_df, pd.DataFrame)
    assert matrix_df.shape == (2, 2)


def test_feature_importance_dataframe(
    trained_model,
    sample_dataset,
    evaluator
):
    model, _ = trained_model
    feature_names = ["feature_a", "feature_b"]

    importance_df = evaluator.feature_importance(model, feature_names)

    assert isinstance(importance_df, pd.DataFrame)
    assert list(importance_df.columns) == ["feature", "importance"]
    assert importance_df.shape[0] <= 2


def test_save_metrics(sample_dataset, trained_model, evaluator):
    model, _ = trained_model
    X = sample_dataset[["feature_a", "feature_b"]]
    y = sample_dataset["target"]

    predictions = evaluator.predict(model, X)
    probabilities = evaluator.predict_probability(model, X)
    metrics = evaluator.evaluate(y, predictions, probabilities)

    saved_paths = evaluator.save_metrics(metrics)

    assert Path(saved_paths["metrics_path"]).exists()
    assert Path(saved_paths["classification_report_path"]).exists()
    assert Path(saved_paths["confusion_matrix_path"]).exists()


def test_run_pipeline(sample_dataset, trained_model, evaluator):
    model, model_path = trained_model
    evaluator.model_path = model_path

    sample_dataset.to_csv(evaluator.dataset_path, index=False)
    result = evaluator.run_pipeline()

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "feature_importance" in result
    assert "saved_paths" in result
    assert result["output_dir"].exists()
    assert result["plots_dir"].exists()
    assert result["dataset_shape"] == sample_dataset.shape
    assert result["prediction_count"] == sample_dataset.shape[0]
    assert result["feature_count"] == 2
