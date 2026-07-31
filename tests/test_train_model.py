"""
Unit tests for the TrainModel class.

Run:

    pytest tests/test_train_model.py -v
"""

# from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train_model import TrainModel


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
            "match_id": [10, 10, 10, 20, 20, 20],
            "index": [1, 2, 3, 4, 5, 6],
            "feature_a": [0.1, 0.5, 0.3, 0.2, 0.8, 0.9],
            "feature_b": [1.0, 2.0, 1.5, 1.2, 2.5, 3.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def train_model(tmp_path):
    """
    TrainModel instance with temporary output paths.
    """
    dataset_path = tmp_path / "engineered_dataset.csv"
    model_path = (
        tmp_path
        / "models"
        / "random_forest_model.pkl"
    )

    return TrainModel(
        dataset_path=dataset_path,
        model_path=model_path,
        test_size=0.33,
        random_state=42,
    )


########################################################################
# Tests
########################################################################

def test_load_dataset(sample_dataset, train_model):
    sample_dataset.to_csv(train_model.dataset_path, index=False)

    dataset = train_model.load_dataset()

    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == sample_dataset.shape


def test_prepare_data(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "target" not in X.columns
    assert X.shape[0] == sample_dataset.shape[0]
    assert y.equals(sample_dataset["target"])


def test_split_data(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)

    X_train, X_test, y_train, y_test = train_model.split_data(X, y)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


def test_train_and_predict(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)
    X_train, X_test, y_train, y_test = train_model.split_data(X, y)

    model = train_model.train(X_train, y_train)
    predictions = train_model.predict(model, X_test)

    assert predictions.shape[0] == X_test.shape[0]
    assert set(predictions).issubset({0, 1})


def test_predict_proba(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)
    X_train, X_test, y_train, y_test = train_model.split_data(X, y)

    model = train_model.train(X_train, y_train)
    probabilities = train_model.predict_proba(model, X_test)

    assert probabilities.shape == (X_test.shape[0], 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_evaluate(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)
    X_train, X_test, y_train, y_test = train_model.split_data(X, y)

    model = train_model.train(X_train, y_train)
    predictions = train_model.predict(model, X_test)
    probabilities = train_model.predict_proba(model, X_test)
    metrics = train_model.evaluate(y_test, predictions, probabilities)

    assert isinstance(metrics, dict)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics


def test_feature_importance(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)
    X_train, X_test, y_train, y_test = train_model.split_data(X, y)

    model = train_model.train(X_train, y_train)
    importance = train_model.feature_importance(model, list(X.columns))

    assert isinstance(importance, pd.DataFrame)
    assert list(importance.columns) == ["feature", "importance"]
    assert (
        importance.iloc[0]["importance"]
        >= importance.iloc[-1]["importance"]
    )


def test_save_model(sample_dataset, train_model):
    X, y = train_model.prepare_data(sample_dataset)
    X_train, _, y_train, _ = train_model.split_data(X, y)

    model = train_model.train(X_train, y_train)
    path = train_model.save_model(model)

    assert path.exists()
    assert path == train_model.model_path


def test_run_pipeline(sample_dataset, train_model):
    sample_dataset.to_csv(train_model.dataset_path, index=False)

    result = train_model.run_pipeline()

    assert "model" in result
    assert "metrics" in result
    assert "feature_importance" in result
    assert "model_path" in result
    assert result["model_path"].exists()
    assert result["X_train_shape"] == (4, 2)
    assert result["X_test_shape"] == (2, 2)
