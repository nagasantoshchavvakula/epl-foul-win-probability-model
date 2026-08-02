"""
Streamlit application for the EPL Foul Win Probability Predictor.

This app loads the trained Random Forest model and evaluation
artifacts from the project and provides an interactive interface
to compute engineered features and predict foul-win probability
for a single event using the project's `FeatureEngineer` logic.

Author: Generated for project
"""
from __future__ import annotations
from src.feature_engineering import FeatureEngineer
from src.logger import get_logger
from pathlib import Path
import math
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import sys

# Ensure project root is importable when running streamlit from repo root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EPL Foul Win Probability Predictor",
    page_icon="⚽",
    layout="wide",
)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CM_PNG = PLOTS_DIR / "confusion_matrix.png"
ROC_PNG = PLOTS_DIR / "roc_curve.png"
PR_PNG = PLOTS_DIR / "precision_recall_curve.png"
FI_PNG = PLOTS_DIR / "feature_importance_top20.png"

# ---------------------------------------------------------------------------
# Caching resources
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("Loading engineered dataset from %s", path)
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    logger.info("Loading trained model from %s", path)
    return joblib.load(path)


@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_feature_importance(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_single_event_df(
    x: float,
    y: float,
    minute: int,
    second: int,
    possession: int,
    event_type_label: str,
    match_id: int = 0,
    index: int = 0,
) -> pd.DataFrame:
    """Create a minimal DataFrame that FeatureEngineer expects.

    The FeatureEngineer requires columns listed in REQUIRED_COLUMNS.
    We provide sensible defaults for non-interactive fields.
    """

    df = pd.DataFrame(
        [
            {
                "location": [x, y],
                "type.name": event_type_label,
                "minute": int(minute),
                "second": int(second),
                "match_id": int(match_id),
                "possession": int(possession),
                "index": int(index),
                "target": 0,
            }
        ]
    )

    return df


def align_features(
    single_row: pd.DataFrame,
    reference: pd.DataFrame,
    feature_engineer: FeatureEngineer
    ) -> pd.DataFrame:
    """Create a full feature vector aligned to the model's trained features.

    Strategy:
    - Engineer features for the single_row using FeatureEngineer.
    - Use the processed reference dataset to determine the numeric feature set
      and to compute fallback values (median) for missing features.
    """

    # Engineer features on single row
    engineered = feature_engineer.create_features(
        single_row
        )

    # Prepare numeric features from reference to learn feature names and medians
    numeric_ref = reference.select_dtypes(include=[np.number]).copy()

    # Remove identifiers and target to mirror training pipeline
    # Use same exclude list as ModelEvaluator.EXCLUDE_COLUMNS indirectly by column names
    # Build feature list from numeric_ref (this mimics training selection)
    feature_columns = [c for c in numeric_ref.columns if c not in {
        "target", "match_id", "id", "index", "Unnamed: 0"
    }]

    # Create feature vector with median fallbacks
    medians = numeric_ref[feature_columns].median()

    feature_vector = pd.DataFrame(columns=feature_columns)

    # For each feature, if available in engineered row use it, else use median
    row = {}
    for col in feature_columns:
        if col in engineered.columns:
            val = engineered.iloc[0][col]
            # Replace NaN with median
            if pd.isna(val):
                val = medians[col]
        else:
            val = medians[col]

        # Ensure numeric
        try:
            row[col] = float(val)
        except Exception:
            # fallback to 0.0
            row[col] = float(medians.get(col, 0.0))

    feature_vector = pd.DataFrame([row])

    return feature_vector, engineered


def plot_horizontal_feature_importance(df: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    top = df.head(20).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

def main():
    # st.sidebar.title("⚽ EPL Foul Win Probability Predictor")
    # st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.write(
        "This app predicts the probability that a player will win a foul after receiving or recovering possession in an EPL match.")

    st.sidebar.markdown("**Model**")
    st.sidebar.info("Random Forest Classifier")

    st.sidebar.markdown("**Dataset**")
    st.sidebar.write("StatsBomb Open Data")
    st.sidebar.write("Matches: 380")
    st.sidebar.write("Events: 1.31 Million")

    st.sidebar.markdown("---")
    # st.sidebar.markdown("**Author**")
    # st.sidebar.write("Nagasantosh Chavvakula")

    threshold = st.sidebar.slider(
        "Prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    # st.sidebar.markdown("---")
    # st.sidebar.caption("Built with Python, scikit-learn, and Streamlit")

    # Header
    st.title("⚽ EPL Foul Win Probability Predictor")
    st.markdown(
        "Use the trained Random Forest model to estimate the probability a player will win a foul after receiving or recovering possession. ``No manual feature engineering required.``"
    )

    # Load resources
    try:
        with st.spinner("Loading model and data..."):
            dataset = load_dataset(DATA_PATH)
            model = load_model(MODEL_PATH)
            metrics_df = load_metrics(METRICS_PATH) if METRICS_PATH.exists() else None
            fi_df = load_feature_importance(FEATURE_IMPORTANCE_PATH) if FEATURE_IMPORTANCE_PATH.exists() else None
            feature_engineer = FeatureEngineer()
    except Exception as exc:
        st.error(f"Error loading project resources: {exc}")
        logger.error("Error loading resources: %s", exc)
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Performance", "Feature Importance", "About"])

    # -------------------------
    # Prediction Tab
    # -------------------------
    with tab1:
        st.subheader("Input Event")

        c1, c2, c3 = st.columns([2, 2, 2])

        with c1:
            event_type = st.selectbox(
                "Event Type",
                options=["Ball Receipt*", "Ball Recovery"],
                format_func=lambda x: x,
            )

            x_coord = st.slider("X Coordinate", 0.0, 120.0, 60.0, step=0.5)
            y_coord = st.slider("Y Coordinate", 0.0, 80.0, 40.0, step=0.5)

        with c2:
            minute = st.slider("Minute", 0, 120, 45)
            second = st.slider("Second", 0, 59, 0)
            possession = st.number_input("Possession Number", min_value=1, value=1, step=1)

        with c3:
            st.markdown("**Event preview**")
            st.write(f"Type: {event_type}")
            st.write(f"Location: ({x_coord:.1f}, {y_coord:.1f})")
            st.write(f"Time: {minute}m {second}s")
            st.write(f"Possession: {possession}")

        st.markdown("---")

        predict_col = st.container()
        with predict_col:
            if st.button("Predict Probability"):
                try:
                    with st.spinner("Engineering features and predicting..."):
                        single = build_single_event_df(
                            x_coord, y_coord, minute, second, possession, event_type
                        )

                        feature_vector, engineered_row = align_features(single, dataset, feature_engineer)

                        # show engineered feature summary
                        st.subheader("Engineered Features")
                        ecols = ["pitch_zone", "distance_to_goal", "goal_angle", "match_time_seconds", "event_type"]
                        engineered_display = {c: engineered_row.iloc[0].get(c, None) for c in ecols}
                        st.metric("Pitch Zone", engineered_display.get("pitch_zone"))
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Distance to Goal", f"{engineered_display.get('distance_to_goal'):.2f}")
                        col2.metric("Goal Angle", f"{engineered_display.get('goal_angle'):.2f}")
                        col3.metric("Match Time (s)", int(engineered_display.get('match_time_seconds')))
                        col4.metric("Encoded Event Type", int(engineered_display.get('event_type')))

                        # Progress simulation
                        progress = st.progress(0)
                        for pct in range(0, 101, 20):
                            progress.progress(pct)
                            time.sleep(0.05)

                        proba = model.predict_proba(feature_vector)[0][1]

                        st.markdown("---")
                        st.subheader("Prediction Result")

                        prob_pct = proba * 100.0
                        st.metric(label="Predicted probability (%)", value=f"{prob_pct:.2f}%")

                        if proba >= threshold:
                            st.success("🟢 High Chance of Winning a Foul")
                        else:
                            st.warning("🔴 Low Chance of Winning a Foul")

                        # Show raw probabilities and small explanation
                        with st.expander("Prediction details"):
                            st.write({"probability": float(proba)})
                            st.caption("Probabilities are model estimates and may require calibration for decision-making.")

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    logger.error("Prediction error: %s", exc)

    # -------------------------
    # Model Performance Tab
    # -------------------------
    with tab2:
        st.subheader("Model Performance")
        if metrics_df is None:
            st.warning("No metrics file found in outputs/; run evaluation pipeline first.")
        else:
            # show key metrics as metric cards
            row = metrics_df.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{row.get('accuracy', math.nan):.4f}")
            m2.metric("Precision", f"{row.get('precision', math.nan):.4f}")
            m3.metric("Recall", f"{row.get('recall', math.nan):.4f}")
            m4.metric("F1 Score", f"{row.get('f1_score', math.nan):.4f}")
            # roc and pr maybe stored under roc_auc and average_precision
            m5.metric("ROC AUC", f"{row.get('roc_auc', row.get('roc_auc_score', math.nan)):.4f}")
            # PR AUC
            pr_col = st.columns(1)[0]
            pr_col.metric("PR AUC", f"{row.get('average_precision', row.get('pr_auc', math.nan)):.4f}")

            st.markdown("---")
            # Display plots if present
            p1, p2, p3 = st.columns(3)
            if CM_PNG.exists():
                p1.image(str(CM_PNG), caption="Confusion Matrix", width="stretch")
            else:
                p1.info("Confusion matrix image not found.")

            if ROC_PNG.exists():
                p2.image(str(ROC_PNG), caption="ROC Curve", width="stretch")
            else:
                p2.info("ROC curve image not found.")

            if PR_PNG.exists():
                p3.image(str(PR_PNG), caption="Precision-Recall Curve", width="stretch")
            else:
                p3.info("Precision-Recall image not found.")

    # -------------------------
    # Feature Importance Tab
    # -------------------------
    with tab3:
        st.subheader("Feature Importance")
        if fi_df is None:
            st.warning("No feature importance file found in outputs/; run evaluation pipeline first.")
        else:
            st.markdown("Top 20 features used by the model")
            st.dataframe(fi_df.head(20).reset_index(drop=True))

            fig = plot_horizontal_feature_importance(fi_df)
            st.pyplot(fig)

    # -------------------------
    # About Tab
    # -------------------------
    with tab4:
        st.subheader("About")
        st.markdown(
            "This project implements an end-to-end machine learning pipeline to predict the probability that a player will win a foul after receiving or recovering possession."
        )

        st.markdown("**Problem statement**: Estimate foul-win probability for event-level possessions in EPL matches.")

        st.markdown("**Dataset**: StatsBomb open event data (engineered for model input).")

        st.markdown("**Target variable**: `target` — indicates whether the event resulted in a foul win.")

        st.markdown("**Model**: Random Forest Classifier (pre-trained).")

        st.markdown("**Pipeline**: Data Understanding → Target Creation → Feature Engineering → Model Training → Evaluation")

        st.markdown("**Technologies**: Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit")

        with st.expander("Project author and notes"):
            st.write("Author: Nagasantosh Chavvakula")
            st.write("Use this dashboard to quickly estimate foul-win probability for tactical analysis and scouting.")

    # Footer
    st.markdown("---")
    st.caption("EPL Foul Win Probability Model — interactive demo")


if __name__ == "__main__":
    main()
