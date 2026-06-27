"""
config.py
=========

Central configuration for the EPL Foul Win Probability Model project.

This module defines project-wide paths so all modules use the same
directory structure regardless of where they are executed from.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------

# epl-foul-prediction/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"

PLOTS_DIR = OUTPUT_DIR / "plots"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# ---------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"