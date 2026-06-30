"""
logger.py
=========

Central logging configuration for the EPL Foul Win Probability Model project.

All modules should import the logger using:

    from src.logger import get_logger

Example
-------
logger = get_logger(__name__)

logger.info("Loading data...")
logger.warning("Missing values detected.")
logger.error("Unable to create target variable.")
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
