from __future__ import annotations

import logging


def configure_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a simple stdout logger with a consistent formatter.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
