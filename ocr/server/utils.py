"""Utils."""
import logging
import logging.config
import os


def get_logger(config: str = "logging.conf") -> logging.Logger:
    """Get logger."""
    logging.config.fileConfig(config)
    logger = logging.getLogger()
    return logger
