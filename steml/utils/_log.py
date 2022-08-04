import sys
import logging.config
from typing import Optional, Dict
from steml.defines import LogLevel


def config_logger(
    log_level: LogLevel,
    log_console: bool = True,
    log_file: Optional[str] = None,
    log_name: Optional[str] = None,
):
    if log_name is None:
        log_name = __name__
    logger = logging.getLogger(log_name)
    try:
        logger_config = _create_logger_config(
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        )
        logging.config.dictConfig(logger_config)
        success_msg = (
            "Logging configuration was loaded. "
            f"Log messages can be found at {log_file}."
        )
        logger.info(success_msg)
    except Exception as error:
        logger.error("Failed to load logging config!")
        raise error


def _create_logger_config(
    log_level: LogLevel,
    log_console: bool = True,
    log_file: Optional[str] = None,
) -> Dict:
    log_level = log_level.name
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "format": (
                    "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                ),
            },
        },
        "handlers": {},
        "loggers": {"": {"handlers": [], "level": log_level}},
    }
    if log_console:
        config["handlers"]["console"] = {
            "level": log_level,
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": sys.stdout,
        }
        config["loggers"][""]["handlers"].append("console")
    if log_file is not None:
        config["handlers"]["file"] = {
            "level": log_level,
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": log_file,
            "mode": "w",
        }
        config["loggers"][""]["handlers"].append("file")
    return config
