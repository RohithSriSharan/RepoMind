"""
Centralized logging configuration for RepoMind.
Every module imports get_logger() from here instead of configuring
logging independently — ensures consistent format and a single log file.
"""
import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers if called multiple times
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        # Console handler - see logs live while developing
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler - permanent record, this is our monitoring story
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger("test")
    logger.info("Logging setup working correctly.")
  