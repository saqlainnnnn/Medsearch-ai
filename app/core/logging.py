import logging
import sys

from app.config import settings


def setup_logging() -> None:
    """
    Configure application-wide logging.
    """

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )