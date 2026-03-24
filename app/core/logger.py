"""
Loguru-based logging configuration.

Two handlers:
  - Console: colourised, DEBUG level
  - File: logs/app.log, rotated at 10 MB, kept 7 days
"""

from loguru import logger
import sys
from pathlib import Path

# Remove default handler
logger.remove()

# Console handler — coloured
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan> | {message}",
    level="DEBUG",
    colorize=True,
)

# File handler — rotation at 10 MB, kept for 7 days
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    encoding="utf-8",
)

__all__ = ["logger"]
