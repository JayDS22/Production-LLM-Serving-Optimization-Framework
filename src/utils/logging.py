"""
Logging configuration for LLM serving framework.
"""
import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
import os

# Remove default handler
logger.remove()


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logger:
    """Setup logger with custom configuration.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Get level from environment or parameter
    log_level = os.getenv("LOG_LEVEL", level).upper()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler if specified
    if log_file or os.getenv("LOG_FILE"):
        log_path = log_file or os.getenv("LOG_FILE")
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="500 MB",
            retention="10 days",
            compression="zip"
        )
    
    return logger.bind(name=name)
