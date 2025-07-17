"""Logging configuration for DDoS.AI platform"""
import logging
import logging.config
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# JSON log format
JSON_LOG_FORMAT = {
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s",
    "name": "%(name)s",
    "message": "%(message)s",
    "module": "%(module)s",
    "function": "%(funcName)s",
    "line": "%(lineno)d"
}


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, fmt_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.fmt_dict = fmt_dict or JSON_LOG_FORMAT
    
    def format(self, record: logging.LogRecord) -> str:
        record_dict = {}
        for key, value in self.fmt_dict.items():
            try:
                record_dict[key] = value % record.__dict__
            except (KeyError, TypeError):
                record_dict[key] = value
        
        # Add exception info if available
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key.startswith("_") and not key.startswith("__"):
                record_dict[key[1:]] = value
        
        return json.dumps(record_dict)


def configure_logging(log_level: str = None, json_logs: bool = False, log_file: str = None) -> None:
    """Configure logging for the application"""
    # Determine log level
    log_level = log_level or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    log_level = getattr(logging, log_level.upper())
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if json_logs:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Set log levels for specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {logging.getLevelName(log_level)}")
    if json_logs:
        logger.info("JSON logging enabled")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


class StructuredLogger:
    """Structured logger for DDoS.AI platform"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log a message with structured data"""
        # Add extra fields with underscore prefix
        extra = {f"_{k}": v for k, v in kwargs.items()}
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log a debug message"""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log an info message"""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log a warning message"""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log an error message"""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log a critical message"""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, exc_info=True, **kwargs) -> None:
        """Log an exception"""
        self._log(logging.ERROR, msg, exc_info=exc_info, **kwargs)