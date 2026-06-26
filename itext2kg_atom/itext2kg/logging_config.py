import logging
import logging.config
import sys
from typing import Optional

import langchain

import warnings
# Suppress LangChain's specific warning about structured output streaming when using ChatOpenAI with Pydantic response_format and 'json_schema' output parser.
warnings.filterwarnings(
    "ignore",
    message="Streaming with Pydantic response_format not yet supported.",
    category=UserWarning,
    module="langchain_openai.chat_models.base"
)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    langchain_level: Optional[str] = None
) -> None:
    """
    Set up logging configuration for iText2KG.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_file: Optional file path to write logs to
        console_output: Whether to output logs to console
        langchain_level: Logging level for langchain libraries. If None, defaults to WARNING (much better performance)
    """

    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Set langchain level - default to WARNING for better performance
    if langchain_level is None:
        langchain_level = "WARNING" if level == "INFO" else level
    numeric_langchain_level = getattr(logging, langchain_level.upper(), None)
    if not isinstance(numeric_langchain_level, int):
        raise ValueError(f'Invalid log level for langchain: {langchain_level}')
    
    # Create formatters
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    root_logger = logging.getLogger("itext2kg")
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure langchain logging - respect user's chosen level for better control
    #logging.getLogger("langchain").setLevel(numeric_langchain_level)
    #logging.getLogger("langchain_core").setLevel(numeric_langchain_level)
    #logging.getLogger("langchain_community").setLevel(numeric_langchain_level)
    
    # Enable langchain debug mode (logs through logging system)
    langchain.debug = False
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Name of the module (usually __name__)
        
    Returns:
        Logger instance
        
    Performance Notes:
        - LangChain loggers are set to WARNING by default when DEBUG is enabled
          to avoid excessive external library logging (which was causing 2x slowdown)
        - To see full DEBUG output from langchain, call setup_logging with langchain_level="DEBUG"
    """
    return logging.getLogger(f"itext2kg.{name}")

