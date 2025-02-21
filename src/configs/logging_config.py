import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging settings."""
    logging_config = {
        "level": getattr(logging, log_level.upper()),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": [logging.StreamHandler(sys.stdout)]
    }
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(logging_config["format"]))
        logging_config["handlers"].append(file_handler)
    
    logging.basicConfig(**logging_config)