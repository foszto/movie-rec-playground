# src/configs/logging_config.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", 
                 log_file: Path = None,
                 clear_handlers: bool = True):
    """Setup logging configuration."""
    
    # Alapformátum definiálása
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Root logger konfigurálása
    root_logger = logging.getLogger()
    
    if clear_handlers:
        # Töröljük a meglévő handlereket
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (ha szükséges)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log level beállítása
    level = getattr(logging, log_level.upper())
    root_logger.setLevel(level)
    
    # Bizonyos modulok logolásának csökkentése
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    # Debug üzenet a konfiguráció megerősítéséhez
    root_logger.debug("Logging setup complete")