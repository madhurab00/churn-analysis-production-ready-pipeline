import logging
from config import get_logging_config

def get_logger(name=__name__):
    log_cfg = get_logging_config()

    level_str = log_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, level_str, logging.INFO)  

    log_format = log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_file = log_cfg.get("file", "pipeline.log")

    logging.basicConfig(
                        level=log_level,
                        format=log_format,
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ]
                        )

    logger = logging.getLogger(name)
    return logger