import logging
import sys
from logging.handlers import RotatingFileHandler

LOG_FILE = "/tmp/abu_vision.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 2


def setup_logger():
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # rotating file — keeps last 10 MB total
    fh = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES,
                             backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)

    # silence noisy libs
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return logging.getLogger("abu")
