import logging
import sys

logging.basicConfig(
    level=logging.WARN,
    )

def get_logger(name: str):
    root = logging.getLogger(name)

    if (root.hasHandlers()):
        root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s","%H:%M:%S")

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(formatter)

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.DEBUG)
    err_handler.setFormatter(formatter)

    root.addHandler(out_handler)
    root.addHandler(err_handler)
    return root