import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] - (%(module)s) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


lx_log = logging.getLogger("luxia")
