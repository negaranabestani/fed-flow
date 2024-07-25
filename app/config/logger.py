import logging

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s:%(lineno)d - %(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
fed_logger = logging.getLogger(__name__)
