import logging
import pandas as pd
from datetime import datetime

#simple logging tool to track actions
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler, need have following log folder structure
handler = logging.FileHandler('logs/tracked_logs.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

# create a stream handler
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

logger.propagate = False

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')