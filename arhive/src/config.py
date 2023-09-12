import os
import json
import logging
# from src.data_types import Parameters
from pathlib import Path


def get_project_root() -> Path:
    """"""
    return Path(__file__).parent.parent


PROJECT_ROOT_DIR = get_project_root()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

'''
with open(os.path.join(PROJECT_ROOT_DIR, "data", "config.json"), "r") as p_f:
    prms_dict = json.load(p_f)

parameters = Parameters.parse_obj(prms_dict)
'''