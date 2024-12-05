import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = Path(ROOT_DIR).parent.absolute()
MQLIB_DIR = os.path.join(PARENT_DIR, 'MQLib')
DATA_DIR = '/lustre/scratch127/qpg/jc59/data'
OUT_DIR = '/lustre/scratch127/qpg/jc59/out'
