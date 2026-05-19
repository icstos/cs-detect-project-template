import os
import sys
from pathlib import Path
import torch
from my_types import RunMode, ImgSize
from config_database import r_cli

RUN_MODE = RunMode.DEV

PERIOD = 'xx'

OPEN_XX = True
if PERIOD == 'xx_2':
    OPEN_XX = False

os.environ['YOLO_VERBOSE'] = 'False'


# -- START: 路径配置 --
# PROJECT_ROOT = sys.path[0]
PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_DIR = Path(PROJECT_ROOT, "output")
LOG_DIR = Path(OUTPUT_DIR, "log")

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path(PROJECT_ROOT, 'xx.engine')
DATA_PATH = Path(PROJECT_ROOT, r'datasets/dataset_config.yaml')
# -- END: 路径配置 --

CATEGORY_NAMES = ["category_1", "category_2", "category_3"]

# --- START: torch ---
DEVICE_ID = 0
# DEVICE = torch.device("cpu")
# torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- END: torch ---

CONF_THRES = 0.25  # 置信度
IOU_THRES = 0.45  # IOU阈值
BATCH_SIZE = 1

CAM_IMG_SIZE = ImgSize(2048, 2048)
MODEL_IMG_SIZE = ImgSize(2048, 2048)
BATCH_SIZE = 1
