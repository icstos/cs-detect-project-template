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
PROJECT_DIR = sys.path[0]
MODEL_PATH = Path(PROJECT_DIR, 'xx.engine')
classes = ["category_1", "category_2", "category_3"]

DATA_PATH = Path(PROJECT_DIR, r'dataset/dataset_config.yaml')
# -- END: 路径配置 --

DEVICE_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONF_THRES = 0.25  # 置信度
IOU_THRES = 0.45  # IOU阈值
BATCH_SIZE = 1

CAM_IMG_SIZE = ImgSize(2048, 2048)
MODEL_IMG_SIZE = ImgSize(2048, 2048)
BATCH_SIZE = 1
