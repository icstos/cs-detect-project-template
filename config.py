import os
import sys
from pathlib import Path
import torch
from my_types import RunMode, ImgSize
from config_database import r
from dataclasses import dataclass, field

RUN_MODE = RunMode.DEV

PERIOD = 'xx'

OPEN_XX = True
if PERIOD == 'xx_2':
    OPEN_XX = False

os.environ['YOLO_VERBOSE'] = 'False'


# -- START: 路径配置 --
# PROJECT_ROOT = sys.path[0]
PROJECT_ROOT = Path(__file__).parent.absolute()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

OUTPUT_DIR = Path(PROJECT_ROOT, "output")
LOG_DIR = Path(OUTPUT_DIR, "log")

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path(PROJECT_ROOT, 'xx.engine')
DATA_PATH = Path(PROJECT_ROOT, r'datasets/dataset_config.yaml')
# -- END: 路径配置 --

CLASSES = ["category_1", "category_2", "category_3"]
CLASS_CONFS = [0.25, 0.25, 0.3]


@dataclass
class Config:
    run_mode: RunMode = RUN_MODE
    period: str = PERIOD
    open_xx: bool = OPEN_XX


cfg_load = Config()
config_file = Path(PROJECT_ROOT, 'config.yaml')
if config_file.exists():
    import yaml

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        cfg_load.run_mode = RunMode(config_data.get('run_mode', RUN_MODE.value))
        cfg_load.period = config_data.get('period', PERIOD)
        cfg_load.open_xx = config_data.get('open_xx', OPEN_XX)
else:
    print(f"配置文件 {config_file} 不存在，使用默认配置。")

# --- START: torch ---
DEVICE_ID = 0
# DEVICE = torch.device("cpu")
# torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
MULTI_GPU = False

if torch.cuda.device_count() > 1:
    MULTI_GPU = True
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HALF = True
# --- END: torch ---

CONF_THRES = 0.25  # 置信度
IOU_THRES = 0.45  # IOU阈值
BATCH_SIZE = 1

CAM_IMG_SIZE = ImgSize(2048, 2048)
MODEL_IMG_SIZE = ImgSize(2048, 2048)
BATCH_SIZE = 1

MODEL_NAME = 'xx'


@dataclass
class Status:
    device: str = ''
    img_rst_deque: list = field(default_factory=list)
    img_byte_deque: list = field(default_factory=list)


STATUS_DICT: dict[str, Status] = {}


if __name__ == "__main__":
    print(f"当前运行模式: {cfg_load.run_mode}")
    print(f"当前周期: {cfg_load.period}")
    print(f"是否开启XX功能: {cfg_load.open_xx}")
