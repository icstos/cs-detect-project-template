from enum import Enum
from dataclasses import dataclass


class RunMode(str, Enum):
    DEV = 'dev'  # 开发模式
    TEST = 'test'  # 测试模式
    RELEASE = 'release'  # 发布模式


@dataclass
class ImgSize:
    width: int = -1
    height: int = -1


@dataclass
class Box:
    x1: int = -1
    y1: int = -1
    x2: int = -1
    y2: int = -1
    category_id: int = -1
    category: str = ''
    msg: str = ''
    conf: float = 0.0
