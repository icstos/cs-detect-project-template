import cv2
import sys
import traceback
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from utils.time import Time
from utils.logger import Logger
import config

LOGGER = Logger("xx.log", name="xx", run_mode=config.RUN_MODE)
LOGGER.info('--- Start Running ---')


MODEL = YOLO(model=config.MODEL_PATH, task="detect")


def get_params_list(model_name):

    xx_thres = config.r.hget(f'cfg_{model_name}', 'xx_thres')
    xx_thres = 0.3 if xx_thres is None else float(xx_thres)

    return [xx_thres]


def get_img_by_redis(model_name):
    img_bytes = config.r.brpop(model_name)[1]
    img_msg = config.r.brpop(model_name)[1]

    return [img_bytes, img_msg]


@Time.timer
def pre_process(img_cv2):
    # 预处理
    pass


@Time.timer
def post_process(rst_list):
    # 后处理
    pass


@Time.timer
def xx_predictor(img: np.ndarray) -> list:
    ng_obj_list = []
    # 核心的推理逻辑
    predictions = MODEL.predict(
        img,
        device=config.DEVICE,
        imgsz=config.MODEL_IMG_SIZE.width,
        conf=config.CONF_THRES,
        iou=config.IOU_THRES,
        # save = True,
        # save_txt = False,
        # line_width = 2,
        # half = False,
        rect=True,
    )
    for _ in predictions:
        final_result_list = []
        box_array = _.boxes.xywh.detach().cpu().numpy()
        box_conf = _.boxes.conf.detach().cpu().numpy()
        box_class = _.boxes.cls.detach().cpu().numpy()
    for box, conf, cls in zip(box_array, box_conf, box_class):
        tmp_dict = {
            "x": float(np.float32(box[0])),
            "y": float(np.float32(box[1])),
            "w": float(np.float32(box[2])),
            "h": float(np.float32(box[3])),
            "conf": float(np.float32(conf)),
            "cls": int(cls),
        }
        final_result_list.append(tmp_dict)
    return final_result_list


def main_detect(
    img: np.ndarray,
    width,
    height,
    stride,
    product_id: str = "unknown",
    logger: Logger = LOGGER,
) -> dict:
    logger.info(f'--- Processing Product ID: {product_id} ---')
    try:
        # --- 业务逻辑1：记录传入的参数 ---
        logger.info(
            f'start detect: {product_id},  {width}, {height}, {sys.getsizeof(img)}  '
        )

        # --- 业务逻辑3：兼容其他大小的图像传入 ---
        if (
            img.shape[-1] != config.CAM_IMG_SIZE.width
            or img.shape[-2] != config.CAM_IMG_SIZE.height
        ):
            img = cv2.resize(
                img, (config.CAM_IMG_SIZE.width, config.CAM_IMG_SIZE.height)
            )
            logger.warning(
                f'img_origin input_img_size != {config.CAM_IMG_SIZE.width} {config.CAM_IMG_SIZE.height}'
            )
    except Exception:
        logger.error(f'main_detect: {product_id}：{traceback.format_exc()}')

    try:
        if len(img.shape) == 3:
            img_detect = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final_result_dict = {}

        ng_obj_list = xx_predictor(img_detect)
        final_result_dict["result"] = "NG" if len(ng_obj_list) != 0 else "OK"
        final_result_dict["box_nums"] = len(ng_obj_list)
        final_result_dict["obj"] = ng_obj_list
    except Exception:
        print(f"Error in main_detect: {traceback.format_exc()}")
        final_result_dict["result"] = "ERROR"
        final_result_dict["obj"] = []
        return final_result_dict

    return final_result_dict


def main_detect_pythonnet(
    img_ptr, width, height, stride, product_id: str = "unknown", logger: Logger = LOGGER
) -> dict:
    import ctypes

    # 指针数据转换为numpy
    img_cv2 = np.ctypeslib.as_array(
        (stride * height * ctypes.c_uint8).from_address(img_ptr)
    ).reshape(height, stride)
    main_detect_result = main_detect(img_cv2, width, height, stride, product_id, logger)
    return main_detect_result


def main():
    # Example usage
    img_path = "path_to_your_image.jpg"
    img = cv2.imread(img_path)
    if img is None:
        LOGGER.info(f"Failed to read image from {img_path}")
        return
    result = main_detect(
        img, img.shape[1], img.shape[0], img.strides[0], "product_00001"
    )
    LOGGER.info(result)
    return result


if __name__ == "__main__":
    main()
