import cv2
import traceback
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import config

MODEL = YOLO(model=config.MODEL_PATH, task="detect")


def get_params_list(model_name):

    xx_thres = config.r_cli.hget(f'cfg_{model_name}', 'xx_thres')
    xx_thres = 0.3 if xx_thres is None else float(xx_thres)

    return [xx_thres]


def xx_predictor(img: np.ndarray) -> list:
    ng_obj_list = []
    return ng_obj_list


def main_detect(img: np.ndarray, product_id: str = "unknown") -> dict:
    print(f'--- Processing Product ID: {product_id} ---')
    try:
        if len(img.shape) == 3:
            img_detect = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final_result_dict = {}

        ng_obj_list = xx_predictor(img_detect)
        final_result_dict["result"] = "NG" if len(ng_obj_list) != 0 else "OK"
        final_result_dict["obj"] = ng_obj_list
    except Exception:
        print(f"Error in main_detect: {traceback.format_exc()}")
        final_result_dict["result"] = "ERROR"
        final_result_dict["obj"] = []
        return final_result_dict

    return final_result_dict


def main():
    # Example usage
    img_path = "path_to_your_image.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image from {img_path}")
        return
    result = main_detect(img)
    print(result)


if __name__ == "__main__":
    main()
