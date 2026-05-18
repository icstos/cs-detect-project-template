from pathlib import Path
import numpy as np
import cv2
import config

import traceback


def get_params_list(model_name):

    xx_thres = config.r_cli.hget(f'cfg_{model_name}', 'xx_thres')
    xx_thres = 0.3 if xx_thres is None else float(xx_thres)

    return [xx_thres]


def xx_predictor(img: np.ndarray) -> list:
    ng_obj_list = []
    return ng_obj_list


def main_detect(img: np.ndarray):
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


if __name__ == "__main__":
    pass
