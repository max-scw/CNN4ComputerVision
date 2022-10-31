from typing import Union, List
import pathlib as pl

import logging
from datetime import datetime
from keras.callbacks import Callback

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.colors as mcolors
from random import shuffle


class Log:
    _filename = None

    def __init__(self, file_name: str = None, print_message: bool = False, level=logging.INFO):
        if file_name:
            self._filename = file_name.strip("*.") + ".log"
            # turn logging on
            logging.basicConfig(filename=self._filename, level=level)
        self._print_message = print_message

    def log(self, message: str, print_message: bool = False):
        # build message
        msg = f"{datetime.now()}: {message}"
        if self._filename:
            logging.info(msg)
        if self._print_message or print_message:
            print(msg)


class StayAliveLoggingCallback(Callback):
    _log_epoch_step = 1
    _external_info = None

    def __init__(self, log_file_name: str = None, epoch_step: int = 1, info: str = None):
        super().__init__()
        self._log_file_name = log_file_name
        self._log_epoch_step = epoch_step
        self._external_info = info
        self._log = Log(file_name=log_file_name, print_message=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self._log_epoch_step) == 0:
            msg = ""
            if self._external_info:
                msg += f"{self._external_info} , "
            msg += f"epoch {epoch}: " +\
                  ", ".join([f"{ky} {round(logs[ky], 4)}" for ky in logs.keys()])
            self._log.log(msg)


def draw_box(image: Image.Image,
             bbox: List[np.ndarray],
             bbox_scores: np.ndarray,
             bbox_classes: np.ndarray,
             color_text: tuple[int, int, int] = (0, 0, 0),
             classes: list = None,
             th_score: float = 0.5
             ) -> Image.Image:
    assert len(bbox) == len(bbox_scores) == len(bbox_classes)
    # apply threshold to exclude uninteresting bounding boxes
    lg = bbox_scores >= th_score
    bbox = bbox[lg]
    bbox_scores = bbox_scores[lg]
    bbox_classes = bbox_classes[lg]

    # create image drawing instance
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # set font (size)
    # (ImageDraw's default font is a bitmap font, and therefore it cannot be scaled. For scaling, you need to select a true-type font.)
    font = ImageFont.truetype(font="../utils/FiraMono-Medium.otf",
                              size=np.floor(0.02 * image.size[1] + 0.5).astype("int32")
                              )
    line_thickness = np.sum(image.size) // 500

    # set colors for classes
    if classes is None:
        classes = np.unique(np.asarray(bbox_classes).flatten()).tolist()

    n_classes = len(classes)
    if n_classes < 10:
        colors = list(mcolors.TABLEAU_COLORS.values())
    else:
        colors = list(mcolors.CSS4_COLORS.values())
        shuffle(colors)

    def _draw_box(box, score, class_prd):
        # box coordinates
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype("int32"))
        left = max(0, np.floor(left + 0.5).astype("int32"))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
        right = min(image.size[0], np.floor(right + 0.5).astype("int32"))

        # label / info
        info = f"{class_prd} {score:.2f}"
        # get size of the label flag
        info_size = draw.textsize(info, font)

        # coordinates for info label
        if top - info_size[1] >= 0:
            text_origin = np.array([left, top - info_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        color = colors[classes.index(class_prd)]
        # draw box (workaround: thickness as multiple rectangles)
        for j in range(line_thickness):
            draw.rectangle((left + j, top + j, right - j, bottom - j), outline=color)
        # draw.rectangle((left, top, right, bottom), outline=color)
        # add label to box
        draw.rectangle((tuple(text_origin), tuple(text_origin + info_size)), fill=color)
        draw.text(tuple(text_origin), info, fill=color_text, font=font)

    for box, score, class_prd in zip(bbox, bbox_scores, bbox_classes):
        _draw_box(box, score, class_prd)

    return img