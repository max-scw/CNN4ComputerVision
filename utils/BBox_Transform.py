"""
    The `coco` format
        `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
    The `pascal_voc` format
        `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
    The `albumentations` format
        is like `pascal_voc`, but normalized,
        in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
    The `yolo` format
        `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
        `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
"""

import numpy as np
from typing import Union, List, Tuple


def swap_array_elements(ary):
    ary = np.asarray(ary)
    sz = ary.shape[1] if len(ary.shape) > 1 else ary.shape[0]
    idx = np.asarray(np.arange(1, sz+1) - [0, 2] * (sz//2), dtype=int)
    return ary[..., idx]


def _yolo_abs2coco(bbox) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    center coordinates to lower-left corner coordinates
    xywh -> min_wh
    yolo -> coco
    """
    bbox = np.array(bbox)
    x0, y0, w, h = bbox
    x_min = x0 - w / 2
    y_min = y0 - h / 2
    bbox = np.array([x_min, y_min, w, h])
    return tuple(__cast2int_if_applicable(bbox))


def yolo2coco(bbox_rel, image_size) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    return scale_box(_yolo_abs2coco(bbox_rel), image_size)


def _coco2yolo_abs(bbox) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    lower-left corner coordinates to center coordinates
    min_wh -> xywh
    coco -> yolo
    """
    x_min, y_min, w, h = np.asarray(bbox)

    x0 = x_min + w / 2
    y0 = y_min + h / 2
    return tuple(__cast2int_if_applicable(np.asarray([x0, y0, w, h])))


def coco2yolo(bbox, image_size) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    return norm_box(_coco2yolo_abs(bbox), image_size)


def _yolo_abs2pascal_voc(bbox_rel) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    center coordinates to corner coordinates
    xywh -> min_max
    yolo -> pascal_voc
    """
    bbox = np.array(bbox_rel)
    x0, y0, w, h = bbox
    w2 = w / 2
    h2 = h / 2
    x_min = x0 - w2  # bbox[0] -= bbox[2]
    y_min = y0 - h2  # bbox[1] -= bbox[3]
    x_max = x0 + w2  # bbox[2] = bbox[0] + 2 * bbox[2]
    y_max = y0 + h2  # bbox[3] = bbox[1] + 2 * bbox[3]
    return tuple(__cast2int_if_applicable(np.asarray([x_min, y_min, x_max, y_max])))


def yolo2pascal_voc(bbox_rel, image_size) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    return scale_box(_yolo_abs2pascal_voc(bbox_rel), image_size)


def coco2pacal_voc(bbox_rel: Union[Tuple[int, int, int, int], List[int], np.ndarray]) -> Tuple[int, int, int, int]:
    """
    lower-left corner coordinates to (global) corner coordinates
    min_wh -> min_max
    coco -> pascal_voc
    """
    bbox = np.array(bbox_rel)
    x_min, y_min, w, h = bbox
    x_max = x_min + w  # bbox[2] = bbox[0] + bbox[2]
    y_max = y_min + h  # bbox[3] = bbox[1] + bbox[3]
    bbox = np.asarray([x_min, y_min, x_max, y_max])
    return bbox


def _pascal_voc2yolo_abs(bbox_rel) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    (global) corner coordinates to center coordinates
    min_max -> xywh
    pascal_voc -> yolo
    """
    x_min, y_min, x_max, y_max = np.array(bbox_rel)
    w = (x_max - x_min)
    h = (y_max - y_min)
    x0 = x_min + w / 2
    y0 = y_min + h / 2
    return tuple(__cast2int_if_applicable(np.asarray([x0, y0, w, h])))


def pascal_voc2yolo(bbox, image_size) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    return norm_box(_pascal_voc2yolo_abs(bbox), image_size)


def pascal_voc2coco(bbox_rel) -> Tuple[int, int, int, int]:
    """
    (global) corner coordinates to lower left corner coordinates
    min_max -> min_wh
    pascal_voc -> coco
    """
    bbox = np.array(bbox_rel)
    x_min, y_min, x_max, y_max = bbox
    w = (x_max - x_min)
    h = (y_max - y_min)
    bbox[2:4] = [w, h]
    return bbox


def __cast2int_if_applicable(out: np.ndarray) -> np.ndarray:
    if not np.mod(out, 1).any():
        out = out.astype(np.int32)
    return out


def scale_box(ary: Union[tuple, list, np.ndarray], image_size: Tuple[int, int]) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    factor = list(image_size) * (len(ary) // 2)
    out = np.asarray(ary) * factor
    return tuple(__cast2int_if_applicable(out))


def norm_box(ary: Union[tuple, list, np.ndarray], image_size: Tuple[int, int]) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    factor = list(image_size) * (len(ary) // 2)
    out = np.asarray(ary) / factor
    return tuple(__cast2int_if_applicable(out))

