import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pathlib as pl
import random
import re

from typing import Union, Tuple, List

import albumentations as A

from utils import voith_colors


class BBoxTransform:
    """
        format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.
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
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
    """
    def __int__(self):
        pass

    @staticmethod
    def __check_format(patterns: List[str], input_string: str) -> bool:
        flag = False
        for el in patterns:
            m = re.search(el, input_string, re.IGNORECASE)
            if m:
                flag = True
                break
        return flag

    def __get_format_mode(self, input_string: str) -> (int, bool, bool):
        pat_space = r"(\\|\-|\s|_)"
        patterns_center_width = ["xywh", "yolo", "x0y0wh"]
        patterns_min_max = ["min(_|\-)?max", "pascal(_voc)?", "x_min, y_min, x_max, y_max", "xyxy", "x1y1x2y2",
                            r"min(\|-s)max", "corner", "albumentations"]
        patterns_min_width = ["min_wh", "coco", "x_min, y_min, width, height", "labelstudio"]

        patterns_norm = [rf"{pat_space}rel", rf"{pat_space}norm", rf"{pat_space}scaled", "albumentations", "yolo",
                         "labelstudio"]
        patterns_yx = [rf"{pat_space}yx"]

        mode_id = -1
        for i, pat in enumerate([patterns_center_width, patterns_min_max, patterns_min_width]):
            if self.__check_format(pat, input_string):
                mode_id = i
                break

        flag_rel = self.__check_format(patterns_norm, input_string)
        flag_yx = self.__check_format(patterns_yx, input_string)
        return mode_id, flag_rel, flag_yx

    def __xy2yx(self, bbox: np.ndarray) -> np.ndarray:
        # xy => yx
        bbox_new = bbox.copy()
        bbox_new[1::2] = bbox[0::2]
        bbox_new[0::2] = bbox[1::2]
        return bbox_new

    def transform(self, bbox: Union[List, Tuple, np.ndarray],
                  format_to: str,
                  format_from: str,
                  image_size: Tuple[int, int]) -> Tuple:
        bbox = np.array(bbox)

        # extract / analyze format strings
        mode_from, flag_rel_from, flag_yx_from = self.__get_format_mode(format_from)
        mode_to, flag_rel_to, flag_yx_to = self.__get_format_mode(format_to)

        # make all input relative
        if not flag_rel_from:
            if flag_yx_from:
                image_size = self.__xy2yx(image_size)
            bbox = self.__abs2rel(bbox, image_size)

        if (np.array(bbox) > 1).any():
            bbox = bbox.astype("float64") / 100.0

        mode = {0: "xywh", 1: "min_max", 2: "min_wh"}
        if mode[mode_to] == mode[mode_from]:
            out = bbox
        elif mode[mode_from] == "xywh":
            if mode[mode_to] == "min_wh":
                out = self._xywh2min_wh(bbox)
            elif mode[mode_to] == "min_max":
                out = self._xywh2min_max(bbox)
            else:
                raise ValueError(f"Unknown format to transform to: {mode_to} = {mode[mode_to]}")
        elif mode[mode_from] == "min_wh":
            if mode[mode_to] == "xywh":
                out = self._min_wh2xywh(bbox)
            elif mode[mode_to] == "min_max":
                out = self._min_wh2min_max(bbox)
            else:
                raise ValueError(f"Unknown format to transform to: {mode_to} = {mode[mode_to]}")
        elif mode[mode_from] == "min_max":
            if mode[mode_to] == "xywh":
                out = self._min_max2xywh(bbox)
            elif mode[mode_to] == "min_wh":
                out = self._min_max2min_wh(bbox)
            else:
                raise ValueError(f"Unknown format to transform to: {mode_to} = {mode[mode_to]}")
        else:
            raise ValueError(f"Unknown format to transform from: {mode_from} = {mode[mode_from]}")

        if not flag_rel_to:
            out = self.__rel2abs(out, image_size)

        if flag_yx_to and not flag_yx_from:
            out = self.__xy2yx(out)

        return tuple(out)

    # [xywh, min_wh, min_max]
    # xywh -> min_wh
    # xywh -> min_max
    # min_wh -> xywh
    # min_wh -> min_max
    # min_max -> xywh
    # min_max -> min_wh

    @staticmethod
    def _xywh2min_wh(bbox_rel):
        """
        center coordinates to lower-left corner coordinates
        xywh -> min_wh
        """
        bbox = np.array(bbox_rel)
        x0, y0, w, h = bbox
        x_min = x0 - w / 2  # bbox[0] -= bbox[2]
        y_min = y0 - h / 2  # bbox[1] -= bbox[3]
        bbox = np.array([x_min, y_min, w, h])
        return bbox

    @staticmethod
    def _xywh2min_max(bbox_rel):
        """
        center coordinates to corner coordinates
        xywh -> min_max
        """
        bbox = np.array(bbox_rel)
        x0, y0, w, h = bbox
        w2 = w / 2
        h2 = h / 2
        x_min = x0 - w2  # bbox[0] -= bbox[2]
        y_min = y0 - h2  # bbox[1] -= bbox[3]
        x_max = x0 + w2  # bbox[2] = bbox[0] + 2 * bbox[2]
        y_max = y0 + h2  # bbox[3] = bbox[1] + 2 * bbox[3]
        return np.array([x_min, y_min, x_max, y_max])

    @staticmethod
    def _min_wh2xywh(bbox_rel):
        """
        lower-left corner coordinates to center coordinates
        xywh -> min_wh
        """
        bbox = np.array(bbox_rel)
        x_min, y_min, w, h = bbox
        w2 = w / 2  # bbox[2] /= 2
        h2 = h / 2  # bbox[3] /= 2
        x0 = x_min + w2  # bbox[0] -= bbox[2]
        y0 = y_min + h2  # bbox[1] -= bbox[3]
        bbox = np.array([x0, y0, w, h])
        return bbox

    @staticmethod
    def _min_wh2min_max(bbox_rel):
        """
        lower-left corner coordinates to (global) corner coordinates
        min_wh -> min_max
        """
        bbox = np.array(bbox_rel)
        x_min, y_min, w, h = bbox
        x_max = x_min + w  # bbox[2] = bbox[0] + bbox[2]
        y_max = y_min + h  # bbox[3] = bbox[1] + bbox[3]
        bbox = np.array([x_min, y_min, x_max, y_max])
        return bbox

    @staticmethod
    def _min_max2xywh(bbox_rel):
        """
        (global) corner coordinates to center coordinates
        min_max -> xywh
        """
        bbox = np.array(bbox_rel)
        x_min, y_min, x_max, y_max = bbox
        w = (x_max - x_min)
        h = (y_max - y_min)
        x0 = x_min + w / 2
        y0 = y_min + h / 2
        bbox = np.array([x0, y0, w, h])
        return bbox

    @staticmethod
    def _min_max2min_wh(bbox_rel):
        """
        (global) corner coordinates to lower left corner coordinates
        min_max -> min_wh
        """
        bbox = np.array(bbox_rel)
        x_min, y_min, x_max, y_max = bbox
        w = (x_max - x_min)
        h = (y_max - y_min)
        bbox[2:4] = [w, h]
        return bbox

    @staticmethod
    def __abs2rel(ary_in: Union[Tuple[int, int, int, int], List[int], np.ndarray],
                  image_size: Tuple[int, int]) -> np.ndarray:
        ary = np.array(ary_in).astype('float64')

        for i in range(0, len(ary), 2):
            ary[i] /= image_size[0]
        for i in range(1, len(ary), 2):
            ary[i] /= image_size[1]
        return ary

    @staticmethod
    def __rel2abs(ary_in: Union[Tuple[int, int, int, int], List[int], np.ndarray],
                  image_size: Tuple[int, int]) -> np.ndarray:
        ary = np.array(ary_in).astype('float64')

        for i in range(0, len(ary), 2):
            ary[i] *= image_size[0]
        for i in range(1, len(ary), 2):
            ary[i] *= image_size[1]
        return ary.round().astype('int64')


def draw_single_bbox(image: Image.Image,
                     bbox,
                     bbox_format: str = "yolo",
                     color=voith_colors.loc["voith_cyan", "RGB"]):
    """Visualizes a single bounding box on the image"""
    xy_min_max = BBoxTransform().transform(bbox,
                                           format_to="coco",
                                           format_from=bbox_format,
                                           image_size=image.size)

    draw = ImageDraw.Draw(image)
    draw.rectangle(xy_min_max, outline=color, width=2)
    # draw.point((x_min + (x_max - x_min)/2, (y_min + (y_max - y_min)/2)), fill=color)

    return image


def draw_bbox(image: Union[Image.Image, np.ndarray], bboxes, category_ids, bbox_format: str = "yolo"):
    if isinstance(image, np.ndarray):
        image_copy = Image.fromarray(image)
    else:
        image_copy = image.copy()
    image_copy = image_copy.convert('RGB')

    for bbox, category_id in zip(bboxes, category_ids):
        # class_name = category_id_to_name[category_id]
        # draw.rectangle((10, 10, 80, 80), fill="red")
        image_copy = draw_single_bbox(image_copy,
                                      bbox,
                                      bbox_format=bbox_format,
                                      color=voith_colors.iloc[category_id]["RGB"])
    image_copy.show()


class ImageDataGenerator4BoundingBoxes:
    _random_seed = 42
    _shuffle = False

    annotations = None
    transform = None

    path_to_data = None
    path_to_annotation = None

    target_size = None
    batch_size = None

    bbox_format = ""
    __bbox_trafo_format = "yolo"
    __label_field = "category_ids"
    _image_file_extension = "jpg"
    _image_format = "RGB"
    __i_batch = 0

    def __init__(
                self,
                path_to_data: Union[str, pl.Path] = None,
                path_to_annotation: Union[str, pl.Path] = None,
                shuffle: bool = False,
                random_seed: int = 42,
                target_size: Tuple[int, int] = (244, 244),
                bbox_format: str = "yolo",
                image_file_extension: str = "jpg",
                batch_size: int = None,
                color_mode: str = "grayscale",
                augment_data: bool = True
                ) -> None:
        self.path_to_data = path_to_data
        self.path_to_annotation = path_to_annotation
        self.annotations = Annotator(self.path_to_annotation)

        self.target_size = target_size
        self._batch_size = batch_size
        self.color_mode = color_mode

        self.bbox_format = bbox_format
        self._image_file_extension = image_file_extension.strip("., \t\r\n")
        self.augment_data = augment_data

        self.reset_seed(random_seed)
        self._shuffle = shuffle



    @property
    def num_classes(self):
        return self.annotations.num_categories

    @property
    def images(self) -> list:
        list_of_images = list(self.path_to_data.glob("*." + self._image_file_extension))
        if not list_of_images:
            raise ValueError("No images found!")
        return list_of_images

    @property
    def num_images(self) -> int:
        return len(self.images)

    def __set_transformation_pipeline(self):
        width, height = self.target_size

        # get original image size
        # image_size_ogl = Image.open(self.images[0]).size
        
        trafo_fncs = []
        if self.augment_data:
            # --- noise
            trafo_fncs.append(A.GaussNoise(var_limit=(10, 50), p=0.2))
            if self.color_mode.lower() == "rgb":
                trafo_fncs.append(A.ISONoise(p=0.5))  # camera sensor noise
            # --- filter
            trafo_fncs.append(A.Sharpen(p=0.2))
            trafo_fncs.append(A.Blur(blur_limit=2, p=0.2))
            # # --- brightness / pixel-values
            # trafo_fncs.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3))
            # A.RandomGamma(p=0.2),
            # A.CLAHE(p=0.2),  # Contrast Limited Adaptive Histogram Equalization
            # A.RGBShift(p=0.1),
            # --- geometry
            # A.RandomSizedCrop((512 - 100, 512 + 100), 512, 512),
            # A.CenterCrop(width=450, height=450)
            trafo_fncs.append(A.HorizontalFlip(p=0.5))
            trafo_fncs.append(A.VerticalFlip(p=0.5),)
            trafo_fncs.append(A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5))
            trafo_fncs.append(A.RandomRotate90(p=1))
            trafo_fncs.append(A.BBoxSafeRandomCrop(p=0.5, erosion_rate=0.01))
        trafo_fncs.append(A.Resize(height=height, width=width, always_apply=True)),
        if self.augment_data:
            trafo_fncs.append(A.PixelDropout(dropout_prob=0.01, p=0.5))

        # compose pipeline
        self.transform = A.Compose(trafo_fncs,
                                   bbox_params=A.BboxParams(format=self.__bbox_trafo_format,
                                                            label_fields=[self.__label_field])
                                   )
        return True

    @property
    def steps_per_epoch(self) -> int:
        return self.num_images // self.batch_size

    @property
    def batch_size(self):
        if self._batch_size is None:
            self._batch_size = self._determine_batch_size(self.num_images)
        return self._batch_size

    @staticmethod
    def _determine_batch_size(n_examples: int, batch_size_max: int = 64) -> int:
        batch_size_max = np.min([batch_size_max, np.max([n_examples // 3, 1])])

        # find most suitable batch size
        batch_size = -1
        remainder = np.inf
        for sz in range(batch_size_max, 0, -1):
            tmp_remainder = n_examples % sz
            if tmp_remainder < remainder:
                remainder = tmp_remainder
                batch_size = sz
                if remainder == 0:
                    break
        return batch_size

    def iter_batchs(self, num_batches: int = None):
        batch_img, batch_bbx, batch_lbl = list(), list(), list()
        for i, (img, bbx, lbl) in enumerate(self._iterimages()):
            batch_img.append(img)
            batch_bbx.append(bbx)
            batch_lbl.append(lbl)
            if (i % self.batch_size) >= (self.batch_size - 1):  # FIXME
                batch_img = np.stack(batch_img, axis=0)
                batch_bbx = np.stack(batch_bbx, axis=0)
                batch_lbl = np.stack(batch_lbl, axis=0)
                yield batch_img, batch_bbx, batch_lbl
                # reset
                batch_img, batch_bbx, batch_lbl = list(), list(), list()
                if num_batches and i/self.batch_size >= num_batches:
                    break

    def iter_epochs(self, n_epochs: int = 10):
        for e in range(n_epochs):
            for b, x in enumerate(self.iter_batchs()):
                yield x
                if b >= self.steps_per_epoch:
                    break

    def iter_epoch(self):
        for b, x in enumerate(self.iter_batchs(self.steps_per_epoch)):
            yield x

    def iter_images(self):
        for i, x in enumerate(self._iterimages()):
            yield x
            if i >= self.num_images:
                break

    def reset_seed(self, seed: int = None) -> bool:
        if seed:
            self._random_seed = seed
        random.seed(self._random_seed)

        self.__set_transformation_pipeline()
        self.__i_batch = 0
        return True

    def _iterimages(self):
        list_of_images = self.images
        num_images = len(list_of_images)

        while True:
            i = self.__i_batch % num_images
            if i == 0 and self._shuffle:
                random.shuffle(list_of_images)
            img_nm = list_of_images[i]
            # load image
            image = Image.open(img_nm).convert(self._image_format)
            # Image.fromarray(img_ary.astype('uint8'), 'RGB')
            # get bounding-box
            bboxes, category_ids = self._find_label_to_image(img_nm)
            # check if there is a bounding box
            if bboxes is None:
                continue

            # transform image
            transformed = self.transform(image=np.array(image),
                                         bboxes=bboxes,
                                         category_ids=category_ids)

            transformed["bboxes"] = self.__trafo_bboxes(transformed["bboxes"],
                                                        image_size=transformed["image"].shape[:2],
                                                        back_trafo=True)
            yield transformed["image"], transformed["bboxes"], transformed["category_ids"]
            # update loop control variable
            self.__i_batch += 1

    def __trafo_bboxes(self, bboxes, image_size, back_trafo: bool = False):
        if back_trafo:
            format_from = self.__bbox_trafo_format
            format_to = self.bbox_format
        else:
            format_from = self.bbox_format
            format_to = self.__bbox_trafo_format

        bboxes_transformed = []
        for bx in bboxes:
            bx_transformed = BBoxTransform().transform(bx,
                                                       format_from=format_from,
                                                       format_to=format_to,
                                                       image_size=image_size)
            bboxes_transformed.append(bx_transformed)
        return bboxes_transformed

    def _find_label_to_image(self, image_name: str) -> Tuple[list, list]:
        bb = self.annotations.find_label_to_image(image_name)
        bboxes, category_ids = self.annotations.get_boundingboxes(bb, bbox_format=self.__bbox_trafo_format)
        return bboxes, category_ids


class Annotator:
    annotation = None
    __idx_annotator = 0
    categories = None
    max_boxes = None
    n_boxes = None
    _path_to_annotation = None

    def __init__(self, path_to_annotation: Union[str, pl.Path]):
        self._path_to_annotation = pl.Path(path_to_annotation)
        self.annotation = pd.read_json(self._path_to_annotation)
        self.annotation["stem"] = self.annotation.image.apply(lambda x: pl.Path(x).stem)

        self._set_categories()

    def __str__(self):
        return f"Annotator({self._path_to_annotation.as_posix()})"

    def get_n_boxes(self, names: List[str] = None) -> Tuple[int, int]:
        if names is None:
            names = self.annotation.stem

        self.max_boxes = 0
        self.n_boxes = 0
        for nm in names:
            lbl = self.find_label_to_image(nm)
        # for lbl in self.annotation["label"]:
            n_boxes = len([el for el in lbl if "rectanglelabels" in el.keys()])
            self.n_boxes += n_boxes

            if n_boxes > self.max_boxes:
                self.max_boxes = n_boxes
        return self.n_boxes, self.max_boxes,

    def find_label_to_image(self, image_name: str) -> list:
        lg = self.annotation["stem"].str.contains(pl.Path(image_name).stem)
        labels = self.annotation.loc[lg, "label"].iat[self.__idx_annotator]
        return labels

    def _set_categories(self):
        label_categories = []
        for idx, row in self.annotation["label"].items():
            for lbl in row:
                for el in lbl["rectanglelabels"]:
                    if el not in label_categories:
                        label_categories.append(el)
        self.categories = label_categories

    def get_category_id(self, category_label: str) -> int:
        return self.categories.index(category_label)

    @property
    def num_categories(self):
        return len(self.categories)

    def get_boundingboxes(self, labels: List[dict], bbox_format: str = "yolo"):
        bboxes = []
        category_ids = []
        for lbl in labels:
            box = [lbl[ky] for ky in ["x", "y", "width", "height"]]  # NORMALIZED DATA
            image_size = (int(lbl["original_width"]), int(lbl["original_height"]))
            bboxes.append(BBoxTransform().transform(bbox=box,
                                                    format_to=bbox_format,
                                                    format_from="labelstudio",
                                                    image_size=image_size
                                                    )
                          )

            category_label = lbl["rectanglelabels"][0]
            category_id = self.get_category_id(category_label)
            category_ids.append(category_id)

            # out.append({"box": box, "class_label": category_label})
        return bboxes, category_ids


if __name__ == "__main__":
    # https://albumentations.ai/docs/examples/example_bboxes/
    x_min = 50
    y_min = 100
    w = 42
    h = 17
    bbox = [x_min, y_min, w, h]
    bdes = [x_min, y_min, x_min + w, y_min + h]
    out = BBoxTransform().transform(bbox,
                                    format_from="min_wh",
                                    format_to="min_max",
                                    image_size=(10, 10)
                                    )
