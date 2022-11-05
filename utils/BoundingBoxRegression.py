import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pathlib as pl
import random

from typing import Union, Tuple, List

import albumentations as A  # expects numpy.ndarray (height, width, channels) as input: (y,x,c)!
# ATTENTION: an Image object has size (width, height) / (x,y) BUT AN ARRAY has shape (height, width) / (y,x) !

import matplotlib.colors as mcolors

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from utils import BBox_Transform as trafo
from utils.utils import determine_batch_size


def draw_box(image: Union[Image.Image, np.ndarray],
             bbox: List[np.ndarray],
             bbox_scores: np.ndarray,
             bbox_classes: np.ndarray,
             color_text: tuple[int, int, int] = (0, 0, 0),
             class_id_to_label: list = None,
             th_score: float = 0.5
             ) -> Image.Image:
    assert len(bbox) == len(bbox_scores) == len(bbox_classes)
    # apply threshold to exclude uninteresting bounding boxes
    lg = np.asarray(bbox_scores) >= th_score
    print(f"Found {lg.sum()} boxes with a score >= {th_score}.")
    bbox = np.asarray(bbox)[lg]
    bbox_scores = np.asarray(bbox_scores)[lg]
    bbox_classes = np.asarray(bbox_classes)[lg]

    # create image drawing instance
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # set font (size)
    # (ImageDraw's default font is a bitmap font, and therefore it cannot be scaled. For scaling, you need to select a true-type font.)
    font = ImageFont.truetype(font=pl.Path(__file__).with_name("FiraMono-Medium.otf").as_posix(),
                              size=np.floor(0.02 * image.size[1] + 0.5).astype("int32")
                              )
    line_thickness = np.sum(image.size) // 500

    # set colors for classes
    if class_id_to_label is None:
        class_id_to_label = np.arange(np.max(bbox_classes) + 1).tolist()

    n_classes = len(class_id_to_label)
    if n_classes < 10:
        colors = list(mcolors.TABLEAU_COLORS.values())
    else:
        colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(colors)

    def draw_one_box(box, color):
        # box coordinates
        # x_min = left
        # y_min = top
        # x_max = right
        # y_max = bottom
        # x_min, y_min, x_max, y_max = box  # PIL
        left, top, right, bottom = box  # PIL

        # draw box (workaround: thickness as multiple rectangles)
        for j in range(line_thickness):
            draw.rectangle((left + j, top + j, right - j, bottom - j), outline=color)

    def draw_label_flag(box, score, class_prd, color):
        # x_min, y_min, x_max, y_max = box  # PIL
        left, top, right, bottom = box  # PIL

        # label / info
        info = f"{class_id_to_label[class_prd]}: {score:.2f}"
        # get size of the label flag
        info_size = draw.textbbox((0, 0), info, font)[2:]

        # coordinates for info label
        if top - info_size[1] >= 0:
            text_origin = np.array([left, top - info_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # add label to box
        draw.rectangle((tuple(text_origin), tuple(text_origin + info_size)), fill=color)
        draw.text(tuple(text_origin), info, fill=color_text, font=font)

    for box, score, class_prd in zip(bbox, bbox_scores, bbox_classes):
        color = colors[class_prd]
        draw_one_box(box, color)
    # draw at foreground
    for box, score, class_prd in zip(bbox, bbox_scores, bbox_classes):
        color = colors[class_prd]
        draw_label_flag(box, score, class_prd, color)

    return img


class ImageDataGenerator4BoundingBoxes:
    __label_field = "category_ids"
    _image_file_extension = "jpg"
    _image_format = "RGB"
    __i_batch = 0

    def __init__(
                self,
                path_to_data: Union[str, pl.Path] = None,
                path_to_annotation: Union[str, pl.Path] = None,
                shuffle: bool = False,
                target_image_size: Tuple[int, int] = (244, 244),
                image_file_extension: str = "jpg",
                batch_size: int = None,
                color_mode: str = "grayscale",
                augment_data: bool = True
                ) -> None:

        self.path_to_data = path_to_data
        self.path_to_annotation = path_to_annotation
        self.annotations = Annotator(self.path_to_annotation)

        self._target_size_pil = target_image_size
        # NOTE: assuming the size that a pillow image has!
        self._target_size_np = trafo.swap_array_elements(self._target_size_pil)
        self._batch_size = batch_size
        self.color_mode = color_mode

        self._image_file_extension = image_file_extension.strip("., \t\r\n")
        self.augment_data = augment_data

        self.reset_seed()
        self._shuffle = shuffle

    def reset_seed(self) -> bool:
        self.__set_transformation_pipeline()
        self.__i_batch = 0
        return True

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
            trafo_fncs.append(A.VerticalFlip(p=0.5), )
            trafo_fncs.append(A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5))
            trafo_fncs.append(A.RandomRotate90(p=1))
            trafo_fncs.append(A.BBoxSafeRandomCrop(p=0.5, erosion_rate=0.01))
        trafo_fncs.append(A.Resize(width=self._target_size_np[1], height=self._target_size_np[0], always_apply=True)),
        if self.augment_data:
            trafo_fncs.append(A.PixelDropout(dropout_prob=0.01, p=0.5))

        # compose pipeline
        self.transform = A.Compose(trafo_fncs,
                                   bbox_params=A.BboxParams(format="pascal_voc",
                                                            label_fields=[self.__label_field])
                                   )
        return True

    @property
    def steps_per_epoch(self) -> int:
        return self.num_images // self.batch_size

    @property
    def batch_size(self):
        if self._batch_size is None:
            self._batch_size = determine_batch_size(self.num_images)
        return self._batch_size

    def iter_batchs(self, num_batches: int = None):
        batch_img, batch_bbx, batch_lbl = list(), list(), list()
        for i, (img, bbx, lbl) in enumerate(self.__iterimages()):
            batch_img.append(img)
            batch_bbx.append(bbx)
            batch_lbl.append(lbl)
            if (i % self.batch_size) >= (self.batch_size - 1):
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
        for i, x in enumerate(self.__iterimages()):
            yield x
            if i >= self.num_images:
                break

    def __iterimages(self):
        list_of_images = self.images
        num_images = len(list_of_images)

        while True:
            i = self.__i_batch % num_images
            if i == 0 and self._shuffle:
                random.shuffle(list_of_images)
            img_nm = list_of_images[i]
            # load image
            image = Image.open(img_nm).convert(self._image_format)
            img_np = np.asarray(image)
            assert all(image.size == trafo.swap_array_elements(img_np.shape[:2]))

            # get bounding-box
            bboxes, category_ids = self._find_label_to_image(img_nm)
            # check if there is a bounding box
            if bboxes is None:
                continue
            # DEBUGGING
            # print(f"bboxes={bboxes} | image.size={image.size} | self._target_size_pil={self._target_size_pil}")
            # draw_box(image, bboxes, np.ones((len(bboxes),)), category_ids).show()

            # transform / augment image
            transformed = self.transform(image=img_np,  # NOTE: numpy-style image but PIL style boxes
                                         bboxes=bboxes,
                                         category_ids=category_ids)
            # # DEBUGGING
            # print(f"transformed['bboxes']={transformed['bboxes']} | transformed['image'].shape={transformed['image'].shape}")
            # draw_box(transformed["image"], transformed["bboxes"], np.ones((len(transformed["bboxes"]),)), transformed["category_ids"]).show()

            yield transformed["image"], transformed["bboxes"], transformed["category_ids"]
            # update loop control variable
            self.__i_batch += 1

    def _find_label_to_image(self, image_name: str) -> Tuple[list, list]:
        bb = self.annotations.find_label_to_image(image_name)
        bboxes, category_ids = self.annotations.get_boundingboxes(bb)
        return bboxes, category_ids


class Annotator:
    __idx_annotator = 0
    categories = None
    max_boxes = None
    n_boxes = None

    def __init__(self, path_to_annotation: Union[str, pl.Path]):
        self._path_to_annotation = pl.Path(path_to_annotation)
        self.annotation = pd.read_json(self._path_to_annotation)
        self.annotation["stem"] = self.annotation.image.apply(lambda x: pl.Path(x).stem)

        self._set_categories()

    def __rpr__(self):
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

    def get_boundingboxes(self, labels: List[dict]):
        bboxes = []
        category_ids = []
        for obj in labels:
            box = np.asarray([obj[ky] for ky in ["x", "y", "width", "height"]]) / 100  # normalized
            image_size = (int(obj["original_width"]), int(obj["original_height"]))

            box_pascal_voc = trafo.scale_box(trafo.coco2pacal_voc(box), image_size)
            bboxes.append(box_pascal_voc)

            category_label = obj["rectanglelabels"][0]
            category_id = self.get_category_id(category_label)
            category_ids.append(category_id)

            # out.append({"box": box, "class_label": category_label})
        return bboxes, category_ids


def calc_iou_between_boxes(b1_min_max: Union[list, tuple, np.array], b2_min_max: Union[list, tuple, np.array]) -> np.array:
    # [(x,y)_min, (x,y)_max]
    # calculate width / height of a box
    b1_min = np.asarray(b1_min_max)[..., :2]
    b1_max = np.asarray(b1_min_max)[..., 2:4]
    b1_wh = b1_max - b1_min

    b2_min = np.asarray(b2_min_max)[..., :2]
    b2_max = np.asarray(b2_min_max)[..., 2:4]
    b2_wh = b2_max - b2_min
    # calculate area
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    # determine area of intersection
    intersect_min = np.maximum(b1_min, b2_min)
    intersect_max = np.minimum(b1_max, b2_max)
    intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # compare area of intersected boxes to the area of the original boxes => IOU = intersection over union
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def determine_iou_and_label(boxes: Union[np.ndarray, list],
                            boxes_prd: np.ndarray,
                            labels: Union[np.ndarray, list],
                            ) -> Tuple[np.array, np.array]:
    # calculate intersection over union (IoU)
    iou_bbx = []
    label_bbx = []
    for b_prd in boxes_prd:
        # intersection over union to all true boxes
        iou = [calc_iou_between_boxes(b_prd, b_true) for b_true in boxes]
        # what is the best true box?
        idx = np.argmax(iou)
        # append to lists
        iou_bbx.append(iou[idx])
        label_bbx.append(labels[idx])
    iou_bbx = np.asarray(iou_bbx)
    label_bbx = np.asarray(label_bbx)
    return iou_bbx, label_bbx


def calc_mean_average_precision(labels: Union[np.ndarray, list],
                                labels_prd: Union[np.ndarray, list],
                                iou: Union[np.ndarray, list],
                                threshold_iou: float = 0.5,
                                num_classes: int = None
                                ) -> float:
    # metrics
    # # mean intersection over union
    # mean_iou = iou_bbx.mean()
    # mean average precision @ X IoU
    lg = np.asarray(iou) >= threshold_iou
    # get predictions above the defined threshold and corresponding classes
    predictions = np.asarray(labels_prd)[lg]
    true_values = np.asarray(labels)[lg]

    # encode to one-hot vectors for multiclass area-under-the-ROC-curve calculation
    # (n_samples, n_classes)
    labels_all = np.concatenate((true_values, predictions))
    y_all = OneHotEncoder(max_categories=num_classes).fit_transform(labels_all.reshape(-1, 1)).toarray()
    y_true = y_all[:len(true_values)]
    y_prd = y_all[len(true_values):]
    auc = roc_auc_score(y_true, y_prd)
    return auc


if __name__ == "__main__":
    path_to_data = pl.Path(r"../Data_ogl_TEST/withBox/Tst")
    path_to_label = pl.Path(r"../Data_ogl_TEST/project-1-at-2022-08-26-14-30-cb5e037c.json")

    bbgen = ImageDataGenerator4BoundingBoxes(path_to_data=path_to_data,
                                             path_to_annotation=path_to_label,
                                             image_file_extension="bmp",
                                             batch_size=1,
                                             target_image_size=(1000, 500),
                                             augment_data=True,
                                             )
    for img, bbx, lbl in bbgen.iter_images():
        # draw_bbox(image, bboxes, category_ids)
        img_out = draw_box(Image.fromarray(img).convert('RGB'), bbx, np.ones(shape=(len(bbx))), lbl,
                           class_id_to_label=bbgen.annotations.categories)
        img_out.show()
        # break

    # https://albumentations.ai/docs/examples/example_bboxes/


