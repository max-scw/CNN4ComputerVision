from TrainModels import TrainModels
from models.yolov3 import YOLOv3, preprocess_true_boxes
from utils.BoundingBoxRegression import ImageDataGenerator4BoundingBoxes, Annotator
from utils.BBox_Transform import swap_array_elements

from keras.optimizers import Adam

from warnings import warn
from typing import Union, Tuple

import numpy as np
import pathlib as pl
import random


class TrainModelsImageBBox(TrainModels):
    model_info = None

    def __init__(self,
                 path_to_data: Union[pl.Path, str],
                 path_to_annotation: Union[pl.Path, str],
                 epochs: int,
                 target_size: Tuple[int, int] = (224, 224),
                 random_seed: int = 42,
                 path_to_save_models: Union[str, pl.Path] = None,
                 verbose: bool = False,
                 file_extension: str = "jpg",
                 log_file_name: str = "TrainModelsImageBBox",
                 use_model_checkpoints: bool = False,
                 max_boxes: int = None,
                 kargs=None,
                 augment_data: bool = True
                 ) -> None:
        super().__init__(path_to_data=path_to_data,
                         epochs=epochs,
                         random_seed=random_seed,
                         path_to_save_models=path_to_save_models,
                         verbose=verbose,
                         file_extension=file_extension,
                         log_file_name=log_file_name,
                         use_model_checkpoints=use_model_checkpoints
                         )
        self._loss = None
        self.max_boxes = max_boxes
        self.path_to_annotation = path_to_annotation
        self.__set_n_classes_and_max_boxes()
        # ensure that the target size is a multiple of 32 for YOLO
        assert np.all(np.asarray(target_size) % 32 == 0), \
            f"Desired image size must be a multiple of 32. target_size was {target_size}"
        self.target_size = target_size
        # print(f"__init__: target_size={self.target_size}") # FIXME: for DEBUGGING
        self._augment_data = augment_data
        self.kargs = kargs

    def __set_n_classes_and_max_boxes(self) -> bool:
        annotation = Annotator(self.path_to_annotation)
        _, self.max_boxes = annotation.get_n_boxes()
        self.n_classes = annotation.num_categories
        return True

    def get_data_generator(self, key: str, shuffle: bool = True):
        # NOTE: wrapper is currently specific for YOLO!s
        # print(f"get_data_generator: target_size={self.target_size}")  # FIXME: for DEBUGGING
        gen = data_generator4yolo(path_to_images=self.paths[key],
                                  path_to_annotation=self.path_to_annotation,
                                  batch_size=self.get_batch_size(key),
                                  target_size=self.target_size,
                                  anchors=self.model_info.anchors,
                                  num_classes=self.n_classes,
                                  max_boxes=self.max_boxes,
                                  num_scale_level=2 if self.__get_from_kargs("tiny_yolo") else 3,
                                  image_file_extension=self._file_extension,
                                  shuffle=shuffle,
                                  augment_data=self._augment_data
                                  )
        return gen

    def _compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=1e-3),
                           loss=self._loss
                           # metrics=["mse"]
                           )

    def __get_from_kargs(self, key):
        if key in self.kargs.keys():
            return self.kargs[key]
        else:
            return None

    def set_model(self, model_name: str) -> bool:
        if self._match_model_name("YOLOv3", model_name):
            # print(f"set_model: target_size={self.target_size}")  # FIXME: for DEBUGGING
            self.model_info = YOLOv3(input_shape=self.target_size,
                                     num_classes=self.n_classes,
                                     anchors=self.__get_from_kargs("anchors"),
                                     tiny_yolo=True if self.__get_from_kargs("tiny_yolo") else False,
                                     max_boxes=self.max_boxes
                                     )

            def _save_model(filename):
                self.model_info.model = self.model
                self.model_info.save_model(filename)

            self._save_model = _save_model

            self.model_name = self.model_info.model_name
            self.model = self.model_info.create_model()
            self._loss = {
                             # use custom yolo_loss Lambda layer.
                             'yolo_loss': lambda y_true, y_pred: y_pred
                         }
            self._log.log(f"{self.model_name} with anchors {self.__get_from_kargs('anchors')}")
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        if self.model_pretrained:
            self.model.trainable = False
            # self.__add_new_model_head()  # TODO: pretrained
        return True


# sdf
def data_generator4yolo(path_to_images: pl.Path,
                        path_to_annotation: pl.Path,
                        batch_size: int,
                        target_size: Tuple[int, int],
                        anchors,
                        num_classes: int,
                        max_boxes: int,
                        num_scale_level: int = 3,
                        image_file_extension: str = "jpg",
                        shuffle: bool = True,
                        random_seed: int = 42,
                        augment_data: bool = True):
    """data generator for fit_generator"""
    # print(f"data_generator4yolo: target_size={target_size}")  # FIXME: for DEBUGGING
    bbgen = ImageDataGenerator4BoundingBoxes(path_to_data=path_to_images,
                                             path_to_annotation=path_to_annotation,
                                             image_file_extension=image_file_extension,
                                             batch_size=batch_size,
                                             target_image_size=target_size,
                                             shuffle=shuffle,
                                             augment_data=augment_data
                                             )
    assert num_classes >= bbgen.num_classes
    for img, bbx, lbl in bbgen.iter_batchs():
        #   - img: numpy.ndarray (batch_size, image_size)   # Numpy/OpenCV style!
        #   - bbx: numpy.ndarray (batch_size, num_boxes, 4)  # PIL style!
        #   - lbl: numpy.ndarray (batch_size, num_boxes)
        # print(f"img.shape={img.shape}, bbx.shape={np.shape(bbx)}, lbl.shape={np.shape(lbl)}")  # FIXME: for DEBUGGING

        img_size_pil = swap_array_elements(img.shape[1:3])
        # print(f"target_size={target_size} == img_size_pil={img_size_pil}")  # FIXME: for DEBUGGING
        if (target_size != img_size_pil).any():
            raise ValueError(f"Mix-up in shape input: target_size={target_size} == img_size_pil={img_size_pil}")
        if (bbx[..., :4] > list(img_size_pil) * 2).any():
            # [draw_box(img[i], bbx[i], np.ones((len(bbx[i]),)), lbl[i]).show() for i in range(img.shape[0])]
            raise ValueError(f"Box {bbx[..., :4].max(axis=1)} size exceeds image size {img_size_pil}! (PIL format)")

        box_data = np.concatenate((bbx, np.expand_dims(lbl, 2)), axis=2)

        assert (box_data > 0).any()
        assert (box_data[..., 0] >= 0).all()
        assert (box_data[..., 1] >= 0).all()
        assert (box_data[..., 2] > 0).all()
        assert (box_data[..., 3] > 0).all()
        assert (box_data[..., 4] >= 0).all()
        assert (box_data[..., :4] <= list(target_size) * 2).all(), "Maximum box value exceeds the image size!"
        # pad data to always have the same maxis shape
        n_box_to_pad = max_boxes - box_data.shape[1]
        if n_box_to_pad < 0:
            warn(f"Input max_boxes too small. Input variable max_boxes ({max_boxes}) should be larger or equal than "
                 f"the number of bounding boxes in the annotation {box_data.shape[1]}!")
            box_data = box_data[:, :max_boxes]
        elif n_box_to_pad > 0:
            box_data = np.pad(box_data, ((0, 0), (0, n_box_to_pad), (0, 0)), mode='constant', constant_values=0)
        # print(f"INFO: shapes: img {img.shape}, box_data {box_data.shape}, lbl {lbl.shape}") # FIXME: for DEBUGGING
        # Absolute x_min, y_min, x_max, y_max, class_id

        # target_size_np = target_size
        # box_data[..., :4] = swap_array_elements(box_data[..., :4])
        # target_size_np = swap_array_elements(target_size)

        # anchors = swap_array_elements(anchors)
        assert (box_data[..., 4] < num_classes).all(), "Class id must be less than num_classes."
        assert (np.asarray(target_size) % 32 == 0).all(), f"Input shape must be a multiple of 32. It is {target_size}."
        assert np.all(anchors <= target_size[::-1]), "Anchors do not match input shape"
        # if (box_data[..., :4] > list(target_size_np) * 2).any():
        #     # [draw_box(img[i], bbx[i], np.ones((len(bbx[i]),)), lbl[i]).show() for i in range(img.shape[0])]
        #     raise ValueError(f"Box {box_data[..., :4].max(axis=1)} size exceeds image size {target_size_np}! (Numpy format)")
        #
        y_true = preprocess_true_boxes(box_data, target_size, anchors, num_classes, num_scale_level=num_scale_level)
        # print(f"data_generator4yolo: shape y_true" + " ".join([str(np.shape(el)) for el in y_true]))  #FIXME: DEBUGGING

        if y_true is None or any([el is None for el in y_true]):
            raise ValueError("y_true = preprocess_true_boxes() is not supposed to return None")
        yield [img, *y_true], np.zeros(batch_size)


if __name__ == "__main__":
    # INPUT SHAPE MUST BE A MULTIPLE OF 32!
    input_shape_pil = (1024, 128)
    # random.seed(42)

    path_to_image_folders = pl.Path(r"Data_ogl_TEST/withBox")
    path_to_label = pl.Path(r"Data_ogl/project-1-at-2022-08-26-14-30-cb5e037c.json")
    train = TrainModelsImageBBox(path_to_data=path_to_image_folders,
                                 file_extension=".bmp",
                                 path_to_annotation=path_to_label,
                                 epochs=2,
                                 target_size=input_shape_pil,
                                 path_to_save_models=pl.Path("trained_models"),
                                 # log_file_name="BoundingBoxRegression2",
                                 random_seed=42,
                                 verbose=True,
                                 kargs={
                                     "anchors": [[1014, 124], [1014, 124]],
                                     # "anchors": [[160, 272], [258, 204], [258,  68], [160,   0], [61,  67], [61, 204],
                                     #             [434, 507], [548, 272], [434,  36], [205,  36], [91, 271], [205, 507]],
                                     # "anchors": [(128, 236), (207, 182), (207, 73), (128, 19), (48, 73), (48, 182),
                                     #             (347, 444), (438, 256), (347, 67), (164, 67), (73, 255), (164, 444)],
                                     "tiny_yolo": True,
                                     # "anchors": [(208, 384), (238, 380), (268, 369), (295, 351), (318, 326),
                                     #             (336, 296), (349, 262), (355, 226), (355, 189), (349, 153),
                                     #             (336, 119), (318, 89), (295, 64), (268, 46), (238, 35), (208, 31),
                                     #             (177, 35), (147, 46), (120, 64), (97, 89), (79, 119), (66, 153),
                                     #             (60, 189), (60, 226), (66, 262), (79, 296), (97, 326), (120, 351),
                                     #             (147, 369), (177, 380)]
                                 },
                                 )
    train.set_model(model_name="YOLOv3")
    train.get_data_generator("training")
    # for i, (yolo_output, tmp) in enumerate(train.get_data_generator("training")):
    #     print(i)
    train.fit()
    # # train.analyze(model_name="YOLOv3")
    # # print("done.")


    # # TEST MODEL
    # model = load_yolov3_model(pl.Path("trained_models") / "2210311301_YOLOv3tiny-1024x1024-12x2_rgb")
    # th_iou = 0.3  # usually > 0.5
    # th_score = 0.7
    #
    # gen = ImageDataGenerator4BoundingBoxes(path_to_data=path_to_image_folders / "Tst",
    #                                                       path_to_annotation=path_to_label,
    #                                                       image_file_extension=".bmp",
    #                                                       batch_size=1,
    #                                                       target_image_size=model.input_shape,
    #                                                       shuffle=True,
    #                                                       augment_data=True
    #                                                       )
    #
    # iou_boxes = []
    # true_labels_per_box = []
    # predicted_labels_per_box = []
    # auc_list = []
    # for img, bbx, lbl in gen.iter_images():
    #     # draw original bounding boxes
    #     draw_box(img, bbx, np.full((len(bbx)), 99), lbl, class_id_to_label=gen.annotations.categories).show()
    #
    #     boxes_prd, scores_prd, classes_prd = model.predict(img, th_score=th_score)
    #
    #     # draw predicted bounding boxes
    #     draw_box(img, boxes_prd, scores_prd, classes_prd,
    #              class_id_to_label=gen.annotations.categories, th_score=th_score).show()
    #
    #     iou_bbx, label_bbx = determine_iou_and_label(bbx, boxes_prd, lbl)
    #
    #     iou_boxes += list(iou_bbx)
    #     true_labels_per_box += list(label_bbx)
    #     predicted_labels_per_box += list(classes_prd)
    #     # calculate AUC-score for particular image
    #     auc_list.append(calc_mean_average_precision(label_bbx, classes_prd, iou_bbx, threshold_iou=th_iou))
    #
    # auc = calc_mean_average_precision(true_labels_per_box, predicted_labels_per_box, iou_boxes, threshold_iou=th_iou)
    # print(f"Mean average precision (mAP) @ IoU={th_iou}: {np.mean(auc_list)}, AUC@IoU={th_iou}: {auc}")


