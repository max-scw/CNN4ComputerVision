from TrainModels import TrainModels
from models.yolov3 import YOLOv3, preprocess_true_boxes, load_yolov3_model
from BoundingBoxRegression import ImageDataGenerator4BoundingBoxes, Annotator, calc_mean_average_precision, determine_iou_and_label, draw_box

from keras.optimizers import Adam

from warnings import warn
from typing import Union, Tuple

import numpy as np
import pathlib as pl


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
                 kargs=None
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

        self.kargs = kargs

    def __set_n_classes_and_max_boxes(self) -> bool:
        annotation = Annotator(self.path_to_annotation)
        _, self.max_boxes = annotation.get_n_boxes()
        self.n_classes = annotation.num_categories
        return True

    # def get_batch_size(self, key: str):
    #     files = [el.stem for el in self.paths[key].glob("*" + self._file_extension)]
    #     n_examples, _ = Annotator(self.path_to_annotation).get_n_boxes(names=files)
    #     return self._determine_batch_size(n_examples, batch_size_max=32)

    def get_data_generator(self, key: str, shuffle: bool = True):

        # FIXME: wrapper is currently specific for YOLO!s
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
                                  random_seed=self._random_seed,
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
                         },
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
                        random_seed: int = 42):
    """data generator for fit_generator"""

    bbgen = ImageDataGenerator4BoundingBoxes(path_to_data=path_to_images,
                                             path_to_annotation=path_to_annotation,
                                             image_file_extension=image_file_extension,
                                             batch_size=batch_size,
                                             target_size=target_size,
                                             bbox_format="min_max",
                                             shuffle=shuffle,
                                             random_seed=random_seed
                                             )
    assert num_classes >= bbgen.num_classes
    for img, bbx, lbl in bbgen.iter_batchs():
        #   - img: numpy.ndarray (batch_size, image_size)
        #   - bbx: numpy.ndarray (batch_size, num_boxes, 4)
        #   - lbl: numpy.ndarray (batch_size, num_boxes)
        box_data = np.concatenate((bbx, np.expand_dims(lbl, 2)), axis=2)
        assert (box_data > 0).any()
        assert (box_data[..., 0] >= 0).all()
        assert (box_data[..., 1] >= 0).all()
        assert (box_data[..., 2] > 0).all()
        assert (box_data[..., 3] > 0).all()
        assert (box_data[..., 4] >= 0).all()
        # pad data to always have the same maxis shape
        n_box_to_pad = max_boxes - box_data.shape[1]
        if n_box_to_pad < 0:
            warn(f"Input max_boxes too small. Input variable max_boxes ({max_boxes}) should be larger or equal than "
                 f"the number of bounding boxes in the annotation {box_data.shape[1]}!")
            box_data = box_data[:, :max_boxes]
        else:
            box_data = np.pad(box_data, ((0, 0), (0, n_box_to_pad), (0, 0)), mode='constant', constant_values=0)
        # print(f"INFO: sahpes: img {img.shape}, box_data {box_data.shape}, lbl {lbl.shape}")
        # Absolute x_min, y_min, x_max, y_max, class_id
        y_true = preprocess_true_boxes(box_data, target_size, anchors, num_classes, num_scale_level=num_scale_level)
        yield [img, *y_true], np.zeros(batch_size)  # (x, y)


if __name__ == "__main__":
    # # INPUT SHAPE MUST BE A MULTIPLE OF 32!
    # input_shape = (1024, 1024)
    #
    path_to_image_folders = pl.Path(r"Data_ogl_TEST/withBox")
    path_to_label = pl.Path(r"Data_ogl/project-1-at-2022-08-26-14-30-cb5e037c.json")
    # train = TrainModelsImageBBox(path_to_data=path_to_image_folders,
    #                              file_extension=".bmp",
    #                              path_to_annotation=path_to_label,
    #                              epochs=2,
    #                              target_size=input_shape,
    #                              path_to_save_models=pl.Path("trained_models"),
    #                              # log_file_name="BoundingBoxRegression2",
    #                              verbose=True,
    #                              kargs={
    #                                  "anchors": [(256, 473), (414, 364), (414, 147), (256, 38), (97, 147), (97, 364),
    #                                              (694, 888), (877, 512), (694, 135), (329, 135), (146, 511), (329, 888)],
    #                                  "tiny_yolo": True,
    #                                  # "anchors": [(208, 384), (238, 380), (268, 369), (295, 351), (318, 326),
    #                                  #             (336, 296), (349, 262), (355, 226), (355, 189), (349, 153),
    #                                  #             (336, 119), (318, 89), (295, 64), (268, 46), (238, 35), (208, 31),
    #                                  #             (177, 35), (147, 46), (120, 64), (97, 89), (79, 119), (66, 153),
    #                                  #             (60, 189), (60, 226), (66, 262), (79, 296), (97, 326), (120, 351),
    #                                  #             (147, 369), (177, 380)]
    #                              },
    #                              )
    # train.set_model(model_name="YOLOv3")
    # # train.analyze(model_name="YOLOv3")
    # # print("done.")

    model = load_yolov3_model(pl.Path("trained_models") / "2210311301_YOLOv3tiny-1024x1024-12x2_rgb")
    th_iou = 0.3  # FIXME: usually > 0.5
    th_score = 0.7

    iou_boxes = []
    true_labels_per_box = []
    predicted_labels_per_box = []
    auc_list = []
    for img, bbx, lbl in ImageDataGenerator4BoundingBoxes(path_to_data=path_to_image_folders / "Tst",
                                                          path_to_annotation=path_to_label,
                                                          image_file_extension=".bmp",
                                                          batch_size=1,
                                                          target_size=model.input_shape,
                                                          bbox_format="min_max",
                                                          shuffle=False,
                                                          augment_data=False
                                                          ).iter_images():
        boxes_prd, scores_prd, classes_prd = model.predict(img, th_score=th_score)

        iou_bbx, label_bbx = determine_iou_and_label(bbx, boxes_prd, lbl)

        iou_boxes += list(iou_bbx)
        true_labels_per_box += list(label_bbx)
        predicted_labels_per_box += list(classes_prd)
        # calculate AUC-score for particular image
        auc_list.append(calc_mean_average_precision(label_bbx, classes_prd, iou_bbx, threshold_iou=th_iou))
        # display bounding boxes
        img_out = draw_box(img, boxes_prd, scores_prd, classes_prd,
                           th_score=th_score, class_id_to_label=["wrong", "correct"])
        img_out.show()

    auc = calc_mean_average_precision(true_labels_per_box, predicted_labels_per_box, iou_boxes, threshold_iou=th_iou)
    print(f"Mean average precision (mAP) @ IoU={th_iou}: {np.mean(auc_list)}, AUC@IoU={th_iou}: {auc}")


