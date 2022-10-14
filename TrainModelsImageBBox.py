from TrainModels import TrainModels
from models.yolov3 import YOLOv3, preprocess_true_boxes
from BoundingBoxRegression import ImageDataGenerator4BoundingBoxes, Annotator

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from PIL import Image
from warnings import warn
from typing import Union, Tuple

import numpy as np
import pathlib as pl
import logging


class TrainModelsImageBBox(TrainModels):
    def __init__(self,
                 path_to_data: Union[pl.Path, str],
                 path_to_annotation: Union[pl.Path, str],
                 epochs: int,
                 random_seed: int = 42,
                 path_to_save_models: Union[str, pl.Path] = None,
                 verbose: bool = False,
                 file_extension: str = "jpg",
                 log_file_name: str = "log",
                 use_model_checkpoints: bool = False,
                 max_boxes: int = None,
                 target_size: Tuple[int, int] = (224, 224)
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
        self.max_boxes = max_boxes
        self.path_to_annotation = path_to_annotation
        self.__set_n_classes_and_max_boxes()
        # ensure that the target size is a multiple of 32 for YOLO
        self.target_size = tuple(32 * (el // 32) for el in target_size)

    def __set_n_classes_and_max_boxes(self):
        annotation = Annotator(self.path_to_annotation)
        _, self.max_boxes = annotation.get_n_boxes()
        self.n_classes = annotation.num_categories

    # def get_batch_size(self, key: str):
    #     files = [el.stem for el in self.paths[key].glob("*" + self._file_extension)]
    #     n_examples, _ = Annotator(self.path_to_annotation).get_n_boxes(names=files)
    #     return self._determine_batch_size(n_examples, batch_size_max=32)

    def get_data_generator(self, key: str, shuffle: bool = True):

        # Annotator(self.path_to_annotation).n_boxes
        # FIXME: wrapper is currently specific for YOLO!s
        gen = data_generator4yolo(path_to_images=self.paths[key],
                                  path_to_annotation=self.path_to_annotation,
                                  batch_size=self.get_batch_size(key),
                                  target_size=self.target_size,
                                  anchors=YOLOv3(input_shape=self.target_size, num_classes=self.n_classes).anchors,
                                  num_classes=self.n_classes,
                                  max_boxes=self.max_boxes,
                                  image_file_extension=self._file_extension,
                                  shuffle=shuffle,
                                  random_seed=self._random_seed
                                  )
        # TODO image size!
        return gen

    def _compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=1e-3),
                           loss={
                                # use custom yolo_loss Lambda layer.
                                'yolo_loss': lambda y_true, y_pred: y_pred
                           },
                           # metrics=["mse"]
                           )

    def set_model(self, model_name: str) -> bool:
        if self._match_model_name("YOLOv3", model_name):
            self.model = YOLOv3(input_shape=self.target_size, num_classes=self.n_classes).create_model()
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
            warn(f"Input max_boxes too small. Input variable max_boxes ({max_boxes}) should be larger or equal than the "
                 f"number of bounding boxes in the annotation {box_data.shape[1]}!")
            box_data = box_data[:, :max_boxes]
        else:
            box_data = np.pad(box_data, ((0, 0), (0, n_box_to_pad), (0, 0)), mode='constant', constant_values=0)

        # Absolute x_min, y_min, x_max, y_max, class_id
        y_true = preprocess_true_boxes(box_data, target_size, anchors, num_classes)
        yield [img, *y_true], np.zeros(batch_size)


if __name__ == "__main__":
    # INPUT SHAPE MUST BE A MULTIPLE OF 32!
    input_shape = (512, 512)

    path_to_image_folders = pl.Path(r"Data_ogl\withBox")
    path_to_label = pl.Path(r"Data_ogl\project-1-at-2022-08-26-14-30-cb5e037c.json")

    train = TrainModelsImageBBox(path_to_data=path_to_image_folders,
                                 path_to_annotation=path_to_label,
                                 epochs=2,
                                 path_to_save_models=pl.Path("trained_models"),
                                 file_extension="bmp",
                                 log_file_name="log_ImageBBox",
                                 target_size=input_shape,
                                 verbose=True
                                 )
    train.analyze(model_name="YOLOv3")

    #
    # #
    # # gen2 = train.get_data_generator("training", shuffle=False)
    # #
    # from MVE import data_generator2
    # #
    # yolo = YOLOv3(input_shape=train.target_size, num_classes=train.n_classes)
    # gen = data_generator2(path_to_images=path_to_image_folders.joinpath("Trn"),
    #                       path_to_annotation=path_to_label,
    #                       batch_size=2,
    #                       target_size=input_shape,
    #                       anchors=yolo.anchors,
    #                       num_classes=2,
    #                       max_boxes=30,
    #                       image_file_extension="bmp")
    # #
    # # for img_ytrue, img_ytrue2 in zip(gen, gen2):
    # #     assert (img_ytrue[0] == img_ytrue2[0]).all()
    # #     assert all([(img_ytrue[i] == img_ytrue2[i]).all() for i in range(len(img_ytrue2))])
    #
    # # yolo = YOLOv3(input_shape=train.target_size, num_classes=train.n_classes)
    # # model = yolo.create_model()
    # train.set_model("YOLOv3")
    # # train._compile_model()
    # # train._fit()
    # train.fit()
    # model = train.model
    #
    # # model.compile(optimizer=Adam(learning_rate=1e-3), loss={
    # #               # use custom yolo_loss Lambda layer.
    # #               'yolo_loss': lambda y_true, y_pred: y_pred},
    # #               # metrics=["mse"]
    # #               )
    # history = model.fit(x=train.get_data_generator("training"),
    #                     steps_per_epoch=train.get_steps_per_epoch("training"),
    #                     epochs=2,
    #                     )
    print("done.")
