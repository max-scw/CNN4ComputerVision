# https://medium.com/analytics-vidhya/training-yolo-with-keras-85c33cdefe21
from models.yolov3 import YOLOv3, preprocess_true_boxes
from BoundingBoxRegression import ImageDataGenerator4BoundingBoxes

from keras.optimizers import Adam

from PIL import Image
from warnings import warn
from typing import Union, Tuple

import numpy as np
import pathlib as pl

# sdf
def data_generator4yolo(path_to_images: pl.Path,
                        path_to_annotation: pl.Path,
                        batch_size: int,
                        target_size: Tuple[int, int],
                        anchors,
                        num_classes: int,
                        max_boxes: int,
                        image_file_extension: str = "jpg"):
    """data generator for fit_generator"""

    bbgen = ImageDataGenerator4BoundingBoxes(path_to_data=path_to_images,
                                             path_to_annotation=path_to_annotation,
                                             image_file_extension=image_file_extension,
                                             batch_size=batch_size,
                                             image_size=target_size,
                                             bbox_format="min_max"
                                             )
    assert num_classes >= bbgen.num_classes
    for img, bbx, lbl in bbgen.iter_batchs():
        #   - img: numpy.ndarray (batch_size, image_size)
        #   - bbx: numpy.ndarray (batch_size, num_boxes, 4)
        #   - lbl: numpy.ndarray (batch_size, num_boxes)
        box_data = np.concatenate((bbx, np.expand_dims(lbl, 2)), axis=2)
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
    input_shape = (416, 416)
    num_classes = 2
    yolo = YOLOv3(input_shape=input_shape, num_classes=num_classes)
    model = yolo.create_model()

    model.compile(optimizer=Adam(learning_rate=1e-3), loss={
                  # use custom yolo_loss Lambda layer.
                  'yolo_loss': lambda y_true, y_pred: y_pred},
                  # metrics=["mse"]
                  )

    num_anchors = len(yolo.anchors)
    """
    args:
         |-> yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
         |-> y_true: list of array, shape like yolo_outputs, xywh are relative value
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        print_loss: boolean, whether to print the loss at each call
    """

    batch_size = 4
    path_to_data = pl.Path(r"Data_ogl\withBox")
    path_to_label = pl.Path(r"Data_ogl\project-1-at-2022-08-26-14-30-cb5e037c.json")

    gen = {ky: data_generator4yolo(path_to_images=path_to_data.joinpath(ky),
                                   path_to_annotation=path_to_label,
                                   batch_size=batch_size,
                                   target_size=(416, 416),
                                   anchors=yolo.anchors,
                                   num_classes=num_classes,
                                   max_boxes=42,
                                   image_file_extension="bmp")
           for ky in ["Trn", "Val"]}

    num_examples = {ky: len(list(path_to_data.joinpath(ky).glob("*.bmp"))) for ky in gen.keys()}
    # number of batches per epoch
    steps_per_epoch = {ky: max(1, num_examples[ky] // batch_size) for ky in num_examples.keys()}

    history = model.fit(x=gen["Trn"],
                        steps_per_epoch=steps_per_epoch["Trn"],
                        validation_data=gen["Val"],
                        validation_steps=steps_per_epoch["Val"],
                        epochs=2,
                        )
    model_name = "YOLOv3"
    model.save(model_name + ".h5")
    history.to_csv(model_name + ".csv", index=False)

