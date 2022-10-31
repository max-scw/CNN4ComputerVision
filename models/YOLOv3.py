"""
YOLO_v3 model defined in Keras.
(inspired) from https://github.com/qqwweee/keras-yolo3
"""
import numpy as np
import pathlib as pl
from PIL import Image

from functools import wraps, reduce
from typing import Union, Tuple, List

from utils.utils import draw_box

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Conv2D,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
    LeakyReLU,
    BatchNormalization,
    Input,
    Lambda,
)
from keras.regularizers import l2


class YOLOv3:
    __color_channels = 3
    model = None

    def __init__(
        self,
            input_shape: Tuple[int, int],
            num_classes: int,
            anchors: Union[List[Tuple[int, int]], np.ndarray] = None,
            tiny_yolo: bool = False,
            max_boxes: int = 100
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.flag_tiny_yolo = True if tiny_yolo else False
        self.num_scale_levels = 2 if tiny_yolo else 3
        self.max_boxes = max_boxes

        if anchors is None:
            """
            We still use k-means clustering to determine our bounding box priors. We just sort of chose 9 clusters and
            3 scales arbitrarily and then divide up the clusters evenly across scales. On the COCO dataset the 9
            clusters were:
            (10x13); (16x30); (33x23); (30x61); (62x45); (59x119); (116x90); (156x198); (373x326).
            """
            if self.flag_tiny_yolo:
                # tiny YOLOv3 has only 2 scale levels
                anchors = [(10, 14), (23, 27), (37, 58),
                           (81, 82), (135, 169), (344, 319)]
            else:
                # the standard YOLOv3 works at 3 scale levels
                anchors = [(10, 13), (16, 30), (33, 23),
                           (30, 61), (62, 45), (59, 119),
                           (116, 90), (156, 198), (373, 326)]

            # factors: [32, 16, 8]
        self.anchors = np.array(anchors).astype(float).reshape(-1, 2)

    @property
    def model_name(self) -> str:
        shape = "x".join([str(el) for el in self.input_shape])
        scale = f"{len(self.anchors)}x{self.num_scale_levels}"

        info = ["tiny"] if self.flag_tiny_yolo else []
        info += [shape, scale]
        return "YOLOv3" + "-".join(info)

    def _build_model_body(self):
        num_anchors_total = len(self.anchors)  # total number of anchors
        num_anchors_per_level = num_anchors_total // self.num_scale_levels

        image_input = Input(shape=(None, None, self.__color_channels))
        if self.flag_tiny_yolo:
            model_body = self.tiny_yolo_body(image_input, num_anchors_per_level)
        else:
            model_body = self.yolo_body(image_input, num_anchors_per_level)
        return model_body

    def create_model(self):
        num_anchors_total = len(self.anchors)  # total number of anchors
        num_anchors_per_level = num_anchors_total // self.num_scale_levels
        model_body = self._build_model_body()

        h, w = self.input_shape
        y_true = []
        for i in range(self.num_scale_levels):
            fct = [32, 16, 8][i]
            y_true.append(Input(shape=(h // fct, w // fct, num_anchors_per_level, self.num_classes + 5)))

        print(f"Create YOLOv3 model with {num_anchors_total} anchors and {self.num_classes} classes.")

        # TODO: if load_pretrained:

        model_loss = Lambda(
            function=self.yolo_loss,
            output_shape=(1,),
            name="yolo_loss",
            # arguments={"ignore_thresh": 0.5},
        )([*model_body.output, *y_true])
        self.model = Model([model_body.input, *y_true], model_loss, name=self.model_name)
        return self.model

    def yolo_body(self, inputs, num_anchors: int):
        """Create YOLO_V3 model CNN body in Keras."""
        num_anchors = len(self.anchors) // self.num_scale_levels

        darknet = Model(inputs, darknet_body(inputs))
        x, y1 = make_last_layers(darknet.output, 512, num_anchors * (self.num_classes + 5))

        x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)
        x = Concatenate()([x, darknet.layers[152].output])
        x, y2 = make_last_layers(x, 256, num_anchors * (self.num_classes + 5))

        x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
        x = Concatenate()([x, darknet.layers[92].output])
        x, y3 = make_last_layers(x, 128, num_anchors * (self.num_classes + 5))
        # NOTE: this creates the num_scale_levels layers. => must be of same size (the standard YOLO has 3 scale levels)
        return Model(inputs, [y1, y2, y3])

    def tiny_yolo_body(self, inputs, num_anchors: int):
        """Create Tiny YOLO_v3 model CNN body in keras."""
        num_anchors = len(self.anchors) // self.num_scale_levels

        x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            DarknetConv2D_BN_Leaky(128, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
        )(inputs)
        x2 = compose(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)),
        )(x1)
        y1 = compose(DarknetConv2D_BN_Leaky(512, (3, 3)), DarknetConv2D(num_anchors * (self.num_classes + 5), (1, 1)))(
            x2
        )

        x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x2)
        y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
            DarknetConv2D(num_anchors * (self.num_classes + 5), (1, 1)),
        )([x2, x1])
        # NOTE: this creates the num_scale_levels layers. => must be of same size (the tiny YOLO has 2 levels of scale)
        return Model(inputs, [y1, y2])

    def yolo_loss(
        self,
        args: Union[list, np.array],
        ignore_thresh: float = 0.5,
        print_loss: bool = False,
    ):
        """
        Return yolo_loss tensor

        Parameters
        ----------
        args:
         |-> yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
         |-> y_true: list of array, shape like yolo_outputs, xywh are relative value
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        print_loss: boolean, whether to print the loss at each call

        Returns
        -------
        loss: tensor, shape=(1,)

        """
        # num_layers = self.num_scale_levels  # default setting
        yolo_outputs = args[:self.num_scale_levels]
        y_true = args[self.num_scale_levels:]

        anchor_mask = np.arange(len(self.anchors)).reshape(self.num_scale_levels, -1)[::-1].tolist()

        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))  # height, width
        grid_shapes = [K.cast(K.shape(yolo_outputs[i])[1:3], K.dtype(y_true[0])) for i in range(self.num_scale_levels)]
        loss = 0
        m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

        def loop_body(b, ignore_msk):
            true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_msk = ignore_msk.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_msk

        for i in range(self.num_scale_levels):
            object_mask = y_true[i][..., 4:5]
            true_class_probs = y_true[i][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = self.yolo_head(
                yolo_outputs[i], self.anchors[anchor_mask[i]], self.num_classes, input_shape, calc_loss=True
            )
            pred_box = K.concatenate([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[i][..., :2] * grid_shapes[i][::-1] - grid
            raw_true_wh = K.log(y_true[i][..., 2:4] / self.anchors[anchor_mask[i]] * input_shape[::-1])
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, "bool")

            _, ignore_mask = tf.while_loop(
                lambda b, *args: b < m, loop_body, [0, ignore_mask]
            )  # TODO: change to keras backend
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            # Note: K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = (
                object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
            )
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            confidence_loss = (
                object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
                + (1 - object_mask)
                * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
                * ignore_mask
            )
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            confidence_loss = K.sum(confidence_loss) / mf
            class_loss = K.sum(class_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
            if print_loss:
                loss = tf.Print(
                    loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message="loss: "
                )
        return loss

    def yolo_head(self, feats, anchors, num_classes: int, input_shape, calc_loss: bool = False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust predictions to each spatial grid point and anchor size.
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def load_model(self, filename_model_weights: Union[str, pl.Path]) -> Model:
        if self.model is None:
            self.model = self._build_model_body() # FIXME: check what kind of model it was
        # load weights
        self.model.load_weights(filename_model_weights)
        return self.model

    def predict(self, image: Union[np.ndarray, Image.Image], th_score: float = 0.5):
        # image must be at the right size and scaled!
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        if np.shape(image)[:2] != self.input_shape:
            image = Image.fromarray(image).resize(self.input_shape)
            # image = np.asarray(img) / 255.0  # RESCALE
        # Add batch dimension
        image_data = np.expand_dims(image, 0)

        out_boxes, out_scores, out_classes = self._call_model(image_data)
        print(f"Found {len(out_boxes)} boxes.")

        # apply threshold
        lg = out_scores >= th_score
        return out_boxes[lg], out_scores[lg], out_classes[lg]

    def _call_model(self, image: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # #(1, 416, 416, 3)
        yolo_outputs = self.model(np.asarray(image))
        out_boxes, out_scores, out_classes = yolo_eval(yolo_outputs,
                                                       self.anchors,
                                                       self.num_classes,
                                                       self.input_shape,
                                                       max_boxes=self.max_boxes,
                                                       score_threshold=0 # FIXME
                                                       )
        # yolo_outputs,
        # anchors,
        # num_classes,
        # image_shape,
        # max_boxes = 20,
        # score_threshold = .6,
        # iou_threshold = .5

        return out_boxes.numpy(), out_scores.numpy(), out_classes.numpy()


def yolo_head(feats, anchors, num_classes: int, input_shape, calc_loss: bool = False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {
        "kernel_regularizer": l2(5e-4),
        "padding": "valid" if kwargs.get("strides") == (2, 2) else "same",
    }
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {"use_bias": False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters: int, num_blocks: int):
    """A series of residual blocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)), DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters: int, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
    )(x)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)), DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = np.arange(len(anchors)).reshape(num_layers, -1)[::-1].tolist()
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(
    true_boxes: Union[List[list], np.ndarray],
    input_shape: Union[List[list], np.ndarray, Tuple[int, int]],
    anchors: Union[List[list], np.ndarray],
    num_classes: int,
    num_scale_level: int = 3
) -> List[np.ndarray]:
    """
    Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5) || (batch_size, max_boxes, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    assert (true_boxes[..., 4] < num_classes).all(), "class id must be less than num_classes"
    num_layers = num_scale_level  # default setting

    anchor_mask = np.arange(len(anchors)).reshape(num_layers, -1)[::-1].tolist()

    # transform from [x1, y1, x2, y2] to [x0, y0, w, h]
    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    xy1 = true_boxes[..., 0:2]
    xy2 = true_boxes[..., 2:4]
    boxes_xy = (xy1 + xy2) // 2
    boxes_wh = xy2 - xy1
    # normalize => make values relative to image size
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # xy1
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # xy2

    # allocate box-mask-vector ?????
    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape // [32, 16, 8][i] for i in range(num_layers)]
    y_true = []
    for i_lyr in range(num_layers):
        grid_width = grid_shapes[i_lyr][0]
        grid_height = grid_shapes[i_lyr][1]
        tmp = np.zeros((batch_size, grid_width, grid_height, len(anchor_mask[i_lyr]), 5 + num_classes), dtype="float32")
        y_true.append(tmp)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_max = anchors / 2.0
    anchor_min = -anchor_max
    # width greater than 0 => masking all non-existing boxes
    valid_mask = boxes_wh[..., 0] > 0

    # loop over all elements in this batch
    for i_bsz in range(batch_size):
        # Extract all rows (boxes) with non-zero width => Discard zero rows.
        wh = boxes_wh[i_bsz, valid_mask[i_bsz]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        # shift width/height to max/min from center point
        box_wh_max = wh / 2.0
        box_wh_min = -box_wh_max
        # calculate intersection with anchor boxes
        iou = calc_iou_between_boxes(box_wh_min, box_wh_max, wh, anchor_min, anchor_max, anchors)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        # iterate over best anchors
        for i_anc, anc in enumerate(best_anchor):
            # iterate over layers (i.e. scaling layers??)
            for i_lyr in range(num_layers):
                # check if anchor is in mask => check if it is looking at the correct mask
                if anc in anchor_mask[i_lyr]:
                    # multiply the (true) box with the different grid shapes for the different scales to update the true
                    # box (or rather to create a target box)
                    # => calculate indices for grid shape
                    i = np.floor(true_boxes[i_bsz, i_anc, 0] * grid_shapes[i_lyr][1]).astype("int32")
                    j = np.floor(true_boxes[i_bsz, i_anc, 1] * grid_shapes[i_lyr][0]).astype("int32")
                    # get index of the (masked) anchor
                    idx_anc = anchor_mask[i_lyr].index(anc)
                    class_id = true_boxes[i_bsz, i_anc, 4].astype("int32")
                    # copy true box coordinates to the output matrix/vector
                    # y_true: list(numpy.ndarray) => [(batch_size, grid_shape, num_anchors, num_classes + 5]
                    y_true[i_lyr][i_bsz, j, i, idx_anc, 0:4] = true_boxes[i_bsz, i_anc, 0:4]
                    y_true[i_lyr][i_bsz, j, i, idx_anc, 4] = 1
                    y_true[i_lyr][i_bsz, j, i, idx_anc, 5 + class_id] = 1
    return y_true


def calc_iou_between_boxes(b1_min, b1_max, b1_wh, b2_min, b2_max, b2_wh):
    # calculate intersection with anchor boxes
    if isinstance(b1_min, tf.Tensor):
        intersect_min = K.maximum(b1_min, b2_min)
        intersect_max = K.minimum(b1_max, b2_max)
        intersect_wh = K.maximum(intersect_max - intersect_min, 0.0)
    elif isinstance(b1_min, np.ndarray):
        intersect_min = np.maximum(b1_min, b2_min)
        intersect_max = np.minimum(b1_max, b2_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
    else:
        raise TypeError("Unknown input type.")
    # compare area of intersected boxes to the area of the original boxes => IOU = intersection over union
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    iou = calc_iou_between_boxes(b1_min, b1_max, b1_wh, b2_min, b2_max, b2_wh)
    return iou


if __name__ == "__main__":
    input_shape = (1024, 1024)
    model = YOLOv3(input_shape=input_shape,
                   num_classes=2,
                   anchors=[(256, 473), (414, 364), (414, 147), (256, 38), (97, 147), (97, 364),
                            (694, 888), (877, 512), (694, 135), (329, 135), (146, 511), (329, 888)],
                   tiny_yolo=True
                   )
    model.load_model(pl.Path("../trained_models") / "2210301424_YOLOv3_rgb.h5")

    for p2img in (pl.Path("../Data_ogl_TEST/withBox") / "Tst").glob("*.bmp"):
        img = Image.open(p2img).convert("RGB")
        boxes, scores, classes = model.predict(img, th_score=0)

        img_out = draw_box(img.resize(input_shape), boxes, scores, classes)
        img_out.show()
