import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from BoundingBoxRegression import BBoxTransform


class MyTestCase(unittest.TestCase):
    x_min = 50
    y_min = 100
    w = 42
    h = 17

    x_max = x_min + w
    y_max = y_min + h
    x0 = int(x_min + (x_max - x_min) / 2)
    y0 = int(y_min + (y_max - y_min) / 2)

    image_size = (1, 10)

    def assert_trafo(self, bbox, bdes, format_from, format_to):
        out = BBoxTransform().transform(bbox,
                                        format_from=format_from,
                                        format_to=format_to,
                                        image_size=self.image_size
                                        )
        assert_almost_equal(bdes, out, decimal=5)

    def test_min_wh2min_max(self):
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.x_max, self.y_max]
        self.assert_trafo(bbox, bdes, "min_wh", "min_max")

    def test_min_wh2xywh(self):
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = [self.x0, self.y0, self.w, self.h]
        self.assert_trafo(bbox, bdes, "min_wh", "xywh")

    def test_xywh2min_max(self):
        bbox = [self.x0, self.y0, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.x_max, self.y_max]
        self.assert_trafo(bbox, bdes, "xywh", "min_max")

    def test_xywh2min_wh(self):
        bbox = [self.x0, self.y0, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.w, self.h]
        self.assert_trafo(bbox, bdes, "xywh", "min_wh")

    def test_min_max2min_wh(self):
        bbox = [self.x_min, self.y_min, self.x_max, self.y_max]
        bdes = [self.x_min, self.y_min, self.w, self.h]
        self.assert_trafo(bbox, bdes, "min_max", "min_wh")

    def test_min_max2xywh(self):
        bbox = [self.x_min, self.y_min, self.x_max, self.y_max]
        bdes = [self.x0, self.y0, self.w, self.h]
        self.assert_trafo(bbox, bdes, "min_max", "xywh")

    def test_coco2yolo(self):
        """
        The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        """
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = np.array([self.x0, self.y0, self.w, self.h]) / (self.image_size * 2) / 100
        self.assert_trafo(bbox, bdes, "coco", "yolo")

    def test_yolo2coco(self):
        """
        The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        """
        bbox = np.array([self.x0, self.y0, self.w, self.h]) / (self.image_size * 2) / 100
        bdes = [self.x_min, self.y_min, self.w, self.h]
        self.assert_trafo(bbox, bdes, "yolo", "coco")

    def test_yolo2yolo(self):
        """
        The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        """
        bbox = [self.x_min, self.y_min, self.w, self.h]
        self.assert_trafo(bbox, np.array(bbox) / 100, "yolo", "yolo")


if __name__ == '__main__':
    unittest.main()