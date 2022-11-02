import unittest
import numpy as np
from numpy.testing import assert_almost_equal

import BBox_Transform as trafo


class MyTestCase(unittest.TestCase):
    x_min = 50
    y_min = 100
    w = 42
    h = 17

    x_max = x_min + w
    y_max = y_min + h
    x0 = x_min + w / 2
    y0 = y_min + h / 2

    image_size = (1, 10)

    # def assert_trafo(self, bbox, bdes, format_from, format_to):
    #     out = trafo.
    #         BBoxTransform().transform(bbox,
    #                                     format_from=format_from,
    #                                     format_to=format_to,
    #                                     image_size=self.image_size
    #                                     )
    #     assert_almost_equal(bdes, out, decimal=5)

    def test_coco2pascal_voc(self):
        """min_wh2min_max"""
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.x_max, self.y_max]
        out = trafo.coco2pacal_voc(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_coco2yolo_abs(self):
        """min_wh2x0y0wh"""
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = [self.x0, self.y0, self.w, self.h]
        out = trafo._coco2yolo_abs(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_yolo_abs2pascal_voc(self):
        """x0y0wh2min_max"""
        bbox = [self.x0, self.y0, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.x_max, self.y_max]
        out = trafo._yolo_abs2pascal_voc(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_yolo_abs2coco(self):
        """x0y0wh2min_wh"""
        bbox = [self.x0, self.y0, self.w, self.h]
        bdes = [self.x_min, self.y_min, self.w, self.h]
        out = trafo._yolo_abs2coco(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_pascal_voc2coco(self):
        """min_max2min_wh"""
        bbox = [self.x_min, self.y_min, self.x_max, self.y_max]
        bdes = [self.x_min, self.y_min, self.w, self.h]
        out = trafo.pascal_voc2coco(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_pascal_voc2yolo_abs(self):
        """min_max2x0y0wh"""
        bbox = [self.x_min, self.y_min, self.x_max, self.y_max]
        bdes = [self.x0, self.y0, self.w, self.h]
        out = trafo._pascal_voc2yolo_abs(bbox)
        assert_almost_equal(bdes, out, decimal=5)

    def test_coco2yolo(self):
        """min_wh2x0y0wh"""
        bbox = [self.x_min, self.y_min, self.w, self.h]
        bdes = np.array([self.x0, self.y0, self.w, self.h]) / (self.image_size * 2)
        out = trafo.coco2yolo(bbox, self.image_size)
        assert_almost_equal(bdes, out, decimal=5)

    def test_yolo2coco(self):
        """x0y0wh2min_wh
        """
        bbox = np.array([self.x0, self.y0, self.w, self.h]) / (self.image_size * 2)
        bdes = [self.x_min, self.y_min, self.w, self.h]
        out = trafo.yolo2coco(bbox, self.image_size)
        assert_almost_equal(bdes, out, decimal=5)


if __name__ == '__main__':
    unittest.main()
