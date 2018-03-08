# -*- coding: utf-8 -*-
import pycolor
import preprocessing
from os.path import abspath, dirname, exists, join
import os
import sys
import cv2
import csv
import timeit
import pandas as pd
import numpy as np
import base64


class ColorDetect:
    """"""
    def __init__(self, **kwargs):
        
        self._mapping_file = self._read_excel("hex_to_color_category.xlsx")

    def _read_excel(self, filename):

        return pd.read_excel(
            self._get_file_abs_path(
                filename
                )
            )

    def _get_file_abs_path(self, filename):

        return join(dirname(abspath(__file__)), filename)
    
    def predict(self, input_file):

        # img = base64.b64decode(input_base64)
        # img_array = np.fromstring(img, np.uint8)
        # input_file = cv2.imdecode(img_array, 1)

        # ip_converted = preprocessing.resizing(input_base64)
        segmented_image = preprocessing.image_segmentation(
                preprocessing.resizing(input_file)
            )
        # processed_image = preprocessing.removebg(segmented_image)
        detect = pycolor.detect_color(
                segmented_image,
                self._mapping_file
            )
        return (detect)


obj = ColorDetect()
print (obj.predict(cv2.imread(sys.argv[1])))
