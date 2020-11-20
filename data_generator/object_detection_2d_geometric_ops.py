'''
Various geometric image transformations for 2D object detection, both deterministic
and probabilistic.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import cv2
import random

from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator

class Resize:
    '''
    Resizes images to a specified height and width in pixels.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 fix_image_ratio=True,
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.fix_image_ratio = fix_image_ratio
    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        kp1_x = self.labels_format['kp1_x']
        kp1_y = self.labels_format['kp1_y']
        kp2_x = self.labels_format['kp2_x']
        kp2_y = self.labels_format['kp2_y']
        kp3_x = self.labels_format['kp3_x']
        kp3_y = self.labels_format['kp3_y']
        kp4_x = self.labels_format['kp4_x']
        kp4_y = self.labels_format['kp4_y']
        kp5_x = self.labels_format['kp5_x']
        kp5_y = self.labels_format['kp5_y']
        kp6_x = self.labels_format['kp6_x']
        kp6_y = self.labels_format['kp6_y']
        kp7_x = self.labels_format['kp7_x']
        kp7_y = self.labels_format['kp7_y']
        kp8_x = self.labels_format['kp8_x']
        kp8_y = self.labels_format['kp8_y']
        kp9_x = self.labels_format['kp9_x']
        kp9_y = self.labels_format['kp9_y']
        kp10_x = self.labels_format['kp10_x']
        kp10_y = self.labels_format['kp10_y']
        kp11_x = self.labels_format['kp11_x']
        kp11_y = self.labels_format['kp11_y']
        kp12_x = self.labels_format['kp12_x']
        kp12_y = self.labels_format['kp12_y']
        kp13_x = self.labels_format['kp13_x']
        kp13_y = self.labels_format['kp13_y']
        kp14_x = self.labels_format['kp14_x']
        kp14_y = self.labels_format['kp14_y']
        kp15_x = self.labels_format['kp15_x']
        kp15_y = self.labels_format['kp15_y']
        kp16_x = self.labels_format['kp16_x']
        kp16_y = self.labels_format['kp16_y']
        kp17_x = self.labels_format['kp17_x']
        kp17_y = self.labels_format['kp17_y']
        kp18_x = self.labels_format['kp18_x']
        kp18_y = self.labels_format['kp18_y']
        kp19_x = self.labels_format['kp19_x']
        kp19_y = self.labels_format['kp19_y']
        kp20_x = self.labels_format['kp20_x']
        kp20_y = self.labels_format['kp20_y']
        kp21_x = self.labels_format['kp21_x']
        kp21_y = self.labels_format['kp21_y']
        kp22_x = self.labels_format['kp22_x']
        kp22_y = self.labels_format['kp22_y']
        kp23_x = self.labels_format['kp23_x']
        kp23_y = self.labels_format['kp23_y']
        kp24_x = self.labels_format['kp24_x']
        kp24_y = self.labels_format['kp24_y']
        kp25_x = self.labels_format['kp25_x']
        kp25_y = self.labels_format['kp25_y']
        kp26_x = self.labels_format['kp26_x']
        kp26_y = self.labels_format['kp26_y']

        if self.fix_image_ratio:
            return image, labels
        else:
            image = cv2.resize(image,
                            dsize=(self.out_width, self.out_height),
                            interpolation=self.interpolation_mode)

            if return_inverter:
                def inverter(labels):
                    labels = np.copy(labels)
                    labels[:, [ymin+1, ymax+1, kp1_y+1, kp2_y+1, kp3_y+1, kp4_y+1, kp5_y+1]] = np.round(labels[:, [ymin+1, ymax+1, kp1_y+1, kp2_y+1, kp3_y+1, kp4_y+1, kp5_y+1]] * (img_height / self.out_height), decimals=0)
                    labels[:, [xmin+1, xmax+1, kp1_x+1, kp2_x+1, kp3_x+1, kp4_x+1, kp5_x+1]] = np.round(labels[:, [xmin+1, xmax+1, kp1_x+1, kp2_x+1, kp3_x+1, kp4_x+1, kp5_x+1]] * (img_width / self.out_width), decimals=0)
                    return labels

            if labels is None:
                if return_inverter:
                    return image, inverter
                else:
                    return image
            else:
                labels = np.copy(labels)
                labels[:, [kp1_y,kp2_y,kp3_y,kp4_y,kp5_y,kp6_y,kp7_y,kp8_y,kp9_y,kp10_y,kp11_y,kp12_y,kp13_y,kp14_y,kp15_y,kp16_y,kp17_y,kp18_y,kp19_y,kp20_y, kp21_y,kp22_y,kp23_y,kp24_y,kp25_y,kp26_y]] = np.round(labels[:, [kp1_y,kp2_y,kp3_y,kp4_y,kp5_y,kp6_y,kp7_y,kp8_y,kp9_y,kp10_y,kp11_y,kp12_y,kp13_y,kp14_y,kp15_y,kp16_y,kp17_y,kp18_y,kp19_y,kp20_y, kp21_y,kp22_y,kp23_y,kp24_y,kp25_y,kp26_y]] * (self.out_height / img_height), decimals=0)
                labels[:, [kp1_x,kp2_x,kp3_x,kp4_x,kp5_x,kp6_x,kp7_x,kp8_x,kp9_x,kp10_x,kp11_x,kp12_x,kp13_x,kp14_x,kp15_x,kp16_x,kp17_x,kp18_x,kp19_x,kp20_x, kp21_x,kp22_x,kp23_x,kp24_x,kp25_x,kp26_x]] = np.round(labels[:, [kp1_x,kp2_x,kp3_x,kp4_x,kp5_x,kp6_x,kp7_x,kp8_x,kp9_x,kp10_x,kp11_x,kp12_x,kp13_x,kp14_x,kp15_x,kp16_x,kp17_x,kp18_x,kp19_x,kp20_x, kp21_x,kp22_x,kp23_x,kp24_x,kp25_x,kp26_x]] * (self.out_width / img_width), decimals=0)

                # if not (self.box_filter is None):
                #     self.box_filter.labels_format = self.labels_format
                #     labels = self.box_filter(labels=labels,
                #                              image_height=self.out_height,
                #                              image_width=self.out_width)

                if return_inverter:
                    return image, labels, inverter
                else:
                    return image, labels

class ResizeRandomInterp:
    '''
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    '''

    def __init__(self,
                 height,
                 width,
                 fix_image_ratio=True,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):
        '''
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers
                that represent valid OpenCV interpolation modes. For example,
                integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.fix_image_ratio=fix_image_ratio
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             fix_image_ratio=self.fix_image_ratio,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)

class Flip:
    '''
    Flips images horizontally or vertically.
    '''
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']
        kp1_x = self.labels_format['kp1_x']
        kp1_y = self.labels_format['kp1_y']
        kp2_x = self.labels_format['kp2_x']
        kp2_y = self.labels_format['kp2_y']
        kp3_x = self.labels_format['kp3_x']
        kp3_y = self.labels_format['kp3_y']
        kp4_x = self.labels_format['kp4_x']
        kp4_y = self.labels_format['kp4_y']
        kp5_x = self.labels_format['kp5_x']
        kp5_y = self.labels_format['kp5_y']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] = img_width - labels[:, [xmax, xmin, kp2_x, kp1_x, kp3_x, kp5_x, kp4_x]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] = img_height - labels[:, [ymax, ymin, kp2_y, kp1_y, kp3_y, kp5_y, kp4_y]]
                return image, labels

class RandomFlip:
    '''
    Randomly flips images horizontally or vertically. The randomness only refers
    to whether or not the image will be flipped.
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Translate:
    '''
    Translates images horizontally and/or vertically.
    '''

    def __init__(self,
                 dy,
                 dx,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            dy (float): The fraction of the image height by which to translate images along the
                vertical axis. Positive values translate images downwards, negative values
                translate images upwards.
            dx (float): The fraction of the image width by which to translate images along the
                horizontal axis. Positive values translate images to the right, negative values
                translate images to the left.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Compute the translation matrix.
        dy_abs = int(round(img_height * self.dy_rel))
        dx_abs = int(round(img_width * self.dx_rel))
        M = np.float32([[1, 0, dx_abs],
                        [0, 1, dy_abs]])

        # Translate the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']
            kp1_x = self.labels_format['kp1_x']
            kp1_y = self.labels_format['kp1_y']
            kp2_x = self.labels_format['kp2_x']
            kp2_y = self.labels_format['kp2_y']
            kp3_x = self.labels_format['kp3_x']
            kp3_y = self.labels_format['kp3_y']
            kp4_x = self.labels_format['kp4_x']
            kp4_y = self.labels_format['kp4_y']
            kp5_x = self.labels_format['kp5_x']
            kp5_y = self.labels_format['kp5_y']

            labels = np.copy(labels)
            # Translate the box coordinates to the translated image's coordinate system.
            labels[:,[xmin,xmax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] += dx_abs
            labels[:,[ymin,ymax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] += dy_abs

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] = np.clip(labels[:,[ymin,ymax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]], a_min=0, a_max=img_height-1)
                labels[:,[xmin,xmax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] = np.clip(labels[:,[xmin,xmax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomTranslate:
    '''
    Randomly translates images horizontally and/or vertically.
    '''

    def __init__(self,
                 dy_minmax=(0.03,0.3),
                 dx_minmax=(0.03,0.3),
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            dy_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that
                determines the minimum and maximum relative translation of images along the vertical
                axis both upward and downward. That is, images will be randomly translated by at least
                `min` and at most `max` either upward or downward. For example, if `dy_minmax == (0.05,0.3)`,
                an image of size `(100,100)` will be translated by at least 5 and at most 30 pixels
                either upward or downward. The translation direction is chosen randomly.
            dx_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that
                determines the minimum and maximum relative translation of images along the horizontal
                axis both to the left and right. That is, images will be randomly translated by at least
                `min` and at most `max` either left or right. For example, if `dx_minmax == (0.05,0.3)`,
                an image of size `(100,100)` will be translated by at least 5 and at most 30 pixels
                either left or right. The translation direction is chosen randomly.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a translated image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if dy_minmax[0] > dy_minmax[1]:
            raise ValueError("It must be `dy_minmax[0] <= dy_minmax[1]`.")
        if dx_minmax[0] > dx_minmax[1]:
            raise ValueError("It must be `dx_minmax[0] <= dx_minmax[1]`.")
        if dy_minmax[0] < 0 or dx_minmax[0] < 0:
            raise ValueError("It must be `dy_minmax[0] >= 0` and `dx_minmax[0] >= 0`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.dy_minmax = dy_minmax
        self.dx_minmax = dx_minmax
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.translate = Translate(dy=0,
                                   dx=0,
                                   clip_boxes=self.clip_boxes,
                                   box_filter=self.box_filter,
                                   background=self.background,
                                   labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']
            kp1_x = self.labels_format['kp1_x']
            kp1_y = self.labels_format['kp1_y']
            kp2_x = self.labels_format['kp2_x']
            kp2_y = self.labels_format['kp2_y']
            kp3_x = self.labels_format['kp3_x']
            kp3_y = self.labels_format['kp3_y']
            kp4_x = self.labels_format['kp4_x']
            kp4_y = self.labels_format['kp4_y']
            kp5_x = self.labels_format['kp5_x']
            kp5_y = self.labels_format['kp5_y']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.translate.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick the relative amount by which to translate.
                dy_abs = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
                dx_abs = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
                # Pick the direction in which to translate.
                dy = np.random.choice([-dy_abs, dy_abs])
                dx = np.random.choice([-dx_abs, dx_abs])
                self.translate.dy_rel = dy
                self.translate.dx_rel = dx

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.translate(image, labels)
                else:
                    # Translate the box coordinates to the translated image's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] += int(round(img_height * dy))
                    new_labels[:, [xmin, xmax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] += int(round(img_width * dx))

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.translate(image, labels)

            # If all attempts failed, return the unaltered input image.
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Scale:
    '''
    Scales images, i.e. zooms in or out.
    '''

    def __init__(self,
                 factor,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            factor (float): The fraction of the image size by which to scale images. Must be positive.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if factor <= 0:
            raise ValueError("It must be `factor > 0`.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.factor = factor
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]
        import pdb
        pdb.set_trace()
        # Compute the rotation matrix.
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=0,
                                    scale=self.factor)

        # Scale the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']
            kp1_x = self.labels_format['kp1_x']
            kp1_y = self.labels_format['kp1_y']
            kp2_x = self.labels_format['kp2_x']
            kp2_y = self.labels_format['kp2_y']
            kp3_x = self.labels_format['kp3_x']
            kp3_y = self.labels_format['kp3_y']
            kp4_x = self.labels_format['kp4_x']
            kp4_y = self.labels_format['kp4_y']
            kp5_x = self.labels_format['kp5_x']
            kp5_y = self.labels_format['kp5_y']

            labels = np.copy(labels)
            # Scale the bounding boxes accordingly.
            # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
            toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:,[xmin, ymin, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:,[xmax, ymax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin, ymax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]] = np.clip(labels[:,[ymin, ymax, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y]], a_min=0, a_max=img_height-1)
                labels[:,[xmin, xmax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]] = np.clip(labels[:,[xmin, xmax, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomScale:
    '''
    Randomly scales images.
    '''

    def __init__(self,
                 min_factor=0.5,
                 max_factor=1.5,
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4, 'kp1_x':5, 'kp1_y':6, 'kp2_x':7, 'kp2_y':8, 'kp3_x':9, 'kp3_y':10, 'kp4_x':11, 'kp4_y':12, 'kp5_x':13, 'kp5_y':14}):
        '''
        Arguments:
            min_factor (float, optional): The minimum fraction of the image size by which to scale images.
                Must be positive.
            max_factor (float, optional): The maximum fraction of the image size by which to scale images.
                Must be positive.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a scaled image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not (0 < min_factor <= max_factor):
            raise ValueError("It must be `0 < min_factor <= max_factor`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.scale = Scale(factor=1.0,
                           clip_boxes=self.clip_boxes,
                           box_filter=self.box_filter,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']
            kp1_x = self.labels_format['kp1_x']
            kp1_y = self.labels_format['kp1_y']
            kp2_x = self.labels_format['kp2_x']
            kp2_y = self.labels_format['kp2_y']
            kp3_x = self.labels_format['kp3_x']
            kp3_y = self.labels_format['kp3_y']
            kp4_x = self.labels_format['kp4_x']
            kp4_y = self.labels_format['kp4_y']
            kp5_x = self.labels_format['kp5_x']
            kp5_y = self.labels_format['kp5_y']
            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.scale.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick a scaling factor.
                factor = np.random.uniform(self.min_factor, self.max_factor)
                self.scale.factor = factor

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.scale(image, labels)
                else:
                    # Scale the bounding boxes accordingly.
                    # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
                    toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
                    bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])

                    # Compute the rotation matrix.
                    M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                angle=0,
                                                scale=factor)

                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T

                    new_labels = np.copy(labels)
                    new_labels[:,[xmin,ymin]] = np.around(new_toplefts, decimals=0).astype(np.int)
                    new_labels[:,[xmax,ymax]] = np.around(new_bottomrights, decimals=0).astype(np.int)

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.scale(image, labels)

            # If all attempts failed, return the unaltered input image.
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Rotate:
    '''
    Rotates images counter-clockwise by 90, 180, or 270 degrees.
    '''

    def __init__(self,
                 angle,
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):
        '''
        Arguments:
            angle (int): The angle in degrees by which to rotate the images counter-clockwise.
                Only 90, 180, and 270 are valid values.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not angle in {0,5,10,15,345,350,355}:
            raise ValueError("`angle` must be in the set {-30 ~ 30 degree}.")
        self.angle = angle
        self.labels_format = labels_format
        

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]
        kp1_x = self.labels_format['kp1_x']
        kp1_y = self.labels_format['kp1_y']
        kp2_x = self.labels_format['kp2_x']
        kp2_y = self.labels_format['kp2_y']
        kp3_x = self.labels_format['kp3_x']
        kp3_y = self.labels_format['kp3_y']
        kp4_x = self.labels_format['kp4_x']
        kp4_y = self.labels_format['kp4_y']
        kp5_x = self.labels_format['kp5_x']
        kp5_y = self.labels_format['kp5_y']
        kp6_x = self.labels_format['kp6_x']
        kp6_y = self.labels_format['kp6_y']
        kp7_x = self.labels_format['kp7_x']
        kp7_y = self.labels_format['kp7_y']
        kp8_x = self.labels_format['kp8_x']
        kp8_y = self.labels_format['kp8_y']
        kp9_x = self.labels_format['kp9_x']
        kp9_y = self.labels_format['kp9_y']
        kp10_x = self.labels_format['kp10_x']
        kp10_y = self.labels_format['kp10_y']
        kp11_x = self.labels_format['kp11_x']
        kp11_y = self.labels_format['kp11_y']
        kp12_x = self.labels_format['kp12_x']
        kp12_y = self.labels_format['kp12_y']
        kp13_x = self.labels_format['kp13_x']
        kp13_y = self.labels_format['kp13_y']
        kp14_x = self.labels_format['kp14_x']
        kp14_y = self.labels_format['kp14_y']
        kp15_x = self.labels_format['kp15_x']
        kp15_y = self.labels_format['kp15_y']
        kp16_x = self.labels_format['kp16_x']
        kp16_y = self.labels_format['kp16_y']
        kp17_x = self.labels_format['kp17_x']
        kp17_y = self.labels_format['kp17_y']
        kp18_x = self.labels_format['kp18_x']
        kp18_y = self.labels_format['kp18_y']
        kp19_x = self.labels_format['kp19_x']
        kp19_y = self.labels_format['kp19_y']
        kp20_x = self.labels_format['kp20_x']
        kp20_y = self.labels_format['kp20_y']
        kp21_x = self.labels_format['kp21_x']
        kp21_y = self.labels_format['kp21_y']
        kp22_x = self.labels_format['kp22_x']
        kp22_y = self.labels_format['kp22_y']
        kp23_x = self.labels_format['kp23_x']
        kp23_y = self.labels_format['kp23_y']
        kp24_x = self.labels_format['kp24_x']
        kp24_y = self.labels_format['kp24_y']
        kp25_x = self.labels_format['kp25_x']
        kp25_y = self.labels_format['kp25_y']
        kp26_x = self.labels_format['kp26_x']
        kp26_y = self.labels_format['kp26_y']
        kp_x = [kp1_x, kp2_x, kp3_x, kp4_x, kp5_x, kp6_x, kp7_x, kp8_x, kp9_x, kp10_x, kp11_x, kp12_x, kp13_x, kp14_x, kp15_x, kp16_x, kp17_x, kp18_x, kp19_x, kp20_x, kp21_x, kp22_x, kp23_x, kp24_x, kp25_x, kp26_x]
        kp_y = [kp1_y, kp2_y, kp3_y, kp4_y, kp5_y, kp6_y, kp7_y, kp8_y, kp9_y, kp10_y, kp11_y, kp12_y, kp13_y, kp14_y, kp15_y, kp16_y, kp17_y, kp18_y, kp19_y, kp20_y, kp21_y, kp22_y, kp23_y, kp24_y, kp25_y, kp26_y]
        # Compute the rotation matrix.
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=self.angle,
                                    scale=1)

        # Get the sine and cosine from the rotation matrix.
        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image.
        img_width_new = int(img_height * sin_angle + img_width * cos_angle)
        img_height_new = int(img_height * cos_angle + img_width * sin_angle)

        # Adjust the rotation matrix to take into account the translation.
        M[1, 2] += (img_height_new - img_height) / 2
        M[0, 2] += (img_width_new - img_width) / 2

        # Rotate the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width_new, img_height_new))

        if labels is None:
            return image
        else:
            
            labels = np.copy(labels)
            rot_point = lambda x, y :(np.dot(M, np.array([x, y, np.ones(labels.shape[0])]))).T
            for x, y in zip(kp_x, kp_y):
                k = rot_point(float(labels[:,x][0]), float(labels[:,y][0]))
                labels[:, [x, y]] = np.round(k.tolist(), decimals =0).astype(np.int).reshape((1,2))

            return image, labels

class RandomRotate:
    '''
    Randomly rotates images counter-clockwise.
    '''

    def __init__(self,
                 angles=[0, 5, 10, 15, 345, 350, 355],
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            angle (list): The list of angles in degrees from which one is randomly selected to rotate
                the images counter-clockwise. Only 90, 180, and 270 are valid values.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        for angle in angles:
            if not angle in {0,5,10,15,345,350,355}:
                raise ValueError("`angles` can only contain the values 90, 180, and 270.")
        self.angles = angles
        self.prob = prob
        self.labels_format = labels_format
        self.rotate = Rotate(angle=15, labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            # Pick a rotation angle.
            self.rotate.angle = random.choice(self.angles)
            self.rotate.labels_format = self.labels_format
            return self.rotate(image, labels)

        elif labels is None:
            return image

        else:
            return image, labels
