'''
The data augmentation operations of the original SSD implementation.

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
import inspect
from random import randrange
from data_generator.object_detection_2d_photometric_ops import ConvertColor, ConvertDataType, ConvertTo3Channels, RandomBrightness, RandomContrast, RandomHue, RandomSaturation, RandomChannelSwap
from data_generator.object_detection_2d_geometric_ops import ResizeRandomInterp, RandomRotate

class Crop:

    def __init__(self, labels_format=None):
        self.labels_format = labels_format
        
    def __call__(self, image, labels=None, center=None, crop_size=256):
        cx, cy = center
        crop_size = crop_size//2

        image = image[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]

        labels = labels.astype(np.float)
        label_mask = np.less(labels, 0).astype(int)
        labels[:,1:52:2] -= cx-crop_size
        labels[:,2:53:2] -= cy-crop_size
        labels[label_mask == 1] = -1
        return image, labels

class SSDRandomCrop:

    def __init__(self, labels_format, random=False):
        self.random = random
        self.labels_format = labels_format
        self.random_crop = Crop(labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        self.random_crop.labels_format = self.labels_format
        h, w, _ = image.shape
        cx, cy = w//2, h//2
        if self.random:
            cx = randrange(cx-10, cx+10)
            cy = randrange(cy-10, cx+10)
            return self.random_crop(image, labels, [cx, cy], 256)
        else:
            crop_size = randrange(256, 280)
            return self.random_crop(image, labels, [cx, cy], crop_size)

class SSDPhotometricDistortions:

    def __init__(self):

        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.0)

        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_channel_swap]

        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.convert_to_float32,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.random_channel_swap]

    def __call__(self, image, labels):

        # Choose sequence 1 with probability 0.5.
        if np.random.choice(2):
            for transform in self.sequence1:
                image, labels = transform(image, labels)
            return image, labels
        # Choose sequence 2 with probability 0.5.
        else:
            for transform in self.sequence2:
                image, labels = transform(image, labels)
            return image, labels

class SSDDataAugmentation:

    def __init__(self,
                 img_height=256,
                 img_width=256,
                 background=(123, 117, 104),
                 rotate_prob=0.4,
                 random_crop_prob=0.5,
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):

        self.labels_format = labels_format
        self.random_crop_prob=random_crop_prob
        self.photometric_distortions = SSDPhotometricDistortions()
        self.random_crop = SSDRandomCrop(labels_format=self.labels_format, random=True)
        self.center_crop = SSDRandomCrop(labels_format=self.labels_format, random=False)
        self.random_rotate = RandomRotate(prob=rotate_prob,labels_format=self.labels_format)
        
        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4],
                                         labels_format=self.labels_format)

        self.sequence = [self.photometric_distortions,
                         self.random_crop,
                         self.random_rotate,
                         self.resize
                        ]

        self.sequence2 = [self.photometric_distortions,
                          self.center_crop,
                          self.random_rotate,
                          self.resize
                          ]

    def __call__(self, image, labels):
        self.random_crop.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        p = np.random.uniform(0,1)
        
        if p >= self.random_crop_prob:
            for transform in self.sequence:
                image, labels = transform(image, labels)
        else:
            for transform in self.sequence2:
                image, labels = transform(image, labels)
        return image, labels
