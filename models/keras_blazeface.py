'''
BlazeFace architecture

Copyright (C) 2019 Viet Anh Nguyen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference: https://arxiv.org/pdf/1907.05047.pdf
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Reshape, Average
from keras import Sequential
import keras.backend as K

from keras_layers.keras_layer_BlazeFace import BlazeFace
from keras_layers.keras_layer_DecodeDetections import DecodeDetections

def blazeface(image_size,
                n_classes,
                mode='training',
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                feature=52):


    n_classes += 1 # Account for the background class.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # Input Preprocessing
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    # Use BlazeFace to extract features
    # [(None, 16, 16, 96), (None, 8, 8, 96)]
    blaze_face = BlazeFace((img_height, img_width, img_channels))(x1)

    landmark_8x8 = Conv2D(filters=feature, kernel_size=(3, 3), strides=(2, 2), padding='same')(blaze_face[0])
    avg_8x8_landmark = AveragePooling2D(pool_size=(4, 4))(landmark_8x8)
    reshape_8x8_landmark = Reshape((-1, feature), name='reshape_8x8_landmark')(avg_8x8_landmark)
    
    landmark_16x16 = Conv2D(filters=feature, kernel_size=(3, 3), strides=(2, 2), padding='same')(blaze_face[1])
    avg_16x16_landmark = AveragePooling2D(pool_size=(8, 8))(landmark_16x16)
    reshape_16x16_landmark = Reshape((-1, feature), name='reshape_16x16_landmark')(avg_16x16_landmark)

    predictions = Average()([reshape_16x16_landmark, reshape_8x8_landmark])   

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               feature=feature,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    return model
