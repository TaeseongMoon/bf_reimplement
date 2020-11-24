'''
A custom Keras layer to decode the raw SSD prediction output. Corresponds to the
`DetectionOutput` layer type in the original Caffe implementation of SSD.

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

NOTICE: This file is a modified version by Viet Anh Nguyen (vietanh@vietanhdev.com)
'''

from __future__ import division
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, InputSpec

class DecodeDetections(Layer):

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):

        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        if coords != 'centroids':
            raise ValueError("The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')
        
        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        '''

        def normalized_coords():
            key_points = []
            for i in range(0,51,2):
                key_points.append(tf.expand_dims(y_pred[..., i] *self.tf_img_width, axis=-1))
                key_points.append(tf.expand_dims(y_pred[..., i+1] * self.tf_img_height, axis=-1))
            
            return  key_points
        def non_normalized_coords():
            key_points = []
            for i in range(0,51,2):
                key_points.append(tf.expand_dims(y_pred[..., i], axis=-1))
                key_points.append(tf.expand_dims(y_pred[..., i+1], axis=-1))
            
            return key_points

        key_points= tf.cond(pred=self.tf_normalize_coords, true_fn=normalized_coords, false_fn=non_normalized_coords)

        
        y_pred = tf.concat(values=key_points, axis=-1)
        
        #####################################################################################
        # 2. Perform confidence thresholding, per-class non-maximum suppression, and
        #    top-k filtering.
        #####################################################################################
        return y_pred
        

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        
        return (batch_size, self.tf_top_k, 12) # Last axis: (class_ID, confidence, 4 box coordinates)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
