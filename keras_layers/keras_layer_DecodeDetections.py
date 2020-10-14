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
    '''
    A Keras layer to decode the raw SSD prediction output.

    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.

    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    '''

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
        '''
        All default argument values follow the Caffe implementation.

        Arguments:
            confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
                positive class in order to be considered for the non-maximum suppression stage for the respective class.
                A lower value will result in a larger part of the selection process being done by the non-maximum suppression
                stage, while a larger value will result in a larger part of the selection process happening in the confidence
                thresholding stage.
            iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
                with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
                to the box score.
            top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
                non-maximum suppression stage.
            nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
                suppression.
            coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
                i.e. the format `(cx, cy, w, h)` (box center coordinates, width, and height). Other coordinate formats are
                currently not supported.
            normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
                and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
                relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
                Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
                coordinates. Requires `img_height` and `img_width` if set to `True`.
            img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
            img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        '''
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

        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################
        print(y_pred.shape)
        # Convert anchor box offsets to image offsets.
        # cx = y_pred[...,-22] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        # cy = y_pred[...,-21] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        # w = tf.exp(y_pred[...,-20] * y_pred[...,-2]) * y_pred[...,-6] # w = exp(w_pred * variance_w) * w_anchor
        # h = tf.exp(y_pred[...,-19] * y_pred[...,-1]) * y_pred[...,-5] # h = exp(h_pred * variance_h) * h_anchor
        
        # Convert 'centroids' to 'corners'.
        # xmin = cx - 0.5 * w
        # ymin = cy - 0.5 * h
        # xmax = cx + 0.5 * w
        # ymax = cy + 0.5 * h
        
        kp1_x = y_pred[...,-18] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
        kp1_y = y_pred[...,-17] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
        kp2_x = y_pred[...,-16] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
        kp2_y = y_pred[...,-15] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
        kp3_x = y_pred[...,-14] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
        kp3_y = y_pred[...,-13] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
        kp4_x = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
        kp4_y = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
        kp5_x = y_pred[...,-10] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
        kp5_y = y_pred[...,-9] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]

        
        # If the model predicts box coordinates relative to the image dimensions and they are supposed
        # to be converted back to absolute coordinates, do that.
        def normalized_coords():
            # xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            # ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            # xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            # ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            kp1_x1 = tf.expand_dims(kp1_x * self.tf_img_width, axis=-1)
            kp1_y1 = tf.expand_dims(kp1_y * self.tf_img_height, axis=-1)
            kp2_x1 = tf.expand_dims(kp2_x * self.tf_img_width, axis=-1)
            kp2_y1 = tf.expand_dims(kp2_y * self.tf_img_height, axis=-1)
            kp3_x1 = tf.expand_dims(kp3_x * self.tf_img_width, axis=-1)
            kp3_y1 = tf.expand_dims(kp3_y * self.tf_img_height, axis=-1)
            kp4_x1 = tf.expand_dims(kp4_x * self.tf_img_width, axis=-1)
            kp4_y1 = tf.expand_dims(kp4_y * self.tf_img_height, axis=-1)
            kp5_x1 = tf.expand_dims(kp5_x * self.tf_img_width, axis=-1)
            kp5_y1 = tf.expand_dims(kp5_y * self.tf_img_height, axis=-1)
            return  kp1_x1, kp1_y1, kp2_x1, kp2_y1, kp3_x1, kp3_y1, kp4_x1, kp4_y1, kp5_x1, kp5_y1
        def non_normalized_coords():
            return tf.expand_dims(kp1_x, axis=-1), tf.expand_dims(kp1_y, axis=-1), tf.expand_dims(kp2_x, axis=-1), tf.expand_dims(kp2_y, axis=-1), tf.expand_dims(kp3_x, axis=-1), tf.expand_dims(kp3_y, axis=-1), tf.expand_dims(kp4_x, axis=-1), tf.expand_dims(kp4_y, axis=-1), tf.expand_dims(kp5_x, axis=-1), tf.expand_dims(kp5_y, axis=-1)

        kp1_x, kp1_y,kp2_x, kp2_y,kp3_x, kp3_y,kp4_x, kp4_y,kp5_x, kp5_y = tf.cond(pred=self.tf_normalize_coords, true_fn=normalized_coords, false_fn=non_normalized_coords)
        
        # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
        y_pred = tf.concat(values=[y_pred[...,:-18], kp1_x, kp1_y,kp2_x, kp2_y,kp3_x, kp3_y,kp4_x, kp4_y,kp5_x, kp5_y], axis=-1)
        
        #####################################################################################
        # 2. Perform confidence thresholding, per-class non-maximum suppression, and
        #    top-k filtering.
        #####################################################################################
        return y_pred
        # batch_size = tf.shape(input=y_pred)[0] # Output dtype: tf.int32
        # n_boxes = tf.shape(input=y_pred)[1]
        # n_classes = y_pred.shape[2] - 10
        # class_indices = tf.range(1, n_classes)
        
        # # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # # - confidence thresholding
        # # - non-maximum suppression (NMS)
        # # - top-k filtering
        # def filter_predictions(batch_item):

        #     # Create a function that filters the predictions for one single class.
        #     def filter_single_class(index):

        #         # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
        #         # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
        #         # confidnece values for just one class, determined by `index`.
        #         confidences = tf.expand_dims(batch_item[..., index], axis=-1)
        #         class_id = tf.fill(dims=tf.shape(input=confidences), value=tf.cast(index, dtype=tf.float32))
        #         box_coordinates = batch_item[...,-10:]
                
                
        #         single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

        #         # Apply confidence thresholding with respect to the class defined by `index`.
        #         threshold_met = single_class[:,1] > self.tf_confidence_thresh
        #         single_class = tf.boolean_mask(tensor=single_class,
        #                                        mask=threshold_met)

        #         # If any boxes made the threshold, perform NMS.
        #         def perform_nms():
        #         #     scores = single_class[...,1]

        #         #     # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
        #         #     xmin = tf.expand_dims(single_class[...,-14], axis=-1)
        #         #     ymin = tf.expand_dims(single_class[...,-13], axis=-1)
        #         #     xmax = tf.expand_dims(single_class[...,-12], axis=-1)
        #         #     ymax = tf.expand_dims(single_class[...,-11], axis=-1)
        #         #     boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)
                    
        #         #     # kp1_x = tf.expand_dims(single_class[...,-10] , axis=-1)
        #         #     # kp1_y = tf.expand_dims(single_class[...,-9] , axis=-1)
        #         #     # kp2_x = tf.expand_dims(single_class[...,-8] , axis=-1)
        #         #     # kp2_y = tf.expand_dims(single_class[...,-7] , axis=-1)
        #         #     # kp3_y = tf.expand_dims(single_class[...,-5] , axis=-1)
        #         #     # kp3_x = tf.expand_dims(single_class[...,-6] , axis=-1)
        #         #     # kp4_x = tf.expand_dims(single_class[...,-4] , axis=-1)
        #         #     # kp4_y = tf.expand_dims(single_class[...,-3] , axis=-1)
        #         #     # kp5_x = tf.expand_dims(single_class[...,-2] , axis=-1)
        #         #     # kp5_y = tf.expand_dims(single_class[...,-1] , axis=-1)
        #         #     # landmarks = tf.concat(values=[kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, kp4_x, kp4_y, kp5_x, kp5_y], axis=-1)
        #         #     # land_coord = tf.reshape(tensor=single_class[...,-10:], shape=[-1,10])
        #         #     #land_coord = tf.expand_dims(single_class[..., -10:], axis=-1)
        #         #     maxima_indices = tf.image.non_max_suppression(boxes=boxes,
        #         #                                                   scores=scores,
        #         #                                                   max_output_size=self.tf_nms_max_output_size,
        #         #                                                   iou_threshold=self.iou_threshold,
        #         #                                                   name='non_maximum_suppresion')
                    
        #             # maxima = tf.gather(params=single_class,
        #             #                    indices=maxima_indices,
        #             #                    axis=0)

                    
        #             # maxima = tf.concat(values=[maxima], axis=-1)
        #             return no_confident_predictions()

        #         def no_confident_predictions():
        #             return tf.constant(value=0.0, shape=(1,12))
                
        #         single_class_nms = tf.cond(pred=tf.equal(tf.size(input=single_class), 0), true_fn=no_confident_predictions, false_fn=perform_nms)


        #         # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
        #         padded_single_class = tf.pad(tensor=single_class_nms,
        #                                      paddings=[[0, self.tf_nms_max_output_size - tf.shape(input=single_class_nms)[0]], [0, 0]],
        #                                      mode='CONSTANT',
        #                                      constant_values=0.0)

        #         return padded_single_class

        #     # Iterate `filter_single_class()` over all class indices.
        #     filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
        #                                         elems=tf.range(1, n_classes),
        #                                         dtype=tf.float32,
        #                                         parallel_iterations=128,
        #                                         back_prop=False,
        #                                         swap_memory=False,
        #                                         infer_shape=True,
        #                                         name='loop_over_classes')

        #     # Concatenate the filtered results for all individual classes to one tensor.
        #     filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1,12))
        #     # import pdb
        #     # pdb.set_trace()
        #     # Perform top-k filtering for this batch item or pad it in case there are
        #     # fewer than `self.top_k` boxes left at this point. Either way, produce a
        #     # tensor of length `self.top_k`. By the time we return the final results tensor
        #     # for the whole batch, all batch items must have the same number of predicted
        #     # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
        #     # predictions are left after the filtering process above, we pad the missing
        #     # predictions with zeros as dummy entries.
        #     def top_k():
                
        #         return tf.gather(params=filtered_predictions,
        #                          indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
        #                          axis=0)
        #     def pad_and_top_k():
 
        #         padded_predictions = tf.pad(tensor=filtered_predictions,
        #                                     paddings=[[0, self.tf_top_k - tf.shape(input=filtered_predictions)[0]], [0, 0]],
        #                                     mode='CONSTANT',
        #                                     constant_values=0.0)
        #         return tf.gather(params=padded_predictions,
        #                          indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
        #                          axis=0)

        #     top_k_boxes = tf.cond(pred=tf.greater_equal(tf.shape(input=filtered_predictions)[0], self.tf_top_k), true_fn=top_k, false_fn=pad_and_top_k)

        #     return top_k_boxes

        # # Iterate `filter_predictions()` over all batch items.
        # output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
        #                           elems=y_pred,
        #                           dtype=None,
        #                           parallel_iterations=128,
        #                           back_prop=False,
        #                           swap_memory=False,
        #                           infer_shape=True,
        #                           name='loop_over_batch')

        # return output_tensor

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
