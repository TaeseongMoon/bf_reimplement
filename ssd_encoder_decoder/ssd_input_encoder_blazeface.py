'''
An encoder that converts ground truth annotations to SSD-compatible training targets.

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
from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:
    '''
    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):


        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # + 1 for the background class
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        # If `scales` is None, compute the scaling factors by linearly interpolating between
        # `min_scale` and `max_scale`. If an explicit list of `scales` is given, however,
        # then it takes precedent over `min_scale` and `max_scale`.
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
        else:
            # If a list of scales is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`.
            self.scales = scales
        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers. If `aspect_ratios_per_layer` is given,
        # however, then it takes precedent over `aspect_ratios_global`.
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # If aspect ratios are given per layer, we'll use those.
            self.aspect_ratios = aspect_ratios_per_layer
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id
 
        # Compute the number of boxes per spatial location for each predictor layer.
        # For example, if a predictor layer has three different aspect ratios, [1.0, 0.5, 2.0], and is
        # supposed to predict two boxes of slightly different size for aspect ratio 1.0, then that predictor
        # layer predicts a total of four boxes at every spatial location across the feature map.
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                self.n_boxes.append(len(aspect_ratios))
        else:
            self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################

        # Compute the anchor boxes for each predictor layer. We only have to do this once
        # since the anchor boxes depend only on the model configuration, not on the input data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.

        self.boxes_list = [] # This will store the anchor boxes for each predicotr layer.

        # The following lists just store diagnostic information. Sometimes it's handy to have the
        # boxes' center points, heights, widths, etc. in a list.
        self.wh_list_diag = [] # Box widths and heights for each predictor layer
        self.steps_diag = [] # Horizontal and vertical distances between any two boxes for each predictor layer
        self.offsets_diag = [] # Offsets for each predictor layer
        self.centers_diag = [] # Anchor box center points as `(cy, cx)` for each predictor layer
        self.anchor = []
        # Iterate over all predictor layers and compute the anchor boxes for each one.
        self.anchor = [x for x in open('anchor_256_fix.txt').readline().split(',') if x != '']
    

    def __call__(self, ground_truth_labels, diagnostics=False):

          
        batch_size = len(ground_truth_labels)

        ##################################################################################
        # Generate the template for y_encoded.
        ##################################################################################

        y_encoded, anchor_tensor = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)
        
        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################


        # y_encoded[:, :, self.background_id] = 1 # All boxes are background boxes by default.
        n_boxes = y_encoded.shape[1] # The total number of boxes that the model predicts per batch item
        class_vectors = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(batch_size): # For each batch item...

            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item
            
            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                labels[:,1:52:2] /= self.img_height # Normalize ymin and ymax relative to the image height
                labels[:,2:53:2] /= self.img_width # Normalize xmin and xmax relative to the image width
            
            for _ in range(y_encoded.shape[1]):
                y_encoded[i,:, :52] = np.tile(labels[:, 1:], (1,2,1))
            y_encoded[:,:, 52:] = np.greater_equal(y_encoded[:, :,:52], 0).astype(int)

        ###################################################s##########`#####################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################

        if self.coords == 'centroids':
            y_encoded[...,:52] -= anchor_tensor[...,:] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            # y_encoded[...,2:54] /= np.tile(y_encoded[...,[-6, -5]], 26) * np.tile(y_encoded[...,[-4,-3]], 26) # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        
        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-18:-14] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded


    def generate_encoding_template(self, batch_size, diagnostics=False):
        
        anchor_tensor = np.tile(self.anchor, (batch_size, 2, 1)).astype(np.float)
        
        # anchor_tensor = np.divide(anchor_tensor, normalize_tensor) # Normalize ymin and ymax relative to the image height
        anchor_tensor[:,0:51:2] /= self.img_height
        anchor_tensor[:,1:52:2] /= self.img_width
        
                
        landmark_tensor = np.zeros_like(anchor_tensor)
        landmark_tensor = np.concatenate((landmark_tensor, landmark_tensor), axis=2)
        # variances_tensor += self.variances # Long live broadcasting
        
        # y_encoding_template = np.concatenate((anchor_tensor, anchor_tensor, variances_tensor), axis=2)

        
        return landmark_tensor, anchor_tensor

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
