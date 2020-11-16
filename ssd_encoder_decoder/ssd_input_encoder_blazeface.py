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

        # predictor_sizes = np.array(predictor_sizes)
        # if predictor_sizes.ndim == 1:
        #     predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # Handle exceptions.
        ##################################################################################

        # if (min_scale is None or max_scale is None) and scales is None:
        #     raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        # if scales:
        #     if (len(scales) != predictor_sizes.shape[0]): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
        #         raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes), but len(scales) == {} and len(predictor_sizes) == {}".format(len(scales), len(predictor_sizes)))
        #     scales = np.array(scales)
        #     for sc in scales:
        #         for s in sc:
        #             if s <= 0:
        #                 raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        # else: # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
        #     if not 0 < min_scale <= max_scale:
        #         raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        # if not (aspect_ratios_per_layer is None):
        #     if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
        #         raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
        #     for aspect_ratios in aspect_ratios_per_layer:
        #         for ar in aspect_ratios:
        #             if ar <= 0:
        #                 raise ValueError("All aspect ratios must be greater than zero.")
        # else:
        #     if (aspect_ratios_global is None):
        #         raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
        #     if np.any(np.array(aspect_ratios_global) <= 0):
        #         raise ValueError("All aspect ratios must be greater than zero.")

        # if len(variances) != 4:
        #     raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        # variances = np.array(variances)
        # if np.any(variances <= 0):
        #     raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        # if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
        #     raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        # if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
        #     raise ValueError("You must provide at least one step value per predictor layer.")

        # if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
        #     raise ValueError("You must provide at least one offset value per predictor layer.")

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
        self.anchor = [x for x in open('anchors.txt').readline().split(',') if x != '']
    

        # for i in range(len(self.predictor_sizes)):
        #     boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
        #                                                                            aspect_ratios=self.aspect_ratios[i],
        #                                                                            scales=self.scales[i],
        #                                                                            this_steps=self.steps[i],
        #                                                                            this_offsets=self.offsets[i],
        #                                                                            diagnostics=True)
        #     self.boxes_list.append(boxes)
        #     self.wh_list_diag.append(wh)
        #     self.steps_diag.append(step)
        #     self.offsets_diag.append(offset)
        #     self.centers_diag.append(center)

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
            
            
            # classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            # labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1) # The one-hot version of the labels for this batch item
            # landmark_one_hot = np.concatenate([classes_one_hot, labels[:, 1:53]], axis=-1)
            
            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            
            # similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-8:-4], coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)
            
            # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            #        This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            # bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            # y_encoded[i, bipartite_matches, :] = labels_one_hot
            
            for _ in range(y_encoded.shape[1]):
                y_encoded[i,:, :52] = labels[:, 1:]
            y_encoded[:,:, 52:] = np.greater_equal(y_encoded[:, :,:52], 0).astype(int)

            # y_encoded[i, 135, :-8] = landmark_one_hot
            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            # similarities[:, bipartite_matches] = 0

            # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #         such ground truth box.

            # if self.matching_type == 'multi':

            #     # Get all matches that satisfy the IoU threshold.
            #     matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

            #     # Write the ground truth data to the matched anchor boxes.
            #     #y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]
            #     y_encoded[i, matches[1], :-8] = landmark_one_hot[matches[0]]
            #     # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            #     similarities[:, matches[1]] = 0

            # # Third: Now after the matching is done, all negative (background) anchor boxes that have
            # #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
            # #        i.e. they will no longer be background boxes. These anchors are "too close" to a
            # #        ground truth box to be valid background boxes.

            # max_background_similarities = np.amax(similarities, axis=0)
            # neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            # y_encoded[i, neutral_boxes, self.background_id] = 0

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

        

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        scales,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):

        size = min(self.img_height, self.img_width)

        wh_list = []
        for scale in scales:
            # Compute the box widths and and heights for all aspect ratios
            for ar in aspect_ratios:
                box_width = scale * size * np.sqrt(ar)
                box_height = scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))


        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        # if self.normalize_coords:
        #     boxes_tensor[:, :, :, [0, 2]] /= self.img_width
        #     boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        
        anchor_tensor = np.tile(self.anchor, (batch_size, 1, 1)).astype(np.float)
        
        # anchor_tensor = np.divide(anchor_tensor, normalize_tensor) # Normalize ymin and ymax relative to the image height
        anchor_tensor[:,0:51:2] /= self.img_height
        anchor_tensor[:,1:52:2] /= self.img_width
        
        

        # Concatenate the anchor tensors from the individual layers to one.
        # boxes_tensor = np.concatenate(boxes_batch, axis=1)
        # classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))
        # landmark_tensor = np.concatenate(ld_batch, axis=1)
        
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
