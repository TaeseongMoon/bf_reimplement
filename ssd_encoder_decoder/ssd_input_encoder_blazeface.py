
from __future__ import division
import numpy as np
import tensorflow as tf
from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:

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

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # + 1 for the background class
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
        else:
            # If a list of scales is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`.
            self.scales = scales
            
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
 
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                self.n_boxes.append(len(aspect_ratios))
        else:
            self.n_boxes = len(aspect_ratios_global)


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
    

    def __call__(self, ground_truth_labels, select_keypoint_label ,diagnostics=False):

          
        batch_size = len(ground_truth_labels)
        
        ##################################################################################
        # Generate the template for y_encoded.
        ##################################################################################

        y_encoded, anchor_tensor = self.generate_encoding_template(batch_size=batch_size, select_keypoint_label=select_keypoint_label, diagnostics=False)
        
        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################

        for i in range(batch_size): # For each batch item...

            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item
            
            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                labels[:,1:52:2] /= self.img_height # Normalize ymin and ymax relative to the image height
                labels[:,2:53:2] /= self.img_width # Normalize xmin and xmax relative to the image width

            labels = labels[:,1:]
            labels = labels[:, [select_keypoint_label]]
            
            for _ in range(y_encoded.shape[1]):
                y_encoded[i,:, :len(select_keypoint_label)] = np.tile(labels, (1,1))
            y_encoded[..., len(select_keypoint_label):] = np.greater_equal(y_encoded[...,:len(select_keypoint_label)], 0).astype(int)
        
        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################
        y_encoded[...,:len(select_keypoint_label)] -= anchor_tensor[...,:] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        
        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-18:-14] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded


    def generate_encoding_template(self, batch_size, select_keypoint_label, diagnostics=False):
        
        anchor_tensor = np.tile(self.anchor, (batch_size, 1)).astype(np.float)
        
        anchor_tensor[...,0:51:2] /= self.img_height
        anchor_tensor[...,1:52:2] /= self.img_width

        anchor_tensor = anchor_tensor[..., [select_keypoint_label]]
                
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
