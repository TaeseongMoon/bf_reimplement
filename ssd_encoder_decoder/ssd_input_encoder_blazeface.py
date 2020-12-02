
from __future__ import division
import numpy as np
import tensorflow as tf

class SSDInputEncoder:

    def __init__(self,
                 img_height,
                 img_width,
                 normalize_coords=True):

        self.img_height = img_height
        self.img_width = img_width
        self.normalize_coords=normalize_coords
        self.boxes_list = [] # This will store the anchor boxes for each predicotr layer.
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

        return landmark_tensor, anchor_tensor

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
