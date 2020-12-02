import cv2
import numpy as np
from models.keras_blazeface import blazeface as BlazeFace
from mtcnn import MTCNN

class blazeface():
    _defaults = {
        "model_path": 'blazeface_model.h5',
        "key_points" : [9, 11, 13, 14, 21, 23, 24, 25, 26],
        "model_image_size" : (256, 256, 3),
        "anchor_path" : 'anchor_256_fix.txt'
    }

    def __init__(self):
        self.__dict__.update(self._defaults)
        self.kp_index = self._get_kp_index()
        self.anchors = self._get_anchors()
        self.blazeface = self.get_model()
        self.detector = self.get_detector()

    def _get_kp_index(self):
        output_index = []
        for i in self.key_points:
            output_index.append(i*2-2)
            output_index.append(i*2-1)
        return output_index

    def _get_anchors(self):
        anchor = np.array([ float(x) for x in open(self.anchor_path).readline().split(',') if x != ''])
        anchor = anchor[self.kp_index]
        return anchor

    
    def get_model(self):
        model = BlazeFace(image_size=self.model_image_size,
                            n_classes=1,
                            mode='inference',
                            normalize_coords=True,
                            subtract_mean=[107, 105, 109],
                            swap_channels=[2, 1, 0],
                            feature=len(self.kp_index))
        model.load_weights(self.model_path, by_name=True)

        return model

    def get_detector(self):
        return MTCNN()

    
    def detect(self, frame):
        face_detect = self.detector.detect_faces(frame)
        if face_detect == []:
            return "No Detection"
        bbox = face_detect[0]['box'] # xmin, ymin, w, h
        
        w  = int(bbox[2])
        h  = int(bbox[3])
        cx = int(bbox[0] + w//2)
        cy = int(bbox[1] + h//2)

        xmin = cx - 128
        ymin = cy - 128
        xmax = cx + 128
        ymax = cy + 128
        bbox = [xmin, ymin, xmax, ymax]
        cropped_image = frame[ymin:ymax, xmin:xmax]
        cropped_image = cv2.resize(cropped_image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        
        landmark = self.detect_landmark(cropped_image, bbox)

        return landmark, bbox

    
    def detect_landmark(self, cropped_image, bbox):
        input_images = np.array([cropped_image])
        y_pred = self.blazeface.predict(input_images)
        output = np.add(y_pred[0][0], self.anchors)
        landmark = [(int(output[idx])+bbox[0], int(output[idx+1])+bbox[1]) for idx in range(0, len(output), 2)]

        return landmark
