
from __future__ import division
import numpy as np
import cv2
import random

class Resize:
    '''
    Resizes images to a specified height and width in pixels.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):

        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        image = cv2.resize(image,
                        dsize=(self.out_width, self.out_height),
                        interpolation=self.interpolation_mode)

        labels = np.copy(labels)
        
        labels[:,1:52:2] = np.round(labels[:,1:52:2].astype(float) * (self.out_height / img_width), decimals=0)
        labels[:,2:53:2] = np.round(labels[:,2:53:2].astype(float) * (self.out_height / img_height), decimals=0)

        return image, labels


class ResizeRandomInterp:
    '''
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 labels_format={'class_id': 0, 'kp1_x':1, 'kp1_y':2, 'kp2_x':3, 'kp2_y':4, 'kp3_x':5, 'kp3_y':6, 'kp4_x':7, 'kp4_y':8, 'kp5_x':9, 'kp5_y':10,
                 'kp6_x':11, 'kp6_y':12, 'kp7_x':13, 'kp7_y':14, 'kp8_x':15, 'kp8_y':16, 'kp9_x':17, 'kp9_y':18, 'kp10_x':19, 'kp10_y':20,
                 'kp11_x':21, 'kp11_y':22, 'kp12_x':23, 'kp12_y':24, 'kp13_x':25, 'kp13_y':26, 'kp14_x':27, 'kp14_y':28, 'kp15_x':29, 'kp15_y':30,
                 'kp16_x':31, 'kp16_y':32, 'kp17_x':33, 'kp17_y':34, 'kp18_x':35, 'kp18_y':36, 'kp19_x':37, 'kp19_y':38, 'kp20_x':39, 'kp20_y':40,
                 'kp21_x':41, 'kp21_y':42, 'kp22_x':43, 'kp22_y':44, 'kp23_x':45, 'kp23_y':46, 'kp24_x':47, 'kp24_y':48, 'kp25_x':49, 'kp25_y':50, 'kp26_x':51, 'kp26_y':52}):

        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = self.interpolation_modes[2]
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels)


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
                if float(labels[:,x][0]) > -1 and float(labels[:,y][0]) > -1:
                    k = rot_point(float(labels[:,x]), float(labels[:,y]))
                    labels[:, [x, y]] = np.round(k.tolist(), decimals =0).astype(int).reshape((1,2))

            return image, labels

class RandomRotate:
    '''
    Randomly rotates images counter-clockwise.
    '''

    def __init__(self,
                 angles=[0, 5, 10, 15, 345, 350, 355],
                 prob=0.4,
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
