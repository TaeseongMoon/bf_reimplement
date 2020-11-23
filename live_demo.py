import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    
import math
import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import cv2
from imageio import imread	
import numpy as np
from matplotlib import pyplot as plt

from models.keras_blazeface import blazeface
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxesBlazeFace import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder_blazeface import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from mtcnn import MTCNN


img_height = 256
img_width = 256

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = blazeface(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode='inference',
                l2_regularization=0.0005,
                scales=[[0.2]], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0]],
                steps=[64],
                offsets=None,
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[107, 105, 109],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.1,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=100)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'checkpoint/new_anchor_256_16_fix_DBset_2_epoch-86_loss-0.5585.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

import random
import pandas as pd
csv = pd.read_csv('/data/tsmoon_set/train_setting_2.csv')
imgs = csv.img_path.tolist()
test_image_name = random.choice(imgs)
# test_image_name = 'Foreigner_Male_1/20200902_day_shadow_01/Class_3/frame/2_20200902_day_shadow_01/2_20200902_day_shadow_01_20 02.jpg'
print(test_image_name)
x = csv[csv.img_path == test_image_name]
xmin, ymin, xmax, ymax = int(x.xmin), int(x.ymin), int(x.xmax), int(x.ymax)
w = xmax - xmin
h = ymax - ymin
cx = int(xmin + w//2)
cy = int(ymin + h//2)
xmin = cx - 128
ymin = cy - 128
xmax = cx + 128
ymax = cy + 128
print(x.iloc[:,1:5])
print(f"xmin,ymin,xmax, ymax: {xmin},{ymin},{xmax},{ymax}")

crop_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.
imgs = csv.img_path.tolist()
# test_image_name = random.choice(imgs)
# We'll only load one image in this example.
img_path = f'/data/{test_image_name}'
# img_path = './test/2.jpg'
# detector = MTCNN()
ori_image = cv2.imread(img_path)
ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
ori_image = ori_image[ymin:ymax, xmin:xmax]
plt.figure(figsize=(20,12))
plt.imshow(ori_image)
ori_image = cv2.resize(ori_image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
input_images = np.array([ori_image])


from time import time

start = time()
# y_pred = model.predict(input_images)|
y_pred = model.predict(input_images)
print(time() - start)

anchor = [ float(x) for x in open('anchor_256_fix.txt').readline().split(',') if x != '']
pred_8x8 = y_pred[0][0]
pred_16x16 = y_pred[0][1]
prediction = (pred_8x8 + pred_16x16)/2
print(prediction)
output = np.add(prediction, anchor)

dbset = [9,11,13,14,21,23,24,25,26]
output_index = []
for i in dbset:
    output_index.append(i*2-2)
    output_index.append(i*2-1)

output = output[output_index]


colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()


current_axis = plt.gca()
height, width, _ = input_images[0].shape

for a in range(0, len(output), 2):
    current_axis.add_patch(plt.Circle((output[a], output[a+1]), 1, color=colors[1]))
