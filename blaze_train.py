import tensorflow as tf
import os
from datetime import datetime
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from models.keras_blazeface import blazeface
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder_blazeface import SSDInputEncoder

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_height = 256 # Height of the model input images
img_width = 256 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [107, 105, 109] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
normalize_coords = True
batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.
# 1: Build the Keras model.
select_keypoint = [9, 11, 13, 14, 19, 21, 23, 24, 25, 26]
output_index = []
for i in select_keypoint:
    output_index.append(i*2-2)
    output_index.append(i*2-1)

with tf.device('/gpu:0'):
    K.clear_session() # Clear previous models from memory.

    model = blazeface(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    feature=len(output_index),
                    swap_channels=swap_channels)

    adam = Adam(0.001)
    #sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    def mse (y_true, y_pred):
        return K.mean(K.square(y_pred -y_true[...,:len(output_index)]), axis=-1)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0, select_keypoint_labels_len=len(output_index))

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=[mse])
    model.summary()

    train_images_dir = "/data/"
    val_images_dir = "/data/"
    train_labels_filename = "/data/tsmoon_set/train_with_general_glasses.csv"
    val_labels_filename = "/data/tsmoon_set/valid_with_general_glasses.csv"
    

    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    train_dataset = DataGenerator(load_images_into_memory=None, hdf5_dataset_path=None, fix_image_ratio=True)
    val_dataset = DataGenerator(load_images_into_memory=None, hdf5_dataset_path=None, fix_image_ratio=True)
    # 2: Parse the image and label lists for the training and validation datasets.


    train_dataset.parse_csv(images_dir=train_images_dir,
                            labels_filename=train_labels_filename,
                            input_format=['image_name','xmin','ymin','xmax','ymax','kp1_x','kp1_y','kp2_x','kp2_y','kp3_x','kp3_y','kp4_x','kp4_y','kp5_x','kp5_y',
                                           'kp6_x','kp6_y','kp7_x','kp7_y','kp8_x','kp8_y','kp9_x','kp9_y','kp10_x','kp10_y','kp11_x','kp11_y','kp12_x','kp12_y','kp13_x',
                                           'kp13_y','kp14_x','kp14_y','kp15_x','kp15_y','kp16_x','kp16_y','kp17_x','kp17_y','kp18_x','kp18_y','kp19_x','kp19_y','kp20_x','kp20_y','kp21_x',
                                           'kp21_y','kp22_x','kp22_y','kp23_x','kp23_y','kp24_x','kp24_y','kp25_x','kp25_y','kp26_x','kp26_y','class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                            include_classes='all')
    val_dataset.parse_csv(images_dir=val_images_dir,
                            labels_filename=val_labels_filename,
                            input_format=['image_name','xmin','ymin','xmax','ymax','kp1_x','kp1_y','kp2_x','kp2_y','kp3_x','kp3_y','kp4_x','kp4_y','kp5_x','kp5_y',
                                            'kp6_x','kp6_y','kp7_x','kp7_y','kp8_x','kp8_y','kp9_x','kp9_y','kp10_x','kp10_y','kp11_x','kp11_y','kp12_x','kp12_y','kp13_x',
                                            'kp13_y','kp14_x','kp14_y','kp15_x','kp15_y','kp16_x','kp16_y','kp17_x','kp17_y','kp18_x','kp18_y','kp19_x','kp19_y','kp20_x','kp20_y','kp21_x',
                                            'kp21_y','kp22_x','kp22_y','kp23_x','kp23_y','kp24_x','kp24_y','kp25_x','kp25_y','kp26_x','kp26_y','class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                            include_classes='all')

    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color,
                                                random_crop_prob=0.5,
                                                rotate_prob=0.4)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                            shuffle=True,
                                            transformations=[ssd_data_augmentation],
                                            label_encoder=ssd_input_encoder,
                                            select_keypoint_label=output_index,
                                            returns={'processed_images',
                                                    'encoded_labels'},
                                            keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=[convert_to_3_channels,
                                                        resize],
                                        label_encoder=ssd_input_encoder,
                                        select_keypoint_label=output_index,
                                        returns={'processed_images',
                                                'encoded_labels'},
                                        keep_images_without_gt=False)
    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size   = val_dataset.get_dataset_size()
    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    model_checkpoint = ModelCheckpoint(filepath='./checkpoint/DB_Set2_with_general_glasses_Avg_2Layer_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)


    log_dir = './logs/scalars/'+ 'DB_Set2_with_general_glass_2Layer_'+datetime.now().strftime("%Y%m%d-%H%M%S")
       
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                tensorboard_callback,
                terminate_on_nan]

    initial_epoch   = 0
    final_epoch     = 160
    steps_per_epoch = train_dataset_size // batch_size

    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=final_epoch,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        validation_data=val_generator,
                        validation_steps=ceil(val_dataset_size/batch_size),
                        )