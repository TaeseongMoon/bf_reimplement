import tensorflow as tf
import math
import os
from datetime import datetime
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from models.keras_blazeface import blazeface
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder_blazeface import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder_blazeface import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 128 # Height of the model input images
img_width = 128 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [121, 111, 105] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales = [[0.2, math.sqrt(0.2 * 0.43)], [0.43, math.sqrt(0.43 * 0.67), 0.67, math.sqrt(0.67 * 0.9), 0.9, math.sqrt(0.9 * 1)]]
aspect_ratios = [[1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]] # The anchor box aspect ratios
steps = [64, 128] # The space between two adjacent anchor box center points for each predictor layer.
offsets = None # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# 1: Build the Keras model.
with tf.device('/gpu:0'):
    K.clear_session() # Clear previous models from memory.

    model = blazeface(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    # 2: Load some weights into the model.

    # TODO: Set the path to the weights you want to load.
    # weights_path = 'blazeface_fddb_07+12_epoch-183_loss-3.9408_val_loss-2.9432.h5'

    # model.load_weights(weights_path, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

    adam = Adam(0.001)
    #sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    model.summary()
    
    train_images_dir = "../BlazeFace/data/WIDER_train/images/"
    val_images_dir = '../BlazeFace/data/WIDER_val/images/'
    train_anno_file = "./data/train_annos_simple.csv"
    # val_anno_file = "./data/val_annos.csv"

    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

    #train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path='wider_train_new.h5')
    #val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path='wider_val_new_v2.h5')
    train_dataset = DataGenerator(load_images_into_memory=None, hdf5_dataset_path=None)
    # val_dataset = DataGenerator(load_images_into_memory=None, hdf5_dataset_path=None)
    # 2: Parse the image and label lists for the training and validation datasets.

    # Ground truth
    train_labels_filename = train_anno_file
    # val_labels_filename   = val_anno_file

    train_dataset.parse_csv(images_dir=train_images_dir,
                            labels_filename=train_labels_filename,
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax','kp1_x','kp1_y','kp2_x','kp2_y','kp3_x','kp3_y','kp4_x','kp4_y','kp5_x','kp5_y','class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                            include_classes='all')

    # val_dataset.parse_csv(images_dir=val_images_dir,
    #                     labels_filename=val_labels_filename,
    #                         input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
    #                         include_classes='all')

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.
    
    # train_dataset.create_hdf5_dataset(file_path='fddb_train.h5',
    #                                   resize=False,
    #                                   variable_image_size=True,
    #                                   verbose=True)

    # val_dataset.create_hdf5_dataset(file_path='fddb_val.h5',
    #                                 resize=False,
    #                                 variable_image_size=True,
    #                                 verbose=True)
    batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.

    # 4: Set the image transformations for pre-processing and data augmentation options.

    # For the training generator:

    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('classes16x16').output_shape[1:3],
                    model.get_layer('classes8x8').output_shape[1:3]]
    
    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                            shuffle=True,
                                            transformations=[ssd_data_augmentation],
                                            label_encoder=ssd_input_encoder,
                                            returns={'processed_images',
                                                    'encoded_labels'},
                                            keep_images_without_gt=False)

    # val_generator = val_dataset.generate(batch_size=batch_size,
    #                                     shuffle=False,
    #                                     transformations=[convert_to_3_channels,
    #                                                     resize],
    #                                     label_encoder=ssd_input_encoder,
    #                                     returns={'processed_images',
    #                                             'encoded_labels'},
    #                                     keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    # val_dataset_size   = val_dataset.get_dataset_size()
    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    # print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    model_checkpoint = ModelCheckpoint(filepath='./checkpoint/blazeface_with_landmark_simple_v3_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                    monitor='loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)
    #model_checkpoint.best = 

    csv_logger = CSVLogger(filename='blazeface_landmark_wider_simple_v3_training_log.csv',
                        separator=',',
                        append=True)

    log_dir = './logs/scalars/'+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                csv_logger,
                tensorboard_callback,
                terminate_on_nan]

    initial_epoch   = 0
    final_epoch     = 250
    steps_per_epoch = train_dataset_size // batch_size

    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=final_epoch,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch
                        )