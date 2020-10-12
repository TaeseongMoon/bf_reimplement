import numpy as np
import os
import tensorflow as tf
from math import ceil
from keras.models import load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, Callback
from keras.optimizers import Adam, SGD

from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd300 import ssd_300

# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.compat.v2.config.experimental.set_memory_growth(gpu, True)


img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
# The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
mean_color = [121, 111, 105]
# The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
swap_channels = [2, 1, 0]
n_classes = 1  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
# The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
# The space between two adjacent anchor box center points for each predictor layer.
steps = [8, 16, 32, 64, 100, 300]
# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True


with tf.device('/gpu:0'):
    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    adam = Adam(0.01)
    # sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    model.summary()
    train_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)
        
    train_images_dir = "./data/widerface/train/images"
    val_images_dir = './data/widerface/val/images'
    # annotation_dir = "./data/FDDB/FDDB-folds/"
    train_anno_file = "./data/train_annos.csv"
    # val_anno_file = "./data/FDDB/FDDB-val.csv"
    val_anno_file = "./data/val_annos.csv"

    train_labels_filename = train_anno_file
    val_labels_filename   = val_anno_file

    train_dataset.parse_csv(images_dir=train_images_dir,
                            labels_filename=train_labels_filename,
                            input_format=['image_name','class_id', 'xmin', 'xmax', 'ymin', 'ymax'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                            include_classes='all')

    val_dataset.parse_csv(images_dir=val_images_dir,
                        labels_filename=val_labels_filename,
                        input_format=['image_name', 'class_id', 'xmin', 'xmax', 'ymin', 'ymax'],
                        include_classes='all')
    # Change the batch size if you like, or if you run into GPU memory issues.
    batch_size = 27

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
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                    model.get_layer('fc7_mbox_conf').output_shape[1:3],
                    model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=True,
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
                                            transformations=[
                                                ssd_data_augmentation],
                                            label_encoder=ssd_input_encoder,
                                            returns={'processed_images',
                                                    'encoded_labels'},
                                            keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=[convert_to_3_channels,
                                                        resize],
                                        label_encoder=ssd_input_encoder,
                                        returns={'processed_images',
                                                'encoded_labels'},
                                        keep_images_without_gt=False)

    train_dataset_size=train_dataset.get_dataset_size()
    val_dataset_size=val_dataset.get_dataset_size()
    
    print(next(train_generator)[1].shape)
    print(next(train_generator)[0].shape)
    model_checkpoint=ModelCheckpoint(filepath='./checkpoint/ssd300_wider_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)
    # model_checkpoint.best =

    csv_logger=CSVLogger(filename='ssd300_wider_training_log.csv',
                        separator=',',
                        append=True)
    class LossHistory(Callback):
        def on_batch_end(self, batch, logs={}):
            print('\ny_true: ',logs.get('y_true'),'\ny_pred :',logs.get('y_pred'))
    
    history = LossHistory()
    terminate_on_nan=TerminateOnNaN()
    callbacks=[ model_checkpoint,
                history,
                csv_logger,
                terminate_on_nan]
    initial_epoch=0
    final_epoch=80
    steps_per_epoch= train_dataset_size // batch_size
    import pdb
    pdb.set_trace()
    history=model.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=final_epoch,
                                callbacks=callbacks,
                                validation_data=val_generator,
                                validation_steps=ceil(
                                    val_dataset_size/batch_size),
                                initial_epoch=initial_epoch)
