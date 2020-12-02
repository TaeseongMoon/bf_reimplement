import tensorflow as tf
from os.path import join
from pathlib import Path
import math
import cv2
import matplotlib.pyplot as plt
from keras import backend as K


def calculate_eye(top, bottom):
    """Calculate EAR(Eye Aspect Ratio) of the driver. 
    Args:
      top : top point of left-side eye.
      bottom : bottom point of left side eye.
    Returns:
      (float) Distance between two point.
    """
    return math.sqrt(math.pow(top[0]-bottom[0], 2) + math.pow(top[1]-bottom[1], 2))


def calculate_mouth(top, bottom):
    """Calculate MAR(Mouth Aspect Ratio) of the driver.
    Args:
      top : top point of mouth.
      bottom : bottom point of mouth.
    Returns:
      (float) Distance between two point.
    """
    return math.sqrt(math.pow(top[0]-bottom[0], 2) + math.pow(top[1]-bottom[1], 2))


def set_videowriter(video, save_path):
    """Connects to the next available port.
    Args:
      minimum: A port value greater or equal to 1024.
    Returns:
      The new minimum port.
    """
    fps = 30
    width = int(video.get(3))
    height = int(video.get(4))
    # width =720
    # height=480
    fcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(save_path, fcc, fps, (width, height))
    
    return writer


def draw_plot(data, save_path):
    """Connects to the next available port.
    Args:
      minimum: A port value greater or equal to 1024.
    Returns:
      The new minimum port.
    """
    plt.figure(figsize=(20, 10))
    
    for key, value in data.items():
        plt.plot(value, label=key)
        plt.ylim((0, 20))
    
    plt.legend(loc='best')
    plt.xlabel('Frame')
    plt.ylabel('Distance')
    plt.savefig(save_path)


def reconstruct_bbox(mtcnn_result):
    """Connects to the next available port.
    Args:
      minimum: A port value greater or equal to 1024.
    Returns:
      The new minimum port.
    """
    bbox = mtcnn_result['box'] # xmin, ymin, w, h

    w = mtcnn_result['box'][2]
    h = mtcnn_result['box'][3]
    cx = int(bbox[0] + w//2)
    cy = int(bbox[1] + h//2)

    xmin = cx - 128
    ymin = cy - 128
    xmax = cx + 128
    ymax = cy + 128

    return [xmin, ymin, xmax, ymax]


def draw_landmark_on_image(image, landmarks, bbox):
    """Connects to the next available port.
    Args:
      minimum: A port value greater or equal to 1024.
    Returns:
      The new minimum port.
    """
    for landmark in landmarks:
        cv2.circle(image, landmark, 3, (0, 255, 255), -1)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (51, 255, 51), 3)
    
    return image

def setup_gpu(gpu_id):
    try:
        visible_gpu_indices = [int(id) for id in gpu_id.split(',')]
        available_gpus = tf.config.list_physical_devices('GPU')
        visible_gpus = [gpu for idx, gpu in enumerate(available_gpus) if idx in visible_gpu_indices]

        if visible_gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs.
                for gpu in available_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Use only the selcted gpu.
                tf.config.set_visible_devices(visible_gpus, 'GPU')
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized.
                print(e)

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(available_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        else:
            tf.config.set_visible_devices([], 'GPU')
    except ValueError:
        tf.config.set_visible_devices([], 'GPU')