import sys
import os
import math
from os.path import join, basename
from pathlib import Path
import argparse
import cv2
from imageio import imread	
import numpy as np
from time import time
from matplotlib import pyplot as plt
import collections
from models.keras_blazeface import blazeface
from utils import set_videowriter, reconstruct_bbox, draw_landmark_on_image, setup_gpu
from mtcnn import MTCNN

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         help='Input Model Path.', default='model.h5', type=str)
    parser.add_argument('--video_path',    help='Video Path', default='', type=str)
    parser.add_argument('--save_path',     help='Video output save Path', default='', type=str)
    parser.add_argument('--anchor_path',    help='Anchor Points txt file Path', default='anchor.txt', type=str)
    parser.add_argument('--gpu',           help='Id of the GPU to use (as reported by nvidia-smi).')
    
    return parser.parse_args(args)


def mtcnn_bbox(args, image, mtcnn_results, model, output_index):

    bbox = reconstruct_bbox(mtcnn_results[0]) # xmin, ymin, xmax, ymax
    
    # input
    crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    crop_image = cv2.resize(crop_image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    input_images = np.array([crop_image])
    
    y_pred = model.predict(input_images)
    anchor = np.array([ float(x) for x in open(args.anchor_path).readline().split(',') if x != ''])
    anchor = anchor[output_index]
    output = np.add(y_pred[0][0], anchor)
    landmarks = [(int(output[idx])+bbox[0], int(output[idx+1])+bbox[1]) for idx in range(0, len(output), 2)]

    return landmarks, bbox


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    if args.gpu is not None:
        setup_gpu(args.gpu)
    select_keypoint = [9, 11, 13, 14, 19, 21, 23, 24, 25, 26]

    output_index = []
    for i in select_keypoint:
        output_index.append(i*2-2)
        output_index.append(i*2-1)
    model = blazeface(image_size=(256, 256, 3),
                n_classes=1,
                mode='inference',
                normalize_coords=True,
                subtract_mean=[107, 105, 109],
                swap_channels=[2, 1, 0],
                feature=len(output_index))
    model.load_weights(args.model, by_name=True)

    face_detector = MTCNN()
    
    video = cv2.VideoCapture(args.video_path)
    video_writer = set_videowriter(video, f'{args.save_path}/{basename(args.video_path).rstrip(".avi")}.mp4')
    
    while True:
        ret, frame = video.read()
        start_fps = time()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detect_results = face_detector.detect_faces(frame)
        if len(detect_results) != 0:
            landmarks, bbox = mtcnn_bbox(args, frame, detect_results, model, output_index)
            frame = draw_landmark_on_image(frame, landmarks, bbox)
            
        end_fps = time()
        fps = 1/(end_fps-start_fps)
        cv2.putText(frame, f'fps :{fps:.2f}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        video_writer.write(frame)

    video.release()
    video_writer.release()
    
if __name__ == "__main__":
    main()
