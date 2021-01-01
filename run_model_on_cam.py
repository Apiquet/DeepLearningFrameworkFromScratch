#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script run neural network model on a camera live stream
"""

from homemade_framework import framework as NN

import argparse
import cv2
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--weight_path",
        required=True,
        type=str,
        help="Path to load weights for the model."
    )
    parser.add_argument(
        "-w",
        "--img_width",
        required=False,
        default=28,
        type=int,
        help="Image width."
    )
    parser.add_argument(
        "-n",
        "--num_classes",
        required=False,
        default=7,
        type=int,
        help="Number of classes."
    )
    parser.add_argument(
        "-c",
        "--crop",
        required=False,
        default=None,
        type=str,
        help="Crop image, format: MinWidth,MaxWidth,MinHeight,MaxHeight.\
              Set -1 for the unchanged ones"
    )
    parser.add_argument(
        "-r",
        "--resize",
        required=False,
        default=None,
        type=str,
        help="Resize shape, format: height,width"
    )
    parser.add_argument(
        "-b",
        "--binarize",
        required=False,
        default=None,
        type=str,
        help="To binarize images, format for thresholding: min,max"
    )
    parser.add_argument(
        "-g",
        "--gray",
        required=False,
        action="store_true",
        help="To save 1-channel images"
    )
    parser.add_argument(
        "-e",
        "--erode",
        required=False,
        default=None,
        type=str,
        help="Erode option, format: kernel_size,iteration"
    )
    parser.add_argument(
        "-d",
        "--dilate",
        required=False,
        default=None,
        type=str,
        help="Dilate option, format: kernel_size,iteration"
    )
    parser.add_argument(
        "-m",
        "--camid",
        required=False,
        default=0,
        type=int,
        help="Camera ID, default is 0"
    )
    parser.add_argument(
        "-t",
        "--tensorflow",
        required=False,
        action="store_true",
        help="To specify if Tensorflow model is used."
    )

    args = parser.parse_args()

    input_size = args.img_width**2
    num_class = args.num_classes
    hidden_size = 128

    if args.tensorflow:
        import tensorflow as tf
        model = tf.keras.models.load_model('models/tf_model/')
    else:
        model = NN.Sequential([NN.Linear(input_size, hidden_size),
                               NN.LeakyReLU(), NN.BatchNorm(),
                               NN.Linear(hidden_size, hidden_size),
                               NN.LeakyReLU(), NN.BatchNorm(),
                               NN.Linear(hidden_size, num_class),
                               NN.Softmax()], NN.LossMSE())
        model.load(args.weight_path)

    cam = cv2.VideoCapture(args.camid)
    ret, frame = cam.read()

    min_height, max_height = 0, frame.shape[0]
    min_width, max_width = 0, frame.shape[1]
    print("Cam resolution: {}x{}".format(max_width, max_height))
    if args.crop is not None:
        res = [int(x) for x in args.crop.split(',')]
        if res[0] != -1:
            min_width = res[0]
        if res[1] != -1:
            max_width = res[1]
        if res[2] != -1:
            min_height = res[2]
        if res[3] != -1:
            max_height = res[3]
        print("Image cropped to minWidth:maxWidth, minHeight:maxHeight: {}:{}\
              , {},{}".format(min_width, max_width, min_height, max_height))
    pause = False
    imgs = []

    print("Model is running")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        if args.crop is not None:
            frame = frame[min_height:max_height, min_width:max_width]
        cv2.imshow("Original image", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == ord('p'):
            # p pressed
            if pause:
                pause = False
            else:
                pause = True

        if not pause:
            if args.gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.binarize:
                frame = cv2.medianBlur(frame, 5)
                min_thresh, max_thresh = [int(x) for x in
                                          args.binarize.split(',')]
                ret, frame = cv2.threshold(frame, min_thresh, max_thresh,
                                           cv2.THRESH_BINARY)
            if args.erode is not None:
                k_size, iteration = [int(x) for x in args.erode.split(',')]
                kernel = np.ones((k_size, k_size), np.uint8)
                frame = cv2.erode(frame, kernel, iterations=int(iteration))
            if args.dilate is not None:
                k_size, iteration = [int(x) for x in args.dilate.split(',')]
                kernel = np.ones((k_size, k_size), np.uint8)
                frame = cv2.dilate(frame, kernel, iterations=int(iteration))

            if args.resize:
                height, width = [int(size) for size in args.resize.split(',')]
                frame = cv2.resize(frame, (height, width),
                                   interpolation=cv2.INTER_AREA)

            image = np.asarray(frame)/255.
            cv2.imshow("Input image for the model", frame)
            image = image.reshape([np.prod(image.shape)])
            if len(imgs) < 3:
                imgs.append(image)
            else:
                if args.tensorflow:
                    print(np.argmax(model(np.asarray(imgs)), axis=1))
                else:
                    NN.get_inferences(model, np.asarray(imgs))
                imgs = imgs[1:]
                imgs.append(image)

            time.sleep(1)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
