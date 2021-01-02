#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script run neural network model on a camera live stream
"""

import argparse
import cv2
import numpy as np
import os
import time
import sys


COMMANDS = {0: "move_forward", 1: "go_down", 2: "rot_10_deg",
            3: "go_up", 4: "take_off", 5: "land", 6: "idle"}


def send_command(anafi, command_id):
    """
    Function to send commands to an Anafi drone in function of the command id
    """
    if command_id not in COMMANDS:
        raise f"Command id not in COMMANDS choices: {command_id}"
    if COMMANDS[command_id] == "idle":
        return

    print("The following command will be sent: ", COMMANDS[command_id])

    if COMMANDS[command_id] == "move_forward":
        anafi.move_relative(dx=1, dy=0, dz=0, dradians=0)
    if COMMANDS[command_id] == "go_down":
        anafi.move_relative(dx=0, dy=0, dz=-0.5, dradians=0)
    if COMMANDS[command_id] == "rot_10_deg":
        anafi.move_relative(dx=0, dy=0, dz=0, dradians=0.785)
    if COMMANDS[command_id] == "go_up":
        anafi.move_relative(dx=0, dy=0, dz=0.5, dradians=0)
    if COMMANDS[command_id] == "take_off":
        anafi.safe_takeoff(5)
    if COMMANDS[command_id] == "land":
        anafi.safe_land(5)
    return


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
        "-a",
        "--pyparrot_path",
        required=True,
        type=str,
        help="Path to pyparrot module downloaded from amymcgovern on github."
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
    parser.add_argument(
        "-z",
        "--number_of_confimation",
        required=False,
        default=3,
        type=int,
        help="Minimum number of identical commands before sending to drone."
    )

    args = parser.parse_args()

    """
    Drone connection
    """
    sys.path.append(args.pyparrot_path)
    from pyparrot.Anafi import Anafi
    print("Connecting to drone...")
    anafi = Anafi(drone_type="Anafi", ip_address="192.168.42.1")
    success = anafi.connect(10)
    print(success)
    print("Sleeping few seconds...")
    anafi.smart_sleep(3)

    """
    Load model
    """
    print("Loading model...")
    input_size = args.img_width**2
    num_class = args.num_classes
    hidden_size = 128

    if args.tensorflow:
        import tensorflow as tf
        model = tf.keras.models.load_model(args.weight_path)
    else:
        script_path = os.path.realpath(__file__)
        sys.path.append(os.path.dirname(script_path) + "/../")
        from homemade_framework import framework as NN
        model = NN.Sequential([NN.Linear(input_size, hidden_size),
                               NN.LeakyReLU(), NN.BatchNorm(),
                               NN.Linear(hidden_size, hidden_size),
                               NN.LeakyReLU(), NN.BatchNorm(),
                               NN.Linear(hidden_size, num_class),
                               NN.Softmax()], NN.LossMSE())
        model.load(args.weight_path)

    """
    Webcam process
    """
    print("Start webcam...")
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
            if len(imgs) < args.number_of_confimation:
                imgs.append(image)
            else:
                if args.tensorflow:
                    results = np.argmax(model(np.asarray(imgs)), axis=1)
                else:
                    results = NN.get_inferences(model, np.asarray(imgs))
                print("Model's output on buffer: ", results)
                if np.unique(results).size == 1:
                    send_command(anafi, results[0])
                imgs = imgs[1:]
                imgs.append(image)

            time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
