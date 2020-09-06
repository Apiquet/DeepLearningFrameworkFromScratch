#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script records images to create a database.
Simple usage for 3 classes:
python db_creation.py -n 3 -o path/to/store/
Optional arguments:
-c to crop images
-r to resize images
-g to convert images to gray scale
-b to binarize images

Once the script is running:
- Press SPACE to start creating the DB
- The images for label 0 will be saved every 0.5s
- Press SPACE once again to start the next label
- Press p to pause and ESC to exit
"""

import argparse
import cv2
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--number_of_classes",
        required=True,
        type=int,
        help="Path to the images or a video."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        default=None,
        type=str,
        help="Path to store images."
    )
    parser.add_argument(
        "-i",
        "--image_name",
        required=False,
        default="img",
        type=str,
        help="Images names, default is 'image'\
             to create img_imagenumber_classenumber.png."
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
        "--resize_fact",
        required=False,
        default=1.0,
        type=float,
        help="Divise image resolution by number specified"
    )
    parser.add_argument(
        "-b",
        "--binarize",
        required=False,
        action="store_true",
        help="To binarize images"
    )
    parser.add_argument(
        "-g",
        "--gray",
        required=False,
        action="store_true",
        help="To save 1-channel images"
    )

    args = parser.parse_args()

    cam = cv2.VideoCapture(0)
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

    img_counter = 0
    pause = False
    label = -1

    print("Press SPACE to create the DB")
    print("The images for label 0 will be saved every 0.5s")
    print("Press SPACE once again to start the next label")
    print("Press p to pause and ESC to exit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        if args.crop is not None:
            frame = frame[min_height:max_height, min_width:max_width]
        cv2.imshow("Images viewer", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            label = label + 1
            if label >= args.number_of_classes:
                print("Last label done, closing...")
                break
        elif k % 256 == ord('p'):
            # p pressed
            if pause:
                pause = False
            else:
                pause = True

        if label > -1 and not pause:
            img_name = args.image_name + "_{:08d}_{:02d}.png".format(
                img_counter, label)
            frame = cv2.resize(frame, (int(frame.shape[1]/args.resize_fact),
                                       int(frame.shape[0]//args.resize_fact)),
                               interpolation=cv2.INTER_AREA)
            if args.gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.binarize:
                frame = cv2.medianBlur(frame, 5)
                frame = cv2.adaptiveThreshold(frame, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
            cv2.imwrite(args.output_path + '/' + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            time.sleep(1)

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
