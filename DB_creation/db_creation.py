#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script records images to create a database.
Usage for 3 classes:
- python db_creation.py -n 3 -o path/to/store/
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

    args = parser.parse_args()

    cam = cv2.VideoCapture(0)

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
        cv2.imshow("Images viewer", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            label = label + 1
            if label>=args.number_of_classes:
                print("Last label done, closing...")
                break
        elif k%256 == ord('p'):
            # p pressed
            if pause:
                pause = False
            else:
                pause = True

        if label > -1 and not pause:
            img_name = args.image_name + "_{:08d}_{:02d}.png".format(img_counter, label)
            cv2.imwrite(args.output_path + '/' + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            time.sleep(1)

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()