#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import cv2
import numpy as np
import time


class DBManager():

    def __init__(self, number_of_classes, output_path, image_name="img",
                 crop=None, resize=None, binarize=None, gray=False,
                 erode=None, dilate=None, camid=0,
                 start_label_idx=0, start_img_idx=0):
        super(DBManager, self).__init__()
        self.number_of_classes = number_of_classes
        self.output_path = output_path
        self.image_name = image_name
        self.crop = crop
        self.resize = resize
        self.binarize = binarize
        self.gray = gray
        self.erode = erode
        self.dilate = dilate
        self.camid = camid
        self.start_label_idx = start_label_idx
        self.start_img_idx = start_img_idx

    def create_db(self, images_name: list):
        """
        Method to get create a database

        Once the function is running:
            - Press SPACE to start
            - The result image will be displayed
            - Press p to start saving the images
            - The images for label 0 will be saved every 0.5s
            - Press SPACE once again to start the next label
            - Press p to pause and ESC to exit
        """
        cam = cv2.VideoCapture(self.camid)
        ret, frame = cam.read()

        min_height, max_height = 0, frame.shape[0]
        min_width, max_width = 0, frame.shape[1]
        print("Cam resolution: {}x{}".format(max_width, max_height))
        if self.crop is not None:
            res = [int(x) for x in self.crop.split(',')]
            if res[0] != -1:
                min_width = res[0]
            if res[1] != -1:
                max_width = res[1]
            if res[2] != -1:
                min_height = res[2]
            if res[3] != -1:
                max_height = res[3]
            print("Image cropped to minWidth:maxWidth, minHeight:maxHeight:\
                  {}:{}, {},{}".format(min_width, max_width,
                                       min_height, max_height))

        img_counter = self.start_img_idx
        pause = True
        label = self.start_label_idx

        print("Press SPACE to create the DB")
        print("The images for label 0 will be saved every 0.5s")
        print("Press SPACE once again to start the next label")
        print("Press p to pause and ESC to exit")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            if self.crop is not None:
                frame = frame[min_height:max_height, min_width:max_width]
            cv2.imshow("Original image", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                label = label + 1
                img_counter = self.start_img_idx
                if label >= self.number_of_classes:
                    print("Last label done, closing...")
                    break
            elif k % 256 == ord('p'):
                # p pressed
                if pause:
                    pause = False
                else:
                    pause = True

            if label > -1:
                img_name = self.image_name + "_{:08d}_{:02d}.png".format(
                    img_counter, label)
                if self.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.binarize:
                    frame = cv2.medianBlur(frame, 5)
                    min_thresh, max_thresh = [int(x) for x in
                                              self.binarize.split(',')]
                    ret, frame = cv2.threshold(frame, min_thresh, max_thresh,
                                               cv2.THRESH_BINARY)
                if self.erode is not None:
                    k_size, iteration = [int(x) for x in self.erode.split(',')]
                    kernel = np.ones((k_size, k_size), np.uint8)
                    frame = cv2.erode(frame, kernel, iterations=int(iteration))
                if self.dilate is not None:
                    k_size, iteration = [int(x)
                                         for x in self.dilate.split(',')]
                    kernel = np.ones((k_size, k_size), np.uint8)
                    frame = cv2.dilate(frame, kernel,
                                       iterations=int(iteration))

                if self.resize:
                    height, width = [int(size)
                                     for size in self.resize.split(',')]
                    frame = cv2.resize(frame, (height, width),
                                       interpolation=cv2.INTER_AREA)

                cv2.imshow("Image to save", frame)

                if not pause:
                    cv2.imwrite(self.output_path + '/' + img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1
                time.sleep(1)
        cam.release()
        cv2.destroyAllWindows()

    def load_data(self, db_path):
        """
        Method to load the database
        N: number of images for a class
        C: number of images channels

        Args:
            - (str) database path
        Return:
            - (numpy array) images (N, C, H, W)
            - (numpy array) classes (N)
        """
        return

    def shuffle_data(self, images, classes):
        """
        Method to shuffle data
        N: number of images for a class
        C: number of images channels

        Args:
            - (numpy array) images (N, C, H, W)
            - (numpy array) classes (N)
        Return:
            - (numpy array) images (N, C, H, W)
            - (numpy array) classes (N)
        """
        return
