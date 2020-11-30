#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import cv2
from glob import glob
import numpy as np
import os
from PIL import Image


class DBManager():

    def __init__(self):
        super(DBManager, self).__init__()

    def load_data(db_path, train_ratio=0.7):
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
        imgs_path = glob(db_path + "/*")
        number_of_imgs = len(imgs_path)
        train_number = int(number_of_imgs*train_ratio)
        random_idx = np.arange(number_of_imgs)
        np.random.shuffle(random_idx)

        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []

        for i, idx in enumerate(random_idx):
            img_path = imgs_path[idx]
            image = Image.open(img_path)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            if i <= train_number:
                train_imgs.append(np.asarray(image).reshape([1, 28, 28]))
                train_labels.append(int(img_basename.split('_')[2]))
            else:
                test_imgs.append(np.asarray(image).reshape([1, 28, 28]))
                test_labels.append(int(img_basename.split('_')[2]))

        return np.array(train_imgs), np.array(train_labels),\
            np.array(test_imgs), np.array(test_labels)
