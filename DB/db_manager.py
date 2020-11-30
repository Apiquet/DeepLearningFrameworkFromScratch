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

    def load_data(db_path):
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
        imgs = []
        labels = []
        random_idx = np.arange(len(imgs_path))
        np.random.shuffle(random_idx)

        for idx in random_idx:
            img_path = imgs_path[idx]
            image = Image.open(img_path)
            imgs.append(np.asarray(image).reshape([1, 28, 28]))
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            labels.append(int(img_basename.split('_')[2]))
        return imgs, labels
