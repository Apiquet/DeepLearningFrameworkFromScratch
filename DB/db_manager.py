#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import cv2
import numpy as np
import time


class DBManager():

    def __init__(self):
        super(DBManager, self).__init__()

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
