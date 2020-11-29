#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to manage created database
"""

import cv2
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
from tqdm import tqdm


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
    return


def shuffle_data(images, classes):
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
