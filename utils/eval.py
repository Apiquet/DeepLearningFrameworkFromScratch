#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Useful functions
"""

import imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def concat4Gif(path_1, path_2, path_3, path_4, out_path, fps=22, line_width=2):
    """
    Function to concat 4 gif files into 1
    path_1 to bottom left
    path_2 to bottom right
    path_3 to top right
    path_4 to top left

    Args:
        - (str) path_1:  path to 1st gif
        - (str) path_2:  path to 2nd gif
        - (str) path_3:  path to 3rd gif
        - (str) path_4:  path to 4th gif
        - (str) out_path:  path to the output gif
        - (int) fps:  number of frames per second for output gif
    """
    gif_1 = imageio.get_reader(path_1, '.gif')
    gif_2 = imageio.get_reader(path_2, '.gif')
    gif_3 = imageio.get_reader(path_3, '.gif')
    gif_4 = imageio.get_reader(path_4, '.gif')

    gif_1_imgs = []
    for frame in gif_1:
        gif_1_imgs.append(frame)

    gif_2_imgs = []
    for frame in gif_2:
        gif_2_imgs.append(frame)

    gif_3_imgs = []
    for frame in gif_3:
        gif_3_imgs.append(frame)

    gif_4_imgs = []
    for frame in gif_4:
        gif_4_imgs.append(frame)

    total_size = (gif_1_imgs[0].shape[0]*2, gif_1_imgs[0].shape[1]*2, 3)
    white_image = 255 * np.ones(total_size, np.uint8)
    white_pil = Image.fromarray(white_image)

    images = []

    for idx in tqdm(range(len(gif_1_imgs))):
        np_img_1 = np.asarray(gif_1_imgs[idx])
        pil_img_1 = Image.fromarray(np_img_1)
        draw = ImageDraw.Draw(pil_img_1)
        min_point = (-10, 0)
        end_point = (pil_img_1.size[0], pil_img_1.size[1]+10)
        draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                       width=line_width)
        white_pil.paste(pil_img_1, (0,  pil_img_1.size[1]))

        np_img_2 = np.asarray(gif_2_imgs[idx])
        pil_img_2 = Image.fromarray(np_img_2)
        draw = ImageDraw.Draw(pil_img_2)
        min_point = (0, 0)
        end_point = (pil_img_2.size[0]+10, pil_img_2.size[1]+10)
        draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                       width=line_width)
        white_pil.paste(pil_img_2, (pil_img_2.size[0], pil_img_2.size[1]))

        np_img_3 = np.asarray(gif_3_imgs[idx])
        pil_img_3 = Image.fromarray(np_img_3)
        draw = ImageDraw.Draw(pil_img_3)
        min_point = (0, -10)
        end_point = (pil_img_3.size[0]+10, pil_img_3.size[1])
        draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                       width=line_width)
        white_pil.paste(pil_img_3, (pil_img_3.size[0], 0))

        np_img_4 = np.asarray(gif_4_imgs[idx])
        pil_img_4 = Image.fromarray(np_img_4)
        draw = ImageDraw.Draw(pil_img_4)
        min_point = (-10, -10)
        end_point = (pil_img_4.size[0], pil_img_4.size[1])
        draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                       width=line_width)
        white_pil.paste(pil_img_4, (0, 0))

        images.append(white_pil.copy())

    images[0].save(out_path, format='GIF',
                   append_images=images[1:],
                   save_all=True, loop=0)
    gif = imageio.mimread(out_path)
    imageio.mimsave(out_path, gif, fps=fps)


def concat4Images(path_1, path_2, path_3, path_4, out_path, line_width=2):
    """
    Function to concat 4 images into 1
    path_1 to bottom left
    path_2 to bottom right
    path_3 to top right
    path_4 to top left

    Args:
        - (str) path_1:  path to 1st image
        - (str) path_2:  path to 2nd image
        - (str) path_3:  path to 3rd image
        - (str) path_4:  path to 4th image
        - (str) out_path:  path to the output image
    """
    pil_img_1 = Image.open(path_1)
    pil_img_2 = Image.open(path_2)
    pil_img_3 = Image.open(path_3)
    pil_img_4 = Image.open(path_4)

    total_size = (pil_img_1.size[0]*2, pil_img_1.size[1]*2, 3)
    white_image = 255 * np.ones(total_size, np.uint8)
    white_pil = Image.fromarray(white_image)

    draw = ImageDraw.Draw(pil_img_1)
    min_point = (-10, 0)
    end_point = (pil_img_1.size[0], pil_img_1.size[1]+10)
    draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                   width=line_width)
    white_pil.paste(pil_img_1, (0,  pil_img_1.size[1]))

    draw = ImageDraw.Draw(pil_img_2)
    min_point = (0, 0)
    end_point = (pil_img_2.size[0]+10, pil_img_2.size[1]+10)
    draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                   width=line_width)
    white_pil.paste(pil_img_2, (pil_img_2.size[0], pil_img_2.size[1]))

    draw = ImageDraw.Draw(pil_img_3)
    min_point = (0, -10)
    end_point = (pil_img_3.size[0]+10, pil_img_3.size[1])
    draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                   width=line_width)
    white_pil.paste(pil_img_3, (pil_img_3.size[0], 0))

    draw = ImageDraw.Draw(pil_img_4)
    min_point = (-10, -10)
    end_point = (pil_img_4.size[0], pil_img_4.size[1])
    draw.rectangle((min_point, end_point), outline=(255, 255, 255),
                   width=line_width)
    white_pil.paste(pil_img_4, (0, 0))

    white_pil.save(out_path)
