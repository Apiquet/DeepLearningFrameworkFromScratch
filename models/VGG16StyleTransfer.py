#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Style Transfer with VGG16:
Style layers = first 6 layers
Content layer = conv5_2
"""

import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm


class VGG16StyleTransfer(tf.keras.Model):

    def __init__(self, tracker_ssd_path, ssd_weights_path=None,
                 n_classes=21, floatType=32):
        """
        Args:
            - (str) tracker_ssd_path: path to github/Apiquet/Tracking_SSD_ReID
            - (str) ssd_weights_path: got from Tracking_SSD_ReID/training.ipynb
            - (int) n_classes: number of target classes
            - (int) floatType: if wanted to se float32 or 16
        """
        super(VGG16StyleTransfer, self).__init__()

        if floatType == 32:
            self.floatType = tf.float32
        elif floatType == 16:
            tf.keras.backend.set_floatx('float16')
            self.floatType = tf.float16
        else:
            raise Exception('floatType should be either 32 or 16')

        sys.path.append(tracker_ssd_path)
        from models.SSD300 import SSD300

        self.n_classes = n_classes
        SSD300_model = SSD300(21, floatType)
        input_shape = (300, 300, 3)
        confs, locs = SSD300_model(tf.zeros([32, 300, 300, 3], self.floatType))
        if ssd_weights_path is not None:
            SSD300_model.load_weights(ssd_weights_path)
        SSD_backbone = SSD300_model.getVGG16()

        from models.VGG16 import VGG16
        self.input_res = input_shape
        self.VGG16 = VGG16(input_shape=input_shape)
        self.VGG16_tilStage5 = self.VGG16.getUntilStage5()

        ssd_seq_idx = 0
        ssd_layer_idx = 0
        for i in range(len(self.VGG16_tilStage5.layers)):
            ssd_layer_idx = i
            if i >= 13:
                ssd_seq_idx = 1
                ssd_layer_idx -= 13
            self.VGG16_tilStage5.get_layer(index=i).set_weights(
                SSD_backbone.get_layer(index=ssd_seq_idx).get_layer(
                    index=ssd_layer_idx).get_weights())
            self.VGG16_tilStage5.get_layer(index=i).trainable = True
        del SSD_backbone
        del SSD300_model

        self.style_layers = []

        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.x = self.VGG16_tilStage5.get_layer(index=0)(self.inputs)

        for i in range(1, 7):
            self.style_layers.append(self.x)
            self.x = self.VGG16_tilStage5.get_layer(index=i)(self.x)

        for i in range(7, len(self.VGG16_tilStage5.layers)-2):
            self.x = self.VGG16_tilStage5.get_layer(index=i)(self.x)

        self.content_layers = [self.x]

        self.model = tf.keras.Model(
            inputs=self.inputs, outputs=self.style_layers+self.content_layers)
        self.model.trainable = False

    def get_features(self, data):
        outputs = self.model(data/255)
        def gram_calc(data):
            return tf.linalg.einsum('bijc,bijd->bcd', data, data) /\
                tf.cast(data.shape[1] * data.shape[2], tf.float32)     
        style_features = [gram_calc(layer) for layer in outputs[:-1]]
        return style_features, outputs[-1]

    def get_loss(self, style_target, style_feature, content_target, content_feature):
        style_loss = tf.add_n([
            tf.reduce_mean(tf.square(features - targets))
            for features, targets in zip(style_feature, style_target)])
        content_loss = tf.add_n([
            0.5 * tf.reduce_sum(tf.square(features - targets))
            for features, targets in zip(style_feature, style_target)])

        return 0.2 * style_loss + 0.001 * content_loss

    def training(self, style_image, content_image, optimizer, epochs=1):
        images = []

        style_targets, _ = self.get_features(style_image)
        _, content_targets = self.get_features(content_image)

        generated_image = tf.cast(content_image, dtype=tf.float32)
        generated_image = tf.Variable(generated_image) 

        images.append(tf.keras.preprocessing.image.array_to_img(tf.squeeze(content_image)))
        for n in tqdm(range(epochs)):
            with tf.GradientTape() as tape:
                style_features, content_features = self.get_features(generated_image)
                loss = self.get_loss(style_targets, style_features, content_targets, content_features)

            gradients = tape.gradient(loss, generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])
            generated_image.assign(
              tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=255.0))

            tmp_img = tf.Variable(generated_image)
            images.append(tf.keras.preprocessing.image.array_to_img(tf.squeeze(tmp_img)))
        return images

    def call(self, style_image, content_image, optimizer, epochs=1):
        return self.training(style_image, content_image, optimizer, epochs)