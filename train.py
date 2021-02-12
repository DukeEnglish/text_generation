# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-02-12
# Description:
# -----------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from model import model_build
import numpy as np
import os
from conf import embedding_dim, rnn_units, BATCH_SIZE, checkpoint_prefix, EPOCHS


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train(dataset, vocab_size):
    model = model_build(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

    # 配置训练
    model.compile(optimizer='adam', loss=loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                             save_weights_only=True)

    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
