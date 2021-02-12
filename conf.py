#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: conf
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-02-12
# Description:
# -----------------------------------------------------------------------#
import os

path_to_file = '/Users/ljy/nlp/text_generator/couplet/train/in.txt'
BATCH_SIZE = 16

# 设定缓冲区大小，以重新排列数据集合
# TF数据被设计为可处理 可能是无限 的序列
# 所以它不会try在内存中shuffle整个序列，
# 它会维持一个缓冲区，在缓冲区shuffle
BUFFER_SIZE = 1000

embedding_dim = 64

rnn_units = 256

# train
EPOCHS = 2

# 配置checkpoint
checkpoint_dir = './checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
