#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: main
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-02-12
# Description:
# -----------------------------------------------------------------------#
from conf import path_to_file, BUFFER_SIZE, BATCH_SIZE
import numpy as np
from train import train
from predict import predict
from model import model_build
import tensorflow as tf


def build_data():
    # 读取并为 py2 compat 解码
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # 文本长度是指文本中的字符个数
    print('Length of text: {} characters'.format(len(text)))

    # 文本中的非重复字符
    vocab = set(text)
    print('{} unique characters'.format(len(vocab)))
    # 创建从非重复字符到索引的映射
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(list(vocab))
    text_as_idx = np.array([char2idx[c] for c in text])

    # 设定每个输入句子长度的最大值，而为了预测出100个char，
    # 实际要输入的是101个，第一个char是trigger
    seq_length = 100
    examples_per_epoch = len(text) // seq_length
    print("examples_per_epoch: ", examples_per_epoch)

    # 创建训练样本/目标
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_idx)

    for i in char_dataset.take(5):
        #     print(i.numpy())
        print(i, i.numpy(), idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    # 对于RNN模型，不仅考虑当前的输入字符，还会考虑上一步的信息
    # （这个信息除了保存在cell中的外，还会有上一步的输出作为本次的输入）
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    vocab_size = len(vocab)

    return dataset, vocab_size, text_as_idx, char2idx, idx2char


if __name__ == '__main__':
    dataset, vocab_size, text_as_idx, char2idx, idx2char = build_data()
    train(dataset, vocab_size)
    predict(vocab_size, char2idx, idx2char)
