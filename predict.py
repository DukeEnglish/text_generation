#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: predict
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-02-12
# Description:
# -----------------------------------------------------------------------#
import tensorflow as tf
from model import model_build
from conf import embedding_dim, rnn_units, checkpoint_dir


def generate_text(model, start_string, char2idx, idx2char):
    # 要生成的字符个数
    num_generate = 1000

    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 空字符串用于存储结果
    text_generated = []

    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定
    temperature = 1.0

    # batchsize = 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        # del the dim of batch
        predictions = tf.squeeze(predictions, 0)

        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))


def predict(vocab_size, char2idx, idx2char):
    model = model_build(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))
    print(generate_text(model, start_string='牛 '))
