# -*- coding:utf-8 -*-
'''
https://github.com/bojone/crf/
'''
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


class CRF(Layer):
    """
    Use Keras to implement CRF Layer
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """
        ignore_last_label：Define to ignore the last label for mask or not.
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """
        Recursion calculate normalization factor
        Point:  1.recursion
               2.use logsumexp to avoid overflow.
        Technique：use expand_dims to align tensor.
        """
        states = tf.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = tf.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = tf.math.reduce_logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]

    def path_score(self, inputs, labels):
        """
        Calculate target path relative possibility（Not normalized）
        Point：Get point from each label and add transition possibility score.
        Technique：use predict dot target to extract score of target path.
        """
        point_score = tf.math.reduce_sum(tf.math.reduce_sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = tf.expand_dims(labels[:, :-1], 3)
        labels2 = tf.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = tf.expand_dims(tf.expand_dims(self.trans, 0), 0)
        trans_score = tf.math.reduce_sum(tf.math.reduce_sum(trans*labels, [2,3]), 1, keepdims=True)
        return tf.math.add(point_score, trans_score) # 两部分得分之和

    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = tf.math.reduce_logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return tf.math.subtract(log_norm, path_score) # 即log(分子/分母)

    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = tf.equal(tf.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = tf.dtypes.cast(isequal, 'float32')
        if mask == None:
            return tf.math.reduce_mean(isequal)
        else:
            return tf.math.reduce_sum(isequal*mask) / tf.math.reduce_sum(mask)

    def get_config(self):
        config = {
            'ignore_last_label': self.ignore_last_label
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))