import tensorflow as tf
import numpy as np
import os
import matplotlib as plt
from tree import *

class Config():


class RNN():

    def load_data(self):
        """Loads training data in correct format"""


    def add_variables(self):
        """
        Add model variables:
        For PROJECTION LAYER (PROJECTION)
            Wp - projection matrix used for projection layer
            bp - bias used for projection layer

        For RECURSIVE LAYER (RECURSIVE)
            Wt - transition matrix used for forward propagation vector features
            bt - bias for forward propagation of vector features

            Wr1 - regression matrix used for calc. salience scores
            Wr2 - regression matrix used for calc. salience scores with raw word features
            Wr3 - regression matrix used for calc. salience scores with raw sentence features
            br - bias for calc. salience scores (regression process)
        """
        with tf.get_variable_scope("PROJECTION"):
            Wp = tf.get_variable("Wp", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, None])
            bp = tf.get_variable("bp", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])

        with tf.get_variable_scope("RECURSIVE"):
            Wt = tf.get_variable("Wt", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])
            bt = tf.get_variable("bt", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])

        with tf.get_variable_scope("REGRESSION"):
            Wr1 = tf.get_variable("Wr1", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])
            Wr2 = tf.get_variable("Wr2", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])
            Wr3 = tf.get_variable("Wr3", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])
            br = tf.get_variable("br", dtype=tf.float32, initializer=tf.truncated_normal_initializer, shape=[None, 1])


    def projection_layer(self, raw_tensor):
        """
        Projection layer : raw features -> hidden features
        :param raw_tensor:
        :return:
        """
        hidden_feature_tensor = None

        with tf.get_variable_scope("PROJECTION", reuse=True):
            Wp = tf.get_variable("Wp", reuse=True)
            bp = tf.get_variable("bp", reuse=True)

            hidden_feature_tensor = tf.tanh(tf.add(tf.matmul(raw_tensor, Wp), bp))

        assert isinstance(hidden_feature_tensor, tf.Tensor)
        return hidden_feature_tensor

    def recursive_layer(self, left_feature_tensor, right_feature_tensor):
        """
        Recursive layer : [left_feature, right_feature] -> parent_features
        :param feature_tensor:
        :return:
        """

        parent_feature_tensor = None
        in_tensor = tf.concat([left_feature_tensor, right_feature_tensor], 1)
        with tf.get_variable_scope("RECURSIVE", reuse=True):
            Wt = tf.get_variable("Wt")
            bt = tf.get_variable("bt")

            parent_feature_tensor = tf.tanh(tf.add(tf.matmul(in_tensor, Wt), bt))

        return parent_feature_tensor

    def regression_layer(self, feature_tensor, word_raw_tensor=None, sentence_raw_tensor=None, tag="rest"):
        """
        Regression layer : calc. salience score
        :param feature_tensor:
        :param word_raw_tensor:
        :param sentence_raw_tensor:
        :param tag:
        :return:
        """

        salience_score_tensor = None

        if tag == "rest":
            with tf.get_variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(tf.add(tf.matmul(feature_tensor, Wr1), br))

        if tag == "pre-termminal":
            with tf.get_variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr2 = tf.get_variable("Wr2")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(word_raw_tensor, Wr2)), br))

        if tag == "root":
            with tf.get_variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr2 = tf.get_variable("Wr3")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(sentence_raw_tensor, Wr3)), br))

        return salience_score_tensor

    def inference(self, tree):
        """
        Build computation graph for given tree
        :param tree:
        :return:
        """
        

    def loss(self, true_salience, calc_salience):
        """
        Loss function for salience scores
        :param true_salience:
        :param calc_salience:
        :return:
        """

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(true_salience, calc_salience))

        return loss