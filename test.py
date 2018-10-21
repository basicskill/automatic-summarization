#! /usr/bin/python

from tree import *
from collections import OrderedDict
import tensorflow as tf
import time
import os
from shutil import rmtree


"""
Parameters of network
"""
learning_rate = 0.01
regularization = 0.1


class RNN():

    def add_variables(self):
        """
        Add model variables:
        For PROJECTION LAYER (PROJECTION)
            Wp - projection matrix used for projection layer
            bp - bias used for projection layer

        For RECURSIVE LAYER (RECURSIVE)
            Wt - transition matrix used for forward propagation vector features
            bt - bias for forward propagation of vector features

        For REGRESSION LAYER (REGRESSION)
            Wr1 - regression matrix used for calc. salience scores
            Wr2 - regression matrix used for calc. salience scores with raw word features
            Wr3 - regression matrix used for calc. salience scores with raw sentence features
            br - bias for calc. salience scores (regression process)
        """

        global Wr1_reg, Wr2_reg, Wr3_reg, Wt_reg, Wp_reg, br_reg, bt_reg, bp_reg

        with tf.variable_scope('PROJECTION'):
            tf.get_variable('Wp', dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(Wp_reg, dtype=tf.float32), shape=[15, 8]))
            tf.get_variable("bp", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(bp_reg, dtype=tf.float32), shape=[1, 8]))

        with tf.variable_scope("RECURSIVE"):
            tf.get_variable("Wt", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(Wt_reg, dtype=tf.float32), shape=[16, 8]))
            tf.get_variable("bt", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(bt_reg, dtype=tf.float32), shape=[1, 8]))

        with tf.variable_scope("REGRESSION"):
            tf.get_variable("Wr1", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(Wr1_reg, dtype=tf.float32), shape=[8, 1]))
            tf.get_variable("Wr2", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(Wr2_reg, dtype=tf.float32), shape=[15, 1]))
            tf.get_variable("Wr3", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(Wr3_reg, dtype=tf.float32), shape=[14, 1]))
            tf.get_variable("br", dtype=tf.float32,
                            initializer=tf.reshape(tf.convert_to_tensor(br_reg, dtype=tf.float32), shape=[1, 1]))

    def projection_layer(self, raw_word_tensor):
        """
        Projection layer : raw features -> hidden features
        :param raw_word_tensor:
        :return: hidden_feature_tensor
        """
        hidden_feature_tensor = None

        with tf.variable_scope('PROJECTION', reuse=tf.AUTO_REUSE):
            Wp = tf.get_variable('Wp')
            bp = tf.get_variable("bp")

            hidden_feature_tensor = tf.tanh(tf.add(tf.matmul(raw_word_tensor, Wp), bp))

        return hidden_feature_tensor

    def recursive_layer(self, left_feature_tensor, right_feature_tensor):
        """
        Recursive layer : [left_feature, right_feature] -> parent_features
        :param left_feature_tensor:
        :param right_feature_tensor:
        :return: parent_feature_tensor
        """
        parent_feature_tensor = None
        in_tensor = tf.concat([left_feature_tensor, right_feature_tensor], 1)

        with tf.variable_scope("RECURSIVE", reuse=True):
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
        :return: salience_score_tensor
        """
        salience_score_tensor = None

        if tag == "rest":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(tf.add(tf.matmul(feature_tensor, Wr1), br))

        if tag == "pre-terminal":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr2 = tf.get_variable("Wr2")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(
                    tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(word_raw_tensor, Wr2)), br))

        if tag == "root":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr3 = tf.get_variable("Wr3")
                br = tf.get_variable("br")

                salience_score_tensor = tf.sigmoid(
                    tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(sentence_raw_tensor, Wr3)), br))

        return salience_score_tensor

    def inference(self, node, sentence_raw_tensor):
        """
        Build computation graph for given node
        :param node:
        :param sentence_raw_tensor:
        :return: feature_tensors_dict, salience_tensors_dict
        """
        feature_tensors_dict = OrderedDict()
        salience_tensors_dict = OrderedDict()

        if node.label == "ROOT":
            left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=sentence_raw_tensor)

            feature_tensor = left_ftrs[node.left]
            salience_tensor = left_sal[node.left]

            feature_tensors_dict.update(left_ftrs)
            feature_tensors_dict[node] = feature_tensor

            salience_tensors_dict.update(left_sal)
            salience_tensors_dict[node] = salience_tensor

        if node.parent is not None:
            if node.parent.label == "ROOT":
                left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=None)
                right_ftrs, right_sal = self.inference(node.right, sentence_raw_tensor=None)

                feature_tensor = self.recursive_layer(left_feature_tensor=left_ftrs[node.left],
                                                      right_feature_tensor=right_ftrs[node.right])
                salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=None,
                                                        sentence_raw_tensor=sentence_raw_tensor, tag="root")

                feature_tensors_dict.update(left_ftrs)
                feature_tensors_dict[node] = feature_tensor
                feature_tensors_dict.update(right_ftrs)

                salience_tensors_dict.update(left_sal)
                salience_tensors_dict[node] = salience_tensor
                salience_tensors_dict.update(right_sal)

        if node.isPreTerminal is True:
            word_raw_tensor = tf.convert_to_tensor(node.left.feature, dtype=tf.float32)
            # assert word_raw_tensor.shape.as_list() == [15, ]
            word_raw_tensor = tf.reshape(word_raw_tensor, shape=[1, 15])
            # assert isinstance(word_raw_tensor, tf.Tensor)

            feature_tensor = self.projection_layer(raw_word_tensor=word_raw_tensor)
            salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=word_raw_tensor,
                                                    sentence_raw_tensor=None, tag="pre-terminal")

            feature_tensors_dict[node] = feature_tensor
            salience_tensors_dict[node] = salience_tensor

        if node.isPreTerminal is not True and node.label != "ROOT" and node.parent.label != "ROOT":
            left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=None)
            right_ftrs, right_sal = self.inference(node.right, sentence_raw_tensor=None)

            # assert left_ftrs[node.left].shape.as_list() == [1, 8]
            # assert right_ftrs[node.right].shape.as_list() == [1, 8]

            feature_tensor = self.recursive_layer(left_feature_tensor=left_ftrs[node.left],
                                                  right_feature_tensor=right_ftrs[node.right])
            salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=None,
                                                    sentence_raw_tensor=None, tag="rest")

            feature_tensors_dict.update(left_ftrs)
            feature_tensors_dict[node] = feature_tensor
            feature_tensors_dict.update(right_ftrs)

            salience_tensors_dict.update(left_sal)
            salience_tensors_dict[node] = salience_tensor
            salience_tensors_dict.update(right_sal)

        return feature_tensors_dict, salience_tensors_dict

    def validate(self, tree):
        """
        Runs validation after one epoch of training
        :return: losses
        """

        global Wr1_reg, Wr2_reg, Wr3_reg, Wt_reg, Wp_reg, br_reg, bt_reg, bp_reg
        out_loss = None

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.add_variables()
                init = tf.global_variables_initializer()
                sess.run(init)
                sentence_raw_tensor = tf.convert_to_tensor(tree.sentence_features, dtype=tf.float32)
                sentence_raw_tensor = tf.reshape(sentence_raw_tensor, shape=[1, 14])

                try:
                    feature_dic, salience_dic = self.inference(tree.root, sentence_raw_tensor=sentence_raw_tensor)
                except:
                    print()
                    return

                calc_saliences = []
                calc_saliences_eval = []
                out = 0
                for key, value in salience_dic.items():
                    calc_saliences.append(value)
                    calc_saliences_eval.append(value.eval())
                true_saliences = tree.getSaliences()
                l = len(true_saliences)
                t_s = tf.convert_to_tensor(true_saliences, dtype=tf.float32)
        
        tf.reset_default_graph()
        return calc_saliences_eval[-1][0,0]



if __name__ == "__main__":
    start = time.time()
    import sys
    import pickle
    Wp_reg = np.zeros((15, 8))
    bp_reg = np.zeros((1, 8))
    Wt_reg = np.zeros((16,8))
    bt_reg = np.zeros((1, 8))
    Wr1_reg = np.zeros((8, 1))
    Wr2_reg = np.zeros((15, 1))
    Wr3_reg = np.zeros((14, 1))
    br_reg = np.zeros((1, 1))
    
    mapwp = np.memmap("./weights/29/wp", dtype='float32', mode='r+', shape=(15, 8))
    Wp_reg[:] = mapwp[:]
    mapbp = np.memmap("./weights/29/bp", dtype='float32', mode='r+', shape=(1, 8))
    bp_reg[:] = mapbp[:]
    mapwt = np.memmap("./weights/29/wt", dtype='float32', mode='r+', shape=(16, 8))
    Wt_reg[:] = mapwt[:]
    mapbt = np.memmap("./weights/29/bt", dtype='float32', mode='r+', shape=(1, 8))
    bt_reg[:] = mapbt[:]
    mapwr1 = np.memmap("./weights/29/wr1", dtype='float32', mode='r+', shape=(8, 1))
    Wr1_reg[:] = mapwr1[:]
    mapwr2 = np.memmap("./weights/29/wr2", dtype='float32', mode='r+', shape=(15, 1))
    Wr2_reg[:] = mapwr2[:]
    mapwr3 = np.memmap("./weights/29/wr3", dtype='float32', mode='r+', shape=(14, 1))
    Wr3_reg[:] = mapwr3[:]
    mapbr = np.memmap("./weights/29/br", dtype='float32', mode='r+', shape=(1, 1))
    br_reg[:] = mapbr[:]

    r = RNN()
    testdir = "/home/anglahel/test/"
    for f in os.listdir(testdir):
        print(f)
        listlisttrees = pickle.load(open(testdir+f, "rb"))
        for listtrees in listlisttrees:
            for tree in listtrees:
                #print(tree.getSaliences())
                out = r.validate(tree)
                tree.root.salience = out
                print("\t" + str(tree.root.salience))
        pickle.dump(listlisttrees, open(testdir+f+"trained", "wb"))
        print(testdir+f+"trained")
    """
    t = pickle.load(open("/home/anglahel/validation/1007.pickle", "rb"))
    dic = r.validate(t)
    for key, value in dic.items():
        print(key.label)
    """
