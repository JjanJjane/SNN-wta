import keras.layers
import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# import tensorflow_probability as tfp

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

#from keras.engine.base_layer import Layer
from keras import backend

import sys

from tensorflow.python.ops import math_ops

# custom gradient
import lib_snn.ops.nn_grad


#

import numpy as np

import matplotlib.pyplot as plt

import collections

#
import lib_snn

#


#from main_hp_tune import conf

from config import conf


from lib_snn.layers import Layer




# abstract class for DNN / SNN
# Activation
class Activation(keras.engine.base_layer.Layer):
    #index = None    # layer index count starts from InputGenLayer

    def __init__(self, act_type=None, loc='HID', **kwargs):

        super(Activation, self).__init__(**kwargs)
        #
        #self.depth = -1

        self.act_type = act_type
        #
        self.conf = conf
        #self.en_snn = (self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn)

        self.depth=Layer.index

        #self.n_name=name
        self.loc=loc

        # ReLU-6
        self.relu_max_value = kwargs.pop('relu_max_value',None)

        # tdbn
        self.tdbn_arg = kwargs.pop('tdbn',None)



        #
        self.output_shape_fixed_batch = None

        #
        self.act = None

        #
        #self.en_record_output = False
        #self.record_output = None
        #self.record_logit = None


    #
    def build(self, input_shapes):
        # super(Conv2D,self).build(input_shapes)
        # print(Conv2D.__mro__[2])
        # tf.keras.layers.Conv2D.build(self,input_shapes)
        # self.__mro__[2].build(self,input_shapes)

        # build ann model
        #print('build layer - {}'.format(self.name))
        super().build(input_shapes)
        # print(super())
        # print(super().build)

        #print(super().build)

        #assert False

        #
        self.output_shape_fixed_batch = input_shapes



        # self.act_snn = lib_snn.layers.Neuron(self.output_shape_fixed_batch,self.conf,\
        # n_type,self.conf.neural_coding,depth,self.name)

        # self.act_dnn = tf.keras.layers.ReLU()

        # if not self.en_snn:
        # self.act = self.act_dnn


        # activation, neuron
        self.act = None

        def identity_func(x):
            return x

        #

        if self.conf.nn_mode=='ANN':
            if self.act_type == 'relu':
                #self.act_dnn = tf.keras.layers.ReLU(max_value=relu_max_value, name=name_act)
                #self.act_dnn = tf.keras.layers.ReLU(max_value=relu_max_value)
                #self.act_dnn = tf.keras.layers.ReLU(max_value=6.0)
                self.act = tf.keras.layers.ReLU(name=self.name)
            elif self.act_type == 'softmax':
                #self.act_dnn = tf.keras.layers.Softmax(name=name_act)
                self.act = tf.keras.layers.Softmax(name=self.name)
            else:
                assert False
        else:
            if self.act_type == 'softmax':
                self.act = tf.keras.layers.Softmax(name=self.name)
            elif self.act_type in {'IF', 'LIF'}:
                self.act = lib_snn.neurons.Neuron(self.output_shape_fixed_batch, self.conf, \
                                              self.act_type, self.conf.neural_coding, self.depth, loc=self.loc, name=self.name)
            else:
                assert False

        self.built = True

    #
    def call(self, inputs, training=None):
    #def call(self, inputs):

        if training is None:
            training = backend.learning_phase()

        #print(self.act)
        #print(self.name)

        #ret = self.act(inputs)
        if isinstance(self.act, tf.keras.layers.Softmax):
            ret = self.act(inputs)
        else:
            ret = self.act(inputs, training)

        return ret

    #
    def get_config(self):
        config = super().get_config()
        config.update({
            "act_type": self.act_type,
            "loc": self.loc
        })
        return config


    #
    def reset(self):
        if hasattr(self.act, 'reset'):
            self.act.reset()

