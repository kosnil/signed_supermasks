
import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras import layers

from custom_layers import DenseExt
from custom_layers import MaskedDense


class FCN(tf.keras.Model):

    def __init__(self,
                 use_bias=False):
        super(FCN,self).__init__()

        self.linear_in = DenseExt(units=300,
                                  use_bias=use_bias)

        self.linear_h1 = DenseExt(units=100,
                                  use_bias=use_bias)

        self.linear_out = DenseExt(units=10,
                                   use_bias=use_bias)

        self.alpha = 1.0


    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)

    def call_with_intermediates(self, inputs):

        layerwise_output = []

        x = self.linear_in(inputs)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.linear_h1(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)
        layerwise_output.append(x)

        return x, layerwise_output

    @tf.function
    def call(self, inputs):

        x = self.linear_in(inputs)
        x = self.activation(x)

        x = self.linear_h1(x)
        x = self.activation(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x


class FCN_Mask(tf.keras.Model):

    def __init__(self,
                 dynamic_scaling=False,
                #  tanh_th=.5,
                 k=0.5,
                 activation_fcn="elu",
                 masking_method="fixed"):

        super(FCN_Mask,self).__init__()

        self.linear_in = MaskedDense(input_dim=784,
                                     units=300,
                                     dynamic_scaling=dynamic_scaling,
                                     k=k,
                                    #  tanh_th=tanh_th,
                                     masking_method=masking_method)

        self.linear_h1 = MaskedDense(input_dim=300,
                                     units=100,
                                     dynamic_scaling=dynamic_scaling,
                                     k=k,
                                    #  tanh_th=tanh_th,
                                     masking_method=masking_method)

        self.linear_out = MaskedDense(input_dim=100,
                                      units=10,
                                      dynamic_scaling=dynamic_scaling,
                                      k=k,
                                    #   tanh_th=tanh_th,
                                      masking_method=masking_method)


        self.alpha = 1.0
        self.activation_fcn = activation_fcn

    def activation(self, x):
        if self.activation_fcn == "elu":
            return tf.keras.activations.elu(x, alpha=self.alpha)
        if self.activation_fcn == "linear":
            return x


    def call_with_intermediates(self,inputs):

        layerwise_output = []

        x = self.linear_in(inputs)
        x = self.activation(x)

        layerwise_output.append(x)

        x = self.linear_h1(x)
        x = self.activation(x)

        layerwise_output.append(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x, layerwise_output

    @tf.function
    def call(self, inputs):

        x = self.linear_in(inputs)
        x = self.activation(x)

        x = self.linear_h1(x)
        x = self.activation(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)


        return x