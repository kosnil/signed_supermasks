import tensorflow as tf
import functools
#import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

from custom_layers import Conv2DExt, DenseExt, MaxPool2DExt, FlattenExt 
from custom_layers import MaskedDense, MaskedConv2D

class Conv2(tf.keras.Model):

    def __init__(self, use_bias=False):
        super(Conv2, self).__init__()  

        
        self.conv_in = Conv2DExt(filters=64, 
                                 kernel_size=3, 
                                 use_bias=use_bias)

        self.conv_second = Conv2DExt(filters=64, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling = MaxPool2DExt(pool_size=(2,2), 
                                    strides=(2,2))
        
        self.flatten = FlattenExt()
        
        self.linear_first = DenseExt(256, 
                                     use_bias=use_bias)
        
        self.linear_second = DenseExt(256, 
                                      use_bias=use_bias)
        
        self.linear_out = DenseExt(10, 
                                   use_bias=use_bias)

        self.alpha = 1.0 
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)

    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        
        x = self.pooling(x)
        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)

        x = self.linear_second(x)
        x = self.activation(x)

        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv4(tf.keras.Model):

    def __init__(self, use_bias=False):
        super(Conv4, self).__init__()  
        
        
        self.conv_in = Conv2DExt(filters=64, 
                                 kernel_size=3, 
                                 use_bias=use_bias)
        
        self.conv_second = Conv2DExt(filters=64, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), 
                                          strides=(2,2))
        
        self.conv_third = Conv2DExt(filters=128, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.conv_fourth = Conv2DExt(filters=128, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), 
                                           strides=(2,2))
        
        self.flatten = FlattenExt()
        
        self.linear_first = DenseExt(256, 
                                     use_bias=use_bias)
        
        self.linear_second = DenseExt(256, 
                                      use_bias=use_bias)
        
        self.linear_out = DenseExt(10, 
                                   use_bias=use_bias)

        self.alpha = 1.0 
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)

    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)

        x = self.conv_second(x)
        x = self.activation(x)
        
        x = self.pooling_first(x)

        x = self.conv_third(x)
        x = self.activation(x)

        x = self.conv_fourth(x)
        x = self.activation(x)

        x = self.pooling_second(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)


        x = self.linear_second(x)
        x = self.activation(x)


        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv6(tf.keras.Model):

    def __init__(self, use_bias=False):
        super(Conv6, self).__init__() 
         
        self.conv_in = Conv2DExt(filters=64, 
                                 kernel_size=3, 
                                 use_bias=use_bias)
        
        self.conv_second = Conv2DExt(filters=64, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), 
                                          strides=(2,2))
        
        self.conv_third = Conv2DExt(filters=128, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.conv_fourth = Conv2DExt(filters=128, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), 
                                           strides=(2,2))
        
        self.conv_fifth = Conv2DExt(filters=256, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.conv_sixth = Conv2DExt(filters=256, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.pooling_third = MaxPool2DExt(pool_size=(2,2), 
                                          strides=(2,2))
        
        self.flatten = FlattenExt()
        
        self.linear_first = DenseExt(256, 
                                     use_bias=use_bias)
        
        self.linear_second = DenseExt(256, 
                                      use_bias=use_bias)
        
        self.linear_out = DenseExt(10, 
                                   use_bias=use_bias)

        self.alpha = 1.0 
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)

    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)

        x = self.conv_third(x)
        x = self.activation(x)
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
        
        x = self.conv_sixth(x)  
        x = self.activation(x)
        x = self.pooling_third(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        
        x = self.linear_second(x)
        x = self.activation(x)
        
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv8(tf.keras.Model):

    def __init__(self, use_bias=False):
        super(Conv8, self).__init__() 
        
         
        self.conv_in = Conv2DExt(filters=64, 
                                 kernel_size=3, 
                                 use_bias=use_bias)
        
        self.conv_second = Conv2DExt(filters=64, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), 
                                          strides=(2,2))
        
        self.conv_third = Conv2DExt(filters=128, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.conv_fourth = Conv2DExt(filters=128, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), 
                                           strides=(2,2))
        
        self.conv_fifth = Conv2DExt(filters=256, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.conv_sixth = Conv2DExt(filters=256, 
                                    kernel_size=3, 
                                    use_bias=use_bias)
        
        self.pooling_third = MaxPool2DExt(pool_size=(2,2), 
                                          strides=(2,2))
        
        self.conv_seventh = Conv2DExt(filters=512, 
                                      kernel_size=3, 
                                      use_bias=use_bias)
        
        self.conv_eighth = Conv2DExt(filters=512, 
                                     kernel_size=3, 
                                     use_bias=use_bias)
        
        self.pooling_fourth = MaxPool2DExt(pool_size=(2,2), 
                                           strides=(2,2))
        
        self.flatten = FlattenExt()
        
        self.linear_first = DenseExt(256, 
                                     use_bias=use_bias)
        
        self.linear_second = DenseExt(256, 
                                      use_bias=use_bias)
        
        self.linear_out = DenseExt(10, 
                                   use_bias=use_bias)


        self.alpha = 1.0 

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)
        
    
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)

        x = self.conv_third(x)
        x = self.activation(x)
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
        
        x = self.conv_sixth(x)  
        x = self.activation(x)
        x = self.pooling_third(x)

        x = self.conv_seventh(x)
        x = self.activation(x)
        
        x = self.conv_eighth(x)  
        x = self.activation(x)
        x = self.pooling_fourth(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = tf.nn.relu(x)
        
        x = self.linear_second(x)
        x = tf.nn.relu(x)
        
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)
    
    


class Conv2_Mask(tf.keras.Model):

    def __init__(self, 
                 input_shape, 
                 dynamic_scaling_cnn=True, 
                 dynamic_scaling_dense=True, 
                 k_cnn=0.4, 
                 k_dense=0.3, 
                 width_multiplier=1, 
                 masking_method="fixed"):
        
        super(Conv2_Mask, self).__init__()

 
        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), 
                                    kernel_size=3, 
                                    input_shape=input_shape, 
                                    dynamic_scaling=dynamic_scaling_cnn, 
                                    masking_method=masking_method,
                                    k=k_cnn,
                                    name="conv_in") 

        conv_in_out_shape = self.conv_in.out_shape


        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_in_out_shape, 
                                        dynamic_scaling=dynamic_scaling_cnn, 
                                        masking_method=masking_method,
                                        k=k_cnn,
                                        name="conv_second") 

        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling = MaxPool2DExt(input_shape = conv_second_out_shape, 
                                    pool_size=(2,2), 
                                    strides=(2,2))
        
        pooling_out_shape= self.pooling.out_shape 
        
        self.flatten = FlattenExt() 
        
        self.linear_first = MaskedDense(input_dim=int(tf.math.reduce_prod(pooling_out_shape[1:]).numpy()),
                                        units=int(256*width_multiplier), 
                                        dynamic_scaling=dynamic_scaling_dense, 
                                        masking_method=masking_method, 
                                        k=k_dense,
                                        name="linear_first")

        self.linear_second = MaskedDense(input_dim=int(256*width_multiplier),
                                         units=int(256*width_multiplier), 
                                         dynamic_scaling=dynamic_scaling_dense, 
                                         masking_method=masking_method, 
                                         k=k_dense,
                                         name="linear_second") 

        self.linear_out = MaskedDense(input_dim=int(256*width_multiplier),
                                      units=10, 
                                      dynamic_scaling=dynamic_scaling_dense, 
                                      masking_method=masking_method, 
                                      k=k_dense,
                                      name="linear_out") 

        self.alpha = 1.0 

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)


    def call_with_intermediates(self, inputs):
        
        layerwise_output = []

        x = self.conv_in(inputs)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_second(x)

        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling(x)
        
        x = self.flatten(x)

        x = self.linear_first(x)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_second(x)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x,layerwise_output

    @tf.function
    def call(self, inputs): 
        

        x = self.conv_in(inputs)
        x = self.activation(x)

        
        x = self.conv_second(x)
        x = self.activation(x)


        x = self.pooling(x)
        
        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)


        x = self.linear_second(x)
        x = self.activation(x)


        x = self.linear_out(x)
        x = tf.nn.softmax(x)
        
        return x 

class Conv4_Mask(tf.keras.Model):
    

    def __init__(self, 
                 input_shape, 
                 dynamic_scaling_cnn=True, 
                 k_cnn=0.4, 
                 k_dense=0.3,
                 dynamic_scaling_dense=True, 
                 width_multiplier=1, 
                 masking_method="fixed"):
        
        super(Conv4_Mask, self).__init__()

        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), 
                                    kernel_size=3, 
                                    input_shape=input_shape, 
                                    k=k_cnn, 
                                    dynamic_scaling=dynamic_scaling_cnn,
                                    masking_method=masking_method, 
                                    name="conv_in")
        
        conv_in_out_shape = self.conv_in.out_shape


        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_in_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_second")

        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, 
                                         pool_size=(2,2), 
                                         strides=(2,2))
        
        pooling_first_out_shape = self.pooling_first.out_shape 

        self.conv_third = MaskedConv2D(filters=int(128*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape = pooling_first_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_third")

        conv_third_out_shape = self.conv_third.out_shape

        self.conv_fourth = MaskedConv2D(filters=int(128*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_third_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_fourth")

        conv_fourth_out_shape = self.conv_fourth.out_shape

        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, 
                                           pool_size=(2,2), 
                                           strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape

        self.flatten = FlattenExt() 

        self.linear_first = MaskedDense(input_dim=int(tf.math.reduce_prod(pooling_second_out_shape[1:]).numpy()),
                                        units=int(256*width_multiplier), 
                                        masking_method=masking_method,
                                        k=k_dense, 
                                        dynamic_scaling=dynamic_scaling_dense, 
                                        name="linear_first")

        self.linear_second = MaskedDense(input_dim=int(256*width_multiplier),
                                         units=int(256*width_multiplier), 
                                         masking_method=masking_method, 
                                         k=k_dense, 
                                         dynamic_scaling=dynamic_scaling_dense, 
                                         name="linear_second")

        self.linear_out = MaskedDense(input_dim=int(256*width_multiplier),
                                      units=10, 
                                      masking_method=masking_method, 
                                      k=k_dense, 
                                      dynamic_scaling=dynamic_scaling_dense,
                                      name="linear_out") 
        
        self.alpha = 1.0 


    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)
    
    def call_with_intermediates(self, inputs):
        
        layerwise_output = []

        x = self.conv_in(inputs)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_second(x)

        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_first(x)
        
        x = self.conv_third(x)

        x = self.activation(x)
        layerwise_output.append(x)
    

        
        x = self.conv_fourth(x)

        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_second(x)

        x = self.flatten(x)

        x = self.linear_first(x)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_second(x)

        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x,layerwise_output


    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        
        x = self.conv_third(x)
        x = self.activation(x)
    
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)

        x = self.linear_second(x)
        x = self.activation(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x 

class Conv6_Mask(tf.keras.Model):
    

    def __init__(self, 
                 input_shape,
                 dynamic_scaling_cnn=True, 
                 k_cnn=0.4, 
                 k_dense=0.3,
                 dynamic_scaling_dense=True, 
                 width_multiplier=1, 
                 masking_method="fixed"):
        
        super(Conv6_Mask, self).__init__()

        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), 
                                    kernel_size=3, 
                                    input_shape=input_shape, 
                                    k=k_cnn, 
                                    dynamic_scaling=dynamic_scaling_cnn,
                                    masking_method=masking_method, 
                                    name="conv_in")
        
        conv_in_out_shape = self.conv_in.out_shape


        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_in_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_second")
        
        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, 
                                         pool_size=(2,2), 
                                         strides=(2,2))
        
        pooling_first_out_shape = self.pooling_first.out_shape 

        self.conv_third = MaskedConv2D(filters=int(128*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape = pooling_first_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_third")
        
        conv_third_out_shape = self.conv_third.out_shape

        self.conv_fourth = MaskedConv2D(filters=int(128*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_third_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_fourth")
        
        conv_fourth_out_shape = self.conv_fourth.out_shape

        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, 
                                           pool_size=(2,2), 
                                           strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape

        self.conv_fifth = MaskedConv2D(filters=int(256*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape=pooling_second_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_fifth")

        conv_fifth_out_shape = self.conv_fifth.out_shape
        
        self.conv_sixth = MaskedConv2D(filters=int(256*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape=conv_fifth_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_sixth")

        conv_sixth_out_shape = self.conv_sixth.out_shape
        
        self.pooling_third = MaxPool2DExt(input_shape=conv_sixth_out_shape, 
                                          pool_size=(2,2), 
                                          strides=(2,2))

        pooling_third_out_shape = self.pooling_third.out_shape

        self.flatten = FlattenExt() 

        self.linear_first = MaskedDense(input_dim=int(tf.math.reduce_prod(pooling_third_out_shape[1:]).numpy()),
                                        units=int(256*width_multiplier), 
                                        masking_method=masking_method, 
                                        k=k_dense, 
                                        dynamic_scaling=dynamic_scaling_dense,
                                        name="linear_first") 
        
        self.linear_second = MaskedDense(input_dim=int(256*width_multiplier),
                                         units=int(256*width_multiplier), 
                                         masking_method=masking_method, 
                                         k=k_dense, 
                                         dynamic_scaling=dynamic_scaling_dense,
                                         name="linear_second") 
        
        self.linear_out = MaskedDense(input_dim=int(256*width_multiplier),
                                      units=10,
                                      masking_method=masking_method, 
                                      k=k_dense, 
                                      dynamic_scaling=dynamic_scaling_dense,
                                      name="linear_out") 

        self.alpha = 1.0 

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)
    
    def call_with_intermediates(self, inputs):
        
        layerwise_output = []

        x = self.conv_in(inputs)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_first(x)
        
        x = self.conv_third(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_sixth(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_third(x)
        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_second(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x,layerwise_output


    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        
        x = self.conv_third(x)
        x = self.activation(x)
    
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
    
        
        x = self.conv_sixth(x)
        x = self.activation(x)
        x = self.pooling_third(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)

        x = self.linear_second(x)
        x = self.activation(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x 

class Conv8_Mask(tf.keras.Model):
    

    def __init__(self, 
                 input_shape,
                 dynamic_scaling_cnn=True, 
                 k_cnn=0.4, 
                 k_dense=0.3,
                 dynamic_scaling_dense=True, 
                 width_multiplier=1, 
                 masking_method="fixed"):
        
        super(Conv8_Mask, self).__init__()


        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), 
                                    kernel_size=3, 
                                    input_shape=input_shape, 
                                    k=k_cnn, 
                                    dynamic_scaling=dynamic_scaling_cnn,
                                    masking_method=masking_method, 
                                    name="conv_in")
        conv_in_out_shape = self.conv_in.out_shape

        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_in_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_second")
        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, 
                                         pool_size=(2,2), 
                                         strides=(2,2))
        pooling_first_out_shape = self.pooling_first.out_shape 

        self.conv_third = MaskedConv2D(filters=int(128*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape = pooling_first_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_third")
        conv_third_out_shape = self.conv_third.out_shape

        self.conv_fourth = MaskedConv2D(filters=int(128*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape = conv_third_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_fourth")

        conv_fourth_out_shape = self.conv_fourth.out_shape

        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, 
                                           pool_size=(2,2), 
                                           strides=(2,2))
        
        pooling_second_out_shape = self.pooling_second.out_shape

        self.conv_fifth = MaskedConv2D(filters=int(256*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape=pooling_second_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn,
                                       masking_method=masking_method, 
                                       name="conv_fifth")

        conv_fifth_out_shape = self.conv_fifth.out_shape
        
        self.conv_sixth = MaskedConv2D(filters=int(256*width_multiplier), 
                                       kernel_size=3, 
                                       input_shape=conv_fifth_out_shape, 
                                       k=k_cnn,
                                       dynamic_scaling=dynamic_scaling_cnn, 
                                       masking_method=masking_method, 
                                       name="conv_sixth")

        conv_sixth_out_shape = self.conv_sixth.out_shape
        
        self.pooling_third = MaxPool2DExt(input_shape=conv_sixth_out_shape, 
                                          pool_size=(2,2), 
                                          strides=(2,2))

        pooling_third_out_shape = self.pooling_third.out_shape

        self.conv_seventh = MaskedConv2D(filters=int(512*width_multiplier), 
                                         kernel_size=3, 
                                         input_shape=pooling_third_out_shape, 
                                         k=k_cnn,
                                         dynamic_scaling=dynamic_scaling_cnn,
                                         masking_method=masking_method, 
                                         name="conv_seventh")

        conv_seventh_out_shape = self.conv_seventh.out_shape
        
        self.conv_eighth = MaskedConv2D(filters=int(512*width_multiplier), 
                                        kernel_size=3, 
                                        input_shape=conv_seventh_out_shape, 
                                        k=k_cnn,
                                        dynamic_scaling=dynamic_scaling_cnn,
                                        masking_method=masking_method, 
                                        name="conv_eighth")

        conv_eighth_out_shape = self.conv_eighth.out_shape
        
        self.pooling_fourth = MaxPool2DExt(input_shape=conv_eighth_out_shape, 
                                           pool_size=(2,2), 
                                           strides=(2,2))

        pooling_fourth_out_shape = self.pooling_fourth.out_shape

        self.flatten = FlattenExt() 

        self.linear_first = MaskedDense(input_dim=int(tf.math.reduce_prod(pooling_fourth_out_shape[1:]).numpy()),
                                        units=256, 
                                        masking_method=masking_method, 
                                        k=k_dense, 
                                        dynamic_scaling=dynamic_scaling_dense,
                                        name="linear_first") 

        self.linear_second = MaskedDense(input_dim=256,
                                         units=256, 
                                         masking_method=masking_method, 
                                         k=k_dense, 
                                         dynamic_scaling=dynamic_scaling_dense,
                                         name="linear_second") 

        self.linear_out = MaskedDense(input_dim=256,
                                      units=10, 
                                      masking_method=masking_method, 
                                      k=k_dense, 
                                      dynamic_scaling=dynamic_scaling_dense,
                                      name="linear_out") 

        self.alpha = 1.0 

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=self.alpha)


    def call_with_intermediates(self, inputs):
        layerwise_output = []

        x = self.conv_in(inputs)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.conv_second(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_first(x)
        
        x = self.conv_third(x)
        x = self.activation(x)
        layerwise_output.append(x)
    
        x = self.conv_fourth(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
        layerwise_output.append(x)
    
        x = self.conv_sixth(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_third(x)

        x = self.conv_seventh(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.conv_eighth(x)
        x = self.activation(x)
        layerwise_output.append(x)

        x = self.pooling_fourth(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_second(x)
        x = self.activation(x)
        layerwise_output.append(x)
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x, layerwise_output

    @tf.function
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        
        x = self.conv_third(x)
        x = self.activation(x)
    
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
    
        x = self.conv_sixth(x)
        x = self.activation(x)
        x = self.pooling_third(x)

        x = self.conv_seventh(x)
        x = self.activation(x)
        
        x = self.conv_eighth(x)
        x = self.activation(x)
        x = self.pooling_fourth(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        
        x = self.linear_second(x)
        x = self.activation(x)
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x
