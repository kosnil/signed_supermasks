import tensorflow as tf
import functools
#import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

from custom_layers import Conv2DExt, DenseExt, MaxPool2DExt, FlattenExt, BatchNormExt, GlobalAveragePooling2DExt
from custom_layers import MaskedDense, MaskedConv2D

class ResnetBlockC_Mask(tf.keras.Model):
    # implementation follows https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
                 filters,
                 input_shape,
                 strides=(1,1),
                 kernel_size = 3):

        super(ResnetBlockC_Mask, self).__init__(name='')

        self.type = "resnet_block"

        #Shortcut
        self.conv_sc = MaskedConv2D(filters=filters*4,
                                    kernel_size=1,
                                    padding="valid",
                                    strides=strides,
                                    input_shape=input_shape)

        self.bn_sc = BatchNormExt(center=False,
                                  scale=False,
                                  trainable=False)

        #Conv1
        self.conv2a = MaskedConv2D(filters=filters,
                                   kernel_size=1,
                                   strides=strides,
                                   padding="valid",
                                   input_shape=input_shape)

        self.bn2a = BatchNormExt(center=False,
                                 scale=False,
                                 trainable=False)

        conv2a_os = self.conv2a.out_shape

        #Conv2
        self.conv2b = MaskedConv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                padding="same",
                                input_shape=conv2a_os)

        self.bn2b = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        conv2b_os = self.conv2b.out_shape

        #Conv3
        self.conv2c = MaskedConv2D(filters=filters*4,
                                kernel_size=1,
                                strides=(1,1),
                                padding="valid",
                                input_shape=conv2b_os)

        self.bn2c = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        self.out_shape = self.conv2c.out_shape

        print("output shape after 3rd conv (1x1): ", self.out_shape)


    @tf.function
    def pad_depth2(self, x, desired_channels):
        delta_channels = desired_channels - x.shape[-1]

        paddings = [[0,0]] * len(x.shape.as_list())
        paddings[-1] = [0,delta_channels]

        return tf.pad(x,paddings)

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)


        # shortcut = self.pad_depth2(input_tensor, x.shape[-1])
        shortcut = self.conv_sc(input_tensor)
        shortcut = self.bn_sc(shortcut, training=training)

        x += shortcut

        return tf.nn.elu(x)

class ResnetBlockI_Mask(tf.keras.Model):
    # implementation follows https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
               filters,
               input_shape,
               kernel_size = 3):

        super(ResnetBlockI_Mask, self).__init__(name='')

        self.type = "resnet_block"

        print("ResnetBlockI...")
        #Shortcut
        # self.conv_sc = MaskedConv2D(filters=filters*4,
        #                             kernel_size=1,
        #                             padding="same",
        #                             input_shape=input_shape)

        # self.bn_sc = BatchNormExt(center=False,
        #                           scale=False,
        #                           trainable=False)

        #Conv1
        print("input shape: ", input_shape)
        self.conv2a = MaskedConv2D(filters=filters,
                                kernel_size=1,
                                strides=(1,1),
                                padding="valid",
                                input_shape=input_shape)

        self.bn2a = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        conv2a_os = self.conv2a.out_shape
        print("output shape of first conv: ", conv2a_os)
        #Conv2
        self.conv2b = MaskedConv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                padding="same",
                                input_shape=conv2a_os)

        self.bn2b = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        conv2b_os = self.conv2b.out_shape

        print("output shape after second cond (kernel size = 3): ", conv2b_os)

        #Conv3
        self.conv2c = MaskedConv2D(filters=filters*4,
                                kernel_size=1,
                                strides=(1,1),
                                padding="valid",
                                input_shape=conv2b_os)

        self.bn2c = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        self.out_shape = self.conv2c.out_shape

        print("output shape after 3rd conv (1x1): ", self.out_shape)

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        return tf.nn.elu(x + input_tensor)


class BasicResnetBlockC(tf.keras.Model):
    # implementation as in https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
               filters,
               strides=(1,1),
               kernel_size = 3):

        super(BasicResnetBlockC, self).__init__(name='')

        self.type = "resnet_block"
        # print("BasicResnetBlockC...")
        #Shortcut
        self.conv_sc = Conv2DExt(filters=filters,
                                    kernel_size=1,
                                    padding="valid",
                                    strides=strides)

        self.bn_sc = BatchNormExt(center=True,
                                  scale=True,
                                  trainable=True)

        #Conv1
        # print("input shape: ", input_shape)
        self.conv2a = Conv2DExt(filters=filters,
                                kernel_size=1,
                                strides=strides,
                                padding="valid")

        self.bn2a = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        # conv2a_os = self.conv2a.out_shape
        # print("output shape of first conv: ", conv2a_os)
        #Conv2
        self.conv2b = Conv2DExt(filters=filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                padding="same")

        self.bn2b = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        # self.out_shape = self.conv2b.out_shape

        # print("output shape after second cond (kernel size = 3): ", self.out_shape)

    @tf.function
    def pad_depth2(self, x, desired_channels):
        delta_channels = desired_channels - x.shape[-1]

        paddings = [[0,0]] * len(x.shape.as_list())
        paddings[-1] = [0,delta_channels]

        return tf.pad(x,paddings)

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.elu(x)

        # shortcut = self.pad_depth2(input_tensor, x.shape[-1])
        shortcut = self.conv_sc(input_tensor)
        shortcut = self.bn_sc(shortcut, training=training)

        x += shortcut

        return tf.nn.elu(x)


class BasicResnetBlockC_Mask(tf.keras.Model):
    # implementation as in https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
               filters,
               input_shape,
               strides=(1,1),
               kernel_size = 3):

        super(BasicResnetBlockC_Mask, self).__init__(name='')

        self.type = "resnet_block"
        # print("BasicResnetBlockC...")
        #Shortcut
        self.conv_sc = MaskedConv2D(filters=filters,
                                    kernel_size=1,
                                    padding="valid",
                                    strides=strides,
                                    input_shape=input_shape)

        self.bn_sc = BatchNormExt(center=False,
                                  scale=False,
                                  trainable=False)

        #Conv1

        self.conv2a = MaskedConv2D(filters=filters,
                                   kernel_size=1,
                                   strides=strides,
                                   padding="valid",
                                   input_shape=input_shape)

        self.bn2a = BatchNormExt(center=False,
                                 scale=False,
                                 trainable=False)

        conv2a_os = self.conv2a.out_shape

        #Conv2
        self.conv2b = MaskedConv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=(1,1),
                                   padding="same",
                                   input_shape=conv2a_os)

        self.bn2b = BatchNormExt(center=False,
                                 scale=False,
                                 trainable=False)

        self.out_shape = self.conv2b.out_shape

    @tf.function
    def pad_depth2(self, x, desired_channels):
        delta_channels = desired_channels - x.shape[-1]

        paddings = [[0,0]] * len(x.shape.as_list())
        paddings[-1] = [0,delta_channels]

        return tf.pad(x,paddings)

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.elu(x)

        # shortcut = self.pad_depth2(input_tensor, x.shape[-1])
        shortcut = self.conv_sc(input_tensor)
        shortcut = self.bn_sc(shortcut, training=training)

        x += shortcut

        return tf.nn.elu(x)


class BasicResnetBlockI(tf.keras.Model):
    # implementation follows https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
                filters,
                kernel_size = 3):

        super(BasicResnetBlockI, self).__init__(name='')

        self.type = "resnet_block"

        #Conv1
        self.conv2a = Conv2DExt(filters=filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                padding="same")

        self.bn2a = BatchNormExt(center=True,
                                 scale=True,
                                 trainable=True)

        #Conv2
        self.conv2b = Conv2DExt(filters=filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                padding="same")

        self.bn2b = BatchNormExt(center=True,
                                 scale=True,
                                 trainable=True)


    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        return tf.nn.elu(x + input_tensor)

class BasicResnetBlockI_Mask(tf.keras.Model):
    # implementation follows https://www.tensorflow.org/tutorials/customization/custom_layers
    def __init__(self,
                 filters,
                 input_shape,
                 kernel_size = 3):

        super(BasicResnetBlockI_Mask, self).__init__(name='')

        self.type = "resnet_block"

        #Conv1
        self.conv2a = MaskedConv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=(1,1),
                                   padding="same",
                                   input_shape=input_shape)

        self.bn2a = BatchNormExt(center=False,
                                 scale=False,
                                 trainable=False)

        conv2a_os = self.conv2a.out_shape

        #Conv2
        self.conv2b = MaskedConv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=(1,1),
                                   padding="same",
                                   input_shape=conv2a_os)

        self.bn2b = BatchNormExt(center=False,
                                 scale=False,
                                 trainable=False)

        self.out_shape = self.conv2b.out_shape




    @tf.function
    def call(self, input_tensor, training=False):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.elu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        return tf.nn.elu(x + input_tensor)

class ResNet20_Mask(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet20_Mask, self).__init__()

        filters = [16, 16, 32, 64]

        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = MaskedConv2D(filters=filters[0],
                                  input_shape=input_shape,
                                  kernel_size=first_kernel_size,
                                  strides=first_stride,
                                  padding="same")

        conv1_os = self.conv1.out_shape

        self.bn1 = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        self.pool1 = MaxPool2DExt(input_shape=conv1_os,
                                  pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC_Mask(filters=filters[1],
                                        strides=(1,1), #resnetc_strides,
                                        input_shape=conv1_os)

        b1_rn1_os = self.b1_rn1.out_shape

        self.b1_rn2 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn1_os)

        b1_rn2_os = self.b1_rn2.out_shape

        self.b1_rn3 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn2_os)

        b1_rn3_os = self.b1_rn3.out_shape


        #Block 2
        self.b2_rn1 = BasicResnetBlockC_Mask(filters=filters[2],
                                        strides=resnetc_strides,
                                        input_shape=b1_rn3_os)

        b2_rn1_os = self.b2_rn1.out_shape

        self.b2_rn2 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn1_os)

        b2_rn2_os = self.b2_rn2.out_shape

        self.b2_rn3 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn2_os)

        b2_rn3_os = self.b2_rn3.out_shape

        #Block 3
        self.b3_rn1 = BasicResnetBlockC_Mask(filters=filters[3],
                                        strides=resnetc_strides,
                                        input_shape=b2_rn3_os)

        b3_rn1_os = self.b3_rn1.out_shape

        self.b3_rn2 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn1_os)

        b3_rn2_os = self.b3_rn2.out_shape

        self.b3_rn3 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn2_os)

        b3_rn3_os = self.b3_rn3.out_shape

        # output block
        self.avgpool = GlobalAveragePooling2DExt(input_shape=b3_rn3_os)

        avgpool_os = self.avgpool.out_shape


        self.fc = MaskedDense(input_dim=avgpool_os[-1],
                              units=num_classes)


    @tf.function
    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x)
        x = self.b1_rn2(x)
        x = self.b1_rn3(x)

        x = self.b2_rn1(x)
        x = self.b2_rn2(x)
        x = self.b2_rn3(x)

        x = self.b3_rn1(x)
        x = self.b3_rn2(x)
        x = self.b3_rn3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)


class ResNet20(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet20, self).__init__()

        filters = [16, 16, 32, 64]
        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = Conv2DExt(filters=filters[0],
                               kernel_size=first_kernel_size,
                               strides=first_stride,
                               padding="same")


        self.bn1 = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        self.pool1 = MaxPool2DExt(pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC(filters=filters[1],
                                        strides=(1,1))

        self.b1_rn2 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn3 = BasicResnetBlockI(filters=filters[1])


        #Block 2
        self.b2_rn1 = BasicResnetBlockC(filters=filters[2],
                                        strides=resnetc_strides)

        self.b2_rn2 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn3 = BasicResnetBlockI(filters=filters[2])


        #Block 3
        self.b3_rn1 = BasicResnetBlockC(filters=filters[3],
                                        strides=resnetc_strides)

        self.b3_rn2 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn3 = BasicResnetBlockI(filters=filters[3])


        self.avgpool = GlobalAveragePooling2DExt()

        self.fc = DenseExt(units=num_classes)


    @tf.function
    def call(self, inputs, training=False):

        # x = tf.cast(inputs, tf.float32)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x, training=training)
        x = self.b1_rn2(x, training=training)
        x = self.b1_rn3(x, training=training)


        x = self.b2_rn1(x, training=training)
        x = self.b2_rn2(x, training=training)
        x = self.b2_rn3(x, training=training)


        x = self.b3_rn1(x, training=training)
        x = self.b3_rn2(x, training=training)
        x = self.b3_rn3(x, training=training)


        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)



class ResNet56_Mask(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet56_Mask, self).__init__()

        filters = [16, 16, 32, 64]

        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = MaskedConv2D(filters=filters[0],
                                  input_shape=input_shape,
                                  kernel_size=first_kernel_size,
                                  strides=first_stride,
                                  padding="same")

        conv1_os = self.conv1.out_shape

        self.bn1 = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        self.pool1 = MaxPool2DExt(input_shape=conv1_os,
                                  pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC_Mask(filters=filters[1],
                                        strides=(1,1), #resnetc_strides,
                                        input_shape=conv1_os)

        b1_rn1_os = self.b1_rn1.out_shape

        self.b1_rn2 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn1_os)

        b1_rn2_os = self.b1_rn2.out_shape

        self.b1_rn3 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn2_os)

        b1_rn3_os = self.b1_rn3.out_shape

        self.b1_rn4 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn3_os)

        b1_rn4_os = self.b1_rn4.out_shape

        self.b1_rn5 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn4_os)

        b1_rn5_os = self.b1_rn5.out_shape

        self.b1_rn6 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn5_os)

        b1_rn6_os = self.b1_rn6.out_shape

        self.b1_rn7 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn6_os)

        b1_rn7_os = self.b1_rn7.out_shape

        self.b1_rn8 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn7_os)

        b1_rn8_os = self.b1_rn8.out_shape

        self.b1_rn9 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn8_os)

        b1_rn9_os = self.b1_rn9.out_shape


        #Block 2
        self.b2_rn1 = BasicResnetBlockC_Mask(filters=filters[2],
                                        strides=resnetc_strides,
                                        input_shape=b1_rn9_os)

        b2_rn1_os = self.b2_rn1.out_shape

        self.b2_rn2 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn1_os)

        b2_rn2_os = self.b2_rn2.out_shape

        self.b2_rn3 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn2_os)

        b2_rn3_os = self.b2_rn3.out_shape

        self.b2_rn4 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn3_os)

        b2_rn4_os = self.b2_rn4.out_shape

        self.b2_rn5 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn4_os)

        b2_rn5_os = self.b2_rn5.out_shape

        self.b2_rn6 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn5_os)

        b2_rn6_os = self.b2_rn6.out_shape

        self.b2_rn7 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn6_os)

        b2_rn7_os = self.b2_rn7.out_shape

        self.b2_rn8 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn7_os)

        b2_rn8_os = self.b2_rn8.out_shape

        self.b2_rn9 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn8_os)

        b2_rn9_os = self.b2_rn9.out_shape

        #Block 3
        self.b3_rn1 = BasicResnetBlockC_Mask(filters=filters[3],
                                        strides=resnetc_strides,
                                        input_shape=b2_rn9_os)

        b3_rn1_os = self.b3_rn1.out_shape

        self.b3_rn2 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn1_os)

        b3_rn2_os = self.b3_rn2.out_shape

        self.b3_rn3 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn2_os)

        b3_rn3_os = self.b3_rn3.out_shape

        self.b3_rn4 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn3_os)

        b3_rn4_os = self.b3_rn4.out_shape

        self.b3_rn5 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn4_os)

        b3_rn5_os = self.b3_rn5.out_shape

        self.b3_rn6 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn5_os)

        b3_rn6_os = self.b3_rn6.out_shape

        self.b3_rn7 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn6_os)

        b3_rn7_os = self.b3_rn7.out_shape

        self.b3_rn8 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn7_os)

        b3_rn8_os = self.b3_rn8.out_shape

        self.b3_rn9 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn8_os)

        b3_rn9_os = self.b3_rn9.out_shape

        # output block
        self.avgpool = GlobalAveragePooling2DExt(input_shape=b3_rn9_os)

        avgpool_os = self.avgpool.out_shape


        self.fc = MaskedDense(input_dim=avgpool_os[-1],
                              units=num_classes)


    @tf.function
    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x)
        x = self.b1_rn2(x)
        x = self.b1_rn3(x)
        x = self.b1_rn4(x)
        x = self.b1_rn5(x)
        x = self.b1_rn6(x)
        x = self.b1_rn7(x)
        x = self.b1_rn8(x)
        x = self.b1_rn9(x)

        x = self.b2_rn1(x)
        x = self.b2_rn2(x)
        x = self.b2_rn3(x)
        x = self.b2_rn4(x)
        x = self.b2_rn5(x)
        x = self.b2_rn6(x)
        x = self.b2_rn7(x)
        x = self.b2_rn8(x)
        x = self.b2_rn9(x)

        x = self.b3_rn1(x)
        x = self.b3_rn2(x)
        x = self.b3_rn3(x)
        x = self.b3_rn4(x)
        x = self.b3_rn5(x)
        x = self.b3_rn6(x)
        x = self.b3_rn7(x)
        x = self.b3_rn8(x)
        x = self.b3_rn9(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)


class ResNet56(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet56, self).__init__()

        filters = [16, 16, 32, 64]

        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = Conv2DExt(filters=filters[0],
                               kernel_size=first_kernel_size,
                               strides=first_stride,
                               padding="same")

        conv1_os = self.conv1.out_shape

        self.bn1 = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        self.pool1 = MaxPool2DExt(input_shape=conv1_os,
                                  pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC(filters=filters[1],
                                        strides=(1,1))

        self.b1_rn2 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn3 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn4 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn5 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn6 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn7 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn8 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn9 = BasicResnetBlockI(filters=filters[1])


        #Block 2
        self.b2_rn1 = BasicResnetBlockC(filters=filters[2],
                                        strides=resnetc_strides)

        self.b2_rn2 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn3 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn4 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn5 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn6 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn7 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn8 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn9 = BasicResnetBlockI(filters=filters[2])


        #Block 3
        self.b3_rn1 = BasicResnetBlockC(filters=filters[3],
                                        strides=resnetc_strides)

        self.b3_rn2 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn3 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn4 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn5 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn6 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn7 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn8 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn9 = BasicResnetBlockI(filters=filters[3])


        # output block
        self.avgpool = GlobalAveragePooling2DExt()
        self.fc = DenseExt(units=num_classes)


    @tf.function
    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x, training=training)
        x = self.b1_rn2(x, training=training)
        x = self.b1_rn3(x, training=training)
        x = self.b1_rn4(x, training=training)
        x = self.b1_rn5(x, training=training)
        x = self.b1_rn6(x, training=training)
        x = self.b1_rn7(x, training=training)
        x = self.b1_rn8(x, training=training)
        x = self.b1_rn9(x, training=training)

        x = self.b2_rn1(x, training=training)
        x = self.b2_rn2(x, training=training)
        x = self.b2_rn3(x, training=training)
        x = self.b2_rn4(x, training=training)
        x = self.b2_rn5(x, training=training)
        x = self.b2_rn6(x, training=training)
        x = self.b2_rn7(x, training=training)
        x = self.b2_rn8(x, training=training)
        x = self.b2_rn9(x, training=training)

        x = self.b3_rn1(x, training=training)
        x = self.b3_rn2(x, training=training)
        x = self.b3_rn3(x, training=training)
        x = self.b3_rn4(x, training=training)
        x = self.b3_rn5(x, training=training)
        x = self.b3_rn6(x, training=training)
        x = self.b3_rn7(x, training=training)
        x = self.b3_rn8(x, training=training)
        x = self.b3_rn9(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)


class ResNet110_Mask(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet110_Mask, self).__init__()

        filters = [16, 16, 32, 64]

        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = MaskedConv2D(filters=filters[0],
                                  input_shape=input_shape,
                                  kernel_size=first_kernel_size,
                                  strides=first_stride,
                                  padding="same")

        conv1_os = self.conv1.out_shape

        self.bn1 = BatchNormExt(center=False,
                                scale=False,
                                trainable=False)

        self.pool1 = MaxPool2DExt(input_shape=conv1_os,
                                  pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC_Mask(filters=filters[1],
                                        strides=(1,1), #resnetc_strides,
                                        input_shape=conv1_os)

        b1_rn1_os = self.b1_rn1.out_shape

        self.b1_rn2 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn1_os)

        b1_rn2_os = self.b1_rn2.out_shape

        self.b1_rn3 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn2_os)

        b1_rn3_os = self.b1_rn3.out_shape

        self.b1_rn4 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn3_os)

        b1_rn4_os = self.b1_rn4.out_shape

        self.b1_rn5 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn4_os)

        b1_rn5_os = self.b1_rn5.out_shape

        self.b1_rn6 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn5_os)

        b1_rn6_os = self.b1_rn6.out_shape

        self.b1_rn7 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn6_os)

        b1_rn7_os = self.b1_rn7.out_shape

        self.b1_rn8 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn7_os)

        b1_rn8_os = self.b1_rn8.out_shape

        self.b1_rn9 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn8_os)

        b1_rn9_os = self.b1_rn9.out_shape


        self.b1_rn10 = BasicResnetBlockC_Mask(filters=filters[1],
                                        strides=(1,1), #resnetc_strides,
                                        input_shape=b1_rn9_os)

        b1_rn10_os = self.b1_rn10.out_shape

        self.b1_rn11 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn10_os)

        b1_rn11_os = self.b1_rn11.out_shape

        self.b1_rn12 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn11_os)

        b1_rn12_os = self.b1_rn12.out_shape

        self.b1_rn13 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn12_os)

        b1_rn13_os = self.b1_rn13.out_shape

        self.b1_rn14 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn13_os)

        b1_rn14_os = self.b1_rn14.out_shape

        self.b1_rn15 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn14_os)

        b1_rn15_os = self.b1_rn15.out_shape

        self.b1_rn16 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn15_os)

        b1_rn16_os = self.b1_rn16.out_shape

        self.b1_rn17 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn16_os)

        b1_rn17_os = self.b1_rn17.out_shape

        self.b1_rn18 = BasicResnetBlockI_Mask(filters=filters[1],
                                        input_shape=b1_rn17_os)

        b1_rn18_os = self.b1_rn18.out_shape


        #Block 2
        self.b2_rn1 = BasicResnetBlockC_Mask(filters=filters[2],
                                        strides=resnetc_strides,
                                        input_shape=b1_rn18_os)

        b2_rn1_os = self.b2_rn1.out_shape

        self.b2_rn2 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn1_os)

        b2_rn2_os = self.b2_rn2.out_shape

        self.b2_rn3 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn2_os)

        b2_rn3_os = self.b2_rn3.out_shape

        self.b2_rn4 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn3_os)

        b2_rn4_os = self.b2_rn4.out_shape

        self.b2_rn5 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn4_os)

        b2_rn5_os = self.b2_rn5.out_shape

        self.b2_rn6 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn5_os)

        b2_rn6_os = self.b2_rn6.out_shape

        self.b2_rn7 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn6_os)

        b2_rn7_os = self.b2_rn7.out_shape

        self.b2_rn8 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn7_os)

        b2_rn8_os = self.b2_rn8.out_shape

        self.b2_rn9 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn8_os)

        b2_rn9_os = self.b2_rn9.out_shape

        self.b2_rn10 = BasicResnetBlockC_Mask(filters=filters[2],
                                        strides=resnetc_strides,
                                        input_shape=b2_rn9_os)

        b2_rn10_os = self.b2_rn10.out_shape

        self.b2_rn11 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn10_os)

        b2_rn11_os = self.b2_rn11.out_shape

        self.b2_rn12 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn11_os)

        b2_rn12_os = self.b2_rn12.out_shape

        self.b2_rn13 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn12_os)

        b2_rn13_os = self.b2_rn13.out_shape

        self.b2_rn14 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn13_os)

        b2_rn14_os = self.b2_rn14.out_shape

        self.b2_rn15 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn14_os)

        b2_rn15_os = self.b2_rn15.out_shape

        self.b2_rn16 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn15_os)

        b2_rn16_os = self.b2_rn16.out_shape

        self.b2_rn17 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn16_os)

        b2_rn17_os = self.b2_rn17.out_shape

        self.b2_rn18 = BasicResnetBlockI_Mask(filters=filters[2],
                                        input_shape=b2_rn17_os)

        b2_rn18_os = self.b2_rn18.out_shape



        #Block 3
        self.b3_rn1 = BasicResnetBlockC_Mask(filters=filters[3],
                                        strides=resnetc_strides,
                                        input_shape=b2_rn18_os)

        b3_rn1_os = self.b3_rn1.out_shape

        self.b3_rn2 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn1_os)

        b3_rn2_os = self.b3_rn2.out_shape

        self.b3_rn3 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn2_os)

        b3_rn3_os = self.b3_rn3.out_shape

        self.b3_rn4 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn3_os)

        b3_rn4_os = self.b3_rn4.out_shape

        self.b3_rn5 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn4_os)

        b3_rn5_os = self.b3_rn5.out_shape

        self.b3_rn6 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn5_os)

        b3_rn6_os = self.b3_rn6.out_shape

        self.b3_rn7 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn6_os)

        b3_rn7_os = self.b3_rn7.out_shape

        self.b3_rn8 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn7_os)

        b3_rn8_os = self.b3_rn8.out_shape

        self.b3_rn9 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn8_os)

        b3_rn9_os = self.b3_rn9.out_shape



        self.b3_rn10 = BasicResnetBlockC_Mask(filters=filters[3],
                                        strides=resnetc_strides,
                                        input_shape=b3_rn9_os)

        b3_rn10_os = self.b3_rn10.out_shape

        self.b3_rn11 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn10_os)

        b3_rn11_os = self.b3_rn11.out_shape

        self.b3_rn12 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn11_os)

        b3_rn12_os = self.b3_rn12.out_shape

        self.b3_rn13 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn12_os)

        b3_rn13_os = self.b3_rn13.out_shape

        self.b3_rn14 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn13_os)

        b3_rn14_os = self.b3_rn14.out_shape

        self.b3_rn15 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn14_os)

        b3_rn15_os = self.b3_rn15.out_shape

        self.b3_rn16 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn15_os)

        b3_rn16_os = self.b3_rn16.out_shape

        self.b3_rn17 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn16_os)

        b3_rn17_os = self.b3_rn17.out_shape

        self.b3_rn18 = BasicResnetBlockI_Mask(filters=filters[3],
                                        input_shape=b3_rn17_os)

        b3_rn18_os = self.b3_rn18.out_shape


        # output block
        self.avgpool = GlobalAveragePooling2DExt(input_shape=b3_rn18_os)

        avgpool_os = self.avgpool.out_shape


        self.fc = MaskedDense(input_dim=avgpool_os[-1],
                              units=num_classes)


    @tf.function
    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x)
        x = self.b1_rn2(x)
        x = self.b1_rn3(x)
        x = self.b1_rn4(x)
        x = self.b1_rn5(x)
        x = self.b1_rn6(x)
        x = self.b1_rn7(x)
        x = self.b1_rn8(x)
        x = self.b1_rn9(x)
        x = self.b1_rn10(x)
        x = self.b1_rn11(x)
        x = self.b1_rn12(x)
        x = self.b1_rn13(x)
        x = self.b1_rn14(x)
        x = self.b1_rn15(x)
        x = self.b1_rn16(x)
        x = self.b1_rn17(x)
        x = self.b1_rn18(x)

        x = self.b2_rn1(x)
        x = self.b2_rn2(x)
        x = self.b2_rn3(x)
        x = self.b2_rn4(x)
        x = self.b2_rn5(x)
        x = self.b2_rn6(x)
        x = self.b2_rn7(x)
        x = self.b2_rn8(x)
        x = self.b2_rn9(x)
        x = self.b2_rn10(x)
        x = self.b2_rn11(x)
        x = self.b2_rn12(x)
        x = self.b2_rn13(x)
        x = self.b2_rn14(x)
        x = self.b2_rn15(x)
        x = self.b2_rn16(x)
        x = self.b2_rn17(x)
        x = self.b2_rn18(x)

        x = self.b3_rn1(x)
        x = self.b3_rn2(x)
        x = self.b3_rn3(x)
        x = self.b3_rn4(x)
        x = self.b3_rn5(x)
        x = self.b3_rn6(x)
        x = self.b3_rn7(x)
        x = self.b3_rn8(x)
        x = self.b3_rn9(x)
        x = self.b3_rn10(x)
        x = self.b3_rn11(x)
        x = self.b3_rn12(x)
        x = self.b3_rn13(x)
        x = self.b3_rn14(x)
        x = self.b3_rn15(x)
        x = self.b3_rn16(x)
        x = self.b3_rn17(x)
        x = self.b3_rn18(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)



class ResNet110(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 first_kernel_size=3,
                 filter_size_multi = 1,
                 first_stride=(1,1)) -> None:
        super(ResNet110, self).__init__()

        filters = [16, 16, 32, 64]

        filters = [int(f * filter_size_multi) for f in filters]

        self.conv1 = Conv2DExt(filters=filters[0],
                               kernel_size=first_kernel_size,
                               strides=first_stride,
                               padding="same")

        conv1_os = self.conv1.out_shape

        self.bn1 = BatchNormExt(center=True,
                                scale=True,
                                trainable=True)

        self.pool1 = MaxPool2DExt(input_shape=conv1_os,
                                  pool_size=(3, 3),
                                  strides=first_stride,
                                  padding="same")

        resnetc_strides = (2,2)

        #Block 1
        self.b1_rn1 = BasicResnetBlockC(filters=filters[1],
                                        strides=(1,1))

        self.b1_rn2 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn3 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn4 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn5 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn6 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn7 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn8 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn9 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn10 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn11 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn12 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn13 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn14 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn15 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn16 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn17 = BasicResnetBlockI(filters=filters[1])

        self.b1_rn18 = BasicResnetBlockI(filters=filters[1])



        #Block 2
        self.b2_rn1 = BasicResnetBlockC(filters=filters[2],
                                        strides=resnetc_strides)

        self.b2_rn2 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn3 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn4 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn5 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn6 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn7 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn8 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn9 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn10 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn11 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn12 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn13 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn14 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn15 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn16 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn17 = BasicResnetBlockI(filters=filters[2])

        self.b2_rn18 = BasicResnetBlockI(filters=filters[2])




        #Block 3
        self.b3_rn1 = BasicResnetBlockC(filters=filters[3],
                                        strides=resnetc_strides)

        self.b3_rn2 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn3 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn4 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn5 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn6 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn7 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn8 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn9 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn10 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn11 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn12 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn13 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn14 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn15 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn16 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn17 = BasicResnetBlockI(filters=filters[3])

        self.b3_rn18 = BasicResnetBlockI(filters=filters[3])


        # output block
        self.avgpool = GlobalAveragePooling2DExt()
        self.fc = DenseExt(units=num_classes)


    @tf.function
    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.elu(x)
        x = self.pool1(x)

        x = self.b1_rn1(x, training=training)
        x = self.b1_rn2(x, training=training)
        x = self.b1_rn3(x, training=training)
        x = self.b1_rn4(x, training=training)
        x = self.b1_rn5(x, training=training)
        x = self.b1_rn6(x, training=training)
        x = self.b1_rn7(x, training=training)
        x = self.b1_rn8(x, training=training)
        x = self.b1_rn9(x, training=training)
        x = self.b1_rn10(x, training=training)
        x = self.b1_rn11(x, training=training)
        x = self.b1_rn12(x, training=training)
        x = self.b1_rn13(x, training=training)
        x = self.b1_rn14(x, training=training)
        x = self.b1_rn15(x, training=training)
        x = self.b1_rn16(x, training=training)
        x = self.b1_rn17(x, training=training)
        x = self.b1_rn18(x, training=training)

        x = self.b2_rn1(x, training=training)
        x = self.b2_rn2(x, training=training)
        x = self.b2_rn3(x, training=training)
        x = self.b2_rn4(x, training=training)
        x = self.b2_rn5(x, training=training)
        x = self.b2_rn6(x, training=training)
        x = self.b2_rn7(x, training=training)
        x = self.b2_rn8(x, training=training)
        x = self.b2_rn9(x, training=training)
        x = self.b2_rn10(x, training=training)
        x = self.b2_rn11(x, training=training)
        x = self.b2_rn12(x, training=training)
        x = self.b2_rn13(x, training=training)
        x = self.b2_rn14(x, training=training)
        x = self.b2_rn15(x, training=training)
        x = self.b2_rn16(x, training=training)
        x = self.b2_rn17(x, training=training)
        x = self.b2_rn18(x, training=training)

        x = self.b3_rn1(x, training=training)
        x = self.b3_rn2(x, training=training)
        x = self.b3_rn3(x, training=training)
        x = self.b3_rn4(x, training=training)
        x = self.b3_rn5(x, training=training)
        x = self.b3_rn6(x, training=training)
        x = self.b3_rn7(x, training=training)
        x = self.b3_rn8(x, training=training)
        x = self.b3_rn9(x, training=training)
        x = self.b3_rn10(x, training=training)
        x = self.b3_rn11(x, training=training)
        x = self.b3_rn12(x, training=training)
        x = self.b3_rn13(x, training=training)
        x = self.b3_rn14(x, training=training)
        x = self.b3_rn15(x, training=training)
        x = self.b3_rn16(x, training=training)
        x = self.b3_rn17(x, training=training)
        x = self.b3_rn18(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return tf.nn.softmax(x)
