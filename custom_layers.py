import tensorflow as tf
import functools
#import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D

# Set Seeds
seed = 7531
np.random.seed(seed)
tf.random.set_seed(seed)


class MaxPool2DExt(tf.keras.layers.MaxPool2D):
    """Extends tf.keras.MaxPool2D class with a type variable which is used in the initialization phase.
    Furthermore, we add a variable which contains the output shape of the layer.
    """
    def __init__(self, 
                 input_shape=None, 
                 pool_size=(2,2), 
                 strides=None, 
                 padding="same", 
                 data_format="channels_last"):
        super(MaxPool2DExt, self).__init__(pool_size=pool_size, 
                                           strides=strides, 
                                           padding=padding, 
                                           data_format=data_format)
        
        self.type="mapo"
        
        if strides is None:
            strides = pool_size
        
        if input_shape is not None:
            if padding == "valid":
                new_rows = int(np.ceil((input_shape[1] - pool_size[0] + 1) / strides[0]))
                new_cols = int(np.ceil((input_shape[2] - pool_size[1] + 1) / strides[1]))

                self.out_shape = (input_shape[0], new_rows, new_cols, input_shape[-1])
            
            if padding == "same":
                new_rows = int(np.ceil(input_shape[1] / strides[0]))
                new_cols = int(np.ceil(input_shape[2] / strides[1]))

                self.out_shape = (input_shape[0], new_rows, new_cols, input_shape[-1])

class BatchNormExt(tf.keras.layers.BatchNormalization):
    """Extends tf.keras.BatchNormalization class with a type variable which is used in the initialization phase
    Not utilized
    """
    def __init__(self, trainable):
        super(BatchNormExt, self).__init__(trainable=trainable)
        self.type="batchnorm"
class FlattenExt(tf.keras.layers.Flatten):
    """Extends tf.keras.Flatten class with a type variable which is used in the initialization phase"""
    def __init__(self):
        super(FlattenExt, self).__init__()
        self.type = "flat"

class Conv2DExt(tf.keras.layers.Conv2D):
    """Extends tf.keras.Conv2D class with a type variable which is used in the initialization phase"""
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 use_bias, 
                 strides=(1, 1), 
                 padding='same', 
                 data_format="channels_last"):
        super(Conv2DExt, self).__init__(filters=filters, 
                                        kernel_size=kernel_size, 
                                        use_bias=use_bias, 
                                        strides=strides, 
                                        padding=padding, 
                                        data_format=data_format)
        self.type = "conv_normal"

class GlobalAveragePooling2DExt(tf.keras.layers.GlobalAveragePooling2D):
    """Extends tf.keras GlobalAveragePooling2D class with a type variable which is used in the initialization phase"""
    def __init__(self):
        super(GlobalAveragePooling2DExt, self).__init__()
        self.type = "globalavgpooling"

class DenseExt(tf.keras.layers.Dense):
    """Extends tf.keras.Dense class with a type variable which is used in the initialization phase"""
    def __init__(self, units, use_bias=True):
        super(DenseExt, self).__init__(units, use_bias=use_bias)
        self.type = "fefo_normal"

class MaskedDense(layers.Layer):
    """Extends the Dense Layer of tf.keras with masking functionality
    """

    def __init__(self,
                 input_dim: int,
                 units: int, 
                 masking_method: str,
                 name=None, 
                 dynamic_scaling=True, 
                 k=0.5,
                 tanh_th=0.01):  

        super(MaskedDense,self).__init__()

        self.out_shape = (None, units)

        self.layer_name = name

        self.dynamic_scaling = dynamic_scaling

        self.type = "fefo"
        
        self.units = units
        self.input_dim = input_dim
        
        self.shape = (input_dim, units)

        self.size = input_dim*units
        
        init_mask = tf.ones_initializer()
        init_w = tf.random_normal_initializer()
        
        self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True,
                                name="mask")

        # self.threshold = 0.5 
        
        # self.multiplier = 1.
        # # self.no_ones = 0.
        
        
        self.w = tf.Variable(init_w(shape=(input_dim, units),dtype='float32'), trainable=False, name="w")

        self.k = k
        self.k_idx =  tf.cast(tf.cast(tf.reshape(self.mask, [-1]).get_shape()[0], tf.float32)*k, tf.int32)
        self.tanh_th = tanh_th

        self.masking_method = masking_method
        # self.masking = self.signed_supermask if masking_method is "fixed" else self.signed_supermask_score

        # print("Masking Method: ", self.masking_method)


    def update_tanh_th(self, percentage=0.75):
        """Updates the threshold for the mask step function. In case of a fixed threshold masking, this function is not used during training

        Args:
            percentage (float, optional): percentage value of maximum weight. Defaults to 0.75.
        """
        tanh_mask = self.mask_activation() 
        mask_max = tf.math.reduce_max(tf.math.abs(tanh_mask))
        
        self.tanh_th = mask_max * percentage
    
    # def set_mask_rand(self):
    #     self.mask = tf.constant(np.random.randint(2, size=(self.input_dim, self.units)).astype("float32"))
    
    def get_shape(self):
        """Returns the layer shape (i.e. shape of the weight/mask matrix)
        """
        return self.shape
    
    def set_mask(self,mask):
        """Set the mask to given values

        Args:
            mask (np.ndarray): new mask values
        """
        self.mask = tf.Variable(tf.cast(mask, "float32"), trainable=True, name="mask")
        self.trainable_weights.append(self.mask) 
    
    def get_mask(self, as_logit=False):
        """ONLY USED WITH BINARY MASKING - NOT IN USE FOR SIGNED SUPERMASKS
        Returns the (logit) mask of the layer

        Args:
            as_logit (bool, optional): True if return mask values as "logit" i.e. the real valued mask values. Defaults to False.

        Returns:
            tf.Variable: mask 
        """
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
    
    def get_bernoulli_mask(self):
        """Returns the effective mask

        Returns:
            tf.Variable: effective mask
        """
        return self.bernoulli_mask  
        
    
    def binary_supermask(self):
        """Generates a binary Supermask out of self.mask

        Returns:
            tf.Variable: effective binary Supermask. Note the straight through estimtator trick
        """
        sigmoid_mask = self.mask #self.mask_activation()

        # bernoulli_sample = tf.random.uniform(shape = self.shape, minval=0., maxval=1.)
        # effective_mask = tf.where(sigmoid_mask > bernoulli_sample, 1., 0.)

        effective_mask = tf.where(sigmoid_mask > self.tanh_th, 1., 0.)
        
        self.bernoulli_mask = effective_mask
        
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) 
    
    def mask_activation(self):
        """Returns the real valued Supermask. In case you want to use some activation function for the mask, this function
        provides some flexibility to that.

        Returns:
            tf.Variable: real-valued mask
        """
        return self.mask
        
        
    def signed_supermask(self):
        """Calculates the signed Supermask (fixed threshold) with a simple step function

        Returns:
            tf.Variable: effective signed Supermask. Note the trick necessary to get the straight through estimator
        """
        
        tanh_mask = self.mask 

        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        self.bernoulli_mask = effective_mask

        #print(self.tanh_th)

        return  tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) 
    
    def signed_supermask_score(self):
        """Calculates the signed Supermask (variable threshold, i.e. in the fashion of Ramarunjan et al).
        Not further investigated in the paper. 

        Returns:
            tf.Variable: effective signed Supermask with variable threshold. 
        """
        
        tanh_mask = self.mask_activation() 

        first_k = min(5000, self.size)
        
        tanh_mask_reshaped = tf.gather(tf.random.shuffle(tf.reshape(tanh_mask, [-1])), np.arange(0,5000))

        sample_k_idx = int(self.k*first_k) + 1
        
        pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, sample_k_idx, sorted=True).values)
        neg_threshold = tf.math.reduce_max(-tf.math.top_k(-tanh_mask_reshaped, sample_k_idx, sorted=True).values)

        effective_mask = tf.where(tanh_mask < neg_threshold, -1., 0.)
        effective_mask = tf.where(tanh_mask > pos_threshold, 1., effective_mask)

        self.bernoulli_mask = effective_mask
        
        return tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) 
    
    def score_mask(self):
        """Calculates the binary Supermask in the fashion of Ramarunjan et al - not used

        Returns:
            tf.Variable: effective binary score Supermask
        """
        sigmoid_mask = tf.math.sigmoid(self.mask)

        threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(sigmoid_mask,[-1]), self.k_idx, sorted=False).values)

        effective_mask = tf.where(sigmoid_mask >= threshold, 1.0, 0.0)

        self.bernoulli_mask = effective_mask

        return tf.stop_gradient(effective_mask) + self.mask - tf.stop_gradient(self.mask) 
    
    def get_normal_weights(self):
        """Returns the weights of the layer"""

        return self.w
    
    def set_normal_weights(self, w):
        """Sets the weights of the layer"""
        self.w = tf.Variable(w.astype("float32"), trainable=False, name="w")
    
    # def reset_mask(self):
    #     self.mask = tf.Variable(np.ones((self.input_dim,self.units), dtype="float32"))
        
    def get_pruned_weights(self):
        """Returns those weight elements that are masked (only used with binary Supermask)

        Returns:
            tf.Variable: reverse-masked weights
        """
        flipped_mask = tf.cast(tf.not_equal(self.bernoulli_mask, 1), tf.float32)
        return tf.multiply(self.w, flipped_mask)
    
    def get_masked_weights(self):
        """Get effective weight matrix

        Returns:
            tf.Variable: effective weight matrix
        """
        return tf.multiply(self.w, self.bernoulli_mask)
    
    def get_nonzero_weights(self):
        """Returns those weights that are not affected by pruning (only used with binary Supermask)

        Returns:
            tf.Variable: weights not affected by pruning
        """
        weights_masked = tf.boolean_mask(self.w, self.bernoulli_mask) 
        return weights_masked 
    
    @tf.function
    def call(self, inputs):
        """Extends the call function of a normal layer by applying the (signed) Supermask before calculating the output

        Args:
            inputs (tf.Variable): input to the layer

        Returns:
            tf.Variable: output of the layer
        """
        inputs = tf.cast(inputs, tf.float32)
        
        #if self.masking_method == "fixed":
        if self.masking_method == "fixed":
            effective_mask = self.signed_supermask()
        elif self.masking_method == "binary":
            effective_mask = self.binary_supermask()
        #else:
        #    sig_mask = self.signed_supermask_score()
        weights_masked = tf.multiply(self.w,effective_mask)
        # if self.dynamic_scaling is True:
            # self.no_ones = tf.reduce_sum(weights_masked)
            # self.multiplier =  tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones) #* (1./self.sigmoid_multiplier)
            # self.multiplier = tf.math.divide(weights_masked - tf.math.reduce_mean(weights_masked), tf.math.reduce_std(weights_masked))
            #print("Multiplier in ", self.name, ": ", self.multiplier) 
            # weights_masked = tf.math.divide(weights_masked , tf.math.reduce_sum(weights_masked))#tf.multiply(tf.stop_gradient( self.multiplier ), weights_masked)

        return tf.matmul(inputs, weights_masked)
    

class MaskedConv2D(tf.keras.layers.Conv2D):

    def __init__(self, 
                 filters: int, 
                 kernel_size: int, 
                 input_shape: int, 
                 masking_method: str, 
                 dynamic_scaling=True, 
                 k=0.5, 
                #  tanh_th=0.01, 
                 padding="same", 
                 strides=(1,1), *args, **kwargs):
        
        super(MaskedConv2D, self).__init__(filters, kernel_size, padding=padding, strides=strides, use_bias=False, *args, **kwargs)
        
        self._uses_learning_phase = True
        self.filters = filters


        self.type = "conv"
        
        self.weight_shape = (kernel_size, kernel_size, input_shape[-1], filters)
        self.size = kernel_size * kernel_size * input_shape[-1] * filters
        
        init_mask = tf.ones_initializer()
        self.mask = tf.Variable(initial_value=init_mask(shape=self.weight_shape, dtype="float32"),name="mask", trainable=True)

        self.total_mask_params= tf.cast(tf.size(self.mask), tf.float32)
        
        # self.threshold = tf.constant(0.5, dtype="float32")
        
        self.bernoulli_mask = self.mask
        
        if padding == "valid": 
            new_rows = int(np.ceil((input_shape[1]-kernel_size+1)/strides[0]))
            new_cols = int(np.ceil((input_shape[2]-kernel_size+1)/strides[1]))
        elif padding == "same":
            new_rows = int(np.ceil(input_shape[1] / strides[0]))    
            new_cols = int(np.ceil(input_shape[2] / strides[1]))
        self.out_shape = (input_shape[0],new_rows,new_cols ,filters)

        init_kernel = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = init_kernel(shape=self.weight_shape, dtype="float32"), name="weights", trainable=False)

        self.dynamic_scaling = dynamic_scaling

        # self.no_ones = 0.
       
        self.k = k 
        self.k_idx =  tf.cast(tf.cast(tf.reshape(self.mask, [-1]).get_shape()[0], tf.float32)*k, tf.int32)
        
        self.tanh_th = 1. #tanh_th
        
        self.masking_method = masking_method
        # self.masking = self.signed_supermask if masking_method is "fixed" else self.signed_supermask_score


    def update_tanh_th(self, percentage=0.75):
        """Updates the threshold for the mask step function. This function is only called once after initialization

        Args:
            percentage (float, optional): percentage value of maximum weight. Defaults to 0.75.
        """
        tanh_mask = self.mask_activation()
        mask_max = tf.math.reduce_max(tf.math.abs(tanh_mask))
        
        self.tanh_th = mask_max * percentage

    def get_output_shape(self):
        """Returns the output shape of the layer

        Returns:
            [Tuple]: output shape of the layer
        """
        return self.out_shape 

    def set_mask(self,mask):
        """Setter for the mask

        Args:
            mask (np.ndarray): mask values as array
        """
        self.mask = tf.Variable(tf.cast(mask, "float32"), name="mask")
        
    def get_mask(self, as_logit=False):
        """ONLY USED WITH BINARY MASKING - NOT IN USE FOR SIGNED SUPERMASKS
        Returns the (logit) mask of the layer

        Args:
            as_logit (bool, optional): True if return mask values as "logit" i.e. the real valued mask values. Defaults to False.

        Returns:
            tf.Variable: mask 
        """
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
    
    def get_bernoulli_mask(self):
        """Returns the effective mask

        Returns:
            tf.Variable: effective mask
        """
        return self.bernoulli_mask
    
    def get_normal_weights(self):
        """Returns the weights of the layer"""
        return self.w
    
    def set_normal_weights(self, w):
        """Sets the weights of the layer"""
        self.w = tf.Variable(w.astype("float32"), trainable=False, name="weights")
    
    # def reset_mask(self):
    #     self.mask = tf.Variable(np.ones((self.input_dim,self.units), dtype="float32"))
    
    def get_pruned_weights(self):
        """Returns those weight elements that are masked (only used with binary Supermask)

        Returns:
            tf.Variable: reverse-masked weights
        """
        flipped_mask = tf.cast(tf.not_equal(self.bernoulli_mask, 1), tf.float32)
        return tf.multiply(self.w, flipped_mask)
    
    def get_masked_weights(self):
        """Get effective weight matrix

        Returns:
            tf.Variable: effective weight matrix
        """
        return tf.multiply(self.w, self.bernoulli_mask)
    
    def get_nonzero_weights(self):
        """Returns those weights that are not affected by pruning (only used with binary Supermask)

        Returns:
            tf.Variable: weights not affected by pruning
        """
        weights_masked = tf.boolean_mask(self.w, self.bernoulli_mask) 
        return weights_masked 


    def build(self, input_shape):
        """Somehow the for-loop below is necessary in order to set the trainable and non-trainable weights of a keras
        Conv layer properly

        Args:
            input_shape (Tuple): input shape to layer
        """
        super(MaskedConv2D, self).build(input_shape)

        for i,var in enumerate( self._trainable_weights ):
            if "kernel" in var.name:
                del self._trainable_weights[i]
                del var

    def binary_supermask(self):
        """Generates a binary Supermask out of self.mask

        Returns:
            tf.Variable: effective binary Supermask. Note the straight through estimtator trick
        """
        sigmoid_mask = self.mask #self.mask_activation() 

        # bernoulli_sample = tf.random.uniform(shape = self.weight_shape, minval=0., maxval=1.)
        # effective_mask = tf.where(sigmoid_mask > bernoulli_sample, 1., 0.)

        effective_mask = tf.where(sigmoid_mask > self.tanh_th, 1., 0.)

        self.bernoulli_mask = effective_mask
    
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) 

    def mask_activation(self):
        """Returns the real valued Supermask. In case you want to use some activation function for the mask, this function
        provides some flexibility to that.

        Returns:
            tf.Variable: real-valued mask
        """
        return self.mask

    def signed_supermask(self):
        """Calculates the signed Supermask (fixed threshold) with a simple step function

        Returns:
            tf.Variable: effective signed Supermask. Note the trick necessary to get the straight through estimator
        """
        
        tanh_mask = self.mask

        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        return  tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask)
    
    def signed_supermask_score(self):
        """Calculates the signed Supermask (variable threshold, i.e. in the fashion of Ramarunjan et al).
        Not further investigated in the paper. 

        Returns:
            tf.Variable: effective signed Supermask with variable threshold. 
        """
        
        tanh_mask = self.mask_activation() 

        first_k = min(5000, self.size)
        tanh_mask_reshaped = tf.gather(tf.random.shuffle(tf.reshape(tanh_mask, [-1])), np.arange(0,5000))
        
        k_idx = int(self.k*first_k) + 1
         
        pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, k_idx, sorted=True).values)
        neg_threshold = tf.math.reduce_max(-tf.math.top_k(-tanh_mask_reshaped, k_idx, sorted=True).values)

        effective_mask = tf.where(tanh_mask < neg_threshold, -1., 0.)
        effective_mask = tf.where(tanh_mask > pos_threshold, 1., effective_mask)

        self.bernoulli_mask = effective_mask

        return tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) 
    
    def score_mask(self):
        """Calculates the binary Supermask in the fashion of Ramarunjan et al - not used

        Returns:
            tf.Variable: effective binary score Supermask
        """
        sigmoid_mask = tf.math.sigmoid(self.mask)

        threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(sigmoid_mask, [-1]), self.k_idx, sorted=False).values)
        
        effective_mask = tf.where(sigmoid_mask >= threshold, 1.0, 0.0)
        self.bernoulli_mask = effective_mask

        return tf.stop_gradient(effective_mask) + self.mask - tf.stop_gradient(self.mask)
        
    @tf.function
    def call(self, inputs):
        """Extends the call function of a normal layer by applying the (signed) Supermask before calculating the output

        Args:
            inputs (tf.Variable): input to the layer

        Returns:
            tf.Variable: output of the layer
        """

        inputs = tf.cast(inputs, tf.float32)

        if self.masking_method == "fixed":
            effective_mask = self.signed_supermask()
        elif self.masking_method == "binary":
            effective_mask = self.binary_supermask()
        #else:
        #    sig_mask = self.signed_supermask_score()
        
        weights_masked = tf.multiply(self.w, effective_mask)

        # if self.dynamic_scaling:
            # single_filter_size = tf.reduce_prod(sig_mask.shape[:-1])
            # reshaped_sig_mask = tf.reshape(sig_mask, (single_filter_size,sig_mask.shape[-1]))
            # print("reshaped sig mask size: ", reshaped_sig_mask.shape)
            # self.no_ones = tf.cast(tf.reduce_sum(reshaped_sig_mask, axis=-1), tf.float32)
            # self.no_ones = tf.cast(tf.reduce_sum(sig_mask), tf.float32)
            # print("No ones cnn multiplier: ", self.no_ones)
            # self.multiplier =  tf.math.divide(self.size, self.no_ones) #* (1./self.sigmoid_multiplier)
            #print("Multiplier in ",self.name," : ", self.multiplier)
            # weights_masked = tf.multiply(self.multiplier, weights_masked)
            # weights_masked = tf.math.divide(weights_masked - tf.math.reduce_mean(weights_masked), tf.math.reduce_std(weights_masked))#tf.multiply(tf.stop_gradient( self.multiplier ), weights_masked)
            # self.no_ones = tf.reduce_sum(sig_mask)
            # self.multiplier = (1/self.sigmoid_multiplier) * tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones)
            # weights_masked = tf.multiply(self.multiplier, weights_masked)


        return self._convolution_op(inputs, weights_masked)
        