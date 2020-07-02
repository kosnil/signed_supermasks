import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

def bernoulli_sampler(p):
    
    #print(f"p shape: {p}")
    
    u = np.random.rand(*p.shape)
    bernoulli = np.zeros(p.shape)
    
    bernoulli[p > u] = 1
    bernoulli[p < u] = 0
    
    return bernoulli

class Linear(layers.Layer):
    
    def __init__(self,input_dim=32,units=32, mask=False, mask_matrix=None):
        super(Linear,self).__init__()
        
        
        self.units = units
        self.input_dim = input_dim
        
        if mask:
            if mask_matrix == None:
                #select random mask
                self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
                self.bias_mask = tf.constant(np.random.randint(2, size=(units,)).astype("float32"))
            else:
                self.mask = tf.constant(mask_matrix.astype("float32"))
        else:
            self.mask = tf.constant(np.ones((input_dim,units), dtype="float32"))
        
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()
        
        self.w = tf.Variable(initial_value=init_w(shape=(input_dim, units),dtype='float32'),trainable=True)
        self.b = tf.Variable(initial_value=init_b(shape=(units,),dtype='float32'),trainable=True)
    
    def set_mask_rand(self):
        self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
    
    def set_mask(self,mask):
        self.mask = tf.constant(tf.cast(mask, "float32"))
    
    def get_mask(self):
        return self.mask
    
    def reset_mask(self):
        self.mask = tf.constant(np.ones((self.input_dim,self.units), dtype="float32"))
        
    def get_all_weights(self):
        return self.w
    
    def get_bias(self):
        return self.b
        
    def get_pruned_weights(self):
        flipped_mask = tf.cast(tf.not_equal(self.mask, 1), tf.float32)
        return tf.multiply(self.w, flipped_mask)
    
    def get_masked_weights(self):
        return tf.multiply(self.w, self.mask)
    
    def get_nonzero_weights(self):
        #weights_masked = tf.multiply(self.w, self.mask)
        weights_masked = tf.boolean_mask(self.w, self.mask) #tf.not_equal(weights_masked, 0)
        return weights_masked #[mask]

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        w_mask = tf.multiply(self.w, self.mask)
        return tf.matmul(inputs, w_mask) + self.b
    

class Linear_Mask(layers.Layer):
    
    def __init__(self,input_dim=32,units=32, sigmoid_multiplier=0.2, mask=False, mask_matrix=None, use_bernoulli_sampler=False, name=None):
        super(Linear_Mask,self).__init__()

        self.layer_name = name

        self.use_bernoulli_sampler = use_bernoulli_sampler

        self.type = "fefo"
        
        
        self.units = units
        self.input_dim = input_dim
        
        self.shape = (input_dim, units)
        #if mask:
        #    if mask_matrix == None:
        #        #select random mask
        #        self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
        #    else:
        #        self.mask = tf.constant(mask_matrix.astype("float32"))
        #else:
        #    self.mask = tf.constant(np.ones((input_dim,units), dtype="float32"))
        
        init_mask = tf.ones_initializer()
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()
        
        #self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True)
        self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True,
                                name="mask")
        #self.bias_mask = tf.Variable(initial_value=init_mask(shape=(units, ), dtype="float32"), trainable=True)
        self.threshold = 0.5 #tf.Variable(initial_value = 0.5, dtype=tf.float32, trainable=True)
        
        #self.multiplier_factor = tf.Variable(initial_value = [1.], trainable=True)
        
        
        self.bernoulli_mask = self.mask
        
        self.sigmoid_multiplier = sigmoid_multiplier
        
        self.multiplier = 1.
        self.no_ones = 0.
        
        #self.sigmoid01 = lambda x: 1 if tf.math.sigmoid(x) >= 0.5 else 0
        
        #self.sig_mask = tf.Variable(initial_value=self.sigmoid01(self.mask), trainable=False)
        
        self.w = tf.Variable(init_w(shape=(input_dim, units),dtype='float32'), trainable=False, name="w")
        self.b = tf.Variable(init_b(shape=(units,),dtype='float32'), trainable=False, name="b")
    

        # self._trainable_weights = []
        # self._trainable_weights.append(self.mask)


    def set_mask_rand(self):
        self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
    
    def get_shape(self):
        return self.shape
    
    def set_mask(self,mask):
        self.mask = tf.Variable(tf.cast(mask, "float32"), trainable=True, name="mask")
        # if self.mask in self.non_trainable_weights:
        #     print("true")
        self.trainable_weights.append(self.mask) 
        self.update_bernoulli_mask()
        #self.mask = tf.Variable(mask.astype("float32"))
    
    def get_mask(self, as_logit=False):
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
        
    def update_bernoulli_mask(self, mask=None):
        
        sigmoid_mask = tf.math.sigmoid(self.mask)
        effective_mask = tf.cast(bernoulli_sampler(sigmoid_mask.numpy()), tf.float32)
        
        self.bernoulli_mask = effective_mask
        
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        
        #no_samples = 1
        #accept_as_one_th = 1
        #relaxation = tf.cast(no_samples - no_samples*accept_as_one_th, tf.int32)
        #return tf.cast(tfp.distributions.RelaxedBernoulli(temperature=0.001, probs=tf.math.sigmoid(self.mask)).sample(), dtype=tf.float32)
        #sig_mask = tf.math.sigmoid(self.mask)
        #bernoulli_samples = tfp.distributions.Bernoulli(probs=sig_mask).sample(no_samples)
        #bernoulli_sample_mask = tf.math.floordiv(tf.math.add(tf.math.reduce_sum(bernoulli_samples,axis=0), relaxation), no_samples)  #(tf.math.reduce_sum(a, axis=0)+30) // 100
        #self.bernoulli_mask = tf.cast(bernoulli_sample_mask, dtype=tf.float32) + sig_mask - tf.stop_gradient(sig_mask)
        #return self.bernoulli_mask
        #bernoulli_mask_lambda_fn = lambda x: tf.cast(tfp.distributions.Bernoulli(probs=x).sample(sample_shape=x.shape), "float32")
        #bernoulli_mask = tf.map_fn(bernoulli_mask_lambda_fn, self.mask)
        #print(bernoulli_mask.numpy().shape)
        #bernoulli_mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.math.sigmoid(self.mask)).sample(sample_shape=1), "float32")
        #return tf.reshape(bernoulli_mask, self.mask.shape)
    
    def grothe_idea(self):
        sig_mask = tf.math.sigmoid(self.mask)
        
        self.bernoulli_mask = tf.cast(tfp.distributions.Bernoulli(probs=sig_mask).sample() , dtype=tf.float32)
        
        return self.bernoulli_mask + sig_mask - tf.stop_gradient(sig_mask)
    
    
    def sigmoid_mask(self, epoch=0):
        sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))
        effective_mask = tf.where(sigmoid_mask >= self.threshold, 1., sigmoid_mask)
        effective_mask = tf.where(effective_mask < self.threshold, 0., effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        #if epoch > 20: # and epoch % 20 == 0:
        #    effective_mask_flipped = tf.math.abs(effective_mask-1)
        #    #return tf.stop_gradient(mask_h * target) + mask * target

        #    leftover_sigmoid_mask = tf.stop_gradient(effective_mask_flipped * sigmoid_mask) + effective_mask*sigmoid_mask
        #else:
        #    leftover_sigmoid_mask = sigmoid_mask
        
        #sig_mask[sig_mask >= 0.5] = 1.
        #sig_mask[sig_mask < 0.5] = 0.
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) # + tf.nn.relu(self.mask) - tf.stop_gradient(tf.nn.relu(self.mask))
    
    def tanh_mask(self):
        
        tanh_mask = tf.math.tanh(tf.multiply(self.mask, 1.))
        
        effective_mask = tf.where(tanh_mask >= 0, 1, tanh_mask)
        effective_mask = tf.where(effective_mask < 0, 0, effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        return effective_mask + tanh_mask - tf.stop_gradient(tanh_mask)
    
    def relu_mask(self):
        
        relu_mask = tf.nn.relu(tf.multiply(self.mask, 1.))
        
        effective_mask = tf.where(relu_mask > 0, 1, relu_mask)
        effective_mask = tf.where(effective_mask <= 0, 0, effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        return effective_mask + relu_mask - tf.stop_gradient(relu_mask)
    
    def get_normal_weights(self):
        return self.w
    
    def set_normal_weights(self, w):
        self.w = tf.Variable(w.astype("float32"), trainable=False, name="w")
        # self.w = tf.constant(w.astype("float32"))
    
    def reset_mask(self):
        self.mask = tf.Variable(np.ones((self.input_dim,self.units), dtype="float32"))
        
    def get_all_weights(self):
        return self.w
    
    def get_bias(self):
        return self.b
        
    def get_pruned_weights(self):
        flipped_mask = tf.cast(tf.not_equal(self.bernoulli_mask, 1), tf.float32)
        return tf.multiply(self.w, flipped_mask)
    
    def get_masked_weights(self):
        return tf.multiply(self.w, self.bernoulli_mask)
    
    def get_nonzero_weights(self):
        #weights_masked = tf.multiply(self.w, self.mask)
        weights_masked = tf.boolean_mask(self.w, self.bernoulli_mask) #tf.not_equal(weights_masked, 0)
        return weights_masked #[mask]

    def call(self, inputs, epoch=0):
        inputs = tf.cast(inputs, tf.float32)
        
        
        if self.use_bernoulli_sampler is True:
            sig_mask = self.update_bernoulli_mask()
        else:
            sig_mask = self.sigmoid_mask(epoch)
        weights_masked = tf.multiply(self.w, sig_mask)
        
        self.no_ones = tf.reduce_sum(sig_mask)
        self.multiplier = (10*self.sigmoid_multiplier) * tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones)
        
        weights_masked = tf.multiply(self.multiplier, weights_masked)

        # print("FF Layer")
        # print("inputs shape")

        return tf.matmul(inputs, weights_masked) #+ self.b
    
        #intermediate_results = []
        
        #def apply_grothe_mask(inputs):
        #    grothe_mask = self.grothe_idea()
        #    weights_masked = tf.multiply(self.w, grothe_mask)
        #    return tf.matmul(inputs, weights_masked)
        
        #repeat=10
        
        #grothe = tf.map_fn(apply_grothe_mask, tf.repeat(tf.expand_dims(inputs, axis=0), repeats=repeat, axis=0), parallel_iterations=repeat, dtype=tf.float32, back_prop=True)
        
        #grothe_mean = tf.reduce_mean(grothe, axis=0)
        
        #return grothe_mean
        
        
        #for i in range(100):
        #    grothe_mask = self.grothe_idea()
        #    weights_masked = tf.multiply(self.w, grothe_mask)
            
        #    intermediate_results.append(tf.matmul(inputs, weights_masked).numpy())
        
        #print(f"Shape of intermediate_results: {np.array(intermediate_results).shape}")
        
        #tf_intermediate_results = tf.reduce_mean(tf.convert_to_tensor(intermediate_results, dtype=tf.float32), axis=0)
        
        
        #return tf_intermediate_results
        

class MaxPool2DExt(tf.keras.layers.MaxPool2D):
    def __init__(self, input_shape, pool_size=(2,2), strides=None, padding="valid", data_format="channels_last"):
        super(MaxPool2DExt, self).__init__(pool_size=pool_size, strides=strides, 
                                           padding=padding, data_format=data_format)
        
        self.type="mapo"
        
        if strides is None:
            strides = pool_size
        
        if padding == "valid":
            new_rows = int(np.ceil((input_shape[1] - pool_size[0] + 1) / strides[0]))
            new_cols = int(np.ceil((input_shape[2] - pool_size[1] + 1) / strides[1]))

            self.out_shape = (input_shape[0], new_rows, new_cols, input_shape[-1])

class FlattenExt(tf.keras.layers.Flatten):
    def __init__(self):
        super(FlattenExt, self).__init__()
        self.type = "flat"

class MaskedConv2D(tf.keras.layers.Conv2D):

    # untrainable original conv2d layer, trainable max
    def __init__(self, filters, kernel_size, input_shape, sigmoid_multiplier=0.2, dynamic_scaling=True,
                padding="valid", strides=(1,2), *args, **kwargs):
        super(MaskedConv2D, self).__init__(filters, kernel_size, padding=padding, strides=strides, *args, **kwargs)
        self._uses_learning_phase = True
        #self.sigmoid_bias = sigmoid_bias # bias to add before rounding to adjust prune percentage
        #self.round_mask = round_mask # round instead of bernoulli sampling
        #self.signed_constant = signed_constant
        #self.const_multiplier = const_multiplier
        self.filters = filters

        self.type = "conv"
        
        self.weight_shape= (kernel_size, kernel_size, input_shape[-1], filters)
        init_mask = tf.ones_initializer()
        self.mask = tf.Variable(initial_value=init_mask(shape=self.weight_shape, dtype="float32"),name="mask", trainable=True)
        
        self.threshold = tf.constant(0.5, dtype="float32")
        self.sigmoid_multiplier = sigmoid_multiplier
        
        self.bernoulli_mask = self.mask
        
        new_rows = int(np.ceil((input_shape[1]-kernel_size+1)/strides[0]))
        new_cols = int(np.ceil((input_shape[2]-kernel_size+1)/strides[1]))
        self.out_shape = (input_shape[0],new_rows,new_cols ,filters)

        init_kernel = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = init_kernel(shape=self.weight_shape, dtype="float32"), name="weights", trainable=False)
        # self.kernel = tf.Variable(initial_value = init_kernel(shape=self.weight_shape, dtype="float32"), name="weights_k", trainable=False)

        init_bias = tf.zeros_initializer()
        self.b = tf.Variable(initial_value =init_bias(shape=(filters,), dtype="float32"), name="bias", trainable=False)
        
        self.dynamic_scaling = dynamic_scaling

        # self._trainable_weights = []
        # self._trainable_weights.append(self.mask)
    
    #def get_shape(self):
    #    return self.shape
    
    def get_output_shape(self):
        return self.out_shape 

    def set_mask(self,mask):
        self.mask = tf.Variable(tf.cast(mask, "float32"), name="mask")
        
        self.sigmoid_mask()
        #self.mask = tf.Variable(mask.astype("float32"))
    
    def get_mask(self, as_logit=False):
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
    
    def get_normal_weights(self):
        return self.w
    
    def set_normal_weights(self, w):
        self.w = tf.Variable(w.astype("float32"), trainable=False, name="weights")
    
    def set_normal_weights_bias(self, wb):
        self.w = tf.constant(wb[0].astype("float32"))
        self.b = tf.constant(wb[1].astype("floar32"))
    
    def reset_mask(self):
        self.mask = tf.Variable(np.ones((self.input_dim,self.units), dtype="float32"))
    
    def get_bias(self):
        return self.b
        
    def get_pruned_weights(self):
        flipped_mask = tf.cast(tf.not_equal(self.bernoulli_mask, 1), tf.float32)
        return tf.multiply(self.w, flipped_mask)
    
    def get_masked_weights(self):
        return tf.multiply(self.w, self.bernoulli_mask)
    
    def get_nonzero_weights(self):
        #weights_masked = tf.multiply(self.w, self.mask)
        weights_masked = tf.boolean_mask(self.w, self.bernoulli_mask) #tf.not_equal(weights_masked, 0)
        return weights_masked #[mask]


    def build(self, input_shape):

        super(MaskedConv2D, self).build(input_shape)


        for i,var in enumerate( self._trainable_weights ):
            if "kernel" in var.name:
                del self._trainable_weights[i]
                del var
                # self._non_trainable_weights.append(var)
                # self._trainable_weights.remove(var)
            else:
                continue


        # self.kernel = self.add_weight(
        #     shape=self.weight_shape,
        #     initializer="random_normal",
        #     trainable=False,
        #     name="kernel_not_in_use"
        # )

    #     #mask_init = tfkeras.initializers.Constant(self.sigmoid_bias)
        # self._trainable_weights = [var for var in self.trainable_weights if "mask" in var.name]
        # for var in self.trainable_weights:
        #     if "kernel" in var.name:
        #         print(var.name)
        #         # self.non_trainable_variables.append(var)
        #         # self.trainable_variables.remove(var)
        #         # var.trainable = False
        #     else:
        #         continue
        # self.add_weight
    #     self.trainable_weights.remove(self.w)
    #     self.non_trainable_weights.append(self.w)
        #if self.use_bias:
        #    self._trainable_weights.remove(self.bias)
        #    self._non_trainable_weights.append(self.bias)

        #self.kernel_mask = tf.get_variable('mask',
        #                                   shape=self.kernel.shape,
        #                                   dtype=self.dtype,
        #                                   initializer=mask_init,
        #                                   trainable=True)
        #self._trainable_weights.append(self.kernel_mask)

    def sigmoid_mask(self):
        sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))

        effective_mask = tf.where(sigmoid_mask >= self.threshold, 1.0, sigmoid_mask)
        effective_mask = tf.where(effective_mask < self.threshold, 0.0, effective_mask)
        
        self.bernoulli_mask = effective_mask
    
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) 
    
    # same as original call() except apply binary mask
    def call(self, inputs):

        sig_mask = self.sigmoid_mask()
        
        # print("Shape mask: ", sig_mask.numpy().shape)
        # print("kernel shape: ", self.w.numpy().shape)
        
        weights_masked = tf.multiply(self.w, tf.cast(sig_mask, tf.float32))
        
        if self.dynamic_scaling:
            self.no_ones = tf.reduce_sum(sig_mask)
            self.multiplier = (10*self.sigmoid_multiplier) * tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones)
            weights_masked = tf.multiply(self.multiplier, weights_masked)
        
        # print("input type: ", inputs.dtype)
        # print("kernel type: ", weights_masked.dtype)
        outputs = self._convolution_op(inputs, weights_masked)
        
        return outputs
        
        #effective_mask = get_effective_mask(self)
        #effective_kernel = self.kernel * effective_mask

        #if self.dynamic_scaling:
        #    self.ones_in_mask = tf.reduce_sum(effective_mask)
        #    self.multiplier = tf.div(tf.to_float(tf.size(effective_mask)), self.ones_in_mask)
        #    effective_kernel = self.multiplier * effective_kernel

        # original code from https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py:
        #outputs = self._convolution_op(inputs, effective_kernel)
        #if self.use_bias:
        #    if self.data_format == 'channels_first':
        #        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
        #if self.activation is not None:
        #    return self.activation(outputs)
        #return outputs
        
        
    
class FCN(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4):
        super(FCN,self).__init__()
                
        self.linear_in = Linear(*layer_shapes[0])
        self.linear_h1 = Linear(*layer_shapes[1])
        self.linear_out = Linear(*layer_shapes[2])
        
        
    
    def call(self, inputs):
        
        
        layerwise_output = []
        layerwise_output.append(tf.reduce_mean(inputs, axis=0))
        
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_h1(x)
        x = tf.nn.relu(x) 
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_out(x)
        x = tf.nn.softmax(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        return x, layerwise_output
        
        #x = self.linear_in(inputs)
        #x = tf.nn.relu(x)
        #x = self.linear_h1(x)
        #x = tf.nn.relu(x) 
        #x = self.linear_out(x)
        #return tf.nn.softmax(x)
    
class FCN_Mask(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4, sigmoid_multiplier=[0.2,0.2,0.2], use_bernoulli_sampler=False):
        super(FCN_Mask,self).__init__()
                
        self.linear_in = Linear_Mask(*layer_shapes[0], sigmoid_multiplier=sigmoid_multiplier[0], use_bernoulli_sampler = use_bernoulli_sampler)
        self.linear_h1 = Linear_Mask(*layer_shapes[1], sigmoid_multiplier=sigmoid_multiplier[1], use_bernoulli_sampler = use_bernoulli_sampler)
        self.linear_out = Linear_Mask(*layer_shapes[2], sigmoid_multiplier=sigmoid_multiplier[2], use_bernoulli_sampler = use_bernoulli_sampler)
    
        self.all_layers = [self.linear_in, self.linear_h1, self.linear_out]
    
    def get_neuron_outputs(self,inputs):
        
        result = []
        result.append(inputs)
        
        x_in = self.linear_in(inputs)
        x_in = tf.nn.relu(x_in)
        result.append(tf.reduce_mean(x_in, axis=0))
        x_hidden = self.linear_h1(x_in)
        x_hidden = tf.nn.relu(x_hidden)
        result.append(tf.reduce_mean(x_hidden, axis=0))
        x_out = self.linear_out(x_hidden)
        x_out = self.nn.softmax(x_out)
        result.append(tf.reduce_mean(x_out, axis=0))
        
        return result
    
    def call(self, inputs):
        
        layerwise_output = []
        layerwise_output.append(tf.reduce_mean(inputs, axis=0))
        
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_h1(x)
        x = tf.nn.relu(x) 
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_out(x)
        x = tf.nn.softmax(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x, layerwise_output
    
class Conv2(tf.keras.Model):

    def __init__(self):

        self.conv_in = tf.keras.layers.Conv2D(filter=64, kernel_size=3, use_bias=False)
        self.conv_second = tf.keras.layers.Conv2D(filter=64, kernel_size=3, use_bias=False)
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.linear_first = tf.keras.layers.Linear(256)
        self.linear_second = tf.keras.layers.Linear(256)
        self.linear_out = tf.keras.layers.Linear(10)
    
    def call(self, inputs):

        x = self.conv_in(inputs)
        x = tf.nn.relu(x)
        x = self.conv_second(x)
        x = tf.nn.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)

        x = self.linear_first(x)
        x = tf.nn.relu(x)
        x = self.linear_second(x)
        x = tf.nn.relu(x)
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

        

class Conv2_Mask(tf.keras.Model):

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2]):
        super(Conv2_Mask, self).__init__()

        self.conv_in = MaskedConv2D(filters=64, kernel_size=3, input_shape=input_shape, use_bias=False,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_in")
        conv_in_out_shape = self.conv_in.out_shape
        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        self.conv_second = MaskedConv2D(filters=64, kernel_size=3, input_shape = conv_in_out_shape,
                                        sigmoid_multiplier=sigmoid_multiplier[1], use_bias=False, name="conv_second")
        conv_second_out_shape = self.conv_second.out_shape
        self.pooling_second = MaxPool2DExt(input_shape = conv_second_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape 
        self.flatten = FlattenExt() 
        # print(pooling_second_out_shape)
        self.linear_first = Linear_Mask(int(tf.math.reduce_prod(pooling_second_out_shape[1:]).numpy()),256,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="linear_first")
        self.linear_second = Linear_Mask(256,256, sigmoid_multiplier=sigmoid_multiplier[3], name="linear_second")
        self.linear_out = Linear_Mask(256,10, sigmoid_multiplier=sigmoid_multiplier[4], name="linear_out")

        # self._trainable_weights = []
        # self._trainable_weights.append(self.conv_in._trainable_weights) 
        # self._trainable_weights.append(self.conv_second._trainable_weights) 
        # self._trainable_weights.append(self.linear_first._trainable_weights) 
        # self._trainable_weights.append(self.linear_second._trainable_weights) 
        # self._trainable_weights.append(self.linear_out._trainable_weights) 
        


    def call(self, inputs):
        # layerwise_output = []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))

        x = self.conv_in(inputs)
        x = tf.nn.relu(x)
        # x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_second(x)
        x = tf.nn.relu(x)
        x = self.pooling_second(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.flatten(x)

        x = self.linear_first(x)
        x = tf.nn.relu(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.linear_second(x)
        x = tf.nn.relu(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x #, layerwise_output



class FCN_Mask4(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4):
        super(FCN_Mask4,self).__init__()
                
        self.linear_in = Linear_Mask(*layer_shapes[0])
        self.linear_h1 = Linear_Mask(*layer_shapes[1])
        self.linear_h2 = Linear_Mask(*layer_shapes[2])
        self.linear_out = Linear_Mask(*layer_shapes[3])
    
        self.all_layers = [self.linear_in, self.linear_h1, self.linear_h2, self.linear_out]
    
    def get_neuron_outputs(self,inputs):
        
        result = []
        result.append(inputs)
        
        x_in = self.linear_in(inputs)
        x_in = tf.nn.relu(x_in)
        result.append(tf.reduce_mean(x_in, axis=0))
        x_hidden_1 = self.linear_h1(x_in)
        x_hidden_1 = tf.nn.relu(x_hidden_1)
        result.append(tf.reduce_mean(x_hidden_1, axis=0))
        x_hidden_2 = self.linear_h1(x_hidden_1)
        x_hidden_2 = tf.nn.relu(x_hidden_2)
        result.append(tf.reduce_mean(x_hidden_2, axis=0))
        x_out = self.linear_out(x_hidden_2)
        x_out = tf.nn.softmax(x_out)
        result.append(tf.reduce_mean(x_out, axis=0))
        
        return result
        
#        for i in range(len(self.all_layers)):
#            important_layers = self.all_layers[:i+1]
            
#            for il in important_layers:
#                x = 
    
    def call(self, inputs):
        
        layerwise_output = []
        layerwise_output.append(tf.reduce_mean(inputs, axis=0))
        
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_h1(x)
        x = tf.nn.relu(x) 
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_h2(x)
        x = tf.nn.relu(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_out(x)
        x = tf.nn.softmax(x)
        layerwise_output.append(tf.reduce_mean(x, axis=0))
        return x, layerwise_output
    
class AE(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes):
        
        self.linear_in = Linear(*layer_shapes[0])
        self.linear_h1 = Linear(*layer_shapes[0])
        self.linear_b = Linear(*layer_shapes[0])
        self.linear_h2 = Linear(*layer_shapes[0])
        self.linear_out = Linear(*layer_shapes[0])
    
    def encode(self, inputs):
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        x = self.linear_h1(x)
        x = tf.nn.relu(x)
        x = self.linear_b(x)
        return tf.nn.relu(x)         
    
    def call(self, inputs):
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        x = self.linear_h1(x)
        x = tf.nn.relu(x)
        x = self.linear_b(x)
        x = tf.nn.relu(x) 
        x = self.linear_h2(x)
        x = tf.nn.relu(x) 
        x = self.linear_out(x)
        return tf.nn.sigmoid(x)
    
    
    #class FCN(tf.keras.Model):
    
   #     def __init__(self, input_dim, layer_shapes, no_layers=4):
   #         super(FCN,self).__init__()

    #        self.layer_shapes = layer_shapes
    #        self.model = {}
    #        for i in range(no_layers):
    #            if i == 0:
                    #tmp_shape_0 = layer_shapes[i][0]
                    #tmp_shape_1 = layer_shapes[i][1]
                    #if layer_shapes[i][0] == "inp":
                    #    tmp_shape_0 = input_dim
                    #if layer_shapes[i][1] == "inp":
                    #    tmp_shape_1 = input_dim

    #                self.model["linear_in"] = Linear(layer_shapes[i][0], layer_shapes[i][1])
    #            elif i < no_layers-1:
    #                if layer_shapes[i][0] == "inp":
    #                    layer_shapes[i][0] = input_dim
    #                self.model["linear_"+str(i)] = Linear(layer_shapes[i][0], layer_shapes[i][1])

    #            else: #if i == no_layers-1:
    #                self.model["linear_out"] = Linear(layer_shapes[i][0], layer_shapes[i][1])


#            self.linear_in = Linear(*layer_shapes[0])
#            self.linear_h1 = Linear(*layer_shapes[1])
#            self.linear_h2 = Linear(*layer_shapes[2])
#            self.linear_out = Linear(*layer_shapes[3])

#        def call(self, inputs):

    #        for name, layer in self.model.items():
    #            if name == "linear_in":
    #                x = layer(inputs)
    #            else:
    #                x = layer(x)

    #            if name != "linear_out":
    #                x = tf.nn.relu(x)

#            x = self.linear_in(inputs)
#            x = tf.nn.relu(x)
#            x = self.linear_h1(x)
#            x = tf.nn.relu(x)
#            x = self.linear_h2(x)
#            x = tf.nn.relu(x)    
#            x = self.linear_out(x)
#            return tf.nn.softmax(x)