import tensorflow as tf
import functools
#import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

def tf_custom_gradient_method(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped

seed = 7531
np.random.seed(seed)
tf.random.set_seed(seed)

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

@tf.custom_gradient
def custom_call_test(inputs):
    def grad(dy, variables=None):
        print("dy shape: ", dy.shape)
        return dy*0., variables
    return inputs, grad

class Dense_Mask(layers.Layer):
    
    def __init__(self,input_dim=32,units=32, sigmoid_multiplier=0.2, mask=False, mask_matrix=None, use_bernoulli_sampler=False, name=None, use_bias=False, dynamic_scaling=True, k=0.5, tanh_th=0.01):
        super(Dense_Mask,self).__init__()

        self.out_shape = (None, units)

        self.layer_name = name

        self.dynamic_scaling = dynamic_scaling
        self.use_bias = use_bias

        self.use_bernoulli_sampler = use_bernoulli_sampler

        self.type = "fefo"
        
        
        self.units = units
        self.input_dim = input_dim
        
        self.shape = (input_dim, units)

        self.size = input_dim*units
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
        
        #self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True)
        self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True,
                                name="mask")
        # if self.use_bias is True:
            # self.mask_b = tf.Variable(initial_value=init_mask(shape=(units, ), dtype="float32"), trainable=True, name="mask_b")
        self.threshold = 0.5 #tf.Variable(initial_value = 0.5, dtype=tf.float32, trainable=True)
        
        #self.multiplier_factor = tf.Variable(initial_value = [1.], trainable=True)
        
        
        self.bernoulli_mask = self.mask
        # if self.use_bias is True:
        #     self.bernoulli_b_mask = self.b_mask

        self.sigmoid_multiplier = sigmoid_multiplier
        
        self.multiplier = 1.
        self.no_ones = 0.
        
        #self.sigmoid01 = lambda x: 1 if tf.math.sigmoid(x) >= 0.5 else 0
        
        #self.sig_mask = tf.Variable(initial_value=self.sigmoid01(self.mask), trainable=False)
        
        self.w = tf.Variable(init_w(shape=(input_dim, units),dtype='float32'), trainable=False, name="w")

        if self.use_bias is True:
            # init_b = tf.keras.initializers.GlorotUniform(seed=7531) #tf.ones_initializer()#zeros_initializer()
            init_bias = tf.ones_initializer()
            self.b = tf.constant(value = init_bias(shape=(units,),dtype='float32'), name="bias_constant")
        # self.b = tf.constant(init_b(shape=(units,) ), dtype="float32", name="bias")

        self.k = k
        self.k_idx =  tf.cast(tf.cast(tf.reshape(self.mask, [-1]).get_shape()[0], tf.float32)*k, tf.int32)
        self.tanh_th = tanh_th
        # self._trainable_weights = []
        # self._trainable_weights.append(self.mask)

    def update_tanh_th(self, percentage=0.75):
        tanh_mask = self.mask_activation() #tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.cos(self.mask)#
        # tanh_mask = 1.7159*tf.math.sin(self.mask)
        mask_max = tf.math.reduce_max(tf.math.abs(tanh_mask))
        
        self.tanh_th = mask_max * percentage

        
    
    def set_mask_rand(self):
        self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
    
    def get_shape(self):
        return self.shape
    
    def update_weight_sign(self, sign):
        
        sign_wo_zero = tf.where(sign == 0., tf.sign( self.w ), sign)
        # self.w = self.w.assign(tf.multiply(self.w, sign_wo_zero))
        # self.w = self.w.assign(tf.cast(tf.multiply(self.w, sign_wo_zero), tf.float32))
        # self.w = self.w.assign(tf.where(tf.sign(self.w) == sign_wo_zero, self.w, tf.math.negative(self.w)))
        # self.w = tf.multiply(self.w, sign_wo_zero)
        # self.w = tf.Variable(tf.multiply(self.w, sign_wo_zero), trainable=False, name="w")
        self.w = tf.Variable(tf.where(tf.sign(self.w) == sign_wo_zero, tf.math.negative(self.w), self.w) , trainable=False, name="w")
        # w = tf.where( tf.equal( tf.sign(self.w),  sign_wo_zero ), self.w, tf.math.negative(self.w))
        # self.set_normal_weights(w.numpy())
        
        # self.w = tf.multiply(self.w, sign)
    
    def set_mask(self,mask):
        self.mask = tf.Variable(tf.cast(mask, "float32"), trainable=True, name="mask")
        # if self.mask in self.non_trainable_weights:
        #     print("true")
        self.trainable_weights.append(self.mask) 
        # self.update_bernoulli_mask()
        # self.sigmoid_mask()
        # self.tanh_mask()
        # self.tanh_score_mask()
        #self.mask = tf.Variable(mask.astype("float32"))
    
    def set_mask_b(self,mask):
        self.mask_b = tf.Variable(tf.cast(mask, "float32"), trainable=True, name="mask")
    
    def get_mask(self, as_logit=False):
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
    
    def get_bernoulli_mask(self):
        return self.bernoulli_mask  
        
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
    
    
    def sigmoid_mask(self):
        # sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))
        # sigmoid_mask = 1.7159 * tf.math.sigmoid(0.67*self.mask)
        sigmoid_mask = self.mask_activation() #tf.math.sigmoid(self.mask)

        bernoulli_sample = tf.random.uniform(shape = self.shape, minval=0., maxval=1.)
        effective_mask = tf.where(sigmoid_mask > bernoulli_sample, 1., 0.)
        
        # effective_mask = tf.where(sigmoid_mask > self.threshold, 1., 0.)
        # effective_mask = tf.where(effective_mask < self.threshold, 0., effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        #sig_mask[sig_mask < 0.5] = 0.
        # return effective_mask + self.mask - tf.stop_gradient(self.mask) #sigmoid_mask - tf.stop_gradient(sigmoid_mask) 
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) # + tf.nn.relu(self.mask) - tf.stop_gradient(tf.nn.relu(self.mask))
    
    def mask_activation(self):
        # return tf.math.cos(self.mask)
        # return tf.math.sin(self.mask)
        # return tf.math.atan((2/3)*self.mask)
        # return tf.nn.softsign(self.mask)
        # return tf.math.sigmoid(self.mask)
        # return tf.math.tanh((2/3)*self.mask) #* 1.7159
        # return tf.nn.elu(self.mask)
        # return tf.keras.activations.elu(self.mask, alpha=1.67326324)
        return self.mask
        
        
    def tanh_mask(self):
        
        # tanh_mask = tf.math.tanh(self.mask)
        # tanh_mask = tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.cos(self.mask)
        # tanh_mask = 1.7159*tf.math.sin(self.mask)
        tanh_mask = self.mask_activation() 
        # th = 0.01
        # bernoulli_sample = tf.random.uniform(shape = self.shape, minval=-1., maxval=1.)
        
        # effective_mask = tf.where(tanh_mask < bernoulli_sample, -1., 0.)
        # effective_mask = tf.where(tanh_mask > bernoulli_sample, 1., effective_mask)
        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        # mask_size = tf.cast(tf.size(effective_mask), tf.float32)
        # print("reduce mean of dense mask layer: ", tf.math.reduce_sum(tf.math.abs(effective_mask))/mask_size)
        
        self.bernoulli_mask = effective_mask
        
        return  tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) #+ self.mask - tf.stop_gradient(self.mask)
        # return  tf.stop_gradient(effective_mask) + (tanh_mask * tf.math.tanh(tanh_mask)) - tf.stop_gradient(tanh_mask * tf.math.tanh(tanh_mask)) #+ self.mask - tf.stop_gradient(self.mask)
    
    def tanh_mask_b(self):
        
        tanh_mask = 1.7159 * tf.math.tanh((2/3)*self.mask_b)
        
        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        return tf.stop_gradient( effective_mask ) + tanh_mask - tf.stop_gradient(tanh_mask) #+ self.mask - tf.stop_gradient(self.mask)
    
    def relu_mask(self):
        
        relu_mask = tf.nn.relu(tf.multiply(self.mask, 1.))
        
        effective_mask = tf.where(relu_mask > 0, 1, relu_mask)
        effective_mask = tf.where(effective_mask <= 0, 0, effective_mask)
        
        self.bernoulli_mask = effective_mask
        
        return effective_mask + relu_mask - tf.stop_gradient(relu_mask)
    
    # @tf.function
    def tanh_score_mask(self):
        
        tanh_mask = self.mask_activation() #tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.tanh(tf.multiply(self.mask, self.sigmoid_multiplier))
        
        # threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(tf.math.abs(tanh_mask), [-1]), self.k_idx, sorted=False).values)

        # print("Threshold: ", threshold)
        
        # effective_mask = tf.where(tanh_mask <= -threshold, -1., 0.)
        # effective_mask = tf.where(tanh_mask > threshold, 1., effective_mask)

        first_k = min(5000, self.size)
        
        # tanh_mask_reshaped = tf.random.shuffle(tf.reshape(tanh_mask, [-1]))[:first_k] 
        tanh_mask_reshaped = tf.gather(tf.random.shuffle(tf.reshape(tanh_mask, [-1])), np.arange(0,5000))

        sample_k_idx = int(self.k*first_k) + 1
        
        pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, sample_k_idx, sorted=True).values)
        neg_threshold = tf.math.reduce_max(-tf.math.top_k(-tanh_mask_reshaped, sample_k_idx, sorted=True).values)
        # pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, self.k_idx, sorted=True).value)
        # neg_threshold = tf.math.reduce_min(-tf.math.top_k(-tanh_mask_reshaped, self.k_idx, sorted=True).values
        # pos_threshold = tf.gather(tf.sort(tanh_mask_reshaped, direction="DESCENDING"), [self.k_idx])
        # neg_threshold = tf.gather(tf.sort(tanh_mask_reshaped, direction="ASCENDING"), [self.k_idx])
        # print("Threshold: ", threshold)
        
        # print("positive th: ", pos_threshold, "--- negative th: ", neg_threshold)

        #print("neg th: ", neg_threshold)
        #print("min val: ", tf.reduce_min(tanh_mask_reshaped).numpy())

        effective_mask = tf.where(tanh_mask < neg_threshold, -1., 0.)
        effective_mask = tf.where(tanh_mask > pos_threshold, 1., effective_mask)

        self.bernoulli_mask = effective_mask
        
        return tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) #+ tanh_mask - tf.stop_gradient(tanh_mask)
    
    def score_mask(self):
        # sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))
        sigmoid_mask = tf.math.sigmoid(self.mask)
        # sigmoid_mask = tf.math.abs(self.mask)

        # print(sigmoid_mask)
        
        
        # sigmoid_mask_flattened = tf.reshape(sigmoid_mask, [-1])
        
        # print("sigmoid mask flattened shape: ", sigmoid_mask_flattened.shape)
        
        # top_k = tf.math.top_k(sigmoid_mask_flattened, k=tf.size(sigmoid_mask_flattened)*k)

        # sigmoid_mask_sorted = tf.sort(sigmoid_mask_flattened, direction="DESCENDING")
        
        # print(sigmoid_mask_sorted)
        # m = v.get_shape()[0]//2
        # sigmoid_mask = tf.math.sigmoid(self.mask)

        
        threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(sigmoid_mask,[-1]), self.k_idx, sorted=False).values)

        # tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)
        #print("DENSE k_idx is: ", k_idx)
        # print("DENSE threshold: ", threshold)
        # k_idx = tf.cast(tf.size(sigmoid_mask_sorted, out_type=tf.float32)*k, tf.int32).numpy()
        # print("k_idx is: ", k_idx, "with value: ", sigmoid_mask_sorted[k_idx])
        # threshold = sigmoid_mask_sorted[k_idx]

        effective_mask = tf.where(sigmoid_mask >= threshold, 1.0, 0.0)
        # effective_mask = tf.where(effective_mask < threshold, 0.0, effective_mask)

        self.bernoulli_mask = effective_mask

        # return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        # return tf.stop_gradient(effective_mask) #+ sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        return tf.stop_gradient(effective_mask) + self.mask - tf.stop_gradient(self.mask) #+ tf.math.sigmoid( self.mask ) - tf.stop_gradient(tf.math.sigmoid( self.mask ))   #+ sigmoid_mask - tf.stop_gradient(sigmoid_mask)
    
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
    
    @tf.custom_gradient
    def custom_call(self, inputs):
        
        score_mask = self.score_mask()
        
        weights_masked = tf.multiply(self.w, score_mask)
        
        outputs = tf.matmul(inputs, weights_masked)
        
        def grad(dy, variables=None):
            # print("gradient of layer: ", self.name)
            print("dy shape: ", dy.shape)
            # print("dy: ")
            print(dy)
            # print("variables: ", len(variables))
            # print(variables)
            # print("dy shape: ", tf.shape(dy))
            # dy = dy * tf.matmul(inputs, self.w)
            # variables_grad = tf.zeros(tf.shape(dy))
            # print("variables grad shape: ", variables_grad.shape)
            return dy*0., variables#, [None]  #.inputs[1]
        
        # print(outputs.shape)
        
        return outputs, grad

    # @tf.function
    def call(self, inputs, weight_grad=False):
        inputs = tf.cast(inputs, tf.float32)
        
        # print("through layer: ", self.name)
        # return self.custom_call(inputs)
        # return self.custom_call(inputs) 
        if self.use_bernoulli_sampler is True:
            sig_mask = self.update_bernoulli_mask()
        else:
            # sig_mask = self.sigmoid_mask()
            # sig_mask = self.score_mask()
            sig_mask = self.tanh_score_mask()
            # sig_mask = self.tanh_mask()
        weights_masked = tf.multiply(self.w, sig_mask)
        # print("Call in fefo")
         
        if self.dynamic_scaling is True:
            # self.no_ones = tf.reduce_sum(weights_masked)
            # self.multiplier =  tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones) #* (1./self.sigmoid_multiplier)
            # self.multiplier = tf.math.divide(weights_masked - tf.math.reduce_mean(weights_masked), tf.math.reduce_std(weights_masked))
            #print("Multiplier in ", self.name, ": ", self.multiplier) 
            weights_masked = tf.math.divide(weights_masked , tf.math.reduce_std(weights_masked))#tf.multiply(tf.stop_gradient( self.multiplier ), weights_masked)

        # print("FF Layer")
        # print("inputs shape")
        outputs = tf.matmul(inputs, weights_masked)
        # import pdb;pdb.set_trace()

        if self.use_bias is True:
        #     bias_masked = tf.multiply(self.b, self.tanh_mask_b())
            outputs = tf.nn.bias_add(outputs, self.b)


        return outputs
        # if weight_grad is False:
        #     return outputs
        # else:
        #     full_output = tf.matmul(inputs, self.w)
        #     # return full_output

        #     return full_output * tf.stop_gradient(outputs/full_output) #tf.stop_gradient(outputs) + full_output*tf.identity(self.bernoulli_mask) - tf.stop_gradient(full_output)
            # return tf.stop_gradient(outputs) + full_output - tf.stop_gradient(full_output)
        # full_output = tf.matmul(inputs, self.w)

        # return outputs * tf.divide(full_output, tf.stop_gradient(full_output))
        # return tf.stop_gradient( outputs ) + full_output - tf.stop_gradient(full_output)#+ self.b

        # return outputs
    
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
    def __init__(self, input_shape=None, pool_size=(2,2), strides=None, padding="same", data_format="channels_last"):
        super(MaxPool2DExt, self).__init__(pool_size=pool_size, strides=strides, 
                                           padding=padding, data_format=data_format)
        
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
    def __init__(self, trainable):
        super(BatchNormExt, self).__init__(trainable=trainable)
        self.type="batchnorm"
class FlattenExt(tf.keras.layers.Flatten):
    def __init__(self):
        super(FlattenExt, self).__init__()
        self.type = "flat"

class Conv2DExt(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, use_bias, strides=(1, 1), padding='same', data_format="channels_last"):
        super(Conv2DExt, self).__init__(filters=filters, kernel_size=kernel_size, use_bias=use_bias, strides=strides, padding=padding, data_format=data_format)
        self.type = "conv_normal"

class GlobalAveragePooling2DExt(tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self):
        super(GlobalAveragePooling2DExt, self).__init__()
        self.type = "globalavgpooling"

class DenseExt(tf.keras.layers.Dense):
    def __init__(self, units, use_bias=True):
        super(DenseExt, self).__init__(units, use_bias=use_bias)
        self.type = "fefo_normal"

class MaskedConv2D(tf.keras.layers.Conv2D):

    # untrainable original conv2d layer, trainable max
    def __init__(self, filters, kernel_size, input_shape, sigmoid_multiplier=0.2, dynamic_scaling=True, use_bias=False, k=0.5, tanh_th=0.01,
                padding="same", strides=(1,1), *args, **kwargs):
        super(MaskedConv2D, self).__init__(filters, kernel_size, padding=padding, strides=strides, use_bias=False, *args, **kwargs)
        self._uses_learning_phase = True
        #self.sigmoid_bias = sigmoid_bias # bias to add before rounding to adjust prune percentage
        #self.round_mask = round_mask # round instead of bernoulli sampling
        #self.signed_constant = signed_constant
        #self.const_multiplier = const_multiplier
        self.filters = filters

        self.use_bias = use_bias

        self.type = "conv"
        
        self.weight_shape = (kernel_size, kernel_size, input_shape[-1], filters)
        self.size = kernel_size * kernel_size * input_shape[-1] * filters
        
        init_mask = tf.ones_initializer()
        self.mask = tf.Variable(initial_value=init_mask(shape=self.weight_shape, dtype="float32"),name="mask", trainable=True)

        self.total_mask_params= tf.cast(tf.size(self.mask), tf.float32)
        
        self.threshold = tf.constant(0.5, dtype="float32")
        self.sigmoid_multiplier = sigmoid_multiplier
        
        self.bernoulli_mask = self.mask
        
        #print("Input shape within Conv layer: ", input_shape)
        if padding == "valid": 
            new_rows = int(np.ceil((input_shape[1]-kernel_size+1)/strides[0]))
            new_cols = int(np.ceil((input_shape[2]-kernel_size+1)/strides[1]))
        elif padding == "same":
            new_rows = int(np.ceil(input_shape[1] / strides[0]))    
            new_cols = int(np.ceil(input_shape[2] / strides[1]))
        self.out_shape = (input_shape[0],new_rows,new_cols ,filters)

        init_kernel = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = init_kernel(shape=self.weight_shape, dtype="float32"), name="weights", trainable=False)
        # self.kernel = tf.Variable(initial_value = init_kernel(shape=self.weight_shape, dtype="float32"), name="weights_k", trainable=False)

        if self.use_bias is True:
            # init_bias = tf.keras.initializers.GlorotUniform(seed=7531) #tf.ones_initializer()#zeros_initializer()
            init_bias = tf.ones_initializer()
        # self.mask_b = tf.Variable(initial_value=init_mask(shape=(filters,), dtype="float32"), name="mask_b", trainable=True)
            # self.b = tf.Variable(initial_value =init_bias(shape=(filters,), dtype="float32"), name="bias", trainable=False)
            self.b = tf.Variable(initial_value = init_bias(shape=(filters, ), dtype=tf.float32), trainable=False, name="bias_constant")
        
        self.dynamic_scaling = dynamic_scaling

        self.no_ones = 0.
       
        self.k = k 
        self.k_idx =  tf.cast(tf.cast(tf.reshape(self.mask, [-1]).get_shape()[0], tf.float32)*k, tf.int32)
        
        self.tanh_th = tanh_th

        # self._trainable_weights = []
        # self._trainable_weights.append(self.mask)
    
    #def get_shape(self):
    #    return self.shape
    
    def update_tanh_th(self, percentage=0.75):
        tanh_mask = self.mask_activation() #tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.cos(self.mask)
        # tanh_mask = 1.7159*tf.math.sin(self.mask)
        mask_max = tf.math.reduce_max(tf.math.abs(tanh_mask))
        
        self.tanh_th = mask_max * percentage

    def get_output_shape(self):
        return self.out_shape 

    def set_mask(self,mask):
        self.mask = tf.Variable(tf.cast(mask, "float32"), name="mask")
        
        # self.tanh_mask()
        # self.tanh_score_mask()
        # self.score_mask()
        # self.sigmoid_mask()
        #self.mask = tf.Variable(mask.astype("float32"))
    
    def update_weight_sign(self, sign):
        
        sign_wo_zero = tf.sign(tf.where(sign == 0., self.w, sign))
        # self.w.assign(tf.where(tf.sign(self.w) == sign_wo_zero, self.w, tf.math.negative(self.w)))
        # self.w = self.w.assign(tf.cast(tf.multiply(self.w, sign_wo_zero), tf.float32))
        # self.w = tf.multiply(self.w, sign_wo_zero)
        self.w = tf.Variable(tf.multiply(self.w, sign_wo_zero), trainable=False, name="w")
        # self.w = tf.Variable(tf.where(tf.sign(self.w) == sign_wo_zero, self.w, tf.math.negative(self.w)), trainable=False, name="w")
        # w = tf.where(tf.sign(self.w) == sign_wo_zero, self.w, tf.math.negative(self.w))
        # self.set_normal_weights(w.numpy())
        # self.w = tf.where(tf.sign(self.w) == sign, self.w, tf.math.negative(self.w))
        # self.w = tf.multiply(self.w, sign)

    def get_mask(self, as_logit=False):
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
    
    def get_bernoulli_mask(self):
        return self.bernoulli_mask
    
    def get_normal_weights(self):
        return self.w
    
    def set_normal_weights(self, w):
        self.w = tf.Variable(w.astype("float32"), trainable=False, name="weights")
    
    def set_normal_weights_bias(self, wb):
        self.w = tf.constant(wb[0].astype("float32"))
        self.b = tf.constant(wb[1].astype("float32"))
    
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

        # print("build called!")

        # for i,var in enumerate(self.trainable_weights):
        #     print(var.name)


        for i,var in enumerate( self._trainable_weights ):
            delete_flag = False
            # print(var.name)
            if "kernel" in var.name:
                # print("kernel in name")
                del self._trainable_weights[i]
                delete_flag = True
                # del var
                # self._non_trainable_weights.append(var)
                # self._trainable_weights.remove(var)
            # else:
            #     continue
            if "bias" in var.name:
                # print("bias in name")
                del self._trainable_weights[i]
                del var
                # self._non_trainable_weights.append(var)
                # self._trainable_weights.remove(var)
            else:
                if delete_flag is True:
                    del var
                else:
                    continue

    def sigmoid_mask(self):
        # sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))
        sigmoid_mask = self.mask_activation() #tf.math.sigmoid(self.mask)

        bernoulli_sample = tf.random.uniform(shape = self.weight_shape, minval=0., maxval=1.)
        effective_mask = tf.where(sigmoid_mask > bernoulli_sample, 1., 0.)

        # effective_mask = tf.where(sigmoid_mask >= self.threshold, 1.0, 0.0)
        # effective_mask = tf.where(effective_mask < self.threshold, 0.0, effective_mask)
        
        self.bernoulli_mask = effective_mask
    
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) #sigmoid_mask - tf.stop_gradient(sigmoid_mask) 

    def mask_activation(self):
        # return tf.math.cos(self.mask)
        # return tf.math.sin(self.mask)
        # return tf.math.atan((2/3)*self.mask)
        # return tf.nn.softsign(self.mask)
        # return tf.math.tanh((2/3)*self.mask) # * 1.7159
        # return tf.nn.elu(self.mask)
        # return tf.keras.activations.elu(self.mask, alpha=1.67326324)
        return self.mask

    def tanh_mask(self):
        
        # tanh_mask = tf.math.tanh(self.mask)
        # tanh_mask = tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.cos(self.mask)
        # tanh_mask = 1.7159*tf.math.sin(self.mask)
        tanh_mask = self.mask_activation()
        # th = 0.04
        
        # bernoulli_sample = tf.random.uniform(shape = self.weight_shape, minval=-1., maxval=1.)
        
        # effective_mask = tf.where(tanh_mask < bernoulli_sample, -1., 0.)
        # effective_mask = tf.where(tanh_mask > bernoulli_sample, 1., effective_mask)
        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        # mask_size = tf.cast(tf.size(effective_mask), tf.float32)
        # print("reduce mean of convolutional mask layer: ", tf.math.reduce_sum(tf.math.abs(effective_mask))/mask_size)

        self.bernoulli_mask = effective_mask
        
        return  tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask)# + self.mask - tf.stop_gradient(self.mask)
        # return  tf.stop_gradient(effective_mask) + (tanh_mask * tf.math.tanh(tanh_mask)) - tf.stop_gradient(tanh_mask * tf.math.tanh(tanh_mask)) #+ self.mask - tf.stop_gradient(self.mask)
    
    def tanh_mask_b(self):
        
        tanh_mask = 1.7159 * tf.math.tanh((2/3)*self.mask_b)
        
        effective_mask = tf.where(tanh_mask < -self.tanh_th, -1., 0.)
        effective_mask = tf.where(tanh_mask > self.tanh_th, 1., effective_mask)
        
        return tf.stop_gradient( effective_mask ) + tanh_mask - tf.stop_gradient(tanh_mask) #+ self.mask - tf.stop_gradient(self.mask)

    # @tf.function
    def tanh_score_mask(self):
        
        tanh_mask = self.mask_activation() #tf.math.tanh((2/3)*self.mask) * 1.7159
        # tanh_mask = tf.math.tanh(tf.multiply(self.mask, self.sigmoid_multiplier))
        # tanh_mask_reshaped = tf.reshape(tanh_mask, [-1]) 
        # pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, self.k_idx, sorted=False).values)
        # neg_threshold = tf.math.reduce_min(-tf.math.top_k(-tanh_mask_reshaped, self.k_idx, sorted=False).values)
        # print("positive th: ", pos_threshold, "--- negative th: ", neg_threshold)
        # print("Threshold: ", threshold)
        # threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(tf.math.abs(tanh_mask), [-1]), self.k_idx, sorted=False).values)
        # effective_mask = tf.where(tanh_mask <= -threshold, -1., 0.)
        # effective_mask = tf.where(tanh_mask > threshold, 1., effective_mask)


        first_k = min(5000, self.size)
        # print("first k: ", first_k)
        # tanh_mask_reshaped = tf.random.shuffle(tf.reshape(tanh_mask, [-1]))[:first_k]
        tanh_mask_reshaped = tf.gather(tf.random.shuffle(tf.reshape(tanh_mask, [-1])), np.arange(0,5000))
        
        k_idx = int(self.k*first_k) + 1
         
        pos_threshold = tf.math.reduce_min(tf.math.top_k(tanh_mask_reshaped, k_idx, sorted=True).values)
        neg_threshold = tf.math.reduce_max(-tf.math.top_k(-tanh_mask_reshaped, k_idx, sorted=True).values)

        effective_mask = tf.where(tanh_mask < neg_threshold, -1., 0.)
        effective_mask = tf.where(tanh_mask > pos_threshold, 1., effective_mask)

        self.bernoulli_mask = effective_mask

        return tf.stop_gradient(effective_mask) + tanh_mask - tf.stop_gradient(tanh_mask) #+ tanh_mask - tf.stop_gradient(tanh_mask)
    
    def score_mask(self):
        # sigmoid_mask = tf.math.sigmoid(tf.multiply(self.mask, self.sigmoid_multiplier))
        sigmoid_mask = tf.math.sigmoid(self.mask)
        # sigmoid_mask = tf.math.abs(self.mask)

        # print(sigmoid_mask)
        
        
        # sigmoid_mask = tf.math.sigmoid(self.mask)
        # sigmoid_mask_flattened = tf.reshape(self.mask, [-1])
        
        # print("sigmoid mask flattened shape: ", sigmoid_mask_flattened.shape)
        
        # top_k = tf.math.top_k(sigmoid_mask_flattened, k=tf.size(sigmoid_mask_flattened)*k)

        # sigmoid_mask_sorted = tf.sort(sigmoid_mask_flattened, direction="DESCENDING")
        
        # print(sigmoid_mask_sorted)
        # m = v.get_shape()[0]//2
        # print(sigmoid_mask_flattened)
        # print("mask has NAN: ", tf.math.is_nan(self.mask).numpy())
        threshold = tf.math.reduce_min(tf.math.top_k(tf.reshape(sigmoid_mask, [-1]), self.k_idx, sorted=False).values)
        # tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)

        # k_idx = tf.cast(tf.size(sigmoid_mask_sorted, out_type=tf.float32)*k, tf.int32).numpy()
        #print("CONV k_idx is: ", k_idx)
        # threshold = sigmoid_mask_sorted[k_idx]
        
        # print("CONV Threshold: ", threshold)
        
        effective_mask = tf.where(sigmoid_mask >= threshold, 1.0, 0.0)
        # effective_mask = tf.where(effective_mask < threshold, 0.0, effective_mask)
        
        #print("Effective Mask: ", effective_mask)
        
        self.bernoulli_mask = effective_mask

        # return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        return tf.stop_gradient(effective_mask) + self.mask - tf.stop_gradient(self.mask) #+ tf.math.sigmoid( self.mask ) - tf.stop_gradient(tf.math.sigmoid( self.mask ))   #+ sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        # return tf.stop_gradient(effective_mask) +  tf.math.sigmoid( self.mask )  - tf.stop_gradient(self.mask) #+ tf.identity(effective_mask) - tf.stop_gradient(tf.identity(effective_mask))   #+ sigmoid_mask - tf.stop_gradient(sigmoid_mask)
        
    @tf.custom_gradient
    def custom_call(self, inputs):
        
        score_mask = self.score_mask()
        
        weights_masked = tf.multiply(self.w, score_mask)
        
        outputs = self._convolution_op(inputs, weights_masked)
        
        def grad(dy, variables=None):
            return (dy, variables)
        
        return outputs, grad
        
    # same as original call() except apply binary mask
    # @tf.function
    def call(self, inputs, weight_grad=False):

        inputs = tf.cast(inputs, tf.float32)

        # print("through layer: ", self.name)
        
        # return self.custom_call(inputs)
        
        
        # sig_mask = self.sigmoid_mask()
        # sig_mask = self.score_mask()
        sig_mask = self.tanh_score_mask()
        # sig_mask = self.tanh_mask()
        
        # print(sig_mask) 
        # print("Shape mask: ", sig_mask.numpy().shape)
        # print("kernel shape: ", self.w.numpy().shape)
        
        # print("call in conv")
        weights_masked = tf.multiply(self.w, sig_mask)#tf.cast(sig_mask, tf.float32))

        # print(self.name, "mask shape: ", sig_mask.shape)
        
            #         self.ones_in_mask = tf.reduce_sum(effective_mask)
            # self.multiplier = tf.div(tf.to_float(tf.size(effective_mask)), self.ones_in_mask)
            # effective_kernel = self.multiplier * effective_kernel


        if self.dynamic_scaling:
            # single_filter_size = tf.reduce_prod(sig_mask.shape[:-1])
            # reshaped_sig_mask = tf.reshape(sig_mask, (single_filter_size,sig_mask.shape[-1]))
            # print("reshaped sig mask size: ", reshaped_sig_mask.shape)
            # self.no_ones = tf.cast(tf.reduce_sum(reshaped_sig_mask, axis=-1), tf.float32)
            # self.no_ones = tf.cast(tf.reduce_sum(sig_mask), tf.float32)
            # print("No ones cnn multiplier: ", self.no_ones)
            # self.multiplier =  tf.math.divide(self.size, self.no_ones) #* (1./self.sigmoid_multiplier)
            #print("Multiplier in ",self.name," : ", self.multiplier)
            # weights_masked = tf.multiply(self.multiplier, weights_masked)
            weights_masked = tf.math.divide(weights_masked - tf.math.reduce_mean(weights_masked), tf.math.reduce_std(weights_masked))#tf.multiply(tf.stop_gradient( self.multiplier ), weights_masked)
            # self.no_ones = tf.reduce_sum(sig_mask)
            # self.multiplier = (1/self.sigmoid_multiplier) * tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones)
            # weights_masked = tf.multiply(self.multiplier, weights_masked)
        # print("input type: ", inputs.dtype)
        # print("kernel type: ", weights_masked.dtype)
        # print(self.name) 
        # print(inputs.shape)
        # print(weights_masked.shape)

        outputs = self._convolution_op(inputs, weights_masked)
        # import pdb; pdb.set_trace()

        if self.use_bias:
            # bias_masked = tf.multiply(self.b, self.tanh_mask_b())
            outputs = tf.nn.bias_add(outputs,self.b)
        
        return outputs
        
        # if weight_grad is False:
        #     return outputs
        # else:
        #     full_output = self._convolution_op(inputs, self.w)
        #     # return full_output
        #     return full_output * tf.stop_gradient(outputs/full_output) #tf.stop_gradient(outputs) + full_output*tf.identity(self.bernoulli_mask) - tf.stop_gradient(full_output)
        # full_output = self._convolution_op(inputs, self.w)
        
        # return tf.divide(tf.multiply(tf.stop_gradient( outputs ),full_output), tf.stop_gradient(full_output))
        # return outputs #+ full_output - tf.stop_gradient(full_output)

        
        #------------------OLD-------------------------------- 
        
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
    
    def __init__(self, input_dim, layer_shapes, no_layers=4, sigmoid_multiplier=[0.2,0.2,0.2], use_bernoulli_sampler=False, dynamic_scaling=True, k=0.5):
        super(FCN_Mask,self).__init__()
                
        self.linear_in = Dense_Mask(*layer_shapes[0], sigmoid_multiplier=sigmoid_multiplier[0], use_bernoulli_sampler = use_bernoulli_sampler, dynamic_scaling=dynamic_scaling, k=k)
        self.linear_h1 = Dense_Mask(*layer_shapes[1], sigmoid_multiplier=sigmoid_multiplier[1], use_bernoulli_sampler = use_bernoulli_sampler, dynamic_scaling=dynamic_scaling, k=k)
        self.linear_out = Dense_Mask(*layer_shapes[2], sigmoid_multiplier=sigmoid_multiplier[2], use_bernoulli_sampler = use_bernoulli_sampler, dynamic_scaling=dynamic_scaling, k=k)
    
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
        x_out = tf.nn.softmax(x_out)
        result.append(tf.reduce_mean(x_out, axis=0))
        
        return result
    
    def call(self, inputs, weight_grad=False):
        
        # layerwise_output = []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))
        
        x = self.linear_in(inputs, weight_grad=weight_grad)
        x = tf.nn.relu(x)
        # print("shape after in: ", x.shape)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_h1(x, weight_grad=weight_grad)
        x = tf.nn.relu(x) 
        # print("shape after h1: ", x.shape)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        x = self.linear_out(x, weight_grad=weight_grad)
        x = tf.nn.softmax(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        # print("x shape: ", x.shape) 
        return x #, layerwise_output
    
class Conv2(tf.keras.Model):

    def __init__(self, use_bias=False, use_dropout=False, dropout_rate=0.2):
        super(Conv2, self).__init__()  

        # self.use_dropout = use_dropout
        # self.dropout_rate = dropout_rate
        
        self.conv_in = Conv2DExt(filters=64, kernel_size=3, use_bias=use_bias)
        self.conv_second = Conv2DExt(filters=64, kernel_size=3, use_bias=use_bias)
        self.pooling = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.flatten = FlattenExt()
        self.linear_first = DenseExt(256, use_bias=use_bias)
        self.linear_second = DenseExt(256, use_bias=use_bias)
        self.linear_out = DenseExt(10, use_bias=use_bias)
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)

    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.activation(x)
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        # if self.use_dropout is True:
        #     x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_second(x)
        x = self.activation(x)
        # if self.use_dropout is True:
        #     x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv4(tf.keras.Model):

    def __init__(self, use_bias=False, use_dropout=False, dropout_rate=.2):
        super(Conv4, self).__init__()  
        
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        
        self.conv_in = Conv2DExt(filters=64, kernel_size=3, use_bias=False)
        self.conv_second = Conv2DExt(filters=64, kernel_size=3, use_bias=False)
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_third = Conv2DExt(filters=128, kernel_size=3, use_bias=False)
        self.conv_fourth = Conv2DExt(filters=128, kernel_size=3, use_bias=False)
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.flatten = FlattenExt()
        self.linear_first = DenseExt(256, use_bias=False)
        self.linear_second = DenseExt(256, use_bias=False)
        self.linear_out = DenseExt(10, use_bias=False)
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)

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
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_second(x)
        x = self.activation(x)
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv6(tf.keras.Model):

    def __init__(self, use_bias=False, use_dropout=False, dropout_rate=.2):
        super(Conv6, self).__init__() 
        
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
         
        self.conv_in = Conv2DExt(filters=64, kernel_size=3, use_bias=use_bias)
        self.conv_second = Conv2DExt(filters=64, kernel_size=3, use_bias=use_bias)
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_third = Conv2DExt(filters=128, kernel_size=3, use_bias=use_bias)
        self.conv_fourth = Conv2DExt(filters=128, kernel_size=3, use_bias=use_bias)
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_fifth = Conv2DExt(filters=256, kernel_size=3, use_bias=use_bias)
        self.conv_sixth = Conv2DExt(filters=256, kernel_size=3, use_bias=use_bias)
        self.pooling_third = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.flatten = FlattenExt()
        self.linear_first = DenseExt(256, use_bias=use_bias)
        self.linear_second = DenseExt(256, use_bias=use_bias)
        self.linear_out = DenseExt(10, use_bias=use_bias)
    
    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)

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
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_second(x)
        x = self.activation(x)
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv8(tf.keras.Model):

    def __init__(self):
        super(Conv8, self).__init__(use_dropout=False, dropout_rate=.2) 
        
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
         
        self.conv_in = Conv2DExt(filters=64, kernel_size=3, use_bias=False)
        self.conv_second = Conv2DExt(filters=64, kernel_size=3, use_bias=False)
        self.pooling_first = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_third = Conv2DExt(filters=128, kernel_size=3, use_bias=False)
        self.conv_fourth = Conv2DExt(filters=128, kernel_size=3, use_bias=False)
        self.pooling_second = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_fifth = Conv2DExt(filters=256, kernel_size=3, use_bias=False)
        self.conv_sixth = Conv2DExt(filters=256, kernel_size=3, use_bias=False)
        self.pooling_third = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.conv_seventh = Conv2DExt(filters=512, kernel_size=3, use_bias=False)
        self.conv_eighth = Conv2DExt(filters=512, kernel_size=3, use_bias=False)
        self.pooling_fourth = MaxPool2DExt(pool_size=(2,2), strides=(2,2))
        self.flatten = FlattenExt()
        self.linear_first = DenseExt(256, use_bias=False)
        self.linear_second = DenseExt(256, use_bias=False)
        self.linear_out = DenseExt(10, use_bias=False)

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)
    
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
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_second(x)
        x = tf.nn.relu(x)
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.linear_out(x)
        
        return tf.nn.softmax(x)

class Conv2_Mask(tf.keras.Model):

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2], use_bias=False, use_dropout=False,dropout_rate=0.2, 
                 dynamic_scaling_cnn=True, dynamic_scaling_dense=True, k_cnn=0.4, k_dense=0.3, width_multiplier=1):
        super(Conv2_Mask, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
 
        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape=input_shape, use_bias=use_bias, k=k_cnn,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_in", dynamic_scaling=dynamic_scaling_cnn)
        # self.bn_1 = BatchNormExt(trainable=False)
        conv_in_out_shape = self.conv_in.out_shape
        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape = conv_in_out_shape, use_bias=use_bias, k=k_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[1], name="conv_second", dynamic_scaling=dynamic_scaling_cnn)
        # self.bn_2 = BatchNormExt(trainable=False)
        conv_in_out_shape = self.conv_in.out_shape
        conv_second_out_shape = self.conv_second.out_shape
        self.pooling = MaxPool2DExt(input_shape = conv_second_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_out_shape= self.pooling.out_shape 
        self.flatten = FlattenExt() 
        # print(pooling_second_out_shape)
        self.linear_first = Dense_Mask(int(tf.math.reduce_prod(pooling_out_shape[1:]).numpy()),int(256*width_multiplier), use_bias=use_bias, dynamic_scaling=dynamic_scaling_dense, k=k_dense,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="linear_first")
        # self.bn_3 = BatchNormExt(trainable=False)
        conv_in_out_shape = self.conv_in.out_shape
        self.linear_second = Dense_Mask(int(256*width_multiplier),int(256*width_multiplier), sigmoid_multiplier=sigmoid_multiplier[3], use_bias=use_bias, dynamic_scaling=dynamic_scaling_dense, name="linear_second", k=k_dense)
        # self.bn_4 = BatchNormExt(trainable=False)
        self.linear_out = Dense_Mask(int(256*width_multiplier),10, sigmoid_multiplier=sigmoid_multiplier[4], use_bias=use_bias, dynamic_scaling=dynamic_scaling_dense, name="linear_out", k=k_dense)

    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)


    def call(self, inputs,  weight_grad=False): #get_intermediate=False,
        
        # if get_intermediate is True:
        #     intermediate =  []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))

        # self.ones_array = []

        x = self.conv_in(inputs, weight_grad=False)
        x = self.activation(x)
        # x = self.bn_1(x)
        # x = self.pooling_first(x)
        # if get_intermediate is True:
        #     # intermediate = tf.ragged.constant(x)
        #     intermediate.append(x)
        
        
        x = self.conv_second(x, weight_grad=False)
        x = self.activation(x)
        # x = self.bn_2(x)

        # if get_intermediate is True:
        #     # intermediate = tf.stack([intermediate, x])
        #     intermediate.append(x)
        x = self.pooling(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.flatten(x)

        x = self.linear_first(x, weight_grad=False)
        x = self.activation(x)
        # x = self.bn_3(x)

        # if get_intermediate is True:
        #     intermediate.append(x)
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.linear_second(x, weight_grad=False)
        x = self.activation(x)
        # x = self.bn_4(x)

        # if get_intermediate is True:
        #     intermediate.append(x)
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.linear_out(x, weight_grad=False)
        # if get_intermediate is True:
        #     intermediate.append(x)
        x = tf.nn.softmax(x)
        
        # if get_intermediate is True: 
        #     self.intermediate = intermediate #tf.ragged.constant(intermediate)
        # self.ones_array.append(self.conv_in.no_ones)
        # self.ones_array.append(self.conv_second.no_ones)
        # self.ones_array.append(self.linear_first.no_ones)
        # self.ones_array.append(self.linear_second.no_ones)
        # self.ones_array.append(self.linear_out.no_ones)
        
        # self.no_ones = tf.reduce_sum(self.ones_array)
        

        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x #, layerwise_output

class Conv4_Mask(tf.keras.Model):
    

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2,0.2,0.2], dynamic_scaling_cnn=True, k_cnn=0.4, k_dense=0.3,
                 dynamic_scaling_dense=True, use_dropout=False, dropout_rate=0.2, width_multiplier=1, use_bias=False):
        super(Conv4_Mask, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape=input_shape, use_bias=use_bias, k=k_cnn, dynamic_scaling=dynamic_scaling_cnn,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_in")
        conv_in_out_shape = self.conv_in.out_shape

        self.bn_1 = BatchNormExt(trainable=False)

        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        # print("conv_in out: ", conv_in_out_shape)
        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape = conv_in_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[1], name="conv_second")
        self.bn_2 = BatchNormExt(trainable=False)
        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_first_out_shape = self.pooling_first.out_shape 
        # print("pooling first out: ", pooling_first_out_shape)
        self.conv_third = MaskedConv2D(filters=int(128*width_multiplier), kernel_size=3, input_shape = pooling_first_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="conv_third")
        self.bn_3 = BatchNormExt(trainable=False)
        conv_third_out_shape = self.conv_third.out_shape
        # print("conv_third out: ", conv_third_out_shape)
        self.conv_fourth = MaskedConv2D(filters=int(128*width_multiplier), kernel_size=3, input_shape = conv_third_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[3], name="conv_fourth")
        self.bn_4 = BatchNormExt(trainable=False)
        conv_fourth_out_shape = self.conv_fourth.out_shape
        # print("conv fourth out: ", conv_fourth_out_shape)
        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape

        self.flatten = FlattenExt() 
        # print("pooling second out: ", pooling_second_out_shape)
        self.linear_first = Dense_Mask(int(tf.math.reduce_prod(pooling_second_out_shape[1:]).numpy()),int(256*width_multiplier),
                                        sigmoid_multiplier=sigmoid_multiplier[4], name="linear_first", k=k_dense, dynamic_scaling=dynamic_scaling_dense, use_bias=use_bias)
        self.bn_5 = BatchNormExt(trainable=False)
        self.linear_second = Dense_Mask(int(256*width_multiplier),int(256*width_multiplier), sigmoid_multiplier=sigmoid_multiplier[5], name="linear_second", k=k_dense, dynamic_scaling=dynamic_scaling_dense, use_bias=use_bias)
        self.bn_6 = BatchNormExt(trainable=False)
        self.linear_out = Dense_Mask(int(256*width_multiplier),10, sigmoid_multiplier=sigmoid_multiplier[6], name="linear_out", k=k_dense, dynamic_scaling=dynamic_scaling_dense, use_bias=use_bias)



    def activation(self, x):
        return x * tf.math.sigmoid(x)
        # return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.leaky_relu(x, alpha=.5)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)

    def call(self, inputs):
        # layerwise_output = []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))

        x = self.conv_in(inputs)
        # x = self.bn_1(x)
        x = self.activation(x)
        # x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_second(x)
        # x = self.bn_2(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_third(x)
        # x = self.bn_3(x)
        x = self.activation(x)
    
        # print(x.shape)
        
        x = self.conv_fourth(x)
        # x = self.bn_4(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        # x = self.bn_5(x)
        x = self.activation(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        # if self.use_dropout is True:
        #     x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_second(x)
        # x = self.bn_6(x)
        x = self.activation(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        # if self.use_dropout is True:
        #     x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x #, layerwise_output

class Conv6_Mask(tf.keras.Model):
    

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2], dynamic_scaling_cnn=True, k_cnn=0.4, k_dense=0.3,
                 dynamic_scaling_dense=True, use_dropout=False, dropout_rate=0.2, width_multiplier=1, use_bias=False):
        super(Conv6_Mask, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.conv_in = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape=input_shape, use_bias=use_bias, k=k_cnn, dynamic_scaling=dynamic_scaling_cnn,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_in")
        conv_in_out_shape = self.conv_in.out_shape
        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        # print("conv_in out: ", conv_in_out_shape)
        self.conv_second = MaskedConv2D(filters=int(64*width_multiplier), kernel_size=3, input_shape = conv_in_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[1], name="conv_second")
        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_first_out_shape = self.pooling_first.out_shape 
        # print("pooling first out: ", pooling_first_out_shape)
        self.conv_third = MaskedConv2D(filters=int(128*width_multiplier), kernel_size=3, input_shape = pooling_first_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="conv_third")
        conv_third_out_shape = self.conv_third.out_shape
        # print("conv_third out: ", conv_third_out_shape)
        self.conv_fourth = MaskedConv2D(filters=int(128*width_multiplier), kernel_size=3, input_shape = conv_third_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[3], name="conv_fourth")
        conv_fourth_out_shape = self.conv_fourth.out_shape
        # print("conv fourth out: ", conv_fourth_out_shape)
        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape

        self.conv_fifth = MaskedConv2D(filters=int(256*width_multiplier), kernel_size=3, input_shape=pooling_second_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_fifth")
        conv_fifth_out_shape = self.conv_fifth.out_shape
        
        self.conv_sixth = MaskedConv2D(filters=int(256*width_multiplier), kernel_size=3, input_shape=conv_fifth_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_sixth")
        conv_sixth_out_shape = self.conv_sixth.out_shape
        
        self.pooling_third = MaxPool2DExt(input_shape=conv_sixth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_third_out_shape = self.pooling_third.out_shape

        self.flatten = FlattenExt() 
        # print("pooling second out: ", pooling_second_out_shape)
        self.linear_first = Dense_Mask(int(tf.math.reduce_prod(pooling_third_out_shape[1:]).numpy()),int(256*width_multiplier), use_bias=use_bias,
                                        sigmoid_multiplier=sigmoid_multiplier[6], name="linear_first", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_second = Dense_Mask(int(256*width_multiplier),int(256*width_multiplier), sigmoid_multiplier=sigmoid_multiplier[7], use_bias=use_bias, name="linear_second", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_out = Dense_Mask(int(256*width_multiplier),10, sigmoid_multiplier=sigmoid_multiplier[8], use_bias=use_bias, name="linear_out", k=k_dense, dynamic_scaling=dynamic_scaling_dense)


    def activation(self, x):
        return x * tf.math.sigmoid(x)
        # return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)


    def call(self, inputs):
        # layerwise_output = []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))

        x = self.conv_in(inputs)
        x = self.activation(x)
        # x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_third(x)
        x = self.activation(x)
    
        # print(x.shape)
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
    
        # print(x.shape)
        
        x = self.conv_sixth(x)
        x = self.activation(x)
        x = self.pooling_third(x)

        x = self.flatten(x)

        x = self.linear_first(x)
        x = self.activation(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_second(x)
        x = self.activation(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x #, layerwise_output

class Conv8_Mask(tf.keras.Model):
    

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2], dynamic_scaling_cnn=True, k_cnn=0.4, k_dense=0.3,
                 dynamic_scaling_dense=True, use_dropout=False, dropout_rate=0.2, width_multiplier=1, use_bias=False):
        super(Conv8_Mask, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.conv_in = MaskedConv2D(filters=64*width_multiplier, kernel_size=3, input_shape=input_shape, use_bias=use_bias, k=k_cnn, dynamic_scaling=dynamic_scaling_cnn,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_in")
        conv_in_out_shape = self.conv_in.out_shape
        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        # print("conv_in out: ", conv_in_out_shape)
        self.conv_second = MaskedConv2D(filters=64*width_multiplier, kernel_size=3, input_shape = conv_in_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[1], name="conv_second")
        conv_second_out_shape = self.conv_second.out_shape
        
        self.pooling_first= MaxPool2DExt(input_shape = conv_second_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_first_out_shape = self.pooling_first.out_shape 
        # print("pooling first out: ", pooling_first_out_shape)
        self.conv_third = MaskedConv2D(filters=128*width_multiplier, kernel_size=3, input_shape = pooling_first_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="conv_third")
        conv_third_out_shape = self.conv_third.out_shape
        # print("conv_third out: ", conv_third_out_shape)
        self.conv_fourth = MaskedConv2D(filters=128*width_multiplier, kernel_size=3, input_shape = conv_third_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[3], name="conv_fourth")
        conv_fourth_out_shape = self.conv_fourth.out_shape
        # print("conv fourth out: ", conv_fourth_out_shape)
        self.pooling_second = MaxPool2DExt(input_shape=conv_fourth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_second_out_shape = self.pooling_second.out_shape

        self.conv_fifth = MaskedConv2D(filters=256*width_multiplier, kernel_size=3, input_shape=pooling_second_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_fifth")
        conv_fifth_out_shape = self.conv_fifth.out_shape
        
        self.conv_sixth = MaskedConv2D(filters=256*width_multiplier, kernel_size=3, input_shape=conv_fifth_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_sixth")
        conv_sixth_out_shape = self.conv_sixth.out_shape
        
        self.pooling_third = MaxPool2DExt(input_shape=conv_sixth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_third_out_shape = self.pooling_third.out_shape

        self.conv_seventh = MaskedConv2D(filters=512*width_multiplier, kernel_size=3, input_shape=pooling_third_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_seventh")
        conv_seventh_out_shape = self.conv_seventh.out_shape
        
        self.conv_eighth = MaskedConv2D(filters=512*width_multiplier, kernel_size=3, input_shape=conv_seventh_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_eighth")
        conv_eighth_out_shape = self.conv_eighth.out_shape
        
        self.pooling_fourth = MaxPool2DExt(input_shape=conv_eighth_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_fourth_out_shape = self.pooling_fourth.out_shape

        self.flatten = FlattenExt() 
        # print("pooling second out: ", pooling_second_out_shape)
        self.linear_first = Dense_Mask(int(tf.math.reduce_prod(pooling_fourth_out_shape[1:]).numpy()),256, 
                                        sigmoid_multiplier=sigmoid_multiplier[6], use_bias=use_bias, name="linear_first", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_second = Dense_Mask(256,256, sigmoid_multiplier=sigmoid_multiplier[7], use_bias=use_bias, name="linear_second", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_out = Dense_Mask(256,10, sigmoid_multiplier=sigmoid_multiplier[8], use_bias=use_bias, name="linear_out", k=k_dense, dynamic_scaling=dynamic_scaling_dense)


    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)


    def call(self, inputs):
        # layerwise_output = []
        # layerwise_output.append(tf.reduce_mean(inputs, axis=0))

        x = self.conv_in(inputs)
        x = self.activation(x)
        # x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_second(x)
        x = self.activation(x)
        x = self.pooling_first(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        x = self.conv_third(x)
        x = self.activation(x)
    
        # print(x.shape)
        
        x = self.conv_fourth(x)
        x = self.activation(x)
        x = self.pooling_second(x)

        x = self.conv_fifth(x)
        x = self.activation(x)
    
        # print(x.shape)
        
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
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_second(x)
        x = self.activation(x)
        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        if self.use_dropout is True:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        # layerwise_output.append(tf.reduce_mean(x, axis=0))
        
        return x #, layerwise_output


class VGG19_Mask(tf.keras.Model):
    

    def __init__(self, input_shape, sigmoid_multiplier=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2], dynamic_scaling_cnn=True, k_cnn=0.4, k_dense=0.3,
                 dynamic_scaling_dense=True, use_dropout=False, dropout_rate=0.2, width_multiplier=1, use_bias=False):
        super(VGG19_Mask, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.conv_1 = MaskedConv2D(filters=64, kernel_size=3, input_shape=input_shape, use_bias=use_bias, k=k_cnn, dynamic_scaling=dynamic_scaling_cnn,
                                    sigmoid_multiplier=sigmoid_multiplier[0], name="conv_1")
        conv_1_out_shape = self.conv_1.out_shape
        # self.pooling_first = MaxPool2DExt(input_shape = conv_in_out_shape, pool_size=(2,2))
        # pooling_first_out_shape = self.pooling_first.out_shape
        # print("conv_in out: ", conv_in_out_shape)
        self.conv_2 = MaskedConv2D(filters=64, kernel_size=3, input_shape = conv_1_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[1], name="conv_2")
        conv_2_out_shape = self.conv_2.out_shape
        
        self.pooling_1= MaxPool2DExt(input_shape = conv_2_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_1_out_shape = self.pooling_1.out_shape 

        self.conv_3 = MaskedConv2D(filters=128, kernel_size=3, input_shape = pooling_1_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[2], name="conv_3")
        conv_3_out_shape = self.conv_3.out_shape
        # print("conv_third out: ", conv_third_out_shape)
        self.conv_4 = MaskedConv2D(filters=128, kernel_size=3, input_shape = conv_3_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                        sigmoid_multiplier=sigmoid_multiplier[3], name="conv_4")
        conv_4_out_shape = self.conv_4.out_shape
        # print("conv 4 out: ", conv_4_out_shape)
        self.pooling_2 = MaxPool2DExt(input_shape=conv_4_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_2_out_shape = self.pooling_2.out_shape

        self.conv_5 = MaskedConv2D(filters=256, kernel_size=3, input_shape=pooling_2_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_5")
        conv_5_out_shape = self.conv_5.out_shape
        
        self.conv_6 = MaskedConv2D(filters=256, kernel_size=3, input_shape=conv_5_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_6")
        conv_6_out_shape = self.conv_6.out_shape
        
        self.conv_7 = MaskedConv2D(filters=256, kernel_size=3, input_shape=conv_6_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_7")
        conv_7_out_shape = self.conv_7.out_shape

        self.conv_8 = MaskedConv2D(filters=256, kernel_size=3, input_shape=conv_7_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_8")
        conv_8_out_shape = self.conv_8.out_shape

        self.pooling_3 = MaxPool2DExt(input_shape=conv_8_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_3_out_shape = self.pooling_3.out_shape

        self.conv_9 = MaskedConv2D(filters=512, kernel_size=3, input_shape=pooling_3_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_9")
        conv_9_out_shape = self.conv_9.out_shape
        
        self.conv_10 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_9_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_10")
        conv_10_out_shape = self.conv_10.out_shape
        
        self.conv_11 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_10_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_11")
        conv_11_out_shape = self.conv_11.out_shape

        self.conv_12 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_11_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_12")
        conv_12_out_shape = self.conv_12.out_shape

        self.pooling_4 = MaxPool2DExt(input_shape=conv_12_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_4_out_shape = self.pooling_4.out_shape

        self.conv_13 = MaskedConv2D(filters=512, kernel_size=3, input_shape=pooling_4_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[4], name="conv_13")
        conv_13_out_shape = self.conv_13.out_shape
        
        self.conv_14 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_13_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_14")
        conv_14_out_shape = self.conv_14.out_shape
        
        self.conv_15 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_14_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_15")
        conv_15_out_shape = self.conv_15.out_shape

        self.conv_16 = MaskedConv2D(filters=512, kernel_size=3, input_shape=conv_15_out_shape, use_bias=use_bias, k=k_cnn,dynamic_scaling=dynamic_scaling_cnn,
                                       sigmoid_multiplier=sigmoid_multiplier[5], name="conv_16")
        conv_16_out_shape = self.conv_16.out_shape

        self.pooling_5 = MaxPool2DExt(input_shape=conv_16_out_shape, pool_size=(2,2), strides=(2,2))
        pooling_5_out_shape = self.pooling_5.out_shape

        self.flatten = FlattenExt() 
        # print("pooling second out: ", pooling_second_out_shape)
        self.linear_1 = Dense_Mask(int(tf.math.reduce_prod(pooling_5_out_shape[1:]).numpy()),256, 
                                        sigmoid_multiplier=sigmoid_multiplier[6], use_bias=use_bias, name="linear_1", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_2 = Dense_Mask(256,256, sigmoid_multiplier=sigmoid_multiplier[7], use_bias=use_bias, name="linear_2", k=k_dense, dynamic_scaling=dynamic_scaling_dense)
        self.linear_out = Dense_Mask(256,10, sigmoid_multiplier=sigmoid_multiplier[8], use_bias=use_bias, name="linear_out", k=k_dense, dynamic_scaling=dynamic_scaling_dense)


    def activation(self, x):
        return tf.keras.activations.elu(x, alpha=1.67326324)
        # return tf.nn.relu(x)
        # return tf.math.sigmoid(x)


    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.activation(x)
        
        x = self.conv_2(x)
        x = self.activation(x)

        x = self.pooling_1(x)
        
        x = self.conv_3(x)
        x = self.activation(x)
    
        x = self.conv_4(x)
        x = self.activation(x)
    
        x = self.pooling_2(x)

        x = self.conv_5(x)
        x = self.activation(x)
    
        
        x = self.conv_6(x)
        x = self.activation(x)
        x = self.conv_7(x)
        x = self.activation(x)
        x = self.conv_8(x)
        x = self.activation(x)

        x = self.pooling_3(x)

        x = self.conv_9(x)
        x = self.activation(x)
        
        x = self.conv_10(x)
        x = self.activation(x)

        x = self.conv_11(x)
        x = self.activation(x)

        x = self.conv_12(x)
        x = self.activation(x)

        x = self.pooling_4(x)

        x = self.conv_13(x)
        x = self.activation(x)

        x = self.conv_14(x)
        x = self.activation(x)

        x = self.conv_15(x)
        x = self.activation(x)

        x = self.conv_16(x)
        x = self.activation(x)

        x = self.pooling_5(x)
        
        x = self.flatten(x)

        x = self.linear_1(x)
        x = self.activation(x)
        
        x = self.linear_2(x)
        x = self.activation(x)

        x = self.linear_out(x)
        x = tf.nn.softmax(x)

        return x #, layerwise_output



class FCN_Mask4(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4):
        super(FCN_Mask4,self).__init__()
                
        self.linear_in = Dense_Mask(*layer_shapes[0])
        self.linear_h1 = Dense_Mask(*layer_shapes[1])
        self.linear_h2 = Dense_Mask(*layer_shapes[2])
        self.linear_out = Dense_Mask(*layer_shapes[3])
    
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