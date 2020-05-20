import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

class Linear(layers.Layer):
    
    def __init__(self,input_dim=32,units=32, mask=False, mask_matrix=None):
        super(Linear,self).__init__()
        
        
        self.units = units
        self.input_dim = input_dim
        
        if mask:
            if mask_matrix == None:
                #select random mask
                self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
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
    
    def __init__(self,input_dim=32,units=32, mask=False, mask_matrix=None):
        super(Linear_Mask,self).__init__()
        
        
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
        
        init_mask = tf.random_normal_initializer()
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()
        
        #self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True)
        self.mask = tf.Variable(initial_value=init_mask(shape=(input_dim, units), dtype="float32"), trainable=True)
        self.threshold = 0.5 #tf.Variable(initial_value = 0.5, dtype=tf.float32, trainable=True)
        
        self.bernoulli_mask = self.mask
        
        self.multiplier = 1.
        self.no_ones = 0.
        
        #self.sigmoid01 = lambda x: 1 if tf.math.sigmoid(x) >= 0.5 else 0
        
        #self.sig_mask = tf.Variable(initial_value=self.sigmoid01(self.mask), trainable=False)
        
        self.w = tf.constant(init_w(shape=(input_dim, units),dtype='float32'))
        self.b = tf.constant(init_b(shape=(units,),dtype='float32'))
    
    def set_mask_rand(self):
        self.mask = tf.constant(np.random.randint(2, size=(input_dim, units)).astype("float32"))
    
    def get_shape(self):
        return self.shape
    
    def set_mask(self,mask):
        self.mask = tf.Variable(tf.cast(mask, "float32"))
        
        self.sigmoid_mask()
        #self.mask = tf.Variable(mask.astype("float32"))
    
    def get_mask(self, as_logit=False):
        if as_logit is True:
            return self.mask
        else:
            return tf.math.sigmoid(self.mask)
        
    def update_bernoulli_mask(self, mask=None):
        no_samples = 1
        accept_as_one_th = 1
        relaxation = tf.cast(no_samples - no_samples*accept_as_one_th, tf.int32)
        #return tf.cast(tfp.distributions.RelaxedBernoulli(temperature=0.001, probs=tf.math.sigmoid(self.mask)).sample(), dtype=tf.float32)
        sig_mask = tf.math.sigmoid(self.mask)
        bernoulli_samples = tfp.distributions.Bernoulli(probs=sig_mask).sample(no_samples)
        bernoulli_sample_mask = tf.math.floordiv(tf.math.add(tf.math.reduce_sum(bernoulli_samples,axis=0), relaxation), no_samples)  #(tf.math.reduce_sum(a, axis=0)+30) // 100
        self.bernoulli_mask = tf.cast(bernoulli_sample_mask, dtype=tf.float32) + sig_mask - tf.stop_gradient(sig_mask)
        return self.bernoulli_mask
        #bernoulli_mask_lambda_fn = lambda x: tf.cast(tfp.distributions.Bernoulli(probs=x).sample(sample_shape=x.shape), "float32")
        #bernoulli_mask = tf.map_fn(bernoulli_mask_lambda_fn, self.mask)
        #print(bernoulli_mask.numpy().shape)
        #bernoulli_mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.math.sigmoid(self.mask)).sample(sample_shape=1), "float32")
        #return tf.reshape(bernoulli_mask, self.mask.shape)
    
    def sigmoid_mask(self):
        sigmoid_mask = tf.math.sigmoid(self.mask)
        effective_mask = tf.where(sigmoid_mask >= self.threshold, 1, sigmoid_mask)
        effective_mask = tf.where(effective_mask < self.threshold, 0, effective_mask)
        
        self.bernoulli_mask = effective_mask
        #sig_mask[sig_mask >= 0.5] = 1.
        #sig_mask[sig_mask < 0.5] = 0.
        return effective_mask + sigmoid_mask - tf.stop_gradient(sigmoid_mask) # + tf.nn.relu(self.mask) - tf.stop_gradient(tf.nn.relu(self.mask))
    
    def get_normal_weights(self):
        return self.w
    
    def set_normal_weights(self, w):
        self.w = tf.constant(w.astype("float32"))
    
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

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        #if self.dynamic_scaling:
        
        sig_mask = self.sigmoid_mask()
        
        weights_masked = tf.multiply(self.w, sig_mask)
        
        
        self.no_ones = tf.reduce_sum(sig_mask)
        self.multiplier = tf.math.divide(tf.size(sig_mask, out_type=tf.float32), self.no_ones)
        
        #print(f"1-masked weights multiplied by: {multiplier:.4f}")
        
        weights_masked = tf.multiply(self.multiplier, weights_masked)
        #mask_fn = lambda x: 1 if tf.math.sigmoid(x) >= 0.5 else 0
        #mask = tf.map_fn(lambda x: mask_fn(x), self.mask, back_prop=True)
        #w_mask = tf.multiply(self.w, self.sigmoid_mask())
        #w_mask = tf.multiply(self.w, tf.math.round(self.sigmoid_mask()))
        return tf.matmul(inputs, weights_masked) #+ self.b
    
    
class FCN(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4):
        super(FCN,self).__init__()
                
        self.linear_in = Linear(*layer_shapes[0])
        self.linear_h1 = Linear(*layer_shapes[1])
        self.linear_out = Linear(*layer_shapes[2])
    
    def call(self, inputs):
        
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        x = self.linear_h1(x)
        x = tf.nn.relu(x) 
        x = self.linear_out(x)
        return tf.nn.softmax(x)
    
class FCN_Mask(tf.keras.Model):
    
    def __init__(self, input_dim, layer_shapes, no_layers=4):
        super(FCN_Mask,self).__init__()
                
        self.linear_in = Linear_Mask(*layer_shapes[0])
        self.linear_h1 = Linear_Mask(*layer_shapes[1])
        self.linear_out = Linear_Mask(*layer_shapes[2])
    
    def call(self, inputs):
        
        x = self.linear_in(inputs)
        x = tf.nn.relu(x)
        x = self.linear_h1(x)
        x = tf.nn.relu(x) 
        x = self.linear_out(x)
        return tf.nn.softmax(x)
    
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