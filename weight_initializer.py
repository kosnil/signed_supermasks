
from typing import Tuple
import tensorflow as tf
import numpy as np
import pickle
import pickletools
from copy import copy

class initializer:
    """Use this class to initialize weights of some tensorflow/keras model
    """
    
    def __init__(self, seed=7531):
        print("initializer")

        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
    def get_fans(self, shape: tuple) -> Tuple[int, int]:
        """Get fan_in, fan_out of a layer with given shape

        Args:
            shape (tuple): shape of layer

        Returns:
            Tuple[int, int]: fan_in, fan_out of given shape
        """
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        return fan_in, fan_out

        
    def initialize_weights(self, 
                           dist: str, 
                           shape: tuple, 
                           method: str, 
                           factor=1.) -> np.ndarray:
        """Initializes weights for a given shape

        Args:
            dist (str): distribution from which the weight values are drawn
            shape (tuple): shape of weights to be initialized
            method (str): either xavier or he. if elus/scaled elus is required, use he in combination with factor
            factor ([type], optional): constant by which initialized weights are multiplied with. Defaults to 1..
            
            
        Returns:
            np.ndarray: initialized weights in given shape
        """
        if dist == "std_normal":
            return np.random.randn(*shape)
        
        if dist == "uniform":
                
            fan_in, fan_out = self.get_fans(shape)
            
            bound = 0.
            
            if method == "xavier": 
                bound = np.sqrt(2) / np.sqrt(fan_in + fan_out)
            if method == "he":
                bound = np.sqrt(2) / np.sqrt(fan_in)
            
            bound *= factor

            return np.random.uniform(-bound, bound, shape)
        
        if dist == "normal":

            fan_in, fan_out = self.get_fans(shape)
            
            if method == "xavier": 
                sigma = np.sqrt(2) / np.sqrt(fan_in + fan_out)
            elif method == "he":
                sigma = np.sqrt(2) / np.sqrt(fan_in)

            sigma *= factor

            return  sigma*np.random.randn(*shape) # always assume mu = 0
        if dist == "zeros":
            return np.zeros(shape)
        if dist == "ones":
            print("Ones")
            return np.ones(shape)
        if dist == "constant":

            fan_in, fan_out = self.get_fans(shape)

            c = 0.

            if method == "xavier":
                c = np.sqrt(2) / np.sqrt(fan_in + fan_out)
            elif method == "he":
                c = np.sqrt(2) / np.sqrt(fan_in)

            c *= factor

            
            return np.ones(shape) * c

        if dist == "signed_constant":
            
            fan_in, fan_out = self.get_fans(shape)
            

            c = 0.

            if method == "xavier": 
                c = np.sqrt(2) / np.sqrt(fan_in + fan_out)
            elif method == "he":
                c = np.sqrt(2) / np.sqrt(fan_in)
            
            c *= factor


            norm = c*np.random.randn(*shape)
            norm[norm >= 0] = c
            norm[norm < 0] = -c

            return norm


    def set_weights_man(self, 
                        model: tf.keras.Model, 
                        layers=None, 
                        dist="normal", 
                        method="xavier",
                        factor=1., 
                        set_mask=False,
                        save_to="", 
                        save_suffix="", 
                        weight_as_constant=False, 
                        layer_shapes=None):
        """Set weights of a given tf model manually (in contrast to letting tensorflow/keras set the weights)

        Args:
            model (tf.keras.Model): model for which weights need to be set
            layers (list, optional): specify specific layers of model whose weights need to be set. If layers != None 
            those layers that are not contained in this list will be ignored. Defaults to None.
            mode (str, optional): distribution of weight initialization. Defaults to "normal".
            mu (int, optional): mean of distribution. Defaults to 0.
            sigma (float, optional): standard deviation of distribution. Defaults to 0.05.
            factor (float, optional): constant initialized weights get multiplied with. Defaults to 1..
            constant (int, optional): [description]. Defaults to 1.
            set_mask (bool, optional): if True, function will set mask weights. If False, function will set "weight weights". Defaults to False.
            mu_bi (list, optional): if mode == "bimodal_normal" mu_bi specifies the distribution. Otherwise ignored. Defaults to [0,0].
            sigma_bi (list, optional): if mode == "bimodal_normal" sigmi_bi specifies the distribution. 
                                       Otherwise ignored. Defaults to [0,0].
            save_to (str, optional): if weights should be saved to a file, specify file path here. 
                                    If not specified, weights will not be saved. Defaults to "".
            save_suffix (str, optional): optional suffix to filename. Defaults to "".
            weight_as_constant (bool, optional): specifies whether the weights should be set to frozen. Defaults to False.
            layer_shapes (list, optional): list of layer shapes. Defaults to None.

        Returns:
            model (tf.keras.Model): model with newly initialized weights
            initialized_weights (list): list of initialized weights
        """
        #i = 0

        initial_weights = []

        if layers == None:
            if weight_as_constant == False:
                layer_shape_counter = 0
                for l in model.layers:
                    
                    if l.type == "fefo":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    shape=[l.input_dim, l.units], 
                                                    factor=factor)
                        if set_mask is False:
                            b = self.initialize_weights("ones", [l.units])
                            initial_weights.append([W,b])
                            # l.set_weights([W,b])
                            # l.set_weights([W])
                            l.set_normal_weights(W)
                            l.trainable = False
                        else:
                            initial_weights.append([W])
                            l.set_mask(W)
                            # l.set_weights([W])
                    elif l.type == "conv":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    shape=list(l.weight_shape),
                                                    factor=factor)
                        if set_mask is False:
                            b = self.initialize_weights("ones", [1,l.filters])
                            initial_weights.append([W,b])
                            # l.set_weights([W,b])
                            # l.set_weights([W])
                            l.set_normal_weights(W)
                            l.trainable = False


                        else:
                            initial_weights.append([W])
                            l.set_mask(W)
                    elif l.type == "fefo_normal":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    dist=layer_shapes[layer_shape_counter],
                                                    factor=factor)
                        l.set_weights([W])
                        layer_shape_counter += 1
                    elif l.type == "conv_normal":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    dist=layer_shapes[layer_shape_counter],
                                                    factor=factor)
                        l.set_weights([W])
                        layer_shape_counter += 1

                    else:
                        continue
            else:
                for l in model.layers:
                    if l.type == "fefo":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    shape=[l.input_dim, l.units],
                                                    factor=factor)
                        initial_weights.append(W)
                        l.set_normal_weights(W)
                    elif l.type == "conv":
                        W = self.initialize_weights(dist=dist, 
                                                    method=method,
                                                    shape=list(l.weight_shape),
                                                    factor=factor)
                        initial_weights.append(W)
                        l.set_normal_weights(W)
                    else:
                        continue
        else:
            layer_counter = 0
            for l in model.layers:
                if l.type is "fefo" or l.type is "conv":
                    W = layers[layer_counter]
                    l.set_weights([W])
                    layer_counter += 1
                # elif l.type is "conv":
                #     W = layer
                else:
                    continue

        if save_to != "":

            with open(save_to+save_suffix+".pkl", 'wb') as handle:
                pickled = pickle.dumps(initial_weights)
                optimized_pickle = pickletools.optimize(pickled)
                handle.write(optimized_pickle)


        return model, initial_weights

    def set_loaded_weights(self, 
                           model: tf.keras.Model, 
                           path:str) -> tf.keras.Model:
        """Sets weights of a specified model from a file

        Args:
            model (tf.keras.Model): model for which the weights need to be specified
            path (str): path to file which holds the weights

        Returns:
            tf.keras.Model: model with set weights
        """

        with open(path, "rb") as f:
            p = pickle.Unpickler(f)
            initial_weights = p.load()
        
        mask_flag = True if "mask" in path else False
        
        layer_counter = 0
        mask_layers = ["fefo", "conv"]
        normal_layers = ["fefo_normal", "conv_normal"]

        for layer in model.layers:
            if layer.type in mask_layers:
                if mask_flag:
                    layer.set_mask(initial_weights[layer_counter][0])
                else:
                    layer.set_normal_weights(initial_weights[layer_counter][0])
                layer_counter += 1
                
            elif layer.type in normal_layers:
                layer.set_weights([initial_weights[layer_counter]])
                layer_counter += 1
        
        return model