import tensorflow as tf
import numpy as np
import pickle
from copy import copy
from custom_nn import FCN


class initializer:
    
    def __init__(self):
        print("initializer")

        self.seed = 7531
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
    def initialize_weights(self, dist, shape, mu=0, sigma=1, mu_bi=[0,0], sigma_bi=[0,0], constant=1):
        if dist == "std_normal":
            return np.random.randn(*shape)
        if dist == "bimodal_normal":
            norm = mu + sigma * np.random.randn(*shape)
            norm_pos = np.where(norm >= 0)
            norm_neg = np.where(norm < 0)

            if mu_bi[0] == 0 and mu_bi[1] == 0:
                mu_neg = norm[norm_neg].mean()
                mu_pos = norm[norm_pos].mean()
            else:
                mu_neg = mu_bi[0]
                mu_pos = mu_bi[1]

            if sigma_bi[0] == 0 and sigma_bi[1] == 0:
                sigma_neg = norm[norm_neg].std()
                sigma_pos = norm[norm_pos].std()
            else:
                sigma_neg = sigma_bi[0]
                sigma_pos = sigma_bi[1]
            
            print(f"Positive: mu={mu_pos}, sigma={sigma_pos}")
            print(f"Negative: mu={mu_neg}, sigma={sigma_neg}")

            norm[norm_pos] = mu_pos + sigma_pos * np.random.randn(*norm_pos[0].shape)
            norm[norm_neg] = mu_neg + sigma_neg * np.random.randn(*norm_neg[0].shape)

            return norm
        
        if dist == "uniform":
            if sigma < 0:
                if sigma == -1:
                    sigma = 2.0 / sum(shape)
                if sigma == -2:
                    sigma = 1.0 / shape[0]
                if sigma == -3:
                    sigma = 1.0 / shape[1]
                if sigma == -4:
                    sigma = 2.0 / shape[0]
                if sigma == -5:
                    sigma = 2.0 / shape[1]
                if sigma == -6:
                    sigma = 3.0 / shape[1]
                if sigma == -7:
                    sigma = np.sqrt(3) / np.sqrt(shape[0:])
                #print("sigma: ", sigma)
                print(f"Glorot uniform with bound {sigma:.4f}")
                return np.random.uniform(-sigma, sigma, shape)
            else:
                return np.random.uniform(-mu,mu,shape)

        if dist == "normal":
            if sigma < 0:
                if sigma == -1:
                    sigma = 2.0 / sum(shape)
                if sigma == -2:
                    sigma = 1.0 / shape[0]
                if sigma == -3:
                    sigma = 1.0 / shape[1]
                if sigma == -4:
                    sigma = 2.0 / shape[0]
                if sigma == -5:
                    sigma = 2.0 / shape[1]
                
                print(f"Glorot normal with sigma {sigma:.4f}")
            return mu + sigma*np.random.randn(*shape)
        if dist == "zeros":
            return np.zeros(shape)
        if dist == "ones":
            return np.ones(shape)
        if dist == "constant":
            return np.ones(shape) * constant
        if dist == "signed_constant":
            if sigma == -1:
                sigma = 2.0 / sum(shape)
            if sigma == -2:
                sigma = 1.0 / shape[0]
            if sigma == -3:
                sigma = 1.0 / shape[1]
            if sigma == -4:
                sigma = 2.0 / shape[0]
            if sigma == -5:
                sigma = 2.0 / shape[1]
            if sigma == -6:
                sigma = 3.0 / shape[1]
            if sigma == -7:
                sigma = np.sqrt(3) / np.sqrt(shape[0:])
            
            if constant == -1:
                norm_glorot = mu + sigma*np.random.randn(*shape)

                norm_glorot_pos = norm_glorot[np.where(norm_glorot > 0)]
                norm_glorot_neg = norm_glorot[np.where(norm_glorot < 0)]


                norm_glorot[norm_glorot >= mu] = np.mean(norm_glorot_pos)
                norm_glorot[norm_glorot < mu] = np.mean(norm_glorot_neg)

                print(f"Signed constant (mean): {np.mean(norm_glorot_pos)}")

                return norm_glorot

            elif constant == -2:
                norm = mu + sigma*np.random.randn(*shape)
                norm[norm >= mu] = sigma
                norm[norm < mu] = -sigma

                print(f"Signed constant (std): {sigma}")

                return norm


    def set_weights_man(self, model, layers=None, mode="normal", mu=0, sigma=0.05, constant=1, set_mask=False,
                        mu_bi=[0,0], sigma_bi=[0,0], save_to="", weight_as_constant=False):
        i = 0
        #len_model = len(model.layers)
        initial_weights = []

        if layers == None:
            if weight_as_constant == False:
                for l in model.layers:
                    if l.type == "fefo":
                        W = self.initialize_weights(mode, [l.input_dim, l.units], mu=mu, sigma=sigma,
                                                    mu_bi=mu_bi, sigma_bi=sigma_bi, constant=constant)
                        if set_mask is False:
                            b = self.initialize_weights("ones", [l.units])
                            initial_weights.append([W,b])
                            l.set_weights([W,b])
                        else:
                            initial_weights.append([W])
                            l.set_mask(W)
                            # l.set_weights([W])
                    elif l.type == "conv":
                        W = self.initialize_weights(mode, list(l.weight_shape), mu=mu, sigma=sigma,
                                                    mu_bi=mu_bi, sigma_bi=sigma_bi, constant=constant)
                        if set_mask is False:
                            b = self.initialize_weights("ones", [1,l.filters])
                            initial_weights.append([W,b])
                            l.set_weights([W,b])
                        else:
                            initial_weights.append([W])
                            l.set_mask(W)

                    else:
                        continue
            else:
                for l in model.layers:
                    if l.type == "fefo":
                        W = self.initialize_weights(mode, [l.input_dim, l.units], mu=mu, sigma=sigma,
                                                    mu_bi=mu_bi, sigma_bi=sigma_bi, constant=constant)
                        initial_weights.append(W)
                        l.set_normal_weights(W)
                    elif l.type == "conv":
                        W = self.initialize_weights(mode, list(l.weight_shape), mu=mu, sigma=sigma,
                                                    mu_bi=mu_bi, sigma_bi=sigma_bi, constant=constant)
                        initial_weights.append(W)
                        l.set_normal_weights(W)
                    else:
                        continue
        else:
            for i,l in enumerate(model.layers):
                if i in layers:
                    W = self.initialize_weights(mode, [l.input_dim, l.units],mu=mu, sigma=sigma,
                                                mu_bi=mu_bi, mu_sigma=sigma_bi, constant=constant)
                    b = self.initialize_weights("zeros", [l.units])
                    initial_weights.append([W,b])
                    l.set_weights([W,b])
                else:
                    continue

        if save_to != "":
            with open(save_to+"_"+mode+".pkl", 'wb') as handle:
                pickle.dump(initial_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)


        return model, initial_weights



class iterative_pruning:
    
    def __init__(self, initial_weights, loss_fn, optimizer):

        self.initializer = initializer()

        self.weights_history = []
        self.weights_masked_history = []
        self.weights_nonzero_history = []
        self.weights_pruned_history = []

        self.weights_delta_history = []

        self.bias_history = []

        self.mask_history = []

        self.loss_history = []

        self.initial_weights = initial_weights
        self.weights_history.append([[w[0] for w in self.initial_weights]])
        
        self.loss_fn = loss_fn
        self.opt = optimizer
        
        self.loss_metric = tf.keras.metrics.Mean()
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy()
        
        self.training_acc_history = []
        self.eval_acc_history = []
        
        self.eval_loss_mean = tf.keras.metrics.Mean()
        self.eval_acc = tf.keras.metrics.CategoricalAccuracy()
    
    def reset_training_metrics(self):
        self.loss_metric.reset_states()
        self.acc_metric.reset_states()
    
    def reset_eval_metrics(self):
        self.eval_loss_mean.reset_states()
        self.eval_acc.reset_states()

    def reset_weight_history(self):
        self.weights_history = []
        self.weights_history.append([[w[0] for w in self.initial_weights]])
        self.weights_masked_history = []
        self.weights_nonzero_history = []
        self.weights_pruned_history = []
        self.weights_delta_history = []

        self.bias_history = []
        
    def reset_acc_history(self):
        self.training_acc_history = []
        self.eval_acc_history = []
    
    def reset_weights_init(self,model, initial_weights=[]):
        if len(initial_weights) is 0:
            for i,l in enumerate(model.layers):
                l.set_weights(self.initial_weights[i])
        else:
            for i,l in enumerate(model.layers):
                l.set_weights(initial_weights[i])
    
    def reset_weights_constant(self, model, keep_sign=True, constant="std"):

        for i, l in enumerate(model.layers):
            if constant == "std":
                w_std = tf.math.reduce_std(l.get_nonzero_weights()).numpy()
            elif constant == "mean":
                w_std = tf.math.reduce_mean(l.get_nonzero_weights().numpy())
            elif constant == "var":
                w_std = tf.math.reduce_variance(l.get_nonzero_weights().numpy())
            if keep_sign is False:
                w = np.multiply(np.ones(l.get_weights()[0].shape), w_std)
            else:
                w_pos_idx = np.where(l.get_all_weights().numpy() > 0)
                w_neg_idx = np.where(l.get_all_weights().numpy() < 0)

                w = np.ones(l.get_weights()[0].shape)

                w[w_pos_idx] = w_std
                w[w_neg_idx] = - w_std

            b = self.initial_weights[i][1]

            l.set_weights([w,b])

    def reset_weights_new_init(self, model):
        MU=0
        MU_BI = [-0.13, 0.13]
        SIGMA=-1 #0.1 IF SIGMA == -1 --> glorot normal
        SIGMA_BI = [-SIGMA, SIGMA]
        model, initial_weights = self.initializer.set_weights_man(model, mode="normal", mu=MU, sigma=SIGMA, mu_bi=MU_BI, sigma_bi=SIGMA_BI, save_to="", weight_as_constant=False)
        #self.initial_weights.append(initial_weights) 
        return model

    def reset_masks(self, model):
        for layer in model.layers:
            layer.reset_mask()
    
    def layerwise_pruning_p(self,model):
        for i,l in enumerate(model.layers):
            total = l.count_params() - l.units
            masked = tf.math.count_nonzero(l.get_masked_weights())
            print(f"{((total-masked)/total) * 100:.3f}% of Layer {i} weights are pruned")
            
    def get_p_smallest_value(self, weights, p, absolute=True):
        w_shape = weights.shape[0]

        if absolute:
            weights_sorted = np.sort(np.absolute(weights))
            weights_unique = np.unique(weights_sorted)

            #weights_sorted_zeros_idx = np.where(weights_sorted == 0.)[0]
            #weights_sorted_nonzero_idx = np.where(weights_sorted != 0.)[0]

            print(f"p-smallest value is (with -2): {weights_unique[int(weights_unique.shape[0]*p)-1]:.3f}.",
            "This means, that weights did either not move more than this value or",
            "the weight itself is smaller than the value (after the last epoch per iteration)")

            return weights_unique[int(weights_unique.shape[0]*p)-2]#weights_sorted[int(weights_sorted_zeros_idx.shape[0] + weights_sorted_nonzero_idx.shape[0]*p)]
    
    def get_same_sign(self, weights, initial_weights):

        w_shape = weights.shape[0]

        weights_idx_pos = np.where(weights > 0)
        weights_idx_neg = np.where(weights < 0)

        initial_weights_idx_pos = np.where(initial_weights > 0)
        initial_weights_idx_neg = np.where(initial_weights < 0)

        pos_intersection = np.intersect1d(weights_idx_pos, initial_weights_idx_pos)
        neg_intersection = np.intersect1d(weights_idx_neg, initial_weights_idx_neg)
        pos_neg_intersection = np.concatenate((pos_intersection, neg_intersection))

        return np.sort(pos_neg_intersection)


    
    def get_p_largest_value(self, weights, p, absolute=True):

        w_shape = weights.shape[0]

        if absolute:
            weights_sorted = np.sort(np.absolute(weights))
            weights_unique = np.unique(weights_sorted)[::-1]

            return weights_unique[int(weights_unique.shape[0]*p)]
            
            
    def mask_globally_magnitude_high(self, model, th=0.5):
        shape_info = [l.weights[0].shape for l in model.layers]
        all_weights_flat = np.concatenate([l.get_masked_weights().numpy().flatten() for l in model.layers])

        mask = np.ones(all_weights_flat.shape)

        p_val = self.get_p_smallest_value(all_weights_flat, p=th)

        all_weights_smaller_p = np.where(np.absolute(all_weights_flat) < p_val)[0]

        mask[all_weights_smaller_p] = 0
    
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model
        
    def mask_globally_magnitude_high(self, model, th=0.5):
        shape_info = [l.weights[0].shape for l in model.layers]
        all_weights_flat = np.concatenate([l.get_masked_weights().numpy().flatten() for l in model.layers])

        mask = np.ones(all_weights_flat.shape)

        p_val = self.get_p_smallest_value(all_weights_flat, p=th)

        all_weights_smaller_p = np.where(np.absolute(all_weights_flat) < p_val)[0]

        mask[all_weights_smaller_p] = 0
    
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model


    def mask_globally_delta_high(self, model, th):
        shape_info = [l.weights[0].shape for l in model.layers]
        all_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][-1]])
        initial_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][0]])

        delta_weights = np.array(initial_weights_flat - all_weights_flat)

        self.weights_delta_history.append(delta_weights)

        delta_weights = np.absolute(delta_weights)
        

        mask = np.ones(all_weights_flat.shape)

        p_val = self.get_p_smallest_value(delta_weights, p=th)

        delta_weights_smaller_p = np.where(delta_weights < p_val)[0]

        pruned_percentage = delta_weights_smaller_p.shape[0] / delta_weights.shape[0] 
        
        print(f"{pruned_percentage*100:.2f}% of the total weights are 0-masked i.e. they are smaller than the p_val")

        mask[delta_weights_smaller_p] = 0
 
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model
    
    def mask_globally_delta_low(self, model, th):
        shape_info = [l.weights[0].shape for l in model.layers]
        all_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][-1]])
        initial_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][0]])

        delta_weights = np.array(initial_weights_flat - all_weights_flat)

        self.weights_delta_history.append(delta_weights)

        delta_weights = np.absolute(delta_weights)
        

        mask = np.ones(all_weights_flat.shape)

        p_val = self.get_p_largest_value(delta_weights, p=th)

        delta_weights_larger_p = np.where(delta_weights > p_val)[0]

        pruned_percentage = delta_weights_larger_p.shape[0] / delta_weights.shape[0] 
        
        print(f"{pruned_percentage*100:.2f}% of the total weights are 0-masked i.e. they are smaller than the p_val")

        mask[delta_weights_larger_p] = 0
 
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model

    def mask_globally_delta_high_same_sign(self, model, th):
        shape_info = [l.weights[0].shape for l in model.layers]
        all_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][-1]])
        initial_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][0]])

        delta_weights = np.array(initial_weights_flat - all_weights_flat)

        self.weights_delta_history.append(delta_weights)
        delta_weights = np.absolute(delta_weights)

        same_sign_idx = self.get_same_sign(all_weights_flat, initial_weights_flat)

        mask = np.zeros(all_weights_flat.shape)

        mask[same_sign_idx] = 1

        p_val = self.get_p_smallest_value(delta_weights, p=th)

        delta_weights_smaller_p = np.where(delta_weights < p_val)[0]

        pruned_percentage = delta_weights_smaller_p.shape[0] / delta_weights.shape[0] 
        
        print(f"{pruned_percentage*100:.2f}% of the total weights are 0-masked i.e. they are smaller than the p_val")

        mask[delta_weights_smaller_p] = 0
 
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model

    def mask_globally_magnitude_high_same_sign(self, model, th):
        
        shape_info = [l.weights[0].shape for l in model.layers]
        
        all_weights_flat = np.concatenate([w.flatten() for w in self.weights_masked_history[-1][-1]])
        initial_weights_flat = np.concatenate([init_w[0].flatten() for init_w in self.initial_weights])

        mask = np.zeros(all_weights_flat.shape)
    
        same_sign_idx = self.get_same_sign(all_weights_flat, initial_weights_flat)

        mask[same_sign_idx] = 1

        p_val = self.get_p_smallest_value(all_weights_flat, p=th)

        all_weights_smaller_p = np.where(np.absolute(all_weights_flat) < p_val)[0]

        mask[all_weights_smaller_p] = 0
    
        for i,s in enumerate(shape_info):
            #print(i,s)
            #print("Mask shape: ", mask.shape)
            old_mask = model.layers[i].get_mask()
            layer_mask = mask[:s[0]*s[1]].reshape(s)
            layer_mask = np.logical_and(old_mask,layer_mask).astype("float32")
            #print(layer_mask.shape)
            model.layers[i].set_mask(layer_mask)
            mask = mask[s[0]*s[1]:]

        return model

    def mask_layerwise(self,model, th=0.2):
        """Applies a mask to each layer in the netwok
        
        Arguments:
            model {tensorflow model} -- [NN to be pruned]
        
        Keyword Arguments:
            th {float} -- [Threshold that will determine which weights are to be pruned] (default: {0.2})
        
        Returns:
            [tensorflow model] -- [returns the model. Not really necessary]
        """
        for l in model.layers:
            weights_shape = l.get_masked_weights().shape
            weights_flat = l.get_masked_weights().numpy().flatten()

            old_mask = l.get_mask()

            mask = np.ones(weights_shape)
            idx_mask_flat = get_margin_min_weights(weights_flat, th)

            idx_mask_unrav = np.unravel_index(idx_mask_flat, weights_shape)
            idx_mask_orig = [list(i) for i in zip(idx_mask_unrav[0], idx_mask_unrav[1])]


            mask[idx_mask_unrav] = 0.
            mask = np.logical_and(old_mask,mask).astype("float32")
            l.set_mask(mask)
            
        return model
    
    def prune_weights(self, model, pruning_th=0.2, pruning_scope="global", pruning_method="magnitude"):
        """Function that calls the proper pruning method function, depending on the parameters
        
        Arguments:
            model {tensorflow model} -- Model that is to be pruned
        
        Keyword Arguments:
            pruning_th {float} -- Threshold that determines which weights are to be pruned (default: {0.2})
            pruning_scope {str} -- Either local (i.e. prune layerwise) or global (flatten weights and then apply threshold) (default: {"global"})
            pruning_method {str} -- Pruning method, i.e. determines the metric of pruning: magnitude or delta (default: {"magnitude"})
        
        Raises:
            NotImplementedError: only global and local pruning implemented
        """
        if pruning_scope == "global":
            if pruning_method == "magnitude":
                self.mask_globally_magnitude_high(model, pruning_th)
            elif pruning_method == "magnitude_same_sign":
                self.mask_globally_magnitude_high_same_sign(model, pruning_th)
            elif pruning_method == "magnitude_init_high":
                self.mask_globally_magnitude_init_high(model, pruning_th)
            elif pruning_method == "delta_high":
                self.mask_globally_delta_high(model, pruning_th)
            elif pruning_method == "delta_high_same_sign":
                self.mask_globally_delta_high_same_sign(model, pruning_th)
            elif pruning_method == "delta_low":
                self.mask_globally_delta_low(model, pruning_th)
        elif pruning_scope == "local":
            self.mask_layerwise(model, pruning_th)
        else:
            print("Pruning Method not implemented, yet.")
            raise NotImplementedError
    
    def burn_in(self, model, dataset_train, epochs, logging=False):
        model, _,_,_,_ = self.train_model(model, dataset_train, epochs, logging)
        return model
    
    def get_zero_rows(self, model):
        rows_to_be_deleted = []
        rows_to_be_kept = []
        for i,layer in enumerate(model.layers):
            a = layer.get_masked_weights().numpy()
            row_delete = np.where(~a.any(axis=1))[0]
            row_keep = np.where(a.any(axis=1))[0]
            rows_to_be_deleted.append(row_delete)
            rows_to_be_kept.append(row_keep)

            col = np.where(~a.any(axis=0))[0]


            print(f"Layer {i}: {row_delete.shape[0]} Rows can be deleted and {col.shape[0]} Columns")
        
        return rows_to_be_deleted, rows_to_be_kept

    def get_pruned_initial_weights(self, old_initial_weights, rows_to_be_kept, keep_input=True):
        new_initial_weights = []
        for i in range(len(rows_to_be_kept)):
            #print(i,"/",len(rows_to_be_kept))
            if i is 0:
                # do not delete any input neurons, yet
                if keep_input is True:
                    w = old_initial_weights[i][0][:, rows_to_be_kept[1]]
                else:
                    w = old_initial_weights[i][0][rows_to_be_kept[i][:,None],rows_to_be_kept[i+1]]
                b = old_initial_weights[i][1][:rows_to_be_kept[i+1].shape[0]]
            elif i is len(rows_to_be_kept)-1:
                w = old_initial_weights[i][0][rows_to_be_kept[i],:]
                b = old_initial_weights[i][1]
            else:
                w = old_initial_weights[i][0][rows_to_be_kept[i][:,None],rows_to_be_kept[i+1]]
                b = old_initial_weights[i][1][:rows_to_be_kept[i+1].shape[0]]


            wb = [w,b]
            new_initial_weights.append(wb)
            
        return new_initial_weights


    def get_new_shape(self, input_dim, rows_to_be_kept, keep_input=True):
        new_shape = []
        for i in range(len(rows_to_be_kept)):
            if i == 0:
                if keep_input is True:
                    tmp = [input_dim, rows_to_be_kept[1].shape[0]]
                else:
                    tmp = [rows_to_be_kept[i].shape[0], rows_to_be_kept[i+1].shape[0]]
            elif i == len(rows_to_be_kept)-1:
                tmp = [rows_to_be_kept[i].shape[0], 10]
            else:
                tmp = [rows_to_be_kept[i].shape[0], rows_to_be_kept[i+1].shape[0]]
            new_shape.append(tmp)
        return new_shape

    def del_input_dimensions(self, image, label, rows_to_be_kept):
        return tf.gather(image, rows_to_be_kept[0], axis=1), label


    def start(self, model, dataset_train, iterations, epochs, dataset_test=None, pruning_scope="global", pruning_method="magnitude", pruning_th=0.2, 
              prune_neurons=False, prune_neuron_iterations=5, keep_input=True, logging=True, reset_masks=False, mask_1_action="re_init", mask_1_action_constant="std", reset_before=False):
        
        if len(self.weights_masked_history) == 0:
            self.weights_masked_history.append([l.get_masked_weights().numpy() for l in model.layers])
        if len(self.weights_nonzero_history) == 0:
            self.weights_nonzero_history.append([l.get_nonzero_weights().numpy() for l in model.layers])
        if len(self.weights_pruned_history) == 0:
            self.weights_pruned_history.append([l.get_pruned_weights().numpy() for l in model.layers])
        if len(self.mask_history) == 0:
            self.mask_history.append([l.get_mask().numpy() for l in model.layers])

        if prune_neurons is False:
            for iteration in range(iterations):
                
                print(f"--- Iteration {iteration+1}/{iterations} started ---")

                if reset_before is True:
                
                    if mask_1_action == "re_init":
                        self.reset_weights_init(model)
                    elif mask_1_action == "new_init":
                        self.reset_weights_new_init(model)
                    elif mask_1_action == "re_shuffle":
                        continue
                    elif mask_1_action == "constant":
                        self.reset_weights_constant(model, constant=mask_1_action_constant)

                model, iter_weights, iter_weights_masked, iter_weights_nonzero, iter_weights_pruned, iter_bias, iter_loss = self.train_model(model, dataset_train, epochs, logging)

                if dataset_test is not None:
                    print("Evaluate before applying a pruning step:") 
                    self.evaluate(model, dataset_test, single_mode=False)

                self.loss_history.append(iter_loss)
                
                self.weights_history.append(iter_weights)
                self.weights_masked_history.append(iter_weights_masked)
                self.weights_nonzero_history.append(iter_weights_nonzero)
                self.weights_pruned_history.append(iter_weights_pruned)

                self.bias_history.append(iter_bias)
                
                self.prune_weights(model, pruning_method=pruning_method, pruning_scope=pruning_scope, pruning_th=pruning_th)

                self.mask_history.append([l.get_mask().numpy() for l in model.layers])


                if pruning_scope == "global":
                    self.layerwise_pruning_p(model)

                if reset_masks is True:
                    self.reset_masks(model)
                
                if reset_before is False:
                    if mask_1_action == "re_init":
                        self.reset_weights_init(model)
                    elif mask_1_action == "new_init":
                        self.reset_weights_new_init(model)
                    elif mask_1_action == "re_shuffle":
                        continue
                    elif mask_1_action == "constant":
                        self.reset_weights_constant(model, constant=mask_1_action_constant)
            
            return model

        else:

            self.initial_weight_history = []

            self.initial_weight_history.append(self.initial_weights)

            input_dim = self.initial_weights[0][0].shape[0]

            new_initial_weights = self.initial_weights

            smaller_model = copy(model)
            
            all_trainable_vars_original = tf.reduce_sum([tf.reduce_prod(v.shape) for v in model.trainable_variables]).numpy()

            for neuron_iteration in range(prune_neuron_iterations):

                self.reset_masks(smaller_model)

                for iteration in range(iterations):
                    
                    print(f"--- Iteration {iteration+1}/{iterations} started ---")
                    
                    self.reset_weights_init(smaller_model, new_initial_weights)
                    smaller_model, iter_weights, iter_weights_masked, iter_weights_nonzero, iter_weights_pruned, iter_bias, iter_loss = self.train_model(smaller_model, dataset_train, epochs, logging)

                    if dataset_test is not None:
                        print("Evaluate before applying a pruning step:") 
                        self.evaluate(smaller_model, dataset_test, single_mode=False)
                    
                    self.loss_history.append(iter_loss)
                    
                    self.weights_history.append(iter_weights)
                    self.weights_masked_history.append(iter_weights_masked)
                    self.weights_nonzero_history.append(iter_weights_nonzero)
                    self.weights_pruned_history.append(iter_weights_pruned)

                    self.bias_history.append(iter_bias)
                    
                    self.prune_weights(smaller_model, pruning_method=pruning_method, pruning_scope=pruning_scope, pruning_th=pruning_th)

                    self.mask_history.append([l.get_mask().numpy() for l in model.layers])

                    if pruning_scope == "global":
                        self.layerwise_pruning_p(smaller_model)

                print("--------- Neuron Pruning ---------")

                rows_to_be_deleted, rows_to_be_kept = self.get_zero_rows(smaller_model)
                new_initial_weights = self.get_pruned_initial_weights(new_initial_weights, rows_to_be_kept, keep_input=keep_input)


                self.initial_weight_history.append(new_initial_weights)

                #if keep_input is False:
                new_shape = self.get_new_shape(input_dim, rows_to_be_kept, keep_input=keep_input)
                print(f"New Shape: {new_shape}")

                if keep_input is True:
                    smaller_model = FCN(input_dim, no_layers=4, layer_shapes=new_shape)
                else:
                    smaller_model = FCN(rows_to_be_kept[0].shape, no_layers=4, layer_shapes=new_shape)
                    
                    lambda_del_input_dimensions = lambda image, label: self.del_input_dimensions(image, label, rows_to_be_kept)

                    dataset_train = dataset_train.map(lambda_del_input_dimensions)
                    dataset_test = dataset_test.map(lambda_del_input_dimensions)

                for i,l in enumerate(smaller_model.layers):
                    l.set_weights(new_initial_weights[i])
                    l.reset_mask()

                all_trainable_vars_small = tf.reduce_sum([tf.reduce_prod(v.shape) for v in smaller_model.trainable_variables]).numpy()

                print(f"Out of a total of {all_trainable_vars_original} weights, {all_trainable_vars_small} are left, which is {(all_trainable_vars_small/all_trainable_vars_original)*100:.2f}% of the original weights")
            
            return smaller_model
            #if dataset_test is not None:
            #    print("Evaluate after applying a pruning step:") 
            #    self.evaluate(model, dataset_test, single_mode=True)

            #if dataset_test is not None:
            #    print("Evaluate after applying a pruning step and resetting weights to Init:") 
            #    self.evaluate(model, dataset_test, single_mode=True)
    
    def train_model(self, model, dataset,epochs, logging=False):
        
        iter_weights = [] #+ [w[0] for w in initial_weights]
        iter_weights_masked = []
        iter_weights_pruned = []
        iter_weights_nonzero = []

        iter_loss = []

        iter_bias = []
        
        iter_weights.append([w[0] for w in self.initial_weights])
        iter_weights_masked.append([l.get_masked_weights().numpy() for l in model.layers])
        iter_weights_nonzero.append([l.get_nonzero_weights().numpy() for l in model.layers])
        iter_weights_pruned.append([l.get_pruned_weights().numpy() for l in model.layers])

        iter_bias.append([w[1] for w in self.initial_weights])
    
        
        # Iterate over epochs.
        for epoch in range(epochs):

            print(f"\t-- Start of epoch {epoch+1}/{epochs} --")

            # Iterate over the batches of the dataset.
            for step, (x_batch_train,y_batch_train) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predicted = model(x_batch_train)
                    # Compute reconstruction loss
                    loss = self.loss_fn(y_batch_train, predicted)
                    #loss += sum(fcn.losses)

                grads = tape.gradient(loss, model.trainable_weights)
                self.opt.apply_gradients(zip(grads, model.trainable_weights))

                self.loss_metric(loss)
                self.acc_metric(y_batch_train,predicted)
                
            

                #if step % 100 == 0:
            iter_weights.append([w.numpy() for w in model.trainable_weights[::2]])
            iter_weights_masked.append([l.get_masked_weights().numpy() for l in model.layers])
            iter_weights_nonzero.append([l.get_nonzero_weights().numpy() for l in model.layers])
            iter_weights_pruned.append([l.get_pruned_weights().numpy() for l in model.layers])

            iter_bias.append([l.get_bias() for l in model.layers])

            iter_loss.append(self.loss_metric.result().numpy())

                
            if logging:
                print('\tAccuracy = %s --- Loss = %s' % (self.acc_metric.result().numpy(),self.loss_metric.result().numpy()))
        

        #iter_weights_delta = #[iter_weights_masked[-1][i].flatten() - iter_weights_masked[0][i].flatten() for i in range(len(iter_weights_masked[len(model.layers)]))]
        
        self.training_acc_history.append(self.acc_metric.result().numpy())

        self.reset_training_metrics()
        
        return model, iter_weights, iter_weights_masked, iter_weights_nonzero, iter_weights_pruned, iter_bias, iter_loss#, iter_weights_delta
    
    def evaluate(self,model,dataset_test, single_mode=False):
        for x_batch_test, y_batch_test in dataset_test:
            test_pred = model(x_batch_test)
            test_loss = self.loss_fn(y_batch_test, test_pred)

            self.eval_loss_mean(test_loss)
            self.eval_acc(y_batch_test, test_pred)
        if single_mode == False:
            self.eval_acc_history.append(self.eval_acc.result().numpy())
        print(f"\t\tTest Loss: {self.eval_loss_mean.result().numpy()}")
        print(f"\t\tTest Accuracy: {self.eval_acc.result().numpy()}")
        self.reset_eval_metrics()
            
    def test_pruning_consistency(self, model, test_loops=10, epochs=10):

        loss_metric = tf.keras.metrics.Mean()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()

        model = model.copy()

        for i,tl in enumerate(test_loops):

            for layer in model.layers:
                layer.reset_mask()

            for epoch in range(epochs):
                continue



    
    def get_weight_history(self):
        return self.weights_history, self.weights_masked_history, self.weights_nonzero_history, self.weights_pruned_history, self.weights_delta_history

    def dump_weight_history(self, path):
        all_w = [self.weights_history, self.weights_masked_history, self.weights_nonzero_history]

        with open(path, 'wb') as handle:
            pickle.dump(all_w, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_acc_history(self):
        return self.training_acc_history, self.eval_acc_history

    def dump_acc_history(self, path):
        all_acc = [self.training_acc_history, self.eval_acc_history]

        with open(path, 'wb') as handle:
            pickle.dump(all_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)