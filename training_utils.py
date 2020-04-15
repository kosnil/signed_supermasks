import tensorflow as tf
import numpy as np


class initializer:
    
    def __init__(self):
        print("initializer")
        
    def initialize_weights(self, dist, shape, mu=0, sigma=1):
        if dist == "std_normal":
            return np.random.randn(*shape)
        if dist == "normal":
            return mu + sigma*np.random.randn(*shape)
        if dist == "zeros":
            return np.zeros(shape)
        if dist == "ones":
            return np.ones(shape)

    def set_weights_man(self, model, layers=None):
        i = 0
        len_model = len(model.layers)
        initial_weights = []

        if layers == None:
            for l in model.layers:
                W = self.initialize_weights("normal", [l.input_dim, l.units], sigma=0.05)
                b = self.initialize_weights("zeros", [l.units])
                initial_weights.append([W,b])
                l.set_weights([W,b])
        else:
            for i,l in enumerate(model.layers):
                if i in layers:
                    W = self.initialize_weights("normal", [l.input_dim, l.units], sigma=0.05)
                    b = self.initialize_weights("zeros", [l.units])
                    initial_weights.append([W,b])
                    l.set_weights([W,b])
                else:
                    continue

        return model, initial_weights

class iterative_pruning:
    
    def __init__(self, initial_weights, loss_fn, optimizer):
        self.weights_history = []
        self.weights_pruned_history = []
        self.weights_nonzero_history = []
        
        self.initial_weights = initial_weights
        self.weights_history.append([w[0] for w in self.initial_weights])
        
        self.loss_fn = loss_fn
        self.opt = optimizer
        
        self.loss_metric = tf.keras.metrics.Mean()
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy()
        
        self.eval_loss_mean = tf.keras.metrics.Mean()
        self.eval_acc = tf.keras.metrics.CategoricalAccuracy()
    
    def reset_weight_history(self):
        self.weights_history = []
        self.weights_pruned_history = []
        self.weights_nonzero_history = []
    
    def reset_weights_init(self,model):
        for i,l in enumerate(model.layers):
            l.set_weights(self.initial_weights[i])

    
    def layerwise_pruning_p(self,model):
        for i,l in enumerate(model.layers):
            total = l.count_params() - l.units
            masked = tf.math.count_nonzero(l.get_masked_weights())
            print(f"{((total-masked)/total) * 100}% of Layer {i} weights are pruned")
            
    def get_p_smallest_value(self, weights, p, absolute=True):
        w_shape = weights.shape[0]

        if absolute:
            weights_sorted = np.sort(np.absolute(weights))
            weights_sorted_zeros_idx = np.where(weights_sorted == 0.)[0]
            weights_sorted_nonzero_idx = np.where(weights_sorted != 0.)[0]

            return np.sort(np.absolute(weights))[int(weights_sorted_zeros_idx.shape[0] + weights_sorted_nonzero_idx.shape[0]*p)]
            
            
    def mask_globally(self, model, th=0.5):
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
    
    def mask_layerwise(self,model, th=0.2):
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
    
    def prune_weights(self, model, pruning_th=0.2, pruning_method="global"):
        
        if pruning_method == "global":
            self.mask_globally(model, pruning_th)
        elif pruning_method == "local":
            self.mask_layerwise(model,pruning_th)
        else:
            print("Pruning Method not implemented, yet.")
            raise NotImplementedError
    
    def start(self, model, dataset_train, iterations, epochs, dataset_test=None, pruning_method="global", pruning_th=0.2, logging=True):
        
        if len(self.weights_pruned_history) == 0:
            self.weights_pruned_history.append([l.get_masked_weights().numpy() for l in fcn.layers])
        if len(self.weights_nonzero_history) == 0:
            self.weights_nonzero_history.append([l.get_nonzero_weights().numpy() for l in fcn.layers])

        for iteration in range(iterations):
            
            print(f"--- Iteration {iteration+1}/{iterations} started ---")
            
            model, iter_weights, iter_weights_pruned, iter_weights_nonzero = self.train_model(model, dataset_train, epochs, logging)
            
            if dataset_test is not None:
                self.evaluate(model, dataset_test)
            
            self.weights_history.append(iter_weights)
            self.weights_pruned_history.append(iter_weights_pruned)
            self.weights_nonzero_history.append(iter_weights_nonzero)
            
            self.prune_weights(model, pruning_method="global", pruning_th=0.2)
            if pruning_method == "global":
                self.layerwise_pruning_p(model)
            
            self.reset_weights_init(model)

    
    def train_model(self, model, dataset,epochs, logging):
        
        iter_weights = [] #+ [w[0] for w in initial_weights]
        iter_weights_pruned = []
        iter_weights_nonzero = []
        iter_weights.append([w[0] for w in self.initial_weights])
        iter_weights_pruned.append([l.get_masked_weights().numpy() for l in model.layers])
        iter_weights_nonzero.append([l.get_nonzero_weights().numpy() for l in model.layers])
    
        
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
            iter_weights_pruned.append([l.get_masked_weights().numpy() for l in model.layers])
            iter_weights_nonzero.append([l.get_nonzero_weights().numpy() for l in model.layers])
                
            if logging:
                print('\tAccuracy = %s --- Loss = %s' % (self.acc_metric.result().numpy(),self.loss_metric.result().numpy()))
                
        return model, iter_weights, iter_weights_pruned, iter_weights_nonzero
    
    def evaluate(self,model,dataset_test):
        for x_batch_test, y_batch_test in dataset_test:
            test_pred = model(x_batch_test)
            test_loss = self.loss_fn(y_batch_test, test_pred)

            self.eval_loss_mean(test_loss)
            self.eval_acc(y_batch_test, test_pred)

        print(f"\t\tTest Loss: {self.eval_loss_mean.result().numpy()}")
        print(f"\t\tTest Accuracy: {self.eval_acc.result().numpy()}")
        
    
    def get_weight_history(self):
        return self.weights_history, self.weights_pruned_history, self.weights_nonzero_history

            
    