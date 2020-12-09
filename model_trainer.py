import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time

class ModelTrainer():
    def __init__(self, model, ds_train, ds_test, optimizer_args={}):
        self.model = model
        
        steps_per_epoch = 390
        
        if optimizer_args["lr_scheduler"] == "exponential_decay":
            if "decay_steps" not in optimizer_args:
                optimizer_args["decay_steps"] = 10  

            lr = tf.keras.optimizers.schedules.ExponentialDecay(float(optimizer_args["lr"]), decay_steps=steps_per_epoch*optimizer_args["decay_steps"], decay_rate=0.96, staircase=True)
            weight_decay = tf.keras.optimizers.schedules.ExponentialDecay(float(optimizer_args["weight_decay"]), decay_steps=steps_per_epoch*optimizer_args["decay_steps"], decay_rate=0.96, staircase=True)
        else:
            lr = float(optimizer_args["lr"])
            weight_decay = float(optimizer_args["weight_decay"])
        
        #print(type(optimizer_args["weight_decay"]))
        
        if isinstance(optimizer_args["weight_decay"], str):
            optimizer_args["weight_decay"] = float(optimizer_args["weight_decay"])
        if isinstance(optimizer_args["lr"], str):
            optimizer_args["lr"] = float(optimizer_args["lr"])
        
        self.train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.test_loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # for key, value in optimizer_args.items():
        #     print(key, type(value))
        
        
        if optimizer_args["type"] == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=optimizer_args["momentum"], nesterov=optimizer_args["nesterov"])
        elif optimizer_args["type"] == "sgdw":
            self.optimizer = tfa.optimizers.SGDW(learning_rate=lr, momentum=optimizer_args["momentum"], nesterov=optimizer_args["nesterov"], weight_decay=weight_decay)
        elif optimizer_args["type"] == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_args["type"] == "adamw":
            self.optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        elif optimizer_args["type"] == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=optimizer_args["momentum"], centered=optimizer_args["centered"])
        
        self.ds_train = ds_train
        self.ds_test = ds_test
        
        self.mask_history = []
        self.train_loss_history = []
        self.train_acc_history = []

        self.test_loss_history = []
        self.test_acc_history = []
        
        self.latest_train_loss = 0.

        self.train_loss_metric = tf.keras.metrics.Mean()
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        self.test_loss_metric = tf.keras.metrics.Mean()
        self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        
        self.current_test_acc = 1.
        self.current_test_precision = 1.
        self.current_test_recall = 1.
        self.current_test_loss = 1.
        
        self.current_one_ratio = 1.
        self.one_ratio_history = []
        
        self.final_masks = []
        
    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            predicted = self.model(x_batch, training=True) 
            loss = self.train_loss_fn(y_batch, predicted)
            #loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables]) * 1e-5
            #loss = self.loss_fn(y_batch, predicted) + loss_l2
        
            gradients = tape.gradient(loss, self.model.trainable_variables)

        gradients = [tf.clip_by_norm(g, .5) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, predicted
    
    # @tf.function
    # def train_step_weights(self, x_batch, y_batch):
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         tape.watch(self.model.non_trainable_variables)
            
    #         predicted = self.model(x_batch, training=True)
    #         loss = self.train_loss_fn(y_batch, predicted)
            
    #         gradients = tape.gradient(loss, self.model.non_trainable_variables)
    #     #print([tf.math.reduce_sum(grad).numpy() for grad in gradients])
    #     self.optimizer.apply_gradients(zip(gradients, self.model.non_trainable_variables))
        
    #     return loss, predicted
    
    def calc_ones_ratio(self):
        
        global_no_ones = np.sum([np.sum(np.abs(layer.tanh_mask())) for layer in self.model.layers if layer.type == "fefo" or layer.type == "conv"])
        global_size = np.sum([tf.size(layer.mask) for layer in self.model.layers if layer.type == "fefo" or layer.type == "conv"])

        remaining_ones_ratio = (global_no_ones/global_size)*100
        # print(f"{remaining_ones_ratio:.2f}% of weights are 'remaining' --- total weights: {global_size}, total weights left: {global_no_ones}")
        
        # self.current_one_ratio = remaining_ones_ratio
        self.one_ratio_history.append(remaining_ones_ratio)

    def train(self, epochs, supermask=True, logging_interval=5, pre_train=False):
        if supermask is True:
            
            # if pre_train:
            #     for i in range(5):
            #         for step, (x_batch_train, y_batch_train) in enumerate(self.ds_train):                    
            #             loss, predicted = self.train_step_weights(x_batch_train, y_batch_train)

            #             self.loss_metric(loss)
            #             self.acc_metric(y_batch_train,predicted)

            #             self.loss_history.append(self.loss_metric.result().numpy())
            #             self.acc_history.append(self.acc_metric.result().numpy())
                
            #         print(f"End of pretraining Epoch {i+1}. Accuracy = {self.acc_metric.result().numpy():.6f} --- Mean Loss = {self.loss_metric.result().numpy():.6f}")
            #         #self.calc_ones_ratio()
            #         self.evaluate()
                
            #         self.loss_metric.reset_states()
            #         self.acc_metric.reset_states()
            
            # self.calc_ones_ratio()
            for epoch in range(epochs):
                # time0 = time.time()
                #for step, (x_batch_train, y_batch_train) in enumerate(self.ds_train):
                for (x_batch_train, y_batch_train) in self.ds_train:
                    loss, predicted = self.train_step(x_batch_train, y_batch_train)
                    
                    self.train_loss_metric(loss)
                    self.train_acc_metric(y_batch_train,predicted)
                
                self.train_loss_history.append(self.train_loss_metric.result().numpy())
                self.train_acc_history.append(self.train_acc_metric.result().numpy())
                

                if epoch % logging_interval == 0:
                    print(f"End of Epoch {epoch}. Accuracy = {self.train_acc_metric.result().numpy():.6f} --- Mean Loss = {self.train_loss_metric.result().numpy():.6f}")
                #     self.calc_ones_ratio()
                #     self.evaluate()
                
                self.train_loss_metric.reset_states()
                self.train_acc_metric.reset_states()
                
                self.evaluate()
                self.calc_ones_ratio()

                # time1 = time.time()
                # print("Time needed for epoch: ", str(time1-time0))
            
            self.final_masks = [layer.bernoulli_mask.numpy() for layer in self.model.layers if layer.type == "fefo" or layer.type == "conv"]

        else:
            # epoch = 0
            # best_test_loss = 0.
            # epoch_since_last_improvement = 0
            # early_stopping = False
            # require_improvement = 15
            # while epoch < epochs and early_stopping == False:
            #     for step, (x_batch_train, y_batch_train) in enumerate(self.ds_train):
            #         loss, predicted = self.train_step(x_batch_train, y_batch_train)

            #         self.train_loss_metric(loss)
            #         self.train_acc_metric(y_batch_train,predicted)

            #     self.train_loss_history.append(self.train_loss_metric.result().numpy())
            #     self.train_acc_history.append(self.train_acc_metric.result().numpy())
                
            #     self.latest_train_loss = self.train_loss_metric.result().numpy()

            #     if epoch % logging_interval == 0:
            #         print(f"End of Epoch: {epoch}: Accuracy = {self.train_acc_metric.result().numpy():.6f} --- Mean Loss = {self.train_loss_metric.result().numpy():.6f}")
            #     #     self.evaluate()
                    
            #     self.train_loss_metric.reset_states()
            #     self.train_acc_metric.reset_states()
    
            #     self.evaluate()

            #     best_loss = np.min(self.train_loss_history)
                
            #     # print("Latest test loss: ", self.latest_test_loss)
            #     # print("Best test loss: ", best_test_loss)

            #     if self.latest_train_loss <= best_loss:
            #         epoch_since_last_improvement = 0
            #         best_loss = self.latest_train_loss
            #     else:
            #         epoch_since_last_improvement += 1
            #         if epoch_since_last_improvement >= require_improvement:
            #             early_stopping = True
            #             print("Early Stopping at epoch", epoch)
            #             break
                
            #     epoch += 1

            
            for epoch in range(epochs):
                for step, (x_batch_train, y_batch_train) in enumerate(self.ds_train):
                    loss, predicted = self.train_step(x_batch_train, y_batch_train)

                    self.train_loss_metric(loss)
                    self.train_acc_metric(y_batch_train,predicted)

                self.train_loss_history.append(self.train_loss_metric.result().numpy())
                self.train_acc_history.append(self.train_acc_metric.result().numpy())
                
                if epoch % logging_interval == 0:
                    print(f"End of Epoch: {epoch}: Accuracy = {self.train_acc_metric.result().numpy():.6f} --- Mean Loss = {self.train_loss_metric.result().numpy():.6f}")
                #     self.evaluate()
                    
                self.train_loss_metric.reset_states()
                self.train_acc_metric.reset_states()
    
                self.evaluate()

    @tf.function
    def evaluate_step(self, x_batch, y_batch):
        test_pred = self.model(x_batch, training=False)
        test_loss = self.test_loss_fn(y_batch, test_pred)

        return test_pred, test_loss
            
            
    def evaluate(self):
        
        for x_batch_test, y_batch_test in self.ds_test:
            
            test_pred, test_loss = self.evaluate_step(x_batch_test, y_batch_test)

            self.test_loss_metric(test_loss)
            self.test_acc_metric(y_batch_test, test_pred)
        
        # self.latest_test_loss = self.test_loss_metric.result().numpy()
        
        self.test_loss_history.append(self.test_loss_metric.result().numpy())
        self.test_acc_history.append(self.test_acc_metric.result().numpy())
        
        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()

        # self.current_test_acc = eval_acc.result().numpy()
        # self.current_test_loss = eval_loss_mean.result().numpy()
        
        # print("Evaluation Loss: ", self.current_test_loss, " Accuracy: ", self.current_test_acc)
        