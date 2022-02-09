import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time

class ModelTrainer():
    """Contains all functions necessary to train and evaluate signed Supermask and "normal" models

    Arguments:
        model (tf.keras.Model): model to be trained
        ds_train (tf.data.Dataset): training dataset
        ds_test (tf.data.Dataset): test dataset
        optimizer_args (dict): specifies parameters for the optimizer used to train model
    """

    def __init__(self, model, ds_train, ds_test, optimizer_args={}, binary_mask=False, dataset_info = {}):
        self.model = model

        if dataset_info:
            steps_per_epoch = dataset_info["ds_size"] #// dataset_info["batch_size"]
        else:
            steps_per_epoch = 390

        if "decay_steps" not in optimizer_args:
            optimizer_args["decay_steps"] = 10

        if optimizer_args["lr_scheduler"] == "exponential_decay":
            if "decay_rate" not in optimizer_args:
                optimizer_args["decay_rate"] = .9
            lr = tf.keras.optimizers.schedules.ExponentialDecay(float(optimizer_args["lr"]),
                                                                decay_steps=steps_per_epoch*optimizer_args["decay_steps"],
                                                                decay_rate=optimizer_args["decay_rate"],
                                                                staircase=False)

            weight_decay = tf.keras.optimizers.schedules.ExponentialDecay(float(optimizer_args["weight_decay"]),
                                                                          decay_steps=steps_per_epoch*optimizer_args["decay_steps"],
                                                                          decay_rate=optimizer_args["decay_rate"],
                                                                          staircase=False)
        elif optimizer_args["lr_scheduler"] == "cosine_decay":

            lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = float(optimizer_args["lr"]),
                                                           decay_steps = steps_per_epoch*optimizer_args["decay_steps"])

            weight_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = float(optimizer_args["weight_decay"]),
                                                           decay_steps = steps_per_epoch*optimizer_args["decay_steps"])

        elif optimizer_args["lr_scheduler"] == "cosine_warm":

            t_mul = 1.0
            m_mul = 2.0

            lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_Rate = float(optimizer_args["lr"]),
                                                                   first_decay_steps = steps_per_epoch * optimizer_args["decay_steps"],
                                                                   t_mul = t_mul,
                                                                   m_mul = m_mul,
                                                                   alpha = 0.)

            weight_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_Rate = float(optimizer_args["lr"]),
                                                                            first_decay_steps = steps_per_epoch * optimizer_args["decay_steps"],
                                                                            t_mul = t_mul,
                                                                            m_mul = m_mul,
                                                                            alpha = 0.)
        else:
            lr = float(optimizer_args["lr"])
            weight_decay = float(optimizer_args["weight_decay"])


        if isinstance(optimizer_args["weight_decay"], str):
            optimizer_args["weight_decay"] = float(optimizer_args["weight_decay"])
        if isinstance(optimizer_args["lr"], str):
            optimizer_args["lr"] = float(optimizer_args["lr"])

        if dataset_info["name"] != "imagenet" and dataset_info["name"] != "cifar100":
            self.train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.test_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:
            self.train_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            self.test_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


        if optimizer_args["type"] == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                                     momentum=optimizer_args["momentum"],
                                                     nesterov=optimizer_args["nesterov"])
        elif optimizer_args["type"] == "sgdw":
            self.optimizer = tfa.optimizers.SGDW(learning_rate=lr,
                                                 momentum=optimizer_args["momentum"],
                                                 nesterov=optimizer_args["nesterov"],
                                                 weight_decay=weight_decay)
        elif optimizer_args["type"] == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_args["type"] == "adamw":
            self.optimizer = tfa.optimizers.AdamW(learning_rate=lr,
                                                  weight_decay=weight_decay)
        elif optimizer_args["type"] == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr,
                                                         momentum=optimizer_args["momentum"],
                                                         centered=optimizer_args["centered"])

        self.lr_exp_decay = optimizer_args["lr_scheduler"] == "exponential_decay"

        self.ds_train = ds_train
        self.ds_test = ds_test

        self.mask_history = []
        self.train_loss_history = []
        self.train_acc_history = []

        self.test_loss_history = []
        self.test_acc_history = []

        self.latest_train_loss = 0.

        self.train_loss_metric = tf.keras.metrics.Mean()
        self.test_loss_metric = tf.keras.metrics.Mean()

        if dataset_info["name"] != "imagenet" and dataset_info["name"] != "cifar100":

            self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
            self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        else:

            self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


        self.current_test_acc = 1.
        self.current_test_precision = 1.
        self.current_test_recall = 1.
        self.current_test_loss = 1.

        self.current_one_ratio = 1.
        self.one_ratio_history = []

        self.final_masks = []

        self.binary_mask = binary_mask

        self.cooldown_counter = 0 #count remaining epochs in cooldown
        self.wait = 0 #count when current loss > last losses
        self.reduction_counter = 0 #count how often lr was reduced
        self.current_best_loss = 0

    @tf.function
    def train_step(self, x_batch, y_batch):
        """Single train step

        Args:
            x_batch (tf.dataset): features
            y_batch (tf.dataset): labels

        Returns:
            float: loss and prediction of the current train step
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:

            predicted = self.model(x_batch, training=True)
            loss = self.train_loss_fn(y_batch, predicted)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        # print("Gradient mean: ", [tf.reduce_mean(g).numpy() for g in gradients])
        # print("Gradient norm: ", [tf.norm(g).numpy() for g in gradients])
        # gradients = [tf.clip_by_norm(g, .5) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, predicted

    def iterator_layers(self, model):
        #print(m.layers)
        for l in model.layers:

            if isinstance(l, tf.keras.Model):
                yield from self.iterator_layers(l)
            else:
                yield l

    def calc_ones_ratio(self):
        """
        Calculates the ratio of remaining weights
        """
        if self.binary_mask == False:
            # global_no_ones = np.sum([np.sum(np.abs(layer.signed_supermask())) for layer in self.model.layers
            #                             if layer.type == "fefo" or layer.type == "conv"])
            global_no_ones = np.sum([np.sum(np.abs(layer.signed_supermask())) for layer in self.iterator_layers(self.model)
                                        if layer.type == "fefo" or layer.type == "conv"])
        else:
            global_no_ones = np.sum([np.sum(layer.binary_supermask()) for layer in self.model.layers
                                        if layer.type == "fefo" or layer.type == "conv"])

        # global_size = np.sum([tf.size(layer.mask) for layer in self.model.layers
        #                       if layer.type == "fefo" or layer.type == "conv"])
        global_size = np.sum([tf.size(layer.mask) for layer in self.iterator_layers(self.model)
                              if layer.type == "fefo" or layer.type == "conv"])

        remaining_ones_ratio = (global_no_ones/global_size)*100

        self.one_ratio_history.append(remaining_ones_ratio)

    def reduce_lr_on_plateau(self,
                             patience:int,
                             factor:float,
                             cooldown=10,
                             min_delta=1e-4):

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        # if self.in_cooldown:
        #     self.wait
        #     return None
        else:
            prev_best_loss = np.min(self.train_loss_history[-patience+1:-1])
            current_loss = self.train_loss_history[-1]

            if np.less(current_loss + min_delta, prev_best_loss):
                return None
            else:
                self.wait += 1
                if self.wait > patience:
                    print("Reducing learning rate")

                    self.optimizer.lr = self.optimizer.lr * factor
                    self.optimizer.weight_decay = self.optimizer.weight_decay * factor

                    self.cooldown_counter = cooldown

                    self.reduction_counter += 1



    def train(self,
              epochs:int,
              supermask=True,
              logging_interval=5,
              reduce_lr_plateau=True,
              patience=10,
              reductions=5,
              lr_reduce_factor=.5):
        """Wrapper function for training and evaluating the model according to specification

        Args:
            epochs (int): Defines the number of epochs a model is to be trained
            supermask (bool, optional): States wether the model to be trained is a signed Supermask model or not. Defaults to True.
            logging_interval (int, optional): Interval for which you want a log. Defaults to 5.

        """

        if self.lr_exp_decay:
            reduce_lr_plateau = False

        if supermask is True:


            for epoch in range(epochs):

                for (x_batch_train, y_batch_train) in self.ds_train:
                    loss, predicted = self.train_step(x_batch_train, y_batch_train)

                    self.train_loss_metric(loss)
                    self.train_acc_metric(y_batch_train,predicted)

                self.train_loss_history.append(self.train_loss_metric.result().numpy())
                self.train_acc_history.append(self.train_acc_metric.result().numpy())


                self.calc_ones_ratio()

                if epoch % logging_interval == 0:
                    print(f"End of Epoch {epoch}. Accuracy = {self.train_acc_metric.result().numpy():.6f} --- Mean Loss = {self.train_loss_metric.result().numpy():.6f}")
                    print(f"One Ratio: {self.one_ratio_history[-1]}")
                self.train_loss_metric.reset_states()
                self.train_acc_metric.reset_states()

                self.evaluate()

                if self.lr_exp_decay != "exponential_decay":
                    if epoch >= 10 and reduce_lr_plateau:
                        self.reduce_lr_on_plateau(patience=patience,
                                                factor=lr_reduce_factor)

                    if self.reduction_counter == reductions:
                        print("Stop learning - learning rate was reduced ",str(reductions)," times.")
                        break
                    else:
                        continue

            self.final_masks = [layer.bernoulli_mask.numpy() for layer in self.iterator_layers(self.model)
                                if layer.type == "fefo" or layer.type == "conv"]


        else:

            for epoch in range(epochs):
                for (x_batch_train, y_batch_train) in self.ds_train: #enumerate(self.ds_train):
                    loss, predicted = self.train_step(x_batch_train, y_batch_train)

                    self.train_loss_metric(loss)
                    self.train_acc_metric(y_batch_train,predicted)

                self.train_loss_history.append(self.train_loss_metric.result().numpy())
                self.train_acc_history.append(self.train_acc_metric.result().numpy())

                if epoch % logging_interval == 0:
                    print(f"End of Epoch: {epoch}: Accuracy = {self.train_acc_metric.result().numpy():.6f} --- Mean Loss = {self.train_loss_metric.result().numpy():.6f}")

                self.train_loss_metric.reset_states()
                self.train_acc_metric.reset_states()

                self.evaluate()

                if self.lr_exp_decay != "exponential_decay":
                    if epoch >= 10 and reduce_lr_plateau:
                        self.reduce_lr_on_plateau(patience=patience,
                                                factor=lr_reduce_factor)

                    if self.reduction_counter == reductions:
                        print("Stop learning - learning rate was reduced 5 times.")
                        break
                    else:
                        continue

    @tf.function
    def evaluate_step(self, x_batch, y_batch):
        """A single evaluation step

        Args:
            x_batch (tf.dataset): a batch of evaluation data
            y_batch (tf.dataset): labels of a batch of evaluation data

        Returns:
            float: returns the test prediction and test loss
        """
        test_pred = self.model(x_batch, training=False)
        test_loss = self.test_loss_fn(y_batch, test_pred)

        return test_pred, test_loss


    def evaluate(self):
        """Evaluates the model on the evaluation dataset
        """
        for x_batch_test, y_batch_test in self.ds_test:

            test_pred, test_loss = self.evaluate_step(x_batch_test, y_batch_test)

            self.test_loss_metric(test_loss)
            self.test_acc_metric(y_batch_test, test_pred)


        self.test_loss_history.append(self.test_loss_metric.result().numpy())
        self.test_acc_history.append(self.test_acc_metric.result().numpy())

        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()
