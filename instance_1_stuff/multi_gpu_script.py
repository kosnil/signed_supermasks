import tensorflow_datasets as tfds
import tensorflow as tf
from resnet_networks import ResNet50_Mask, ResNet18_Mask, ResNet18B_Mask, ResNet34_Mask
from conv_networks import Conv4_Mask, Conv6_Mask, Conv8_Mask, VGGS
from data_preprocessor import data_handler, data_handler_imagenet
from model_trainer import ModelTrainer
import numpy as np
import tensorflow_addons as tfa
from weight_initializer import initializer

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# ---------------------------TRAIN PARAMS------------------------------------------------
opt_config = {"type": "sgdw",
              "lr_scheduler": "None",
              "lr": 4.,
              "decay_steps": 5,
              "decay_rate": .9,
              "weight_decay": 1e-4,
              "momentum": .9,
              "nesterov": True,
              "centered": True #only for rms_prop,
             }

model_config = {"binary_mask": False}

train_config = {"epochs": 40,
                "reduce_on_plateau": True,
                "patience":10,
                "lr_reduce_factor":.5}

BATCH_SIZE_PER_DEVICE = 64
BATCH_SIZE = BATCH_SIZE_PER_DEVICE * strategy.num_replicas_in_sync

NUM_CLASSES = 1000

# ---------------------------DATA------------------------------------------------
ds_train, ds_test = data_handler_imagenet(batch_size=BATCH_SIZE)

# distributed data across machines
train_dist_dataset = strategy.experimental_distribute_dataset(ds_train)
test_dist_dataset = strategy.experimental_distribute_dataset(ds_test)

INPUT_SHAPE = iter(ds_train).next()[0].numpy().shape
INPUT_SHAPE = (BATCH_SIZE_PER_DEVICE, INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]) 

# ---------------------------MODEL------------------------------------------------
NUM_CLASSES = 1000 #iter(ds_train).next()[1].numpy().shape[1]

init_create = initializer()

def iterate_layers(m):
    #print(m.layers)
    for l in m.layers:
        #print(l)
        if isinstance(l, tf.keras.Model):
            yield from iterate_layers(l)
        else:
            #if l.type == "fefo" or l.type == "conv":
            yield l

with strategy.scope():
    model = ResNet50_Mask(input_shape=INPUT_SHAPE,
                          num_classes=NUM_CLASSES,
                          first_kernel_size=7,
                          first_stride=(2,2))
    
    model, initial_weights = init_create.set_weights_man(model, 
                                                     dist="signed_constant", 
                                                     method="he",
                                                     factor=np.sqrt(2), # .57
                                                     single_value=False,
                                                     save_to="",  
                                                     weight_as_constant=True, 
                                                     set_mask=False) 
    
    #mask
    model, initial_weights = init_create.set_weights_man(model, 
                                                         dist="uniform", 
                                                         method="xavier", 
                                                         factor= 1.,
                                                         save_to="",  
                                                         weight_as_constant=False, 
                                                         set_mask=True) 
#     if opt_config["type"] == "sgdw":
    optimizer = tfa.optimizers.SGDW(learning_rate=opt_config["lr"], 
                                    momentum=opt_config["momentum"], 
                                    nesterov=opt_config["nesterov"], 
                                    weight_decay=opt_config["weight_decay"])   

    
    for l in iterate_layers(model):
        if l.type == "fefo" or l.type == "conv":
            l.update_tanh_th(percentage=.7)


  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
    

@tf.function
def train_step(inputs):
    """Single train step

    Args:
        x_batch (tf.dataset): features
        y_batch (tf.dataset): labels

    Returns:
        float: loss and prediction of the current train step
            
    """
    x_batch, y_batch = inputs
        
    with tf.GradientTape(watch_accessed_variables=True) as tape:
            
        predicted = model(x_batch, training=True) 
        loss = compute_loss(y_batch, predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_accuracy.update_state(y_batch, predicted)

    return loss

#@tf.function
def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    
    return strategy.reduce(tf.distribute.ReduceOp.SUM, 
                           per_replica_losses,
                           axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

# Plateau

cooldown_counter= 0 #count remaining epochs in cooldown
wait = 0 #count when current loss > last losses
reduction_counter = 0 #count how often lr was reduced
current_best_loss = 0
class learning_rate_reducer():
    
    def __init__(self, 
                 cooldown_counter=0, 
                 wait=0, 
                 reduction_counter=0, 
                 current_best_loss=0):
        self.cooldown_counter = cooldown_counter
        self.wait = wait
        self.reduction_counter = reduction_counter
        self.current_best_loss = current_best_loss
        
    def reduce_lr_on_plateau(self,
                             loss_history,
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
                    
                    
# calculate one ratio
def calc_ones_ratio(model):
    """
    Calculates the ratio of remaining weights
    """
    if model_config["binary_mask"] == False: 
        # global_no_ones = np.sum([np.sum(np.abs(layer.signed_supermask())) for layer in self.model.layers 
        #                             if layer.type == "fefo" or layer.type == "conv"])
        global_no_ones = np.sum([np.sum(np.abs(layer.signed_supermask())) for layer in iterate_layers(model)
                                    if layer.type == "fefo" or layer.type == "conv"])
    else:
        global_no_ones = np.sum([np.sum(layer.binary_supermask()) for layer in model.layers 
                                    if layer.type == "fefo" or layer.type == "conv"])

    # global_size = np.sum([tf.size(layer.mask) for layer in self.model.layers 
    #                       if layer.type == "fefo" or layer.type == "conv"])
    global_size = np.sum([tf.size(layer.mask) for layer in iterate_layers(model)
                          if layer.type == "fefo" or layer.type == "conv"])

    remaining_ones_ratio = (global_no_ones/global_size)*100

    return remaining_ones_ratio

calc_ones_ratio(model)

total_size_trainable = 0
for tw in model.trainable_weights:
    total_size_trainable += tf.size(tw).numpy()
total_size_trainable

total_size_nontrainable = 0
for tw in model.non_trainable_weights:
    total_size_nontrainable += tf.size(tw).numpy()
total_size_nontrainable

assert total_size_trainable == total_size_nontrainable

import time


train_loss_history = []
test_loss_history = []

train_acc_history = []
test_acc_history = []

one_ratio_history = []

lr_reducer = learning_rate_reducer()

template = ("Epoch {}, Loss: {}, Accuracy: {}%, Test Loss: {}, "
                  "Test Accuracy: {}%, Time: {}min")


for epoch in range(train_config["epochs"]):
    
    print("start epoch", epoch)
    
    now = time.time()

    total_loss = 0.0
    num_batches = 0


    for data in train_dist_dataset:
        #start_batch = time.time()
        total_loss += distributed_train_step(data)
        # otherwise div by zero
        num_batches += 1

        if num_batches % 100 == 0:
            print(num_batches)
        #print("Batch took: {} sec".format(int(time.time()-start_batch)))

    train_loss = total_loss / num_batches
    print("start testing...")
    # TEST LOOP
    for x in test_dist_dataset:
        distributed_test_step(x)


    # write logs
    train_loss_history.append(train_loss.numpy())
    test_loss_history.append(test_loss.result().numpy())
    train_acc_history.append(train_accuracy.result().numpy()*100)
    test_acc_history.append(test_accuracy.result().numpy()*100)
    
    one_ratio = calc_ones_ratio(model)
    one_ratio_history.append(one_ratio)

    print(test_loss_history)

    if epoch > 10 and train_config["reduce_on_plateau"]:
        lr_reduction = lr_reducer.reduce_lr_on_plateau(train_loss_history, 
                                            patience=train_config["patience"], 
                                            factor=train_config["lr_reduce_factor"])

        optimizer.lr = optimizer.lr*lr_reduction




    print(template.format(epoch,
                          round(train_loss_history[-1], 4),
                          round(train_acc_history[-1], 4),
                          round(test_loss_history[-1], 4),
                          round(test_acc_history[-1], 4),
                          round(int(time.time()-now)/60), 2))

    print("One Ratio: ",round(one_ratio,4))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
        