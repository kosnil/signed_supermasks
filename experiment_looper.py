from ast import parse
import numpy as np
import tensorflow as tf

import time

import argparse
from tensorflow_datasets.core import dataset_info
import yaml

import pickle
import pickletools

from model_trainer import ModelTrainer
from weight_initializer import initializer
from data_preprocessor import data_handler

from conv_networks import Conv2, Conv4, Conv6, Conv8
from conv_networks import Conv2_Mask, Conv4_Mask, Conv6_Mask, Conv8_Mask #, VGG16_Mask, VGG19_Mask
from resnet_networks import ResNet110, ResNet20_Mask, ResNet20, ResNet56, ResNet56_Mask, ResNet110_Mask

from dense_networks import FCN, FCN_Mask

def parse_config_file(path: str) -> dict:
    """Parse and load the config file

    Args:
        path (str): path to file

    Returns:
        dict: config
    """

    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def network_builder(config: dict) -> tf.keras.Model:
    """Given the config dictionary, this function builds the there defined tensorflow model accordingly.
    It is possible to select FCN, Conv2, Conv4, Conv6 and Conv8

    Args:
        config (dict): configuration in which the model is defined

    Returns:
        tf.keras.Model: model
    """

    #depending on the dataset the model is trained on, choose the appropriate input shape.
    if config["data"] == "cifar":
        input_shape = (128,32,32,3)
    elif config["data"] == "mnist":
        input_shape = (128,784)

    #go through necessary properties in config to build up the network step by step

    #baseline
    if config["baseline"] == True:
        if config["model"]["type"] == "FCN":
            model = FCN(use_bias=False)
        elif config["model"]["type"] == "Conv2":
            model = Conv2(use_bias=False)
        elif config["model"]["type"] == "Conv4":
            model = Conv4(use_bias=False)
        elif config["model"]["type"] == "Conv6":
            model = Conv6(use_bias=False)
        elif config["model"]["type"] == "Conv8":
            model = Conv8(use_bias=False)
        elif config["model"]["type"] == "ResNet20":
            if "filter_size_multi" not in config["model"]:
                config["model"]["filter_size_multi"] = 1.
            model = ResNet20(num_classes=10,
                             filter_size_multi=config["model"]["filter_size_multi"])
        elif config["model"]["type"] == "ResNet56":
            model = ResNet56(num_classes=10)
        elif config["model"]["type"] == "ResNet110":
            model = ResNet110(num_classes=10)

        else:
            print("Please define a model")
            return 0

        model.build(input_shape=input_shape)

        return model
    #signed supermask
    else:
        if config["model"]["type"] == "FCN":
            model = FCN_Mask(masking_method=config["model"]["masking_method"],
                            #  tanh_th=config["model"]["tanh_th"],
                             k=config["model"]["k_dense"],
                             dynamic_scaling=config["model"]["dynamic_scaling_dense"])
        elif config["model"]["type"] == "Conv2":
            model = Conv2_Mask(input_shape=input_shape,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"])
        elif config["model"]["type"] == "Conv4":
            model = Conv4_Mask(input_shape=input_shape,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"])

        elif config["model"]["type"] == "Conv6":
            model = Conv6_Mask(input_shape=input_shape,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"])

        elif config["model"]["type"] == "Conv8":
            model = Conv8_Mask(input_shape=input_shape,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"])

        elif config["model"]["type"] == "ResNet20":
            if "filter_size_multi" not in config["model"]:
                config["model"]["filter_size_multi"] = 1.
            model = ResNet20_Mask(input_shape=input_shape,
                                  num_classes=10,
                                  filter_size_multi=config["model"]["filter_size_multi"])

        elif config["model"]["type"] == "ResNet56":
            model = ResNet56_Mask(input_shape=input_shape,
                                  num_classes=10)

        elif config["model"]["type"] == "ResNet110":
            model = ResNet110_Mask(input_shape=input_shape,
                                  num_classes=10)
        # elif config["model"]["type"] == "VGG16":
        #     model = VGG16_Mask(input_shape=input_shape,
        #                         masking_method=config["model"]["masking_method"],
        #                         tanh_th=config["model"]["tanh_th"],
        #                         k_cnn=config["model"]["k_cnn"],
        #                         k_dense=config["model"]["k_dense"],
        #                         dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
        #                         dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
        #                         width_multiplier=config["model"]["width_multiplier"])

        # elif config["model"]["type"] == "VGG19":
        #     model = VGG19_Mask(input_shape=input_shape,
        #                         masking_method=config["model"]["masking_method"],
        #                         tanh_th=config["model"]["tanh_th"],
        #                         k_cnn=config["model"]["k_cnn"],
        #                         k_dense=config["model"]["k_dense"],
        #                         dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
        #                         dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
        #                         width_multiplier=config["model"]["width_multiplier"])

        else:
            print("Please define a model")
            return 0

        # if config["model"]["masking_method"] == "fixed":
        #     print("Fixed Threshold...updating tanh_th")
        #     for layer in model.layers:
        #         if layer.type == "fefo" or layer.type == "conv":
        #             layer.update_tanh_th(percentage=config["model"]["tanh_th"])

        return model

def initialize_model(model:tf.keras.Model,
                     config:dict,
                     run_number:int,
                     on_the_fly=False) -> tf.keras.Model:
    """Loads the weights and mask values defined in the config file

    Args:
        model (tf.keras.Model): model to be trained
        config (dict): configuration of model and training
        run_number (int): number of experiment (There are only 50 pre-defined weight and mask tensors)

    Returns:
        tf.keras.Model: model with initialized weight and mask values
    """

    SEED = 7531 + run_number

    init = initializer(seed=SEED)

    if on_the_fly == True:

        if config["baseline"] == False:
            #weights
            model, _ = init.set_weights_man(model,
                                            dist=config["init"]["weight"]["dist"],
                                            method=config["init"]["weight"]["method"],
                                            factor=np.sqrt(config["init"]["weight"]["factor"]), # .57
                                            single_value=False,
                                            save_to="",
                                            weight_as_constant=True,
                                            set_mask=False)

            #mask
            model, _ = init.set_weights_man(model,
                                            dist=config["init"]["mask"]["dist"],
                                            method=config["init"]["mask"]["method"],
                                            factor= config["init"]["mask"]["factor"],
                                            save_to="",
                                            weight_as_constant=False,
                                            set_mask=True)
        else:

            layer_shapes = []
            for layer in iterate_layers(model):
                if layer.type == "conv_normal" or layer.type == "fefo_normal":
                    shape = layer.get_weights()[0].shape
                    layer_shapes.append(shape)

            model, _ = init.set_weights_man(model,
                                            dist=config["init"]["weight"]["dist"],
                                            method=config["init"]["weight"]["method"],
                                            factor=np.sqrt(config["init"]["weight"]["factor"]), # .57
                                            single_value=False,
                                            save_to="",
                                            weight_as_constant=False,
                                            set_mask=False,
                                            layer_shapes=layer_shapes)

    else:

        weight_file_name = config["model"]["type"] + "_weights_" + str(run_number) + ".pkl"
        mask_file_name = config["model"]["type"] + "_mask_" + str(run_number) + ".pkl"

        model = init.set_loaded_weights(model = model,
                                        path = config["path_weights"]+weight_file_name)

        if config["baseline"] == False:
            model = init.set_loaded_weights(model = model,
                                            path = config["path_masks"]+mask_file_name)


    return model

def iterate_layers(m):
    for l in m.layers:
        if isinstance(l, tf.keras.Model):
            yield from iterate_layers(l)
        else:
            yield l

def repeat_experiment(config:dict) -> list:
    """Loads the dataset and then loops through each experiment for in the config defined amount of runs.
    After loading the data (which is always the same), the order is as follows:
    Build model (network_builder) --> Initialize model (initialize_model) --> Initialize Modeltrainer -->
    Train Model (mt.train) --> Append intermediate results to the "results"-array, which holds all results

    Args:
        config (dict): config file

    Returns:
        [list]: basic list that holds the results of all runs for the given model
    """

    print("Loading dataset...")
    ds_train, ds_test = data_handler(config["data"])
    print("Dataset loaded!")

    results = []

    # steps_per_epoch = 390

    for i in range(config["training"]["no_experiments"]):

        model = network_builder(config)

        print("-------------------------------------------------------")
        print("Starting Experiment", i,"...")
        print("-------------------------------------------------------")

        intermediate_results = {}

        model = initialize_model(model,
                                 config,
                                 run_number=i,
                                 on_the_fly=config["init"]["on_the_fly"])

        if config["model"]["masking_method"] == "fixed" and config["baseline"] is False:
            # or config["model"]["masking_method"] == "binary"):
            print("Fixed Threshold...updating tanh_th")
            for l in iterate_layers(model):
                if l.type == "fefo" or l.type == "conv":
                    l.update_tanh_th(percentage=config["model"]["tanh_th"])
            # for layer in model.layers:
                # if layer.type == "fefo" or layer.type == "conv":
                    # layer.update_tanh_th(percentage=config["model"]["tanh_th"])

        print("Model initialized!")

        train_w_binary_mask = True if config["model"]["masking_method"] == "binary" else False


        dataset_info = {
            "ds_size": ds_train.cardinality().numpy(),
            "name": "cifar",
            "batch_size": 128,
        }

        mt = ModelTrainer(model,
                          ds_train = ds_train,
                          ds_test = ds_test,
                          optimizer_args = config["optimizer"],
                          dataset_info=dataset_info,
                          binary_mask = train_w_binary_mask)

        time0 = time.time()
        print("Start training...")

        if "patience" not in config["training"]:
            config["training"]["patience"] = 20

        if "reductions" not in config["training"]:
            config["training"]["reductions"] = 5

        if "lr_reduce_factor" not in config["training"]:
            config["training"]["lr_reduce_factor"] = .2

        if config["baseline"] is False:

            mt.calc_ones_ratio()

            mt.train(epochs=config["training"]["epochs"],
                     patience=config["training"]["patience"],
                     reductions=config["training"]["reductions"],
                     lr_reduce_factor=config["training"]["lr_reduce_factor"],
                     logging_interval=20,
                     supermask=True)

        else:
            mt.train(epochs=config["training"]["epochs"],
                     patience=config["training"]["patience"],
                     reductions=config["training"]["reductions"],
                     lr_reduce_factor=config["training"]["lr_reduce_factor"],
                     logging_interval=20,
                     supermask=False)

        print("Training successful!")
        time1 = time.time()
        print("Time needed for training: ", str(time1-time0))

        intermediate_results["train_loss"] = mt.train_loss_history
        intermediate_results["train_acc"] = mt.train_acc_history

        intermediate_results["test_loss"] = mt.test_loss_history
        intermediate_results["test_acc"] = mt.test_acc_history

        intermediate_results["one_ratio"] = mt.one_ratio_history

        intermediate_results["final_masks"] = mt.final_masks

        intermediate_results["training_time"] = time1 - time0

        # intermediate_results["test_acc"] = mt.current_test_acc
        # intermediate_results["test_loss"] = mt.current_test_loss
        # intermediate_results["ones_ratio"] = mt.current_one_ratio

        results.append(intermediate_results)

    return results

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def save_results(results: dict,
                 filename: str):
    """This function saves the results obtrained from training a model

    Args:
        results (dict): results that are to be saved
        filename (str): name of the file that holds results
    """

    with open("./results/"+filename+".pkl", 'wb') as handle:
        pickled = pickle.dumps(results)
        optimized_pickle = pickletools.optimize(pickled)
        handle.write(optimized_pickle)

def main_pipeline(config_path: str):
    """Pipeline that laods the config file, created and initializes the model, trains it and finally saves the results

    Args:
        config_path (str): path to config file
    """
    print("Load config...")
    config = parse_config_file(path = config_path)
    print("Config loaded!")
    print(" ")

    results = repeat_experiment(config)

    config_name = config_path[findnth(config_path, "/", 1)+1:config_path.rfind(".")]
    print("Saving results...")
    save_results(results=results,
                 filename=config_name)
    print("Results saved!")

