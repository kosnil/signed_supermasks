from ast import parse
import numpy as np
import tensorflow as tf

import time

import argparse
import yaml

import pickle
import pickletools

from model_trainer import ModelTrainer
from weight_initializer import initializer
from data_preprocessor import data_handler

from conv_networks import Conv2, Conv4, Conv6, Conv8
from conv_networks import Conv2_Mask, Conv4_Mask, Conv6_Mask, Conv8_Mask

from dense_networks import FCN, FCN_Mask

def parse_config_file(path: str) -> dict:
    
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
    

def network_builder(config: dict) -> tf.keras.Model:
    

    if config["data"] == "cifar":
        input_shape = (128,32,32,3)
    elif config["data"] == "mnist":
        input_shape = (128,784)
    
    #go through necessary properties in config to build up the network step by step
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
        else:
            print("Please define a model")
            return 0
        
        model.build(input_shape=input_shape)

        return model     

    else:
        if config["model"]["type"] == "FCN":
            model = FCN_Mask(masking_method=config["model"]["masking_method"],
                            #  tanh_th=config["model"]["tanh_th"],
                             k=config["model"]["k_dense"],
                             dynamic_scaling=config["model"]["dynamic_scaling_dense"])
        elif config["model"]["type"] == "Conv2":
            model = Conv2_Mask(input_shape=input_shape,
                                use_bias=False,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"],
                                use_dropout=config["model"]["use_dropout"])
        elif config["model"]["type"] == "Conv4":
            model = Conv4_Mask(input_shape=input_shape,
                                use_bias=False,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"],
                                use_dropout=config["model"]["use_dropout"])

        elif config["model"]["type"] == "Conv6":
            model = Conv6_Mask(input_shape=input_shape,
                                use_bias=False,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"],
                                use_dropout=config["model"]["use_dropout"])

        elif config["model"]["type"] == "Conv8":
            model = Conv8_Mask(input_shape=input_shape,
                                use_bias=False,
                                masking_method=config["model"]["masking_method"],
                                # tanh_th=config["model"]["tanh_th"],
                                k_cnn=config["model"]["k_cnn"],
                                k_dense=config["model"]["k_dense"],
                                dynamic_scaling_cnn=config["model"]["dynamic_scaling_cnn"],
                                dynamic_scaling_dense=config["model"]["dynamic_scaling_dense"],
                                width_multiplier=config["model"]["width_multiplier"],
                                use_dropout=config["model"]["use_dropout"])

        else:
            print("Please define a model")
            return 0
 
        if config["model"]["masking_method"] is "variable":
            for layer in model.layers:
                if layer.type == "fefo" or layer.type == "conv":
                    layer.update_tanh_th(percentage=config["model"]["tanh_th"])

        return model
    
def initialize_model(model:tf.keras.Model, config:dict, run_number:int) -> tf.keras.Model:    

    init = initializer()
    
    weight_file_name = config["model"]["type"] + "_weights_" + str(run_number) + ".pkl"
    mask_file_name = config["model"]["type"] + "_mask_" + str(run_number) + ".pkl"
    
    model = init.set_loaded_weights(model = model, path = config["path_weights"]+weight_file_name)
    
    if config["baseline"] == False:
        model = init.set_loaded_weights(model = model, path = config["path_masks"]+mask_file_name)
    
    return model     

def repeat_experiment(config:dict):
    
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
        
        model = initialize_model(model, config, run_number=i)
        
        print("Model initialized!")

        mt = ModelTrainer(model, ds_train = ds_train, ds_test = ds_test, optimizer_args=config["optimizer"])
        
        time0 = time.time()
        print("Start training...")
        if config["baseline"] == False:
            mt.train(epochs=config["training"]["epochs"], logging_interval=20)
        else:
            mt.train(epochs=config["training"]["epochs"], logging_interval=20, supermask=False)
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

def save_results(results: dict, filename: str):
    
    with open("./results/"+filename+".pkl", 'wb') as handle:
        pickled = pickle.dumps(results)
        optimized_pickle = pickletools.optimize(pickled)
        handle.write(optimized_pickle)

def main_pipeline(config_path: str):
    print("Load config...")
    config = parse_config_file(path = config_path)
    print("Config loaded!")
    print(" ")
    
    results = repeat_experiment(config)
    
    config_name = config_path[findnth(config_path, "/", 1)+1:config_path.rfind(".")]
    print("Saving results...")
    save_results(results=results, filename=config_name)
    print("Results saved!")
# if __name__ == '__main__':
#     #import argparse

#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--config', required=True,
#                         help='path to config file')
#     args = parser.parse_args()
#     main_pipeline(config_path=args.config_path)

