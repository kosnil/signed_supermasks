import numpy as np

from weight_initializer import initializer as w_init
from dense_networks import FCN, FCN_Mask
from conv_networks import Conv2_Mask, Conv4_Mask, Conv6_Mask, Conv8_Mask

mask_distributions = {"uniform": ["xavier"]} #{"uniform": ["he", "xavier"], "normal": ["he", "xavier"]}
weight_distributions = {"signed_constant": ["elu_scaled"]} #{"constant": ["he", "xavier"], "signed_constant": ["he", "xavier"], "uniform": ["he", "xavier"]}

def create_weight_files(net_type, mask_dist, weight_dist, no_runs=5):
    
    if net_type == "FCN":
        INPUT_SHAPE = (128, 784)
    elif "Conv" in net_type:
        INPUT_SHAPE = (128, 32, 32, 3)
    else:
        print("The network type you specified is not implemented...")
        return 0
    
    #FACTOR = np.sqrt(3) # multiplier for ELUS
    
    init_create = w_init()
    
    #FCN
    if net_type == "FCN":
        model = FCN_Mask() #FCN(layer_shapes=[(784,300), (300,100), (100,10)]) 
        model.build(input_shape=INPUT_SHAPE)
    elif net_type == "Conv2":
        model = Conv2_Mask(input_shape=INPUT_SHAPE)
    elif net_type == "Conv4":
        model = Conv4_Mask(input_shape=INPUT_SHAPE)
    elif net_type == "Conv6":
        model = Conv6_Mask(input_shape=INPUT_SHAPE)
    elif net_type == "Conv8":
        model = Conv8_Mask(input_shape=INPUT_SHAPE)
    #CNN
    #model = Conv2_Mask(input_shape=INPUT_SHAPE, use_bias=False)
    
    model_str = net_type
    
    for dist in weight_dist:
        print(dist.upper())
        for specific in weight_dist[dist]:
            print("---",specific.upper())
            FACTOR = 1.
            if specific == "elu_scaled":
                sigma = -8
                FACTOR = np.sqrt(3)
            elif specific == "he":
                sigma = -8
            elif specific == "xavier":
                sigma = -7
            
            for i in range(no_runs):
                print("--- ---weight run ",str(i))
                model, _ = init_create.set_weights_man(model, 
                                                       mode=dist, 
                                                       constant=-2, 
                                                       mu=0, 
                                                       sigma=sigma, 
                                                       factor=FACTOR,
                                                       save_to="./example_weights/"+model_str+"/weight/"+dist+"/"+specific+"/"+model_str+"_weights", 
                                                       save_suffix="_"+str(i), 
                                                       weight_as_constant=True, 
                                                       set_mask=False) 
                                                                     
    for dist in mask_dist:
        print(dist.upper())
        for specific in mask_dist[dist]:
            print("---",specific.upper())
            FACTOR = 1.
            if specific == "elu_scaled":
                sigma = -8
                FACTOR = np.sqrt(3)
            elif specific == "he":
                sigma = -8
            elif specific == "xavier":
                sigma = -7
            for i in range(no_runs):
                print("--- ---mask run ",str(i))
                model, _ = init_create.set_weights_man(model, 
                                                       mode=dist, 
                                                       constant=-2, 
                                                       mu=0, 
                                                       sigma=sigma, 
                                                       factor=FACTOR, 
                                                       save_to="./example_weights/"+model_str+"/mask/"+dist+"/"+specific+"/"+model_str+"_mask", 
                                                       save_suffix="_"+str(i), 
                                                       weight_as_constant=False, 
                                                       set_mask=True)
  
    print("Weight Initialization successful")
                                                                          
# As we specify the weight files for both the masked and normal version of a given architecture, we drop the "_Mask" suffix
# in this case to maintain clarity
create_weight_files(net_type="FCN", 
                    weight_dist=weight_distributions, 
                    mask_dist=mask_distributions)
