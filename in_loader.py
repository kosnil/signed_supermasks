import tensorflow as tf
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds
import os
cwd = os.getcwd()

print("----CWD----",cwd)

BASEDIR = "./data/tfds_data"
DOWNLOADIR = "./data/imagenet_raw"


[ds_train, ds_test], ds_info = tfds.load("imagenet2012", split=['train', 'validation'],
                                        data_dir=BASEDIR, download=True, shuffle_files=True,
                                        as_supervised=True, with_info=True, 
                                        download_and_prepare_kwargs= {'download_dir':DOWNLOADIR})