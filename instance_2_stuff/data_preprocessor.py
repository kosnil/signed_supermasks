import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

def data_loader(dataset:str):
    """Loads the specified dataset

    Args:
        dataset (str): name of the dataset, either "mnist" or "cifar"

    Returns:
        [tf.dataset]: training and testing set with an additional object "ds_info" that holds more information about the dataset
    """
    if dataset == "mnist":
        (ds_train, ds_test), ds_info = tfds.load('mnist',
                                                split=['train', 'test'],
                                                shuffle_files=True,
                                                as_supervised=True,
                                                with_info=True)
        return ds_train, ds_test, ds_info

    elif dataset == "cifar":
        (ds_train, ds_test), ds_info = tfds.load('cifar10',
                                                split=['train', 'test'],
                                                shuffle_files=True, 
                                                as_supervised=True,
                                                with_info=True)
    
        return ds_train, ds_test, ds_info
    
    elif dataset == "cifar100":

        (ds_train, ds_test), ds_info = tfds.load('cifar100',
                                                split=['train', 'test'],
                                                shuffle_files=True, 
                                                as_supervised=True,
                                                with_info=True)
        
        return ds_train, ds_test, ds_info

    elif dataset == "imagenet":
        BASEDIR = "data/tfds/tfds_data"
        DOWNLOADIR = "data/raw"


        (ds_train, ds_test), ds_info = tfds.load("Imagenet2012", split=['train', 'validation'],
                                                 data_dir=BASEDIR, download=True, shuffle_files=True,
                                                 as_supervised=True, with_info=True, 
                                                 download_and_prepare_kwargs= {'download_dir':DOWNLOADIR})
            
        return ds_train, ds_test, ds_info

# @tf.function
def normalize_cifar10(image, label):
    """Normalizes the CIFAR-10 Dataset by casting all values to flat and then standardizing each image separately
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, tf.one_hot(label,10)

def normalize_cifar10_resnet(image, label):
    """Normalizes the CIFAR-10 Dataset for ResNets by casting all values to flat and then standardizing each image separately
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.pad(image, [[4,4], [4,4], [0,0]])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, [32,32,3])

    return image, tf.one_hot(label,10)

def normalize_pacm(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, tf.one_hot(label,2)

def normalize_cifar100(image, label):
    """Normalizes the CIFAR-100 Dataset by casting all values to flat and then standardizing each image separately
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, label #tf.one_hot(label,100)

def normalize_cifar100_train(image, label):
    image = tf.cast(image, tf.float32)
    
    image = tf.image.per_image_standardization(image)
    image = tf.pad(image, [[4,4], [4,4], [0,0]])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, [32,32,3])
    
    return image, label
    
    
def normalize_mnist(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, [tf.shape(image)[0]*tf.shape(image)[1]]), tf.one_hot(label,10)

# @tf.function
def resize_image_keep_aspect(image, lo_dim):
    # Take width/height
    
    initial_width = tf.cast(tf.shape(image)[0], tf.float32)
    initial_height = tf.cast(tf.shape(image)[1], tf.float32)

    # Take the shorter side, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = min_ / lo_dim #tf.cast(lo_dim, dtype=tf.float32)
    # ratio = tf.cast(min_, dtype=tf.float32) / lo_dim #tf.cast(lo_dim, dtype=tf.float32)

    new_width = tf.math.round(initial_width / ratio)
    new_height = tf.math.round(initial_height / ratio)

    return tf.image.resize(image, [new_width, new_height])

# @tf.numpy_function
def resize_image_keep_aspect2(image, lo_dim):
    # Take width/height
    initial_width = image.numpy().shape[0]
    initial_height = image.numpy().shape[1]

    # Take the shorter side, and use it for the ratio
    min_ = min(initial_width, initial_height)
    ratio = min_ / lo_dim

    new_width = round(initial_width / ratio)
    new_height = round(initial_height / ratio)

    return tf.image.resize(image, [new_width, new_height])

def normalize_imagenet(image, label):
    # image = tf.cast(image, tf.float32)
    #tf.keras.preprocessing.image.smart_resize(image, [])
    image = resize_image_keep_aspect(image, 256.)
    #image = tf.image.resize(image, [256, 256], preserve_aspect_ratio=True)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, [224, 224, 3])
    # image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image, label #tf.one_hot(label,1000)

def normalize_imagenet_test(image, label):
    # image = tf.cast(image, tf.float32)
    #image = tf.image.resize(image, [224, 224])
    # image = tf.image.random_flip_up_down(image)
    image = resize_image_keep_aspect(image, 224)
    image = tf.image.random_crop(image, [224, 224, 3])
    # image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image, label #tf.one_hot(label,1000)
# @tf.function
def prep_data(ds, ds_info, batch_size, testset=False, distributed=False, verbose=0):
    """Prepares the dataset

    Args:
        ds (tf.dataset): dataset to be used
        ds_info (tf.dataset.info): info about the dataset

    Returns:
        tf.dataset: the standardized and batched dataset
    """
    if ds_info.name == "mnist":
        ds = ds.map(normalize_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # elif ds_info.name == "cifar10" and testset == False:
    elif ds_info.name == "cifar10":
        ds = ds.cache()
        if testset == False: 
            ds = ds.shuffle(ds_info.splits["train"].num_examples)#.repeat()
        else:
            ds = ds.shuffle(ds_info.splits["test"].num_examples)#.repeat()
        ds = ds.map(normalize_cifar10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        
    # elif ds_info.name == "cifar10" and testset == True:
    #     ds = ds.map(normalize_cifar10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ds_info.name == "cifar100":
        ds = ds.cache()
        if testset == False: 
            ds = ds.shuffle(ds_info.splits["train"].num_examples).repeat(2)
            ds = ds.map(normalize_cifar100_train, num_parallel_calls=TF_AUTOTUNE)
        else:
            ds = ds.shuffle(ds_info.splits["test"].num_examples)#.repeat()
            ds = ds.map(normalize_cifar100, num_parallel_calls=TF_AUTOTUNE)
        #ds = ds.cache()
    elif ds_info.name == "imagenet2012" and testset == False:
        # ds = ds.shuffle(1024)
        # ds = ds.cache()
        ds = ds.map(normalize_imagenet, num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache()
        ds = ds.apply(tf.data.experimental.ignore_errors())
    elif ds_info.name == "imagenet2012" and testset == True:
        # ds = ds.shuffle(64)
        # ds = ds.cache()
        ds = ds.map(normalize_imagenet_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        ds = ds.apply(tf.data.experimental.ignore_errors())
    #ds = ds.cache()
    #ds = ds.shuffle(ds_info.splits['train'].num_examples)
    
    ds = ds.batch(batch_size)
    
    return ds.prefetch(TF_AUTOTUNE)

def data_handler(dataset: str, batch_size=128, distributed=False, verbose=0):
    """Pipeline that loads and prepares the dataset

    Args:
        dataset (str): name of the dataset, either "mnist" or "cifar"

    Returns:
        tf.dataset: standardized and batched dataset
    """
    ds_train, ds_test, ds_info = data_loader(dataset)

    ds_train = prep_data(ds=ds_train, 
                         ds_info=ds_info, 
                         batch_size=batch_size,
                         distributed=distributed, 
                         verbose=verbose)
    
    ds_test = prep_data(ds=ds_test, 
                        ds_info=ds_info, 
                        batch_size=batch_size, 
                        testset = True,
                        distributed=distributed, 
                        verbose=verbose)

    return ds_train, ds_test, ds_info


# IMAGENET SPECIFIC
################################################################################



def prep_imagenet(ds, batch_size):
    ds = ds.shuffle(batch_size)
    ds = ds.map(normalize_imagenet, num_parallel_calls = TF_AUTOTUNE)
    #ds = ds.cache()
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.batch(batch_size)#.prefetch(TF_AUTOTUNE)

def prep_imagenet_test(ds, batch_size):
    ds = ds.map(normalize_imagenet_test, num_parallel_calls = TF_AUTOTUNE)
    ds = ds.cache()
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.batch(batch_size)#.prefetch(TF_AUTOTUNE)

def data_handler_imagenet(batch_size=128):
    
    BASEDIR = "data/tfds/tfds_data"
    DOWNLOADIR = "data/raw"

    (ds_train, ds_test) = tfds.load("imagenet2012", 
                                    split=['train', 'validation'],
                                    data_dir=BASEDIR, 
                                    download=True, 
                                    shuffle_files=True,
                                    as_supervised=True, 
                                    with_info=False, 
                                    download_and_prepare_kwargs= {'download_dir':DOWNLOADIR})
    
    ds_train = prep_imagenet(ds_train, batch_size=batch_size)
    ds_test = prep_imagenet_test(ds_test, batch_size=batch_size)

    return ds_train, ds_test