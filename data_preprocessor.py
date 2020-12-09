import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def data_loader(dataset:str):
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

# @tf.function
def normalize_cifar(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, tf.one_hot(label,10)

def normalize_mnist(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, [tf.shape(image)[0]*tf.shape(image)[1]]), tf.one_hot(label,10)


# @tf.function
def prep_data(ds, ds_info):
    
    if ds_info.name == "mnist":
        ds = ds.map(normalize_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.map(normalize_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    #ds = ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.batch(128)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds

def data_handler(dataset: str):
    ds_train, ds_test, ds_info = data_loader(dataset)
    ds_train = prep_data(ds_train, ds_info)
    ds_test = prep_data(ds_test, ds_info)

    return ds_train, ds_test