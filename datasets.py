from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
import numpy as np


def read_data(type):
    if type == 'mnist':
        # MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)
        num_classes = 10
    else:
        # CIFAR-10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        num_classes = 10

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, num_classes)
    Y_test = to_categorical(y_test, num_classes)

    return (X_train, Y_train), (X_test, Y_test)