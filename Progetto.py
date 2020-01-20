import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 784,)
    x_test = x_test.reshape(x_test.shape[0], 784,)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    mean_vals = np.mean(x_train, axis=0)
    std_val = np.std(x_train)
    
    X_train_centered = (x_train - mean_vals)/std_val
    X_test_centered = (x_test - mean_vals)/std_val
    
    return X_train_centered, y_train, X_test_centered, y_test
