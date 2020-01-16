from tensorflow import keras
import numpy as np
fashion_mnist = keras.datasets.fashion_mnist

def prep_data():

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255

    train_images = np.reshape(train_images, (60000,1,-1))

    one_hot_train_labels = []


    for i in train_labels:
        answer = np.zeros(10)
        answer[i] = 1
        one_hot_train_labels.append([answer])

    train_labels = np.array(one_hot_train_labels)

    return train_images, train_labels
