import numpy as np
import math
import time
from prep_data import prep_data
input_layer_neurons = 784
l1_neurons = 80
l2_neurons = 100
l3_neurons = 50
output_layer_neurons = 10

test_x, test_y = prep_data()
l1_w = np.random.rand(input_layer_neurons,l1_neurons) / 1000
l2_w = np.random.rand(l1_neurons,l2_neurons) / 1000
l3_w = np.random.rand(l2_neurons,l3_neurons) / 1000
l4_w = np.random.rand(l3_neurons,output_layer_neurons) / 1000


l1_b = np.random.rand(1,l1_neurons) / 1000
l2_b = np.random.rand(1,l2_neurons) / 1000
l3_b = np.random.rand(1,l3_neurons) / 1000
l4_b = np.random.rand(1,output_layer_neurons) / 1000


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def alt_activation(x): #Sigmoid
    z = list(map(sigmoid, x[0]))
    return np.array([z])

def activation(x): #Sigmoid
    x = np.clip(x,-500,500)
    x = 1 / (1 + np.exp(-x))
    return x
lr = 1


def sigmoid_prime(x):
    return math.exp(-x) / ((1 + math.exp(-x)) ** 2)

def alt_inverse_activation(x): #Sigmoid
    z = list(map(sigmoid_prime, x[0]))
    return np.array([z])

def inverse_activation(x): #Sigmoid
    x = np.clip(x,-500,500)
    x = np.exp(-x) / ((1 + np.exp(-x))^2)
    return x



def backpropogation(x,epoch):
    guess = []
    actual_gt = []
    global l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b
    for (i,sample) in enumerate(x):

        x1 = np.matmul(sample,l1_w)
        x1 = x1 + l1_b
        act1 = activation(x1)

        x2 = np.matmul(act1,l2_w)
        x2 = x2 + l2_b
        act2 = activation(x2)

        x3 = np.matmul(act2,l3_w)
        x3 = x3 + l3_b
        act3 = activation(x3)

        x4 = np.matmul(act3,l4_w)
        x4 = x4 + l4_b
        y_hat = activation(x4)


        g_t = test_y[i]


        db4 = (g_t - y_hat) * inverse_activation(x4)
        dw4 = np.matmul(act3.T , db4)

        db3 = np.matmul(db4, l4_w.T) * inverse_activation(x3)
        dw3 = np.matmul(act2.T, db3)

        db2 = np.matmul(db3, l3_w.T) * inverse_activation(x2)
        dw2 = np.matmul(act1.T, db2)

        db1 = np.matmul(db2, l2_w.T) * inverse_activation(x1)                           #DIS THE PATTERN
        dw1 = np.matmul(sample.T, db1)


        l1_w += dw1 * lr
        l1_b += db1 * lr

        l2_w += dw2 * lr
        l2_b += db2 * lr

        l3_w += dw3 * lr
        l3_b += db3 * lr

        l4_w += dw4 * lr
        l4_b += db4 * lr
        if i % 1000 == 0:
            print("Epoch:",epoch,"\nBatch:",i,"\nCurrent prediction:", y_hat, "\nActual", g_t, "\n\n")
    return guess,actual_gt


def display(current_guess, ground_truth):
    global iter
    #if iter == 20:
    if True:
        for (i,y_hat) in enumerate(current_guess):
            print(i,"Current prediction:", y_hat, "actual", ground_truth[i])

        print("\n\n")
        time.sleep(1)
        iter = 0
    else:
        iter +=1










epoch = 1
while True:
    backpropogation(test_x,epoch)
    epoch += 1
