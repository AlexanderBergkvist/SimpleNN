import numpy as np
import math
import time
input_layer_neurons = 2
l1_neurons = 2
l2_neurons = 2


test_x = np.array([[[0,1]],
          [[1,0]],
          [[1,1]],
          [[0,0]]])

test_y = np.array([[[0,1]],
          [[1,0]],
          [[1,0]],
          [[0,1]]])

l1_w = np.random.rand(input_layer_neurons,l1_neurons)
l2_w = np.random.rand(l1_neurons,l2_neurons)
l1_b = np.random.rand(1,l1_neurons)
l2_b = np.random.rand(1,l2_neurons)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def activation(x): #Sigmoid
    [[x1,x2]] = x
    z = np.array([[sigmoid(x1), sigmoid(x2)]])
    return z


lr = 10


def sigmoid_prime(x):
    return math.exp(-x) / ((1 + math.exp(-x)) ** 2)

def inverse_activation(x): #Sigmoid
    [[x1,x2]] = x
    z = np.array([[sigmoid_prime(x1), sigmoid_prime(x2)]])
    return z


def backpropogation(x):
    global l2_w, l2_b, l1_w
    for (i,sample) in enumerate(x):
        x1 = np.matmul(sample,l1_w)
        x1 = x1 + l1_b
        act1 = activation(x1)

        x2 = np.matmul(act1,l2_w)
        x2 = x2 + l2_b
        y_hat = activation(x2)
        g_t = test_y[i]

        print(i,"Current prediction:", y_hat, "actual", g_t)
        dw5 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * act1[0][0] * lr
        dw7 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * act1[0][1] * lr

        dw6 = (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * act1[0][0] * lr
        dw8 = (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * act1[0][1] * lr

        db3 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * lr
        db4 = (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * lr

        d_l2_w = np.array([[dw5,dw6],[dw7,dw8]])
        d_l2_b = np.array([[db3,db4]])




        dw1 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * l2_w[0][0] * inverse_activation(x1)[0][0] * sample[0][0] * lr
        dw1 += (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * l2_w[0][1] * inverse_activation(x1)[0][0] * sample[0][0] * lr

        dw2 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * l2_w[1][0] * inverse_activation(x1)[0][1] * sample[0][0] * lr
        dw2 += (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * l2_w[1][1] * inverse_activation(x1)[0][1] * sample[0][0] * lr

        dw3 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * l2_w[0][0] * inverse_activation(x1)[0][0] * sample[0][1] * lr
        dw3 += (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * l2_w[0][1] * inverse_activation(x1)[0][0] * sample[0][1] * lr

        dw4 = (g_t[0][0] - y_hat[0][0]) * inverse_activation(x2)[0][0] * l2_w[1][0] * inverse_activation(x1)[0][1] * sample[0][1] * lr
        dw4 += (g_t[0][1] - y_hat[0][1]) * inverse_activation(x2)[0][1] * l2_w[1][1] * inverse_activation(x1)[0][1] * sample[0][1] * lr



        d_l1_w = np.array([[dw1,dw2],[dw3,dw4]])

        #print(d_l1_w,"\n")
        #print(d_l2_w,"\n")
        #print(d_l2_b,"\n\n\n")


        l1_w += d_l1_w
        l2_w += d_l2_w
        l2_b += d_l2_b
while True:
    #predict(test_x)
    time.sleep(0.1)
    backpropogation(test_x)
    print("\n")
