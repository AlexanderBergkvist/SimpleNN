import numpy as np
import math
import time
l1_prev_layer = 3
l2_prev_layer = 3
l3_prev_layer = 3

l1_neurons = 3
l2_neurons = 3
l3_neurons = 3

test_x = np.array([[0,0,1],
          [0,1,0],
          [0,1,1],
          [1,0,0],
          [1,0,1],
          [1,1,0],
          [1,1,1]])

test_y = np.array([[0,0,1],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0],
          [1,0,0],
          [1,0,0]])

l1_w = np.random.rand(l1_prev_layer,l1_neurons)
l2_w = np.random.rand(l2_prev_layer,l2_neurons)
l3_w = np.random.rand(l3_prev_layer,l3_neurons)

def softmax(x):
    total = sum(x)
    for (i,val) in enumerate(x):
        x[i] = val / total
    return x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def activation(x): #Sigmoid
    ans = []
    for set in x:
        set_ans = []
        for v in set:
            set_ans.append(sigmoid(v))
        set_ans = softmax(set_ans)
        ans.append(set_ans)
    return np.array(ans)

def neural_net(x):
    x = np.matmul(x,l1_w)
    x = activation(x)

    x = np.matmul(x,l2_w)
    x = activation(x)

    x = np.matmul(x,l3_w)
    x = activation(x)
    return x

y_hat = neural_net(test_x)

lr = 0.01

def inverse_sigmoid(x):
    return math.exp(-x) / (1 + math.exp(-x))**2

def cost_function(y_hat,y_act):
    answer = []
    for (i,_) in enumerate(y_hat):
        c = (y_hat[i] - y_act[i]) ** 2
        answer.append(sum(c))
    return answer

    #return sum((y_hat - y_act) ** 2)

cost = cost_function(y_hat, test_y)
def inverse_activation(x): #Sigmoid
    ans = []
    for set in x:
        set_ans = []
        for v in set:
            set_ans.append(inverse_sigmoid(v))
        set_ans = softmax(set_ans)
        ans.append(set_ans)

    return np.array(ans)

def back_propagation(xx):
    global l1_w, l2_w, l3_w, cost
    for (i,a) in enumerate(xx):
        x = np.array([a])

        x1 = np.matmul(x,l1_w)
        act1 = activation(x1)

        x2 = np.matmul(act1,l2_w)
        act2= activation(x2)

        x3 = np.matmul(act2,l3_w)

        dw1 = 2 * cost[i] * inverse_activation(x3) * l3_w * l2_w * inverse_activation(x1) * x
        dw2 = 2 * cost[i] * inverse_activation(x3) * l3_w * (inverse_activation(x2) * act1 + l2_w * inverse_activation(x1) * x)
        dw3 = 2 * cost[i] * inverse_activation(x3) * (act2 + l3_w * (l2_w * inverse_activation(x1) * x))


        l1_w -= dw1 * lr
        l2_w -= dw2 * lr
        l3_w -= dw3 * lr

while True:
    time.sleep(1)
    print("\n\n\n\n\n")

    y_hat = neural_net(test_x)
    for (i, _) in enumerate(y_hat):
        print(y_hat[i], "\t\t\tactual", test_y[i], "cost:", cost[i])
    print(l1_w,"\n")
    print(l2_w,"\n")
    print(l3_w,"\n")
    back_propagation(test_x)
