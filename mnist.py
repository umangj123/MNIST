import sys
import numpy as np

input_list = []
output_list = []
test_input_list = []
test_output_list = []
output_value = []
weight_list = [None]
bias_list = [None]
a = []


def activation(x): return 1 / (1 + np.exp(-x))


def derivative_activation(x): return activation(x) * (1 - activation(x))


def generate_random(network):
    for e in range(0, len(network) - 1):
        weight_layer = 2 * np.random.rand(network[e], network[e + 1]) - 1
        weight_list.append(weight_layer)
    for f in range(0, len(network) - 1):
        bias_layer = 2 * np.random.rand(1, network[f + 1]) - 1
        bias_list.append(bias_layer)


def backprop(weights, biases, x, y, learning_rate):
    for i in range(0, 1):
        a.clear()
        a.append(x)
        dot = [None]
        delta_list = []
        for e in weights:
            delta_list.append(0)
        for l in range(1, len(weights)):
            new_dot = a[l - 1] @ weights[l] + biases[l]
            dot.append(new_dot)
            a.append(activation(new_dot))
        delta = derivative_activation((dot[len(dot) - 1])) * (y - a[len(a) - 1])
        delta_list[len(delta_list) - 1] = delta
        for b in range(len(weights) - 2, 0, -1):
            delta = derivative_activation(dot[b]) * (delta_list[b + 1] @ weights[b + 1].transpose())
            delta_list[b] = delta
        for c in range(1, len(weights)):
            biases[c] = biases[c] + learning_rate * delta_list[c]
            weights[c] = weights[c] + learning_rate * (a[c - 1].transpose() @ delta_list[c])
    return weights, biases


def sigmoid_pnet(A, input, weight, bias):
    new_f = np.vectorize(A)
    b = input
    for c in range(1, len(weight)):
        b = new_f(b @ weight[c] + bias[c])
    return b


#w = np.loadtxt("blay3.txt")
file = "mnist_train.csv"
with open(file) as f:
    for line in f:
        pixels = line.split(",")
        y = int(pixels[0])
        input_matrix = np.random.rand(1, len(pixels) - 1)
        output = np.random.rand(1, 10)
        for i in range(0, len(pixels) - 1):
            input_matrix[0][i] = int(pixels[i + 1])
        for b in range(0, 10):
            if b == y:
                output[0][b] = 1
            else:
                output[0][b] = 0
        input_list.append(input_matrix)
        output_list.append(output)
file = "mnist_test.csv"
with open(file) as f:
    for line in f:
        pixels = line.split(",")
        y = int(pixels[0])
        output_value.append(y)
        input_matrix = np.random.rand(1, len(pixels) - 1)
        output = np.random.rand(1, 10)
        for i in range(0, len(pixels) - 1):
            input_matrix[0][i] = int(pixels[i + 1])
        for b in range(0, 10):
            if b == y:
                output[0][b] = 1
            else:
                output[0][b] = 0
        test_input_list.append(input_matrix)
        test_output_list.append(output)
#print(test_input_list)
print("pre-process complete")
generate_random([784, 500, 150, 10])
learning_rt = .05
for z in range(0, 100):
    count = 0
    for i in range(0, len(input_list)):
        # sigmoid_pnet(activation, input_list[i], circle_weight_list, circle_bias_list)
        w1, b1 = backprop(weight_list, bias_list, input_list[i], output_list[i], learning_rt)
        weight_list = w1
        bias_list = b1
    if z % 2 == 0:
        for g in range(0, len(test_input_list)):
            out = sigmoid_pnet(activation, test_input_list[g], weight_list, bias_list)
            max = -10
            index = -1
            for x in range(0, 10):
                if out[0][x] > max:
                    max = out[0][x]
                    index = x
            if index != output_value[g]:
                count += 1
        print("Misclassified", count)
    print("Finished Epoch", z)
    print(bias_list[3])
    np.savetxt('wlay1.txt', weight_list[1], fmt='%.2e')
    np.savetxt('wlay2.txt', weight_list[2], fmt='%.2e')
    np.savetxt('wlay3.txt', weight_list[3], fmt='%.2e')
    np.savetxt('blay1.txt', bias_list[1], fmt='%.2e')
    np.savetxt('blay2.txt', bias_list[2], fmt='%.2e')
    np.savetxt('blay3.txt', bias_list[3], fmt='%.2e')
# for i in range(1,len(weights)):
#     np.savetxt('wlay1.txt', weights[i], fmt='%.2e')

# print(weights)
# file2 = open("wlay1.txt", "w")
# for i in range(1,10):
#     file2.write(weights[1][i])
