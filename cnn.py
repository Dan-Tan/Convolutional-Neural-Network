# -*- coding: utf-8 -*-
"""CNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sw2j45Vs2hr986qmAzIEzdBrxlEJAKMX
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive/')
import tensorflow as tf
tf.test.gpu_device_name()

class Convolutional_Layer():

    def __init__(self, kernel_shape, padding, learning_rate, momentum):

        self.kernel_shape = kernel_shape
        self.padding = padding
        self.learning_rate = learning_rate
        self.momentum = momentum

        # kernels are initalized in a 2-d array where the number of rows is the number of columns to allow convolutions to be caclulated using the dot product
        self.kernels = np.random.normal(0, 1, (self.kernel_shape[0], self.kernel_shape[1] * self.kernel_shape[2] * self.kernel_shape[3]))
        self.leaky_av = np.zeros_like(self.kernels)
        # counter for momentum to be able to initalize the first step
        self.counter = 0

    def add_padding(self, inputs):

        depth, col, row = inputs.shape
        a = self.padding
        padded = np.zeros((depth, col + 2 * a, row + 2 * a))
        for f in range(depth):
            padded[f, a : a + col, a : a + row] = inputs[f, :, :]

        return padded

    def ReLU(self, z):
        z[z < 0] = 0 
        return z

    def ReLU_prime(self, z):
        z[z > 0] = 1
        z[z < 0] = 0
        return z

    def forward(self, inputs):

        # trying to implement convolution through the dot product by using vectorisation
        inputsp = self.add_padding(inputs) / inputs.max()
        # dimensions of output for reshaping to propagate to next layer
        depth, col, row = (self.kernel_shape[0], inputsp.shape[1] - self.kernel_shape[2] + 1, inputsp.shape[2] - self.kernel_shape[3] + 1)
        # saving indices for use in backpropagation
        self.col = col
        self.row = row
        self.depth = depth
        output = np.zeros((self.kernel_shape[0], self.kernel_shape[1] * col * row))
        len_window = self.kernel_shape[1] * self.kernel_shape[2] * self.kernel_shape[3]
        vecinput = np.zeros((len_window, col * row))

        # counter for indexing vector input
        counter = 0
        for b in range(col):
            for c in range(row):
                #vectorize input and kernels
                vecinput[:, counter] = inputsp[:, b : b + self.kernel_shape[2], c : c + self.kernel_shape[3]].ravel().reshape((len_window,))
                counter += 1

        output = np.dot(self.kernels, vecinput)
        output = self.ReLU(output)
        self.outputs = output
        # storing current shape for use in back propagation (forward will always be called before backpropagation)
        self.im2colshape = output.shape
        self.vecinput = vecinput
        output = output.reshape((self.kernel_shape[0], col, row))
        
        #storing object parameters for use in other methods hence forward must be called before back_prop
        #output = self.ReLU(output)

        self.inputsp = inputsp
        self.output = output

        return output

    def back_prop(self, dLdO):

        vecinput = self.vecinput
        dLdO = dLdO.reshape(self.im2colshape)
        ReLU_primeoutput = self.ReLU_prime(self.outputs)
        dLdO = dLdO * ReLU_primeoutput
        dLdK = np.dot(dLdO, vecinput.T)
        tempstor = np.zeros_like(self.inputsp)
        dLdX = np.dot(self.kernels.T, dLdO)
        dLdXs = np.zeros_like(self.inputsp)
        counter = 0
        for a in range(self.col):
            for b in range(self.row):
                tempstor[:, a : a + self.kernel_shape[2], b : b + self.kernel_shape[3]] = dLdX[:, counter].reshape((self.kernel_shape[1], self.kernel_shape[2], self.kernel_shape[3]))
                counter += 1
                dLdXs += tempstor
                tempstor = np.zeros_like(self.inputsp)

        if self.counter == 0:
            dLdK = dLdK
            self.leaky_av = dLdK
            self.counter += 1
            
        else: 
            self.leaky_av = self.momentum * self.leaky_av + (1 - self.momentum) * dLdK

        self.kernels -= self.learning_rate * self.leaky_av


        return dLdXs

class Max_Pool_Layer():

    def __init__(self, shape, stride):

        self.shape = shape
        self.stride = stride

    def forward(self, inputs):
        
        self.inputs = inputs
        depth , col, row = inputs.shape
        output = np.zeros((depth, col // self.stride, row // self.stride))
        self.output = output
        back = np.zeros_like(inputs)

        for a in range(depth):
            for b in range(col // self.stride):
                i = b
                b = b * self.stride
                for c in range(row // self.stride):
                    j = c
                    c = c * self.stride
                    d = inputs[a, b : b + self.shape[0], c : c + self.shape[1]]
                    f = d.max()
                    for cols, e in enumerate(d):
                        for rows, g in enumerate(e):
                            if g == f:
                                max_col, max_row = b + cols, c + rows
                    back[a, max_col, max_row] = 1
                    output[a, i, j] = f
        self.back = back
        return output

    def back_prop(self, dLdO):
        dLdX = np.zeros_like(self.inputs)
        for g in range(dLdO.shape[0]):
            for f in range(dLdO.shape[1]):
                i = f * self.stride
                for e in range(dLdO.shape[2]):
                    j = e * self.stride

                    check = self.back[g, i : i + self.shape[0], j : j + self.shape[1]]
                    for cols, a in enumerate(check):
                        for rows, b in enumerate(a):
                            if b == 1:
                                dLdX[g, i + cols, j + rows] = dLdO[g, f, e]
        #print(dLdX)

        return dLdX

class Fully_Connected_Layer():

    def __init__(self, inputlayersize, hiddenlayersize, no_classes, learning_rate, momentum):

        self.inputlayersize = inputlayersize
        self.hiddenlayersize = hiddenlayersize
        self.outputlayersize = no_classes
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.W1 = np.random.randn(self.inputlayersize, self.hiddenlayersize) /100
        self.W2 = np.random.randn(self.hiddenlayersize, self.outputlayersize) /100
        self.leaky_av_W1 = np.zeros_like(self.W1)
        self.leaky_av_W2 = np.zeros_like(self.W2)
        self.counter = 0
    #activation functions and derivatives
    def ReLU(self, z):
        z[z < 0] = 0
        return z

    def ReLU_prime(self, z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


    def forward(self, inputs):

        self.inputs = inputs / inputs.max()

        #saving activities and totals for use in backprop
        self.z2 = np.dot(inputs, self.W1)
        self.a2 = self.ReLU(self.z2).reshape((1, self.hiddenlayersize))
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.softmax(self.z3)
        yHat = self.a3
        return yHat

    def cross_entropy(self, yHat, gtruth):
        
        for ind, e in enumerate(gtruth):
            if e == 1:
                if yHat[:, ind] < 0.0000000001:
                    yHat[:, ind] = 0.0000000001
                # print(yHat[:, ind])
                loss = -np.log(yHat[:, ind])
        return loss
    
    def cross_entropy_prime(self, yHat, gtruth):
        
        for ind, e in enumerate(gtruth):
            if e == 1:
                self.gtruthind = ind
                outc = -(1/yHat[:, ind])

        return outc

    def cost_function(self, yHat, gtruths):
        cost = np.sum((gtruths - yHat)**2)
        return cost

    def cost_function_back(self, yHat, gtruths):
        return (gtruths - yHat)

    def softmax(self, z):

        self.exp_z = np.exp(z - np.max(z))
        self.exp_z_sum = np.sum(self.exp_z)
        
        return (self.exp_z / self.exp_z_sum)
    
    def softmax_prime(self, z, gtruth):

        b = self.softmax(z)

        dOdS = np.zeros((1, len(gtruth)))
        for ind in range(len(gtruth)):
            if ind == self.gtruthind:
                dOdS[:, ind] = b[:, ind] * (1 - b[:, ind])
            else:
                dOdS[:, ind] = - b[:, self.gtruthind] * b[:, ind]

        return dOdS

    def back_prop(self, yHat, gtruths):

        self.loss = self.cross_entropy(yHat, gtruths)
        # print(self.loss)
        gradinit = self.cross_entropy_prime(yHat, gtruths)

        delta3 = (gradinit * self.softmax_prime(self.z3, gtruths))
        dLdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.ReLU_prime(self.z2)
        dLdW1 = np.dot(self.inputs.T.reshape((self.inputs.shape[0], 1)), delta2)

        dLdX = np.dot(delta2, self.W1.T)

        if self.counter == 0:
            self.leaky_av_W1 = dLdW1
            self.leaky_av_W2 = dLdW2
            self.counter += 1

        else:
            self.leaky_av_W1 = self.momentum * self.leaky_av_W1 + (1 - self.momentum) * dLdW1
            self.leaky_av_W2 = self.momentum * self.leaky_av_W2 + (1 - self.momentum) * dLdW2

        #update the weights (stochastic gradient descent)
        self.W1 -= self.learning_rate * self.leaky_av_W1
        self.W2 -= self.learning_rate * self.leaky_av_W2

        return dLdX

class Convolutional_Neural_Network():

    def __init__(self, c11, max1, c12, max2, fc):
        self.c11 = c11
        self.max1 = max1
        self.c12 = c12
        self.max2 = max2
        self.fc = fc

    def forward(self, inputs):

        #the forward propagation of entire network layers
        self.inputs = inputs
        self.fm11 = self.c11.forward(inputs)
        self.in2 = self.max1.forward(self.fm11)
        self.fm12 = self.c12.forward(self.in2)
        self.infc = self.max2.forward(self.fm12)
        output = self.infc.ravel()
        self.out = self.fc.forward(output)
        out = self.out
        return out


    # dropout helps reduce overfitting stopping the network from 'relying on one kernel'
    # often said to be training multiple large networsk at once, ignoring random outputs

    def dropout(self, dropout):

        c11_indn = int(round(dropout * self.c11.kernel_shape[0], 0))
        c12_indn = int(round(dropout * self.c12.kernel_shape[0], 0))
        W1_indn = int(round(dropout * self.fc.hiddenlayersize, 0))
        W2_indn = int(round(dropout * self.fc.outputlayersize, 0))

        c11_dropout = np.random.choice(range(self.c11.kernel_shape[0]), c11_indn, replace = False)
        c12_dropout = np.random.choice(range(self.c12.kernel_shape[0]), c12_indn, replace = False)
        W1_dropout = np.random.choice(range(self.fc.hiddenlayersize), W1_indn, replace = False)
        W2_dropout = np.random.choice(range(self.fc.outputlayersize), W2_indn, replace = False)

        c11_tempkernel = np.zeros((c11_indn, self.c11.kernel_shape[1] * self.c11.kernel_shape[2] * self.c11.kernel_shape[3]))
        c12_tempkernel = np.zeros((c12_indn, self.c12.kernel_shape[1] * self.c12.kernel_shape[2] * self.c12.kernel_shape[3]))
        W1_temp = np.zeros((self.fc.inputlayersize, W1_indn))
        W2_temp = np.zeros((self.fc.hiddenlayersize, W2_indn))
        
        # for ind, e in enumerate(c11_dropout):
        #     c11_tempkernel[ind, :] = self.c11.kernels[e, :]
        #     self.c11.kernels[e, :] = 0 
                                                   
        # for ind, e in enumerate(c12_dropout):
        #     c12_tempkernel[ind, :] = self.c12.kernels[e, :]
        #     self.c12.kernels[e, :] = 0
                                                   
        for ind, e in enumerate(W1_dropout):
            W1_temp[:, ind] = self.fc.W1[:, e]
            self.fc.W1[:, e] = 0

        for ind, e in enumerate(W2_dropout):
            W2_temp[:, ind] = self.fc.W2[:, e]
            self.fc.W2[:, e] = 0

        return c11_dropout, c12_dropout, W1_dropout, W2_dropout, c11_tempkernel, c12_tempkernel, W1_temp, W2_temp

    def pick_up(self, cache):

        c11_dropout, c12_dropout, W1_dropout, W2_dropout, c11_tempkernel, c12_tempkernel, W1_temp, W2_temp = cache

        # for ind, e in enumerate(c11_dropout):
        #     self.c11.kernels[e, :] = c11_tempkernel[ind, :]

        # for ind, e in enumerate(c12_dropout):
        #     self.c12.kernels[e, :] = c12_tempkernel[ind, :]

        for ind, e in enumerate(W1_dropout):
            self.fc.W1[:, e] = W1_temp[:, ind]

        for ind, e in enumerate(W2_dropout):
            self.fc.W2[:, e] = W2_temp[:, ind]

    def back_prop(self, inputs, gtruths, dropout):

        # cache = self.dropout(dropout)

        self.yHat = self.forward(inputs)
        dLdX = self.fc.back_prop(self.yHat, gtruths)
        dLdX = dLdX.reshape(self.max2.output.shape)
        din2 = self.max2.back_prop(dLdX)
        dinX12 = self.c12.back_prop(din2)
        din1 = self.max1.back_prop(dinX12)
        _ = self.c11.back_prop(din1)

        # self.pick_up(cache)

        if np.array_equal(self.yHat.round(0).reshape((self.fc.outputlayersize, )), gtruths):
            return 1
        else:
            return 0

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

c11 = Convolutional_Layer((32, 2, 5, 5), 2, 0.01, 0.9)
max1 = Max_Pool_Layer((2, 2), 2)
c12 = Convolutional_Layer((64, 32, 5, 5), 2, 0.01, 0.9)
max2 = Max_Pool_Layer((2, 2), 2)
fc = Fully_Connected_Layer(3136, 128, 10, 0.01, 0.9)

CNN = Convolutional_Neural_Network(c11, max1, c12, max2, fc)

class Trainer():

    def __init__(self, CNN):

        self.CNN = CNN


    def callback_function(self, cost_sum, num_corr, epoch):

        av_cost = cost_sum /10000
        acc = num_corr / 10000 * 100

        print("epoch:", epoch+1)
        print("av_loss:", av_cost)
        print("accuracy:", acc, "%")     

        # perhaps should be greater then 85%

        if acc > 95:
            return 1
        else:
            return 0


    def train(self, inputs, gtruths, epochs, dropout):
        self.inputs = inputs

        # self.CNN.c11.kernels = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/kernelsc11.npy')
        # self.CNN.c12.kernels = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/kernelsc12.npy')
        # self.CNN.fc.W1 = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/weight1.npy')
        # self.CNN.fc.W2 = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/weight2.npy')

        for e in range(epochs):
            permutation = np.random.permutation(len(inputs))
            inputs = inputs[permutation]
            gtruths = gtruths[permutation]
            cost = 0
            corr = 0

            if e == 1:
                self.CNN.c11.learning_rate = 0.0001
                self.CNN.c12.learning_rate = 0.0001
                self.CNN.fc.learning_rate = 0.0001
            elif e == 3:
                self.CNN.c11.learning_rate = 0.00001
                self.CNN.c12.learning_rate = 0.00001
                self.CNN.fc.learning_rate = 0.00001
            elif e > 4:
                self.CNN.c11.learning_rate *= 0.01
                self.CNN.c12.learning_rate *= 0.01
                self.CNN.fc.learning_rate *= 0.01
       

            for ind, (example, gtruth) in enumerate(zip(inputs, gtruths)):

                if ind % 1000 == 0 and ind != 0:
                    print("av_cost:", cost / ind)
                    print("accuracy:", corr / ind * 100, "%")
                a = self.CNN.back_prop(example, gtruth, dropout)

                cost += self.CNN.fc.cost_function(self.CNN.out, gtruth)
                corr += a

            np.save('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/{}'.format("kernelsc11"), CNN.c11.kernels)
            np.save('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/{}'.format("kernelsc12"), CNN.c12.kernels)
            np.save('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/{}'.format("weight1"), CNN.fc.W1)
            np.save('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/{}'.format("weight2"), CNN.fc.W2)

            finish = self.callback_function(cost, corr, e)



            if finish == 1:
                break

Trainer_test = Trainer(CNN)

inputs = x_train[0:10000] 

inputss = np.zeros((10000, 2, 28, 28))

for ind, f in enumerate(inputs):
    inputss[ind, 0, :, :] = f
    inputss[ind, 1, :, :] = f

inputss = inputss / 255

output = np.zeros((10000, 10))
for ind, f in enumerate(y_train[0:10000]):
    output[ind, f] = 1

Trainer_test.train(inputss, output, 10, 0.2)

im = 1

test_input = inputss[im, :, :, :]
plt.imshow(inputss[im, 0, :, :])
test = CNN.forward(test_input)
print(test.round(0))
fm11 = CNN.fm11[6, :, :] * 100
fm12 = CNN.fm12[12, :, :] * 100

fig = plt.figure(figsize = (8, 8))
h = 28
for i in range(1, 32):
    img = CNN.fm11[i, :, :]
    fig.add_subplot(4, 8, i)
    plt.imshow(img)

jig = plt.figure(figsize = (8, 8))
h = 28
for i in range(1, 64):
    img = CNN.fm12[i, :, :]
    jig.add_subplot(8, 8, i)
    plt.imshow(img)

def Validation(test_examples, gtruths, CNN):

    CNN.c11.kernels = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/kernelsc11.npy')
    CNN.c12.kernels = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/kernelsc12.npy')
    CNN.fc.W1 = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/weight1.npy')
    CNN.fc.W2 = np.load('/content/drive/My Drive/Colab Notebooks/Neural Network stuff/CNN better/weight2.npy')

    num_corr = 0
    for ind, (test, gtruth) in enumerate(zip(test_examples, gtruths)):
        yHat = CNN.forward(test)
        maxy = yHat.max()
        for ind, e in enumerate(yHat[0, :]):
            if e == maxy:
                yHat[:, ind] = 1
            else: 
                yHat[:, ind] = 0
        if np.array_equal(yHat.reshape((CNN.fc.outputlayersize, )).round(0), gtruth):
                num_corr += 1

    return num_corr / 1000 * 100

inputs = x_test[0:1000] 

inputss = np.zeros((1000, 2, 28, 28))

for ind, f in enumerate(inputs):
    inputss[ind, 0, :, :] = f
    inputss[ind, 1, :, :] = f

inputss = inputss / 255

output = np.zeros((1000, 10))
for ind, f in enumerate(y_test[0:1000]):
    output[ind, f] = 1

Accuracy = Validation(inputss, output, CNN)
print(Accuracy)

a = np.random.choice(range(10), 5, replace = False)
print(a)