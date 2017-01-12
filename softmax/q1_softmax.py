import  numpy as np
import unittest

'''
    change the variable to the maritx
'''
def sigmod(x):
    return 1./(1+np.exp(-x))

def sigmod_prime(x):
    return sigmod(x)*(1-sigmod(x))

def softmax(x):
    '''
    compute the softmax function for each row of input x
    :param x:
    :return:
    '''
    softMaxValue=[]
    if len(x.shape) > 1:   # x is not a vector
        x = np.exp(x)
        tmp = np.sum(x,axis=1)
        for index in np.arange(x.shape[0]):
            temp = x[index,::]/tmp[index]
            softMaxValue.append(temp)
     #   softMaxValue = x/tmp   #boarding is valid?
    else:
        x = np.exp(x)
        tmp = np.sum(x)
        softMaxValue = x/tmp
    return softMaxValue

class neutral_network():
    '''
    data is like this "[[[1,1,1,1],[0]],[[1,1,1,1],[1]]]",first is data x,last is the label,size is like this '[2,3,2]',like first layer size is 2
    and the second layer size is 3,the last layer size is 2
    '''
    def __init__(self,data,size):
        self.size = size
        self.numlayer = len(size)
        self.SampleDataSize = len(data)
        self.initWeigth()
        self.processData(data)


    def processData(self,data):
        x_data = []
        y_data = []
        for x in range(self.SampleDataSize):
            x_data.append(data[x][0])
            y_data.append(data[x][1])
        self.datax = np.array(x_data)
        self.datay = np.array(y_data)

    def initWwight(self):
        self.weights = [np.random.randn(y,x) for x,y in zip(self.size[:-1],self.size[1:])]
        self.bias = [np.random.randn(y,1) for y in self.size[1:]]

    def feedforward(self):
        tem = self.datax
        for w,b in zip(self.weights,self.bias):
            tem = sigmod(np.dot(w,tem)+b)
        return tem

    def cost_derivation(self,lastLaterOutput):
        return softmax(lastLaterOutput) - self.datay

    def backprop(self):
        nable_b = [np.zeros(b.shape) for b in self.bias]  # 临时变量
        nable_w = [np.zeros(w.shape) for w in self.weights]  # 临时变量
        active = self.datax
        activation = [active]
        neutral_output = []
        for w,b in zip(self.weights,self.bias):
            z = np.dot(w,active)+b
            neutral_output.append(z)
            active = sigmod(z)
            activation.append(active)
        delta = self.cost_derivation(activation[-1])*sigmod_prime(neutral_output[-1])
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activation[-2].transpose())
        for l in range(2, self.numlayer):
            z = neutral_output[-l]
            delta = (self.weights[-l + 1].transpose()) * sigmod_prime(z)
            nable_b[-l] = delta
            nable_w[-l] = np.dot(activation[-l - 1], delta)
        return (nable_b, nable_w, activation)



