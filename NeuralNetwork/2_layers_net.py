#!/usr/bin/env
# -*- coding:utf-8 -*-
'''
commented by xxxmy.
email:371802059@qq.com
'''
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x) )

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'): #layers :层数，包含输入层。为list,元素为节点数，不包含bias
        if activation == 'Logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        #for i in range(len(layers)-1):#这里len(layers)=3,i=0,1
            # [0,1) * 2 - 1 => [-1,1) => * 0.25 => [-0.25,0.25)  
            #self.weights.append( (2*np.random.random((layers[i-1] + 1, layers[i]+1 ))-1 ) * 0.25 )#random_sample:Return random floats in the half-open interval [0.0, 1.0).
            #self.weights.append((2 * np.random.random( (layers[i]+1, (layers[i+1])) - 1) * 0.25 )
        for i in range(0, len(layers)-1):
             m = layers[i]  # 第i层节点数
             n = layers[i+1]  # 第i+1层节点数
             wm = m + 1    
             wn = n + 1     #把bias加上
             if i == len(layers)-2:
                 wn = n
             weight = np.random.random((wm, wn)) * 2 - 1
             self.weights.append(0.25 * weight)
    def fit(self, X, y, learning_rate=0.2, epochs = 10000):    #参数X为样本数据，y为标签数据，learning_rate 为学习率默认0.2，epochs 为迭代次数默认值10000
        X = np.atleast_2d(X)
        X = np.column_stack((X, np.ones(len(X))))#这里加一层bias #列合并/扩展：np.column_stack()；行合并/扩展：np.row_stack()
        y = np.array(y)#数据预处理

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # shape[0]返回行数，随机产生一个数，对应行号，即数据集编号 #If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
            a = [X[i]]# 抽出这行的数据集,作为输入的训练数据集a^1
            # 正向计算a^l,得到所有a[l]
            for l in range(len(self.weights)): #l=0，1
                a.append(self.activation( np.dot(a[l], self.weights[l])))#得到a^2，a^3,a=[a^1,a^2,a^3]

            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])] #delta^

            ## 求deltas=[delata^3,delta^2]
            layerNum = len(a) - 2  #len(a)=3
            for j in range(layerNum, 0, -1): # 倒数第二层开始,-1表示反序，所以求出的delta顺序应该反的
               # deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))
                 deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))
            
            deltas.reverse() #反向排序,deltas=[delata^2,delta^3]
            # 梯度下降法更新权值
            for i in range(len(self.weights)):
                al = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * al.T.dot(delta)  

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__=='__main__':
    nn = NeuralNetwork([2, 2, 1], 'tanh')  
    x = np.array([[0,0],[0,1],[1,0],[1,1]])  
    y = np.array([0,1,1,0])  
    nn.fit(x, y)  
    for i in [[0,0],[0,1],[1,0],[1,1]]:  
        print(i,nn.predict(i))  





