
import random, math
import numpy as np
import pandas as pd
from numpy import linalg as LA


def g(x):
    a = 1
    return (1/(1+math.e**(-1*a*x)))

df = pd.read_csv("iris.csv")
Y = np.array(df["species"])
X = np.array(df.drop("species",axis=1))

#X = np.array([[1,2,3],[2,3,4],[4,5,6]])
#Y = np.array([10,13,1])

class Network:
    def __init__(self,nhidden,neurons_per_layer,g,epsilon,alpha) -> None:
        self._hipotesis = None
        self._W = []
        self.A = []
        self.W = []
        self.B = []
        self.per_layer = neurons_per_layer #[l0,l1,..,lf]
        self.hidden = nhidden
        self.g = g
        self.errors = []
        self.epsilon = epsilon
        self.alpha = alpha

    def train(self,it,X,Y):
        for x in range(it):
            print(x)
            self.initialize_network(X,Y)
            for row in X:
                self.initilize_input_layer(row)
                self.propagate_inputs()
                self.propagate_deltas_output(Y)
                self.propagate_deltas_on_layers(Y)
                self.update_weights(X)
            if LA.norm(self.errors[self.hidden]) < self.epsilon:
                break
        return self.W

    def get_output(self, Y):
        neurons = len(Y)
        rows = len(Y[0])
        out = []
        for x in range(neurons):
            out.append(np.array([random.random()]*rows))
        return out

    def initialize_network(self,X,Y):
        if self.hidden == 0:
            a = np.array([random.random()]*self.per_layer[len(self.per_layer)-1])
            self.W = [a]
            self.A = [X,a]
            self.B = [a]
            self.errors = [a]
        else:
            a = np.array([random.random()]*self.per_layer[0])
            self.W = [a]
            self.A = [X]
            self.B = [a]
            self.errors = [a]
            for x in range(1,len(self.per_layer)):
                b = np.array([random.random()]*self.per_layer[x])
                c = []
                for u in range(0,self.per_layer[x]):
                    c.append([random.random()]*self.per_layer[x-1])
                d = np.array(c)
                self.W.append(d)
                #e = np.array([random.random()]*(self.per_layer[x]-1))
                self.A.append(b)
                self.B.append(b)
                self.errors.append(d)
            self.A.pop(len(self.A)-1)
            self.A.append(self.get_output(Y))

    def initilize_input_layer(self,X):
        self.A[0] = X

    def propagate_inputs(self):
        for l in range(1,len(self.A)):
            for x in range(0,len(self.W[l])):
                Z = np.matmul(self.W[l][x], self.A[l-1]) + self.B[l][x]
                if l == len(self.A)-1:
                    
                    for p in range(len(self.A[l])):
                        self.A[l][p][x] = self.g(Z)
                    #self.A[l][1][x] = self.g(Z)
                    #self.A[l][2][x] = self.g(Z)
                else:
                    self.A[l][x] = self.g(Z)

    def propagate_deltas_output(self,Y):
        last = self.per_layer[len(self.per_layer)-1]
        #cambiado
        for y in range(last):
            for p in range(len(Y)):
                self.errors[len(self.per_layer)-1][y] = np.matmul(np.transpose(self.A[len(self.A)-1][p]),(Y[p][y]-self.A[len(self.A)-1][p]))
                self.B[len(self.per_layer)-1][y] = Y[p][y] - self.A[len(self.A)-1][p][y]

    def propagate_deltas_on_layers(self,Y):
        last = len(self.per_layer)-1  #cambiado
        #d0 = Y[0]-self.A[last][0]
        #d1 = Y[1]-self.A[last][1]
        #d2 = Y[2]-self.A[last][2]
        for x in range(len(self.per_layer)-2,1,-1):
            for p in range(len(Y)):
                a = np.multiply(self.A[x][p],(1-self.A[x][p]))
                d0 = Y[p] - self.A[x][p]
                b = np.matmul(d0 , np.transpose(self.W[x]))
                c = np.multiply(a,b)
                self.errors[x][p] = c

    def update_weights(self,X):
        last = len(self.per_layer)
        for x in range(last-1):
            if x == last-1:
                print("----")
                for p in range(len(Y)):
                    self.W[x] += self.alpha*np.matmul(self.A[x][p],self.errors[x][p])
            else:
                self.W[x] = self.W[x] + self.alpha*np.matmul(self.A[x],self.errors[x])
            #self.W[x] += self.alpha*np.multiply(self.A[x],self.errors[x])
        
#net = Network(2,[3,3,4,3],g,0.004,0.02)
#net.initialize_network(X)
#w = net.train(1000,X,Y)
#print(w)
#print(net.A)

class Classifier:
    def __init__(self) -> None:
        self._hipotesis = None

    def errors(self,predicted,actual):
        tests_results = list(zip(predicted,actual))
        relative_errors = [abs(predicted-actual)/actual for predicted, actual in tests_results]
        ERM = sum(relative_errors)/len(tests_results)
        return ERM

    def modify_Y_binary(self,Y,species):
        f = lambda x: 1 if x == species else 0
        np.vectorize()
        my_func = np.vectorize(f)
        return np.array(my_func(Y))

    def binary(self,df,species):
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y = self.modify_Y_binary(Y,species)
        layer_0 = 4 #ncolumnas de X
        layer_j = 150 #nfilas de Y
        network = Network(1, [4,5,150], g, 0.004, 0.02)
        network.initialize_network(X,[Y])
        w = network.train(20,X,[Y])
        print(network.A)

    def modify_Y_multiclass(self,df):
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y0 = self.modify_Y_binary(Y,"Iris-setosa")
        Y1 = self.modify_Y_binary(Y,"Iris-versicolor")
        Y2 = self.modify_Y_binary(Y,"Iris-virginica")
        return X, [Y0,Y1,Y2]

    def multiclass(self,df):
        X,Y = self.modify_Y_multiclass(df)
        layer_0 = 4 #ncolumnas de X
        layer_j = 150 #nfilas de Y
        network = Network(1, [4,5,150], g, 0.004, 0.02)
        network.initialize_network(X,Y)
        w = network.train(20,X,Y)
        print(network.A[len(network.A)-1])
        print("DONE")


df = pd.read_csv("iris.csv")
Y = np.array(df["species"])
X = np.array(df.drop("species",axis=1))

a = Classifier()
#a.binary(df,"Iris-setosa")
a.multiclass(df)