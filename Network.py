import random, math
import numpy as np
import pandas as pd
from numpy import linalg as LA
random.seed(5) 

def g(x):
    a = 1
    return 1/(1+np.exp(-x))

def gp(x):
    return (x*(1-x))

class Layer():
    def __init__(self,nrow,ncol) -> None:
        self.W = np.random.uniform(0,1,(nrow,ncol))
        self.A = None
        self.error = None
        self.B = np.random.uniform(0,1,(ncol,))
        self.bias_gradient = None 
        self.weight_gradient = None

class Layer_Output(Layer):
    def __init__(self,nrow,ncol):
        super().__init__(nrow,ncol)
        self.a0 = None

    def propagate_input(self,a0,f):
        res = (a0 @ self.W) + self.B
        self.A = f(res)
        self.a0 = res

    def propagate_deltas_output(self, previous_activation, Y):
        resta = Y-self.A #self.A-Y
        error = (resta.T @ resta)/len(Y)
        error_gradient = (self.A - Y)*2/len(Y)
        layer_gradient = error_gradient @ self.W.T
        self.weight_gradient = previous_activation.T @ error_gradient
        self.bias_gradient = np.sum(error_gradient, axis=0, keepdims=True)
        self.error = error
        return layer_gradient

    def update_weights(self,alpha):
        self.W = self.W - alpha*self.weight_gradient
        self.B = self.B - alpha*self.bias_gradient


class Hidden_Layers():
    def __init__(self):
        self.layers = []
        self.ini = []
    def add_layer(self,layer):
        self.layers.append(layer)
    def propagate_inputs(self,X,f):
        a_0 = X
        res = X
        for layer in self.layers:
            self.ini.append(a_0)
            res = (a_0 @ layer.W) + layer.B  #res = (layer.W @ a_0) + layer.B
            layer.A = f(res)
            a_0 = layer.A
        return a_0

    def propagate_deltas_on_layers(self,upstream_gradient):
        activation_derivative = gp
        for layer,ini in reversed(list(zip(self.layers,self.ini))):
            activation_gradient = activation_derivative(layer.A) * upstream_gradient
            layer_gradient = activation_gradient @ layer.W.T
            weight_gradient = ini.T @ activation_gradient
            bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
            layer.weight_gradient = weight_gradient
            layer.bias_gradient = bias_gradient
            upstream_gradient = layer_gradient


    def update_weights(self,alpha):
        for layer in self.layers:
            layer.W = layer.W - alpha*layer.weight_gradient
            layer.B = layer.B - alpha*layer.bias_gradient


class Network:
    def __init__(self,neurons_per_layer : list[tuple[int,int]],g,epsilon : float,alpha : float) -> None:
        self.hidden = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.epsilon = epsilon
        self.g = g
        self.alpha = alpha
        self.layer_output = [] 
        self.hidden_layers = Hidden_Layers()

    def initialize_network(self):
        for x,y in self.neurons_per_layer[:-1]:
            self.hidden_layers.add_layer(Layer(x,y))
        x,y = self.neurons_per_layer[-1]
        self.layer_output = Layer_Output(x,y)

    def propagate_inputs(self,X):
        if len(self.hidden_layers.layers) != 0:
            a0 = self.hidden_layers.propagate_inputs(X,self.g)
        else:
            a0 = X
        self.layer_output.propagate_input(a0,self.g) 

    def propagate_deltas_on_layers(self,grad):
        self.hidden_layers.propagate_deltas_on_layers(grad)

    def train(self,it,X,Y):
        self.initialize_network()
        for x in range(it):
            self.forward_propagation(X)
            self.backwards_propagation(X,Y)
            self.update_weights()
            #if LA.norm(self.layer_output.error) < self.epsilon:
            #   print("Termina por epsilon")
            #   break

    def predict(self,X):
        self.forward_propagation(X)
        return (self.layer_output.A)

    def forward_propagation(self,X):
        self.propagate_inputs(X)

    def backwards_propagation(self,X,Y):
        previous_act = X if len(self.hidden_layers.layers) == 0 else self.hidden_layers.layers[-1].A
        layer_grad = self.layer_output.propagate_deltas_output(previous_act,Y)
        self.propagate_deltas_on_layers(layer_grad)

    def update_weights(self):
        self.layer_output.update_weights(self.alpha)
        self.hidden_layers.update_weights(self.alpha)


class Classifier:
    def __init__(self) -> None:
        self._hipotesis = None

    def modify_Y_binary(self,Y,species):
        f = lambda x: 1 if x == species else 0
        np.vectorize()
        my_func = np.vectorize(f)
        array = np.array(my_func(Y))
        return np.reshape(array, (len(array),1))

    def binary(self,df,species):
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y = self.modify_Y_binary(Y,species)
        #[(ncolumnas,x),(x,y),..,(y,1)]
        network = Network([(4,4),(4,1)], g, 0.0004, 0.1)
        w = network.train(1000,X,Y)
        print(network.layer_output.A)
        self.test(network.layer_output.A,Y)

    def test(self,predictions,test):
        RSS = sum(np.square(predictions - test))
        TSS = sum(np.square(predictions - np.average(predictions)))
        R2_coeff = 1 - RSS/TSS
        print(R2_coeff)

    def modify_Y_multiclass(self,df):
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y0 = self.modify_Y_binary(Y,"Iris-setosa")
        Y1 = self.modify_Y_binary(Y,"Iris-versicolor")
        Y2 = self.modify_Y_binary(Y,"Iris-virginica")
        return X, [Y0,Y1,Y2]

    def multiclass(self,df):
        X,Y = self.modify_Y_multiclass(df)
        Y = np.array(Y)
        z1 = np.reshape(Y[0].T, (120,1))
        z2 = np.reshape(Y[1].T, (120,1))
        z3 = np.reshape(Y[2].T, (120,1))
        Y = np.reshape(np.array([z1,z2,z3]), (120,3) )
        # X = (120,4)   Y=(120,3)
        network = Network([(4,3)], g, 0.0004, 0.1)
        network.train(100,X,Y)
        print(network.layer_output.A)
        self.test(network.layer_output.A,Y)

df = pd.read_csv("iris_train.csv")
Y = np.array(df["species"])
X = np.array(df.drop("species",axis=1))

a = Classifier()
#a.binary(df,"Iris-setosa")
a.multiclass(df)