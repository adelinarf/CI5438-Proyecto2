import numpy as np
import pandas as pd
from Network import Network, g, gp

class Classifier:
    def __init__(self) -> None:
        self._hipotesis = None
        self.prediction = None
        self.network = None

    def modify_Y_binary(self,Y,species):
        f = lambda x: 1 if x == species else 0
        
        my_func = np.vectorize(f)
        array = np.array(my_func(Y))
        return np.reshape(array, (len(array),1))

    def binary(self, df, specie, hidden_layers=[], iterations=5000, epsilon=0.0004, alpha=0.1):
        """Entrena un modelo binario con los datos de las flores. 
        
        df: Dataframe con las flores
        specie: Especie con la que se entrenará el clasificador
        hidden_layers: Lista de tuplas donde cada tupla representa la cantidad de conexiones
            que entran y salen de cada capa
        iterations: Cantidad de iteraciones durante el entrenamiento
        epsilon: Condicion de parada para el entrenamiento
        alpha: Constante de aprendizaje
        """
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y = self.modify_Y_binary(Y,specie)
        
        network = Network([(4,4), *hidden_layers, (4,1)], g, epsilon, alpha)
        network.train(iterations, X, Y)
        errors = self.get_errors(network.layer_output.A, Y)
        self.network = network
        return errors

    def test(self, predictions, test):
        RSS = sum(np.square(predictions - test))
        TSS = sum(np.square(predictions - np.average(predictions)))
        R2_coeff = 1 - RSS/TSS
        return R2_coeff

    def get_errors(self, predictions, test):
        errors = np.abs(predictions - test)
        max_error = max(errors)
        min_error = min(errors)
        avg_error = sum(errors)/len(errors)
        R2_error = self.test(predictions, test)
        return [max_error, min_error, avg_error, R2_error] 
        

    def modify_Y_multiclass(self,df):
        Y = np.array(df["species"])
        X = np.array(df.drop("species",axis=1))
        Y0 = self.modify_Y_binary(Y,"Iris-setosa")
        Y1 = self.modify_Y_binary(Y,"Iris-versicolor")
        Y2 = self.modify_Y_binary(Y,"Iris-virginica")
        return X, [Y0,Y1,Y2]

    def multiclass(self, df, hidden_layers=[], iterations=5000, epsilon=0.0004, alpha=0.1):
        X,Y = self.modify_Y_multiclass(df)
        Y = np.array(Y)
        z1 = np.reshape(Y[0].T, (120,1))
        z2 = np.reshape(Y[1].T, (120,1))
        z3 = np.reshape(Y[2].T, (120,1))
        SALIDA = []
        for x in range(len(z1)):
            A = np.append([z1[x]],[z2[x],z3[x]])
            SALIDA.append(A)
        SALIDA = np.array(SALIDA)
        network = Network([(4,5), *hidden_layers, (6,3)], g, epsilon, alpha)
        network.train(iterations, X, SALIDA)
        self.network = network


if __name__ == '__main__':
    df = pd.read_csv("iris data\\iris train data.csv")
    Y = np.array(df["species"])
    X = np.array(df.drop("species",axis=1))
    a = Classifier()    
    a.multiclass(df)





    print(w)