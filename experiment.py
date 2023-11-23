"""Modulo principal con todos los experimentos realizados para obtener los datos para el informe"""
from src.Classifier import Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_binary_prediction(predictions, real, threshold):
    """Comprueba la precision del modelo binario comparando las predicciones 
    con los datos reales
    predictions: Arreglo de predicciones
    real: Arreglo de datos reales
    threshold: Umbral a partir del que se considera que una predicción binaria
        es o no es la clase.
    """
    pred_and_real = list(zip(predictions, real))
    
    results = list(map(lambda data: (data[0] >= threshold, data[1]), pred_and_real))
    results = pd.DataFrame(results, columns=['prediction', 'real'])
    
    false_positives = len(results[(results['prediction']==True) & (results['real']==False)])
    false_negatives = len(results[(results['prediction']==False) & (results['real']==True)])
    return [false_positives, false_negatives]

def test_multiclass_prediction(predictions,actual,threshold):
    falses = []
    for x in range(len(actual)):
        pred = np.array(list(predictions[x]))
        y = np.array([item for sublist in actual[x] for item in sublist])
        falses.append(test_binary_prediction(pred, y, threshold))
    return falses


def train_test_binary(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs):
    classifier.binary(train_df, specie, alpha=alpha, hidden_layers=hidden_layers)

    real_y = list(map(lambda row: row == specie, test_df["species"]))
    
    predictions = classifier.network.predict(test_df.drop("species", axis=1)).values.tolist()

    # Esto retorna una lista de listas, se aplana
    flatten_predictions = [item for sublist in predictions for item in sublist]

    test_results = test_binary_prediction(flatten_predictions, real_y, 0.85)
    errors = classifier.get_errors(np.array(flatten_predictions), np.array(real_y))

    iteration_error = classifier.network.errors
    it_error = list(map(lambda x : x[0][0],iteration_error))

    #PLOTS   [max_error, min_error, avg_error, R2_error]
    plt.plot(list(range(0,len(it_error))), it_error)
    plt.xlabel('Iteración')
    plt.ylabel('Error en la iteración') 
    plt.title("Curva de entrenamiento \n alpha={} specie='{}' Capas ocultas={}".format(alpha,specie,hidden_layers))
    plt.savefig("plots/plot_alpha={}_specie={}_hidden_layers={}.jpg".format(alpha,specie,hidden_layers), dpi=300)
    plt.clf()

    outputs.append([specie, alpha, hidden_layers, *errors, *test_results])
    return outputs


def train_test_multiclass(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs_multiclass):
    classifier.multiclass(train_df, alpha=alpha, hidden_layers=hidden_layers)

    predictions = classifier.network.predict(test_df.drop("species", axis=1))

    X, real_y = classifier.modify_Y_multiclass(test_df)

    Y = []
    for x in range(len(real_y[0])):
        A = np.append([real_y[0][x]],[real_y[1][x],real_y[2][x]])
        Y.append(A)
    Y = np.array(Y)

    test_results_multiclass = test_multiclass_prediction(predictions,real_y,0.85)
    errors_multiclass = classifier.get_errors_multiclass(predictions,Y)   

    iteration_error = classifier.network.errors
    L = np.array(list(map(lambda x : np.diag(x),iteration_error)))

    #PLOTS
    plt.plot(list(range(0,len(L[:,0]))), L[:,0], label="Iris-setosa")
    plt.legend()
    plt.plot(list(range(0,len(L[:,1]))), L[:,1], label="Iris-virginica")
    plt.legend()
    plt.plot(list(range(0,len(L[:,2]))), L[:,2], label="Iris-versicolor")
    plt.legend()
    plt.xlabel('Iteración')
    plt.ylabel('Error en la iteración') 
    plt.title("Curva de entrenamiento \n alpha={} Multiclase Capas ocultas={}".format(alpha,hidden_layers))
    plt.savefig("plots/multiclass_{}_{}.jpg".format(alpha,hidden_layers), dpi=300)
    plt.clf()

    outputs_multiclass[0].append([specie, alpha, hidden_layers, *list(errors_multiclass[0]), *test_results_multiclass[0]])
    outputs_multiclass[1].append([specie, alpha, hidden_layers, *list(errors_multiclass[1]), *test_results_multiclass[1]])
    outputs_multiclass[2].append([specie, alpha, hidden_layers, *list(errors_multiclass[2]), *test_results_multiclass[2]])
    return outputs_multiclass


# Entrenamiento de los binarios

species = ['Iris-setosa', 'Iris-virginica','Iris-versicolor']
alphas = [0.1, 0.01, 0.001]
hidden_layers = [[], [4], [5,6,7]]

configurations = [(specie, alpha, hidden_layer) for specie in species for alpha in alphas for hidden_layer in hidden_layers]


outputs = []
outputs_multiclass = [[],[],[]]
for specie, alpha, hidden_layers in configurations:
    classifier = Classifier()
  
    train_df = pd.read_csv("iris data\\iris train data.csv")
    test_df = pd.read_csv("iris data\\iris test data.csv")

    outputs = train_test_binary(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs)
    outputs_multiclass = train_test_multiclass(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs_multiclass)

predictions_data = pd.DataFrame(outputs, 
    columns=["specie", "alpha", "hidden layers", "max error", "min error", "avg error", "R2 error", "false positives", "false negatives"])

predictions_data.to_csv("experiment_results.csv", index=False)

predictions_class_1 = pd.DataFrame(outputs_multiclass[0], 
    columns=["specie", "alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

predictions_class_1.to_csv("experiment_results_multiclass_0.csv", index=False)

predictions_class_2 = pd.DataFrame(outputs_multiclass[1], 
    columns=["specie", "alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

predictions_class_2.to_csv("experiment_results_multiclass_1.csv", index=False)

predictions_class_3 = pd.DataFrame(outputs_multiclass[2], 
    columns=["specie", "alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

predictions_class_3.to_csv("experiment_results_multiclass_2.csv", index=False)