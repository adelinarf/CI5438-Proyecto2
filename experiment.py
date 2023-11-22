"""Modulo principal con todos los experimentos realizados para obtener los datos para el informe"""
from src.Classifier import Classifier
import pandas as pd
import numpy as np

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

# Entrenamiento de los binarios
species = ['Iris-setosa', 'Iris-virginica','Iris-versicolor']
alphas = [0.1, 0.01, 0.001]
hidden_layers = [[], [(4, 4)], [(4, 10), (10, 4)]]

configurations = [(specie, alpha, hidden_layer) for specie in species for alpha in alphas for hidden_layer in hidden_layers]


outputs = []
for specie, alpha, hidden_layers in configurations:
    classifier = Classifier()
    # Obtener los errores del conjunto de entrenamiento
    train_df = pd.read_csv("iris data\\iris train data.csv")
    train_errors = classifier.binary(train_df, specie, alpha=alpha, hidden_layers=hidden_layers)
    train_errors = [error for array in train_errors for error in array]

    # Obtener los errores del conjunto de pruebas
    test_df = pd.read_csv("iris data\\iris test data.csv")
    real_y = list(map(lambda row: row == specie, test_df["species"]))    
    predictions = classifier.network.predict(test_df.drop("species", axis=1)).values.tolist()    
    flatten_predictions = [item for sublist in predictions for item in sublist]

    # Obtener los falsos positivos y falsos negativos
    test_results = test_binary_prediction(flatten_predictions, real_y, 0.85)

    # Retornar los errores
    errors = classifier.get_errors(np.array(flatten_predictions), np.array(real_y))
    errors.extend(train_errors)
    outputs.append([specie, alpha, hidden_layers, *errors, *test_results])

predictions_data = pd.DataFrame(outputs, 
    columns=["specie", "alpha", "hidden layers", "test max error", "test min error", 
             "test avg error", "test R2 error", "train max error", "train min error", 
             "train avg error", "train R2 error","false positives", "false negatives"])

predictions_data.to_csv("experiment_results.csv", index=False)


# Entrenamiento del multiclase
