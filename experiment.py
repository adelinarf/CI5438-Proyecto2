"""Modulo principal con todos los experimentos realizados para obtener los datos para el informe"""
from src.Classifier import Classifier
import pandas as pd

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
    max_conf = max(predictions)
    min_conf = min(predictions)
    avg_conf = sum(predictions)/len(predictions)
    return [false_positives, false_negatives, max_conf, min_conf, avg_conf]

# Entrenamiento de los binarios

species = ['Iris-setosa', 'Iris-virginica','Iris-versicolor']
alphas = [0.1, 0.01, 0.001]
hidden_layers = [[], [(4, 4)], [(4, 4), (4, 4)]]

configurations = [(specie, alpha, hidden_layer) for specie in species for alpha in alphas for hidden_layer in hidden_layers]


outputs = []
for specie, alpha, hidden_layers in configurations:
    classifier = Classifier()
  
    train_df = pd.read_csv("iris data\\iris train data.csv")
    classifier.binary(train_df, specie, alpha=alpha, hidden_layers=hidden_layers)

    test_df = pd.read_csv("iris data\\iris test data.csv")
    real_y = list(map(lambda row: row == specie, test_df["species"]))
    
    predictions = classifier.network.predict(test_df.drop("species", axis=1)).values.tolist()
    # Esto retorna una lista de listas, se aplana
    flatten_predictions = [item for sublist in predictions for item in sublist]

    test_results = test_binary_prediction(flatten_predictions, real_y, 0.85)    
    outputs.append([specie, alpha, hidden_layers, *test_results])

predictions_data = pd.DataFrame(outputs, 
    columns=["specie", "alpha", "hidden layers",  "false positives", "false negatives", "max conf", "min conf", "avg conf"])

predictions_data.to_csv("experiment_results.csv", index=False)