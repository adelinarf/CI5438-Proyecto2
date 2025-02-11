# """Modulo principal con todos los experimentos realizados para obtener los datos para el informe"""
from src.Classifier import Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_binary_prediction(predictions, real, threshold):
    """Comprueba la precision del modelo binario comparando las predicciones 
    con los datos reales
    predictions: Arreglo de predicciones
    real: Arreglo de datos reales
    threshold: Umbral classifier partir del que se considera que una predicción binaria
        es o no es la clase.
    """
    pred_and_real = list(zip(predictions, real))
    
    results = list(map(lambda data: (data[0] >= threshold, data[1]), pred_and_real))
    results = pd.DataFrame(results, columns=['prediction', 'real'])
    
    false_positives = len(results[(results['prediction']==True) & (results['real']==False)])
    false_negatives = len(results[(results['prediction']==False) & (results['real']==True)])
    return [false_positives, false_negatives]

# def test_multiclass_prediction(predictions,actual,threshold):
#     falses = []
#     for x in range(len(actual)):
#         pred = np.array(list(predictions[x]))
#         y = np.array([item for sublist in actual[x] for item in sublist])
#         falses.append(test_binary_prediction(pred, y, threshold))
#     return falses


# def accuracy(false_positives,false_negatives,total):
#     # total - falso / total
#     return (total - (false_positives+false_negatives))/total


# def train_test_binary(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs):
#     classifier.binary(train_df, specie, alpha=alpha, hidden_layers=hidden_layers)

#     real_y = list(map(lambda row: row == specie, test_df["species"]))
    
#     predictions = classifier.network.predict(test_df.drop("species", axis=1)).values.tolist()

#     # Esto retorna una lista de listas, se aplana
#     flatten_predictions = [item for sublist in predictions for item in sublist]

#     test_results = test_binary_prediction(flatten_predictions, real_y, 0.85)
#     accuracy_ = accuracy(test_results[0],test_results[1],len(test_df))
#     print("Accuracy =",accuracy_)

#     errors = classifier.get_errors(np.array(flatten_predictions), np.array(real_y))

#     iteration_error = classifier.network.errors
#     it_error = list(map(lambda x : x[0][0],iteration_error))

#     #PLOTS   [max_error, min_error, avg_error, R2_error]
#     plt.plot(list(range(0,len(it_error))), it_error)
#     plt.xlabel('Iteración')
#     plt.ylabel('Error en la iteración') 
#     plt.title("Curva de entrenamiento \n alpha={} specie='{}' Capas ocultas={}".format(alpha,specie,hidden_layers))
#     plt.savefig("plots/plot_alpha={}_specie={}_hidden_layers={}.jpg".format(alpha,specie,hidden_layers), dpi=300)
#     plt.clf()

#     outputs.append([specie, alpha, hidden_layers, *errors, *test_results])
#     return outputs, accuracy_


# def train_test_multiclass(train_df,test_df,alpha,hidden_layers,classifier,outputs_multiclass):
#     classifier.multiclass(train_df, alpha=alpha, hidden_layers=hidden_layers)

#     predictions = classifier.network.predict(test_df.drop("species", axis=1))

#     X, real_y = classifier.modify_Y_multiclass(test_df)

#     Y = []
#     for x in range(len(real_y[0])):
#         A = np.append([real_y[0][x]],[real_y[1][x],real_y[2][x]])
#         Y.append(A)
#     Y = np.array(Y)

#     test_results_multiclass = test_multiclass_prediction(predictions,real_y,0.85)
#     accuracy_ = []
#     for x in range(len(test_results_multiclass)):
#         accuracy_.append(accuracy(test_results_multiclass[x][0],test_results_multiclass[x][1],len(test_df)))
#         print(f'Accuracy Multiclass {x} = {accuracy_}')
#     errors_multiclass = classifier.get_errors_multiclass(predictions,Y)   

#     iteration_error = classifier.network.errors
#     L = np.array(list(map(lambda x : np.diag(x),iteration_error)))

#     #PLOTS
#     plt.plot(list(range(0,len(L[:,0]))), L[:,0], label="Iris-setosa")
#     plt.legend()
#     plt.plot(list(range(0,len(L[:,1]))), L[:,1], label="Iris-virginica")
#     plt.legend()
#     plt.plot(list(range(0,len(L[:,2]))), L[:,2], label="Iris-versicolor")
#     plt.legend()
#     plt.xlabel('Iteración')
#     plt.ylabel('Error en la iteración') 
#     plt.title("Curva de entrenamiento \n alpha={} Multiclase Capas ocultas={}".format(alpha,hidden_layers))
#     plt.savefig("plots/multiclass_{}_{}.jpg".format(alpha,hidden_layers), dpi=300)
#     plt.clf()

#     outputs_multiclass[0].append([alpha, hidden_layers, *list(errors_multiclass[0]), *test_results_multiclass[0]])
#     outputs_multiclass[1].append([alpha, hidden_layers, *list(errors_multiclass[1]), *test_results_multiclass[1]])
#     outputs_multiclass[2].append([alpha, hidden_layers, *list(errors_multiclass[2]), *test_results_multiclass[2]])
#     return outputs_multiclass, accuracy_


# def plot_accuracy_binary():
#     for s in d.keys():
#         for h in d[s].keys():
#             alphas_ = []
#             accurate = []
#             for a in d[s][h].items():
#                 alphas_.append(a[0])
#                 accurate.append(a[1][0])
#             plt.bar(alphas_, accurate, width = 0.1,label=s+" Ocultas="+h)
#     plt.legend(bbox_to_anchor=(1.02, 1.1), loc='upper left', borderaxespad=0)
#     plt.xlabel('alpha')
#     plt.ylabel('Exactitud') 
#     plt.title("Exactitud de Clasificador Binario")
#     plt.savefig("plots/Exactitud_Binaria.jpg", dpi=300,bbox_inches='tight')
#     plt.clf()

# def plot_accuracy_multiclass():
#     for h in ac.keys():
#         alphas_ = []
#         accurate_0 = []
#         accurate_1 = []
#         accurate_2 = []
#         for a in ac[h].items():
#             alphas_.append(a[0])
#             accurate_0.append(a[1][0][0])
#             accurate_1.append(a[1][0][1])
#             accurate_2.append(a[1][0][2])
#         plt.bar(alphas_,accurate_0, width = 0.1,label="Iris-setosa Ocultas="+h)
#         plt.bar(alphas_,accurate_1, width = 0.1,label="Iris-virginica Ocultas="+h)
#         plt.bar(alphas_,accurate_2, width = 0.1,label="Iris-versicolor Ocultas="+h)
#     plt.legend(bbox_to_anchor=(1.02, 1.1), loc='upper left', borderaxespad=0)
#     plt.xlabel('alpha')
#     plt.ylabel('Exactitud') 
#     plt.title("Exactitud de Clasificador Multiclase")
#     plt.savefig("plots/Exactitud_Multiclase.jpg", dpi=300,bbox_inches='tight')
#     plt.clf()

# # Entrenamiento de los binarios

# species = ['Iris-setosa', 'Iris-virginica','Iris-versicolor']

# def file_name(specie,n):
#     if n==0:
#         #train
#         return 'iris data\\only_'+specie+"_train.csv"
#     else:
#         #test
#         return 'iris data\\only_'+specie+"_test.csv"


# alphas = [2,0.5,0.1, 0.01, 0.001, 0.0001]
# hidden_layers = [[], [4], [5],[10],[20],[5,6],[1,5],[10,8]]

# configurations = [(specie, alpha, hidden_layer) for specie in species for alpha in alphas for hidden_layer in hidden_layers]
# configurations2 = [(alpha, hidden_layer) for alpha in alphas for hidden_layer in hidden_layers]

# d = {}
# for specie in species:
#     d[specie] = {}
#     for layer in hidden_layers:
#         d[specie][str(layer)] = {}
#         for a in alphas:
#             d[specie][str(layer)][a] = []
# ac = {}
# for layer in hidden_layers:
#     ac[str(layer)] = {}
#     for a in alphas:
#         ac[str(layer)][a] = []

# outputs = []
# outputs_multiclass = [[],[],[]]


# for specie, alpha, hidden_layers in configurations:
#     print(specie,alpha,hidden_layers)
#     classifier = Classifier()
  
#     train_df = pd.read_csv(file_name(specie,0))
#     test_df = pd.read_csv(file_name(specie,1))

#     outputs, accuracy_ = train_test_binary(train_df,test_df,specie,alpha,hidden_layers,classifier,outputs)
#     d[specie][str(hidden_layers)][alpha].append(accuracy_)

# plot_accuracy_binary()

# #Entrenamiento Multiclase

# for alpha, hidden_layers in configurations2:
#     print(alpha,hidden_layers)
#     classifier = Classifier()
  
#     train_df = pd.read_csv("iris data\\iris train data.csv")
#     test_df = pd.read_csv("iris data\\iris test data.csv")

#     outputs_multiclass, accuracy_ = train_test_multiclass(train_df,test_df,alpha,hidden_layers,classifier,outputs_multiclass)
#     ac[str(hidden_layers)][alpha].append(accuracy_)

# plot_accuracy_multiclass()

# predictions_data = pd.DataFrame(outputs, 
#         columns=["alpha", "hidden layers", "test max error", "test min error", 
#                 "test avg error", "test R2 error", "train max error", "train min error", 
#                 "train avg error", "train R2 error","false positives", "false negatives"])

# predictions_data.to_csv("experiment_results.csv", index=False)

# predictions_class_1 = pd.DataFrame(outputs_multiclass[0], 
#     columns=["alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

# predictions_class_1.to_csv("experiment_results_multiclass_0.csv", index=False)

# predictions_class_2 = pd.DataFrame(outputs_multiclass[1], 
#     columns=["alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

# predictions_class_2.to_csv("experiment_results_multiclass_1.csv", index=False)

# predictions_class_3 = pd.DataFrame(outputs_multiclass[2], 
#     columns=["alpha", "hidden layers", "max error", "min error", "avg error", "false positives", "false negatives"])

# predictions_class_3.to_csv("experiment_results_multiclass_2.csv", index=False)


# Entrenamiento del clasificador de SPAM
from src.Network import Network, g

alphas = [2,0.5,0.1, 0.01, 0.001, 0.0001]
hidden_layers = [[], [20], [40], [60], [20, 20], [40, 40], [60, 60]]

configurations = [(alpha, hidden_layer)  
                  for alpha in alphas for hidden_layer in hidden_layers]

outputs = []
for alpha, hidden_layer in configurations:
    classifier = Classifier()
    train_data = pd.read_csv("spambase\spam_train.csv")

    # Conseguir los errores de entrenamiento
    X = np.array(train_data.drop("57",axis=1))
    Y = np.array(train_data["57"])
    Y = np.reshape(Y, (len(Y),1))
    prev = len(X[0])
    tuples = []
    for neuron in hidden_layer:
        tuples.append((prev,neuron))
        prev = neuron
    tuples.append((prev,1))
    
    network = Network(
        tuples, 
        g, epsilon=0.0004,alpha=alpha)
    
    network.train(3000, X, Y)
    errors = classifier.get_errors(network.layer_output.A, Y)
    errors = np.reshape(np.array(errors), (1, len(errors)))

    # Conseguir los errores de prueba
    test_data = pd.read_csv("spambase\spam_test.csv")
    X = test_data
    predictions = network.predict(test_data.drop("57", axis=1)).values.tolist()
    flatten_predictions = [item for sublist in predictions for item in sublist]    
    # Calcular los errores de prediccion
    false_positives_and_negatives = test_binary_prediction(
        flatten_predictions, list(test_data["57"]), 0.8)
    
    test_errors = classifier.get_errors(np.array(flatten_predictions), list(test_data["57"]))
    
    outputs.append([alpha, hidden_layer, *test_errors, *errors[0], *false_positives_and_negatives])

    predictions_data = pd.DataFrame(outputs, 
        columns=["alpha", "hidden layers", "test max error", "test min error", 
                "test avg error", "test R2 error", "train max error", "train min error", 
                "train avg error", "train R2 error","false positives", "false negatives"])

    predictions_data.to_csv("spam_experiment_results_3000_it.csv", index=False)
    iteration_error = network.errors
    it_error = list(map(lambda x : x[0][0],iteration_error))

    #PLOTS   [max_error, min_error, avg_error, R2_error]
    plt.plot(list(range(0,len(it_error))), it_error)
    plt.xlabel('Iteración')
    plt.ylabel('Error en la iteración') 
    plt.title("Curva de entrenamiento \n alpha={} Capas ocultas={}".format(alpha,hidden_layer))
    plt.savefig("plots/plot_spam_alpha={}_hidden_layers={}.jpg".format(alpha, hidden_layer), dpi=300)
    plt.clf()
