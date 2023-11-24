import pandas as pd
import numpy as np
X1 = pd.read_csv("Resultados de los experimentos/experiment_results_multiclass_0.csv")
X2 = pd.read_csv("Resultados de los experimentos/experiment_results_multiclass_1.csv")
X3 = pd.read_csv("Resultados de los experimentos/experiment_results_multiclass_2.csv")

alphas = [2,0.5,0.1, 0.01, 0.001, 0.0001]

def generate_lists(L):
	L = np.array(L)
	lista = []
	for a in alphas:
		new_arr = np.array(list(filter(lambda x: x[0] if x[0] == a else [], L)))
		maxe = new_arr[:,2]
		mine = new_arr[:,3]
		avgerror = new_arr[:,4]
		pos = new_arr[:,5]
		neg = new_arr[:,6]
		lista.append([a,np.average(maxe),np.average(mine),np.average(avgerror),np.average(pos),np.average(neg)])
	return lista

C1=generate_lists(X1)
C2=generate_lists(X2)
C3=generate_lists(X3)


p1 = pd.DataFrame(C1, 
    columns=["alpha", "max error", "min error", "avg error", "false positives", "false negatives"])
p2 = pd.DataFrame(C2, 
    columns=["alpha", "max error", "min error", "avg error", "false positives", "false negatives"])
p3 = pd.DataFrame(C3, 
    columns=["alpha", "max error", "min error", "avg error", "false positives", "false negatives"])

p1.to_csv("setosa-multiclass-resumen.csv", index=False)
p2.to_csv("virginica-multiclass-resumen.csv", index=False)
p3.to_csv("versicolor-multiclass-resumen.csv", index=False)
