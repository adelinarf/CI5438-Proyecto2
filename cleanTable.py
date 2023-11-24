import pandas as pd
import numpy as np
X = pd.read_csv("experiment_results.csv")

clean = np.array(X.drop("R2 error",axis=1))

setosa = clean[:48]
virginica = clean[48:96]
versicolor = clean[96:]

alphas = [2,0.5,0.1, 0.01, 0.001, 0.0001]

def generate_lists(L,name):
	lista = []
	for a in alphas:
		new_arr = np.array(list(filter(lambda x: x[1] if x[1] == a else [], L)))
		maxe = new_arr[:,3]
		mine = new_arr[:,4]
		avgerror = new_arr[:,5]
		pos = new_arr[:,6]
		neg = new_arr[:,7]
		lista.append([name,a,np.average(maxe),np.average(mine),np.average(avgerror),np.average(pos),np.average(neg)])
	return lista

C1 = generate_lists(setosa,"Iris-setosa")
C2=generate_lists(virginica,"Iris-virginica")
C3=generate_lists(versicolor,"Iris-versicolor")

p1 = pd.DataFrame(C1, 
    columns=["clase", "alpha", "max error", "min error", "avg error", "false positives", "false negatives"])
p2 = pd.DataFrame(C2, 
    columns=["clase", "alpha", "max error", "min error", "avg error", "false positives", "false negatives"])
p3 = pd.DataFrame(C3, 
    columns=["clase", "alpha", "max error", "min error", "avg error", "false positives", "false negatives"])

p1.to_csv("setosa-resumen.csv", index=False)
p2.to_csv("virginica-resumen.csv", index=False)
p3.to_csv("versicolor-resumen.csv", index=False)
