from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


dataset={'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]] }

novas_features= [5,7]

#distancia_euclidiana = sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)
def k_nearest_neighboor(data, predict, k=3):
	if len(data) >= k:
		warnings.warm('K ta muito baixo para fazer algo relevante!')
	distancia=[]
	for group in data:
		for features in data[group]:
			distancia_euclidiana = np.linalg.norm(np.array(features)-np.array(predict))
			distancia.append([distancia_euclidiana, group])
	votos=[i[1] for i in sorted(distancia) [:k]]
	#print(Counter(votos).most_common(1))
	votos_resultado= Counter(votos).most_common(1) [0] [0]
	return votos_resultado

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1, inplace=True)
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)

test_size= 0.2
train_set ={2:[],4:[]}
test_set ={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct=0
total=0
for group in test_set:
	for data in test_set[group]:
		voto= k_nearest_neighboor(train_set, data, k=5)
		if group == voto:
			correct+=1
		total +=1
print('Precisao', correct/total)