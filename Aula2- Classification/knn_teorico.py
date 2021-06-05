from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('ggplot')

dataset={'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]] }

novas_features= [5,7]

#distancia_euclidiana = sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(novas_features[0], novas_features[1], s=100, color= 'green')
# plt.show()

def k_nearest_neighboor(data, predict, k=3):
	if len(data) >= k:
		warnings.warm('K ta muito baixo para o numero de grupos votantes')

	knnalgos
	return vote_result