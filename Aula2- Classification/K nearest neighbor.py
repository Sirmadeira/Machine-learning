# Knn- Seria a associacao feita, por um conjunto de ponto que nos grafico tem uma distancia euclidiana aproximada
# Ele e muito bom para a definicao de classificacao
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999 ,inplace= True)
df.drop(['id'], 1 , inplace= True)
# O 1 e destacando a primeira coluna


X= np.array(df.drop(['class'],1))

y=np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

precisao=clf.score(X_test,y_test)

amostra_exemplo=np.array([[4,2,1,1,1,2,3,2,1]])
amostra_exemplo= amostra_exemplo.reshape(1,-1)
# Utilizada para testar aquilo que tipo celular que nao existe na nossa data

predicao=clf.predict(amostra_exemplo)
print(precisao)
print(predicao)