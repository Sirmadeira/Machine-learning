import pandas as pd
import quandl
import math
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression


#Vamo comecar pegando a base de dados, voce pode usar notebook aqui para ver a dataset mas facilmente.
#Simplifique o seus dados, e evite dados inuteis.
# Organizando os dados
df= quandl.get('WIKI/GOOGL')

df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# Selecionando partes especfiicas da database. Especialmente as adjusted volumes, que seria o valor da acao ajustado.

df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/ df['Adj. Close'] * 100.0
# Porcentagem de volatilidade
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/ df['Adj. Open'] * 100.0



df= df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]

forecast_col= 'Adj. Close'
# Features, seria o conjunto de dados que voce esta inserindo para ser capaz de formatar a analise

df.fillna(-99999,inplace= True)
# Esse codigo serve para substituir, os valores vazios

forecast_out = math.ceil(0.01*len(df))
#Ceil retorna o primeiro valor integro, nesse caso ele ta pegando o length da database e fazendo ele retornar o ultimo valor integral, vezes 0.01
# Que seria o numero de dias a frente
print(forecast_out)


df['label']= df[forecast_col].shift(-forecast_out)
# Label seria aquilo que voce esta tentando olha para o futuro, ou seja, isso daki seria os valores futorus a ser  identificado
# a forecast_col = adj close que seria o fechamento da acao, acrescentando a forecast out que seria os dias a frente. Tentando definir o adj close deles


df.dropna(inplace=True)

X= np.array(df.drop(['label'],1))
# Convertendo a dataframe, eliminando os labelse e transformanmdo em array
y=np.array(df['label'])

X=preprocessing.scale(X)
# Preprocessing, seria associar um conjunto de dados a um axis, centerizando ele de acordo a media e a relacao de variancia entre variaveis
# Nao recomendado para modelos com alto influxo devido a fragilidade desse processo, que pode causar data leaks. Ou antes de treinar e testar a data
# Pode fazer com que eles se misturem, oque e nono

df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# Dividindo os eixo em data de treinamento e data de teste


clf= LinearRegression(n_jobs=-1)
# N jobs seria o quanto do seu cpu tu quer usar para o treinamento -1 seria tudo
clf.fit(X_train,y_train)

precisao=clf.score(X_test,y_test)
print(precisao)
