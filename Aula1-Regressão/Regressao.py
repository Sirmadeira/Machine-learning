import pandas as pd
import quandl
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


print(df.head())
