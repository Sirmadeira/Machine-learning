#A data se encontra em https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmJhWC1HT2pHLTlQR0ZDNC15VDFXUnY5U2FFd3xBQ3Jtc0tscVRVY1Ruam9wWHJJX21JSndpYnZQQzN4a1J6TFFjdVVxc216LTBlbmpYanFLSFFYYUpwcDNPcS1ucHY1UDBFaGdrN2ttV25SWmgzV1ZjVlFyYV9RUmM2Y1JWTGpoQ0RWTEVPRGRxX280VFFvd1NCYw&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, train_test_split

df=pd.read_excel('titanic.xls')
df.drop(['body','name'],1, inplace= True)

for n in df.columns.values:
     df[n] = pd.to_numeric(df[n],errors="ignore")

df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

df.drop(['sex','boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])



clf = KMeans(n_clusters=2)
clf.fit(X)


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1


print(correct/len(X))

# Ultrapassado user dummies https://pbpython.com/categorical-encoding.html