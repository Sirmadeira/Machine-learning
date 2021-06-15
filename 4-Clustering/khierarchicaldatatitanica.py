import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, train_test_split

df=pd.read_excel('titanic.xls')
original_df=pd.DataFrame.copy(df)

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

df.drop(['ticket','home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])



clf = MeanShift()
clf.fit(X)

labels=clf.labels_
cluster_centers=clf.cluster_centers_

original_df['cluster_group']= np.nan

# Povoando as partes dos labels
original_df['cluster_group'].iloc[:] = labels

n_clusters_=len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    #Dataframe temporaria quando  o cluster group e 0
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate=len(survival_cluster)/len(temp_df)
    survival_rates[i]=survival_rate

print(survival_rates)
#Printando grupo de individuos com maior sobrevivencia
print(original_df[(original_df['cluster_group']==1)].describe())