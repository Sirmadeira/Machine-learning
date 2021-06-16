import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets import make_blobs
import random

centros=random.randrange(2,8)
print(centros)
X,y = make_blobs(n_samples=50,centers=centros, n_features=2)
# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

colors = 10*["g","r","c","b","k"]

class Mean_shift:
    # Nesse caso eu fiz por hard code, ou seja eu tenho que definir o radius, algo que e nono
    # Agora vou tentar fazer por norm_steps que seria em partes muito pequenas que vao engolindo uma a outra.
    # Ou melhor conhecido como dynamic bandwidth,
    def __init__(self,radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
    def fit(self,data):
        if self.radius == None:
            all_data_centroid = np.average(data,axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids ={}
        # Id do centroides, e o valor desse id sera o valor do data
        for i in range(len(data)):
            centroids[i]=data[i]

        pesos=[i for i in range(self.radius_norm_step)][::-1]
        # Basicamente de 1 a 100, revertendo a lista
        while True:
            novo_centroids=[]
            for i in centroids:
                in_bandwidth=[]
                pesos_lista=[]
                # Isso daki vai ser populado com todos os featuresets, em nosso radius
                centroid=centroids[i]
                #Pegando o valor do centroide
                for featureset in data:
                    distancia= np.linalg.norm(featureset-centroid)
                    # Distancia entre os pontos
                    if distancia == 0:
                        distancia=0.00001
                        # Como eu disse eu quero fazer passobem pequenos logo distancia bem pequena 
                    peso_index=int(distancia/self.radius) 
                    if peso_index >self.radius_norm_step-1:
                        #Definindo para caso akl distancia ultrapasse o limite de 100
                        # Nesse caso seria o limite
                        peso_index = self.radius_norm_step-1 
                    pesos_lista.append(pesos[peso_index])
                    in_bandwidth.append(featureset)
                    #to_add= (pesos[peso_index]**2)*[featureset]
                    #in_bandwidth+= to_add
                    # Posso fazer plus equals  porque e uma lista dentro de uma lista
                # Juncionando os bandwith caso ele se encaixe dentro do radius
                novo_centroid=np.average(in_bandwidth,axis=0,weights=np.array(pesos_lista))
                # Pegando a media dos centroides dentro da bandwidth
                novo_centroids.append(tuple(novo_centroid))
            uniques= sorted(list(set(novo_centroids))) # Eu faco isso daki por que tuplas podem te dar um set de valores unicos
            to_pop=[]
            for i in uniques:
                if i in to_pop:
                    pass
                for ii in uniques:
                    if i ==ii:
                        pass
                    #Se eles forem identicos a gente passa
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop:
                        #Se a distancia entre os unicos forem menores do que o radius. eu acrescento eles a lista
                        # Dei uma melhorada para nao adicionar akls que ja tao no radius
                        to_pop.append(ii)
                        # Eu fiz desse jeito poruqe eu nao posso inteirar pela lista enquanto eu passo por ela
                        break
            for i in to_pop:
                uniques.remove(i)
                #Removendo eles
            prev_centroids = dict(centroids)
            centroids={}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i],prev_centroids[i]):
                    # Se eles nao forem iguais significa que eles nao se juntaram no mean logo falso
                    optimized= False
                if not optimized:
                    break
            if optimized:
                break
                    #Distancia euclidiana np.linalg.norm(featureset-centroid)
        self.centroids=centroids
        # Resetando os centroides
        self.classifications ={}

        for i in range(len(self.centroids)):
            self.classifications[i] =[]

        for featureset in data:
            distancias= [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification= distancias.index(min(distancias))
            self.classifications[classification].append(featureset)

        def predict(self,data):
            distancias= [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification= distancias.index(min(distancias))
            return classification

clf= Mean_shift()
clf.fit(X)
centroids=clf.centroids
for classification in clf.classifications:
    color=colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker='x', color=color, s=150, linewidths=5)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k',marker='*', s= 200)
plt.show()