import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0],X[:, 1], s=150)
# Datapoint inicial para analise
plt.show()
colors= 10*['g','r','c','b','k']

class K_means:
    def __init__(self,k=2,tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol 
        self.max_iter = max_iter
    def fit(self,data):
        # Limpandos os centroids sucessores
        self.centroids={}
        # Selecionando os primeiro centroids, nao e randomico, pode ser feito por shuffling no data type mas nao quis
        for i in range(self.k):
            self.centroids[i] = data[i]
        # Para ver  as mudancas entre os centroides, troque self.max_iter por 1, 2 ,3 dependendo do numero de vezes
        for i in range(self.max_iter):
            self.classifications = {}
            # Criando classificadores de acordo com o numero de ks
            for i in range (self.k):
                self.classifications[i] = []
            for featureset in data:
                #Basicamente formatando a distrancias entre os centroides, ja estabelecendo um novo centroid de imediato por isso o centroid em um for loop com os centroides feitos
                # Ae formula-se a algebla linear estabelece a normativa do vetor e boom, voce tem as distancias cool right?
                distancias= [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # Definindo a postura de classificacao de acordo com o minimo das distancias
                classification=distancias.index(min(distancias))
                self.classifications[classification].append(featureset)
            # Definindo os centroides no passado, por algum motivo so dar call self.centroids nao funciona
            # Descobri que e pq altera de acordo, com a mudanca do centroides, logo ao formular por dict eu consertei 
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # Achando a media entre as classficacoes, e redefinindo novos centroids
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            # Famosa frase  inocente ate provado culpado
            optimized= True
            for c in self.centroids:
                centroide_original = prev_centroids[c]
                centroide_atual = self.centroids[c]
                # Se alguns dos centroides for maior do que a tolerancia, definida la em cima nos nao vamos estar optimizado
                if np.sum(( centroide_atual - centroide_original)/centroide_original*100.0) > self.tol:
                    # Isso daki vai me dizer quantas interacoes eu passei por
                    print(np.sum(( centroide_atual - centroide_original)/centroide_original*100.0))
                    optimized= False
            if optimized:
                break
                # Heell yeah babey
    def predict(self,data):
        distancias= [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification=distancias.index(min(distancias))
        return classification


clf = K_means()

clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],marker='o',color='k',s= 150, linewidths=5)
for classification in clf.classifications:
    color = colors[classification]
    # Basicamente expondo os centroids do seuguinte features
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker='x', color=color, s=150,linewidths=5)
# Alguns arrays doidos para testes
numseis = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4],])
for numsei in numseis:
    classification=clf.predict(numsei)
    plt.scatter(numsei[0], numsei[1], marker='*', color=colors[classification], s=150,linewidths=5)
plt.show()