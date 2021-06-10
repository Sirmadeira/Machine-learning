import matplotlib.pyplot as plt 	
from matplotlib import style
import numpy as np
style.use('ggplot')

class support_vector_machine:
	# Nenhum dos metodos de uma classe irao rodar alem do init metodo
	def __init__(self,visualization=True):
		self.visualization= visualization
		self.colors = {1:'r',-1:'k'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)
			# Isso daki e utilizado para visualizar o grafico, e o seu subplot

	def fit(self, data):
		self.data = data
		# { a magnitude de w : [w,b]  }
		opt_dict = {}
		transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]
		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
		# Pegando features do dataset
		self.max_feature_value = max(all_data)
		self.min_feature_value  = min(all_data)
		# Pegando valor maximo e minimo da data agariada
		all_data = None 
		step_sizes=[self.max_feature_value * 0.1,self.max_feature_value * 0.01,self.max_feature_value * 0.001]
		# Tambem evitar point of expense, basicamente o custo para rodar os steps
		# Tamanho dos pulas pulas iniciais do grafico, vai diminuindo a medida que vamo achando
		# Evitar min local
		# Como saber se eu tenho que fazer mais passos,
		# Lembre-se yi(xi. w+b) = 1
		# Voce vai saber quando tanto a suas classes positivas e negativas forem mais proximas de 1
		# B RANGE E EXTREMAMENTE CARO EM RELACAO A CPU, E ELE N VALE TANTO QUANTO ACHAR VETOR W
		b_range_multiple=5
		#
		b_multiple = 5
		# O primeiro elemento do vetor w
		latest_optimun = self.max_feature_value*10
		for step in step_sizes:	
			w=np.array([latest_optimun,latest_optimun])
			# A gente pode fazer isso por causa da biblioteca convex, que seria o fato de a gente saber que estamos otimizado
			optimized= False
			while not optimized:
				#Arange e a mesma coisa que range so q mais eficaz e definicao de passos o quanto pula pode ser qlqr coisa nao somente ints
				# A gente faz isso pq a gente n precisa por steps tao grandes quanto o do w

				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
					for transformation in transforms:
						w_t = w*transformation
						# Transformando os w para todos os valores possiveis
						found_option = True
						# Parte mais fraca em relacao a optimizacao do SVM
						# Smo tenta consertar isso daki
						#yi(xi.w+b) >=1
						for i in self.data:
							for xi in self.data[i]:
								yi=i
								if not yi*(np.dot(w_t,xi)+b) >=1:
									found_option = False
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]
							# Como fazer a maginutude de um veto
				if w[0]<0:
					optimized= True
					print('	Otimizado um passo')
				else:
					w= w - step

			norms= sorted([n for n in opt_dict])
			# Lista organizada de todas as magnitudes
			opt_choice = opt_dict[norms[0]]
			# Melhor escolha, onde o w e o menor possivel
			# Lembre-se o dicionario de magnitude w e assim, /w/: [w,b]
			# Logo opt_dict norm 0 e o w
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimun = opt_choice[0][0]+step*2


	def predict(self,features):
		# sign(x.w+b)
		# Funcao base para definir os tracejados da estrada
		classification = np.sign(np.dot(np.array(features),self.w)+self.b)
		if classification !=0 and self.visualization:
			self.ax.scatter(features[0],features[1], s=200, marker='*', c= self.colors[classification])
		# Dot e o produto de dois arrays
		return classification
	def visualize(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
		# Hiperplano = x.w+b
		# v =x.w+b
		#psv=1
		#nsv=-1
		#dec = 0
		def hyperplane(x,w,b,v):
			return(-w[0]*x-b+v / w[1])
			# Funcao feita apra desenhar hiperplano
		datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
		# Feito para limitar o grafico
		hyp_x_min= datarange[0]
		hyp_x_max= datarange[1]
		# Limitando o limite da data
		#(wx+b) = 1
		# Vetores de suporte positivos no hiperplano
		psv1=hyperplane(hyp_x_min,self.w,self.b,1)
		psv2=hyperplane(hyp_x_max,self.w,self.b,1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

		#(wx+b) = -1
		# Vetores de suporte negativo no hiperplano
		nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
		nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
		self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])

		#(wx+b) = 0
		# Vetores de decisao
		db1=hyperplane(hyp_x_min,self.w,self.b,0)
		db2=hyperplane(hyp_x_max,self.w,self.b,0)
		self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])

		plt.show()


data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = support_vector_machine()
svm.fit(data=data_dict)
svm.visualize()

# 