import matplotlib.pyplot as plt 	
from matplotlib import style
import numpy as np
style.use('ggplot')

class support_vector_machine:
	# Nenhum dos metodos de uma classe irao rodar alem do init metodo
	def __init__(self,visualization=True):
		self.visualization= visualization
		self.color = {1:'r',-1:'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)
			# Isso daki e utilizado para visualizar o grafico, e o seu subplot

	def fit(self, data):
		self.data = data
		# { a magnitude de w : [w,b]  }
		opt_dict = {}
		transforms = {[[1,1],[-1,1],[1,-1],[-1,-1]]}
		all_data = []
		for yi in self.data:
			for featureset in self.data[y1]:
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
					w=w-step




	def predict(self,features):
		# sign(x.w+b)
		# Funcao base para definir os tracejados da estrada
		classification = np.sign(np.dot(np.array(features),self.w)+self.b)
		# Dot e o produto de dois arrays
		return classification



data_dict= {-1:np.array([[1,7],[2,8],[3,8],]),1:np.array([[5,1],[6,-1],[7,3],])}

