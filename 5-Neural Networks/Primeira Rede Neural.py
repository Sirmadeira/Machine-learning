import tensorflow as tf
#Essa rede neural tera como funcao identificar o carater que foi escrito,atraves de pixels da imagen
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("/tmp/data", one_hot=True)
#One hot significa um elementa da imagen tera valor 1, enquanto o resto tera valor 0
#Exemplo: 0=[1,0,0,0,0,0,0,0,0,0], 1=[0,1,0,0,0,0,0,0,0,0] e vai indo
#A gente vai ter 10 classes divisivas, entre 0-9
n_nodes_hl1 = 500
n_nodes_hl2 = 500
m_nodes_hl3 = 500
#Numero de nodes no hidden layer, como voce pode ver ela e deep, devido a ter um alto numero de nodes
n_classes = 10
batch_size = 100
#Tamanho da quantidade de features que entra nos inputs nodes
x = tf.placeholder('float',[None, 784])
#input data, o segundo parametro vai esmagar a data em 784 pixels ele representa altura x width da data. E ja que a nossa data nao e 3d nao precisa de altura
y = tf.placeholder()
#Labels, para identificar a data

def neural_network_model(data):
	#Lembrete=(input_data * weights) + bias
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}
	#Isso daki e simplesmente um conjunto de numero randomicos. Que vao atuar como nossos weights, depois a medida que treinamos fica tudo certinho
	#Biases, nao precisam de um multiplo
	
	l1=tf.add(tf.matmul(data, hidden_1_layer['weights'])+hidden_1_layer['biases'])
	#l1=LAYER1
	l1=tf.nn.relu(l1)
	#Funcao de ativacao
	l2=tf.add(tf.matmul(l1, hidden_2_layer['weights'])+hidden_2_layer['biases'])
	l2=tf.nn.relu(l2)
	#Como voce pode ver ele esta recebendo  a data depois de passar por layer 1 e sua funcao de ativacao

	l3=tf.add(tf.matmul(l2, hidden_3_layer['weights'])+hidden_3_layer['biases'])
	l3=tf.nn.relu(l3)

	output=tf.matmul(l3, output_layer['weights'])+output_layer['biases']

	return output