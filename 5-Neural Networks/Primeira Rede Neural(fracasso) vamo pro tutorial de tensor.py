import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#Essa rede neural tera como funcao identificar o carater que foi escrito,atraves de pixels da imagen
import tensorflow_datasets


mnist = tensorflow_datasets.load('mnist')

#One hot significa um elementa da imagen tera valor 1, enquanto o resto tera valor 0
#Exemplo: 0=[1,0,0,0,0,0,0,0,0,0], 1=[0,1,0,0,0,0,0,0,0,0] e vai indo
#A gente vai ter 10 classes divisivas, entre 0-9
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
#Numero de nodes no hidden layer, como voce pode ver ela e deep, devido a ter um alto numero de nodes
n_classes = 10
batch_size = 100
#Tamanho da quantidade de features que entra nos inputs nodes
x = tf.placeholder('float', [None, 784] )
#input data, o segundo parametro vai esmagar a data em 784 pixels ele representa altura x width da data. E ja que a nossa data nao e 3d nao precisa de altura
y = tf.placeholder('float')
#Labels, para identificar a data

def neural_network_model(data):
	#Lembrete=(input_data * weights) + bias
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}
	#Isso daki e simplesmente um conjunto de numero randomicos. Que vao atuar como nossos weights, depois a medida que treinamos fica tudo certinho
	#Biases, nao precisam de um multiplo
	
	l1=tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
	#l1=LAYER1
	l1=tf.nn.relu(l1)
	#Funcao de ativacao
	l2=tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2=tf.nn.relu(l2)
	#Como voce pode ver ele esta recebendo  a data depois de passar por layer 1 e sua funcao de ativacao

	l3=tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3=tf.nn.relu(l3)

	output=tf.matmul(l3, output_layer['weights'])+output_layer['biases']

	return output

def train_neural_network(x):
	prediction= neural_network_model(x)
	#Isso daki seria eu avaliando a preidcao de acordo com output data da funcao acima
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	#Essa funcao custo ira avaliar o quao errado a gente esta, basicamente a nossa loss function
	#Ela seria o que a gente gostaria de minimizar com a alteracao de nossos pesos e bias
	optimizer = tf.train.AdamOptimizer().minimize(cost)
    #Para minimizar ela a gente se utiliza do adamoptimizer que famosamente um bom optimizer
    #(entender adam optmizer)
	hm_epochs=10
    #Numero de vezes que o algoritmo de aprendizado ira passar pela data, numero de feed forwards e backprops
    #Aviso essa rede neural e feed forward, logo os dados nao irao passa retiliamente
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
    	#Inicializando as variaveis que iremos passar por
    	#A sessao comeco aqui
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train_num_examples/batch_size)):
				x, y = mnist.train.next_batch(batch_size)
				#Aviso _ seria variavel que eu n me importo com
				_, c = sess.run([optimizer,cost],feed_dict = {x:y,y:y})
				#C seria cost
				epoch_loss += c
			print('Epoch', epoch,'completada',hm_epochs,'loss:',epoch_loss)
		#Para cada epoch e cada batch em nossa data a gente vai rodar um optimizer e avaliar o nosso custo
		#Para manter um tracking de quanto a gente ta ganhando por custo basicamente, ou perdendo nesse caso
		#Para cada epoch a gente da um output de loss que deveria estar diminuindo e bom para notar se ta tendo diminishing returns

		correct= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		#Isso daki vai nos falar quanas predicoes estiveram corretas com os labels
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		#Cast seria eu modificando o valor de predicao de argmax, para um float
		print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

		
train_neural_network(x)