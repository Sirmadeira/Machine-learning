#Aula1 Informacoes principais-Introducao
#Uma rede neural consiste de nodes, e conexoes entre os nodes.
#Toda rede neural comeca com parametros nao identificados.
#Parametros- Seriam  os numero identificado, ao encaixar uma linha de tendencia a equacao
#Basicamente, akl formulinha do tracejado
#Esses parametro sao identificados por um metodo chamado backpropagation
#Existe muitas linhas que a gente pode escolher para formatar os parametros
#Existe a softplus, bentline , Rectified linear unit ou uma funcao sigmoid
# O nome dado para essas linhas e activation function
#Na pratica se utiliza muita mais Relu ou softplus do que a sigmoid que geralmente e ensinada
# Uma rede neural sempre comeca com inputs e termina com output, geralmente tem multiplas conexoes e multiplos inputs
# O nome dos nodes entre output e input e hidden nodes, o nome da 'linha' vertical entre nodes e hidden layer
# Entre a conexao entre o input node e um hidden node, se multiplica ele para ele ser encaixado, na activation function
# Os valores pelo quais ele e multiplicado, e achado pelo fenomeno de backpropagation
# Depois disso, acha-se o eixo y atraves da funcao definida da activation. Basicamente aplicase a formula escolhida se e relu formula de relu softplus softplus etc
# Logo em seguida, se formula uma nova funcao de acordo com os valores y achados. Depois disso voce multiplica, pelo outro parametro achado pelo backpropagation
#Formulando uma nova funcao, depois disso voce e capaz de notar que as funcoes y e x se encaixam em certas partes das funcoes de ativacao
# Depois disso, soma-se o eixos y achados, na conexoes hideens superiores na funcao do eixo inferior
#Formulando assim a funcao perfeitinha
# Logo para fazer a predicao e so analisar o  eixo x dado pela funcao final
#Ou so inserir ele no input layer e tudo esse processo sera refeito
#Aula 2 Infos Principais-Backpropagation, introducao
#Os parametros definido por backpropagation, sao divididos em pesos e bias
#Os pesos sao facilmente definido,pelo fato deles geralmente terem um x do lado
#Enquanto o bias sao os que sao usados para fazer a transformacao na activation function
#Ou seja, na conexao sempre se comeca com peso e termina com bias
#Bias, geralmente comecam com o valor 0
#Nos podemos calcular o quao bom a funcao final encaixa na data de treinamento
#Fazendo a soma dos valores residuais ao quadrado, residuo sera a diferenca entre o valor observado - o predito
#Depois disso, achamo um novo ponto numa funcao SSR-SUM OF SQUARED RESIDUALS, por bias final.
#Escalamos, o b3 multiplicamos ele pela funcao final e achamos uma nova funcao.
#Ate diminuirmo, a soma do valores residuais ao maximo.
#A gente sabe a qual seria o bfinal adequado atraves do ponto, no grafico ssr onde o bfinal  esta no minimo.
#Relembrando a funcao final seria a soma das funcoes dadas pelos pesos, anteriores
#Sendo assim, e sabendo que temos que achar o valor bfinal. Temos que achar o derivado, do ssr do bfinal
#Que consiste de duas parte como apresentado pela funcao, o derivado ssr/d bfinal = d SSR/d predfito + d predito/d b3, achamos essa formula atraves da chain rule
#CHAIN RULE
#Lembrete- Quando uma equacao passa pela origen ela n tem intercept
#Se voce tive duas equacoes com um valor semelhante por exemplo. Peso identifica altura e altura identifica tamanho de sapato
#Voce posse associa-las, geralmente achando uma relacao
#Essa relacao e a base da chain rule
#Exemplo, d= derivado, dTamanho/dPeso=(dTamanho/dAltura *dTamanho/dPeso). Seria utilizada, para funcoes que tem as mesmas verosimilhancas
#O problema e quando nao e obvio a relacoes entre as funcoes
#Para identificar a relacao, que de inicio nao e obvio tente achar parenteses
#Depois simplifica por exemplo, dCraves/dTime = (dCraves/dInside * dInside/dTempo)
#Esse inside seria as coisa dentro do parenteses
#Se utilizamo disso, para achar exatamente a relacao entre o residuo ao quadrado identificado e o seu interceptador
#Tentando achar o quando o residuo quadrado e 0 obviamente
#A funcao seria, d Residual``2/dIntercept=(d Residual``2/dResidual * dResidual*/dIntercept)
#GRADIENTE DESCENDENTE
#Gradiente descendente serve para achar o valor adequado para o interceptador e a inclinacao
#Um belo jeito de achar o melhor gradiente, seria aumentar gradualmente o intercept, e calcular constantemente qual e a melhor
#Soma dos residuos ao quadrado
#Vale lembra que geralmente, o gradiente ao calcular a soma dos residuos ao quadrado. Ele comeca aumentando o intercept grandiosamente
#E depois diminui o intercept consideralvemente 
#Lembrete= Soma dos residuos ao quadrado= (x do ponto-(intercept+y*x))**2
#Depois disso a gente aplica a d/d intercept Soma dos residuos ao quadrado= d/d intercept das partes de cada ponto que no caso seria, a soma dos residuos ao quadrado de cada ponto encontrado
#Gradiente descente e utilizado quando nao existe uma derivada=0 ou seja quando nao passa pela os eixos x
#Quando a inclinacao da funcao da soma dos residuos ao quadrado pelo intercept,
#estiver chegando perto de 0 e quando a gente comeca a dar os pequenos passos. Pq estamos chegando perto do ponto ideal
#Geralmente define-se isso quando o step size for menor que 0.001, em pratica. E tbm caso chegue a um limite
# A soma dos residuos ao quadrado tambem e chamada de loss function
#A gente tambem pode se utilizar o descente gradiente com duas variaveis ao mesmo tempo
#O intercept e o slope
#A gente basicamente define eles como uma constante, ao tirar a chain rule e seus devidos derivados. Basicamente nulificando os
#Ou seja quando eu faco d/d intercept a derivado da minha inclinacao  sera uma constante logo 0
#Quando voce tem mais de dois ou dois derivados na mesma funcao eles sao chamado de gradiente a gente vai se utilizar desse gradiente
#Para atingir o menor ponto da loss function, logo gradiente descente
#No final, o primeiro passo = Pegar o derivado da loss function para cada parametro
#O segundo e pegar valores random para os parametro
#O terceiro e por os parametros nas derivadas, ou melhor gradiente
#O quarto calcular os step sizes = inclinacao * aprendizado
#O quinto calcular os novos parametros = velho parametro- step size
#Repita o terceiro passo ate vc atinger o step size bem pequeno
#Para calcular o gradiente descente de milhoes de pontos se usa de stochastic gradiente descendente
#Aula 3 ReLu em acao
#Agora a gente vai substituir a funcao softplus para a funcao de ativacao mais utilizada hoje em dia relu
#A funcao relu tem como output, o maior valor das coordenedas
#Lembrete depois de fazer a primeira conexao com hidden layer e formatar a nova funcao
#A gente multiplica todas as coordenadas y pelo parametro achado pela backpropagation
#No parametro final depois da somataria, que junciona as duas funcoes. a gente aplica a funcao relu dnv
#Achando algo quase perfeito
#Aviso a relu funcao de ativacao nao e curvada, ela e retada.Logo a derivada nao e definida quando a funcao tem sua quebra
#Para evitar isso defina a derivada do ponto 0 a 1 se nao da erro
#Aula4- Multiplos inputs
#Nada dms, so aumento o numero de dimensoes.
#E o raw output fica meio zuado tema da proxima aula
#Aula5-Argmax e softmax
#Para corrigir o output se utiliza de dois layers, argmax e softmax
#O argmax define o maior valor para 1 e os menores valores para 0
#Devido ao output do argmax ser constante, nao podemos se utilizar dele para tentar corrigir os parametros antecessores
#Porque a derviado dele e 0
#Logo nao podemos utilizar dele para backpropagation
#Agora a softmax function que, nao e a softplus, pode
#logo ela e mais adequada para treinamento enquanto a argmax quando ja ta pronto	
#Na softmax, a gente poim um euler number embaixo de cada output fazendo deles elevados
#Aviso nao ponha muita confianca nas probabibilidades dados pelo softmax, elas nao necessariamente sao as mais
#Precisas por assim dizer, devido a varianca dos parametros no treinamento. Que sao randomicamente selecionados
#Aviso a gente se utilizava de soma dos residuos ao quadrado para avaliar o quao bem a data era treinada ou se encaixava
#Devido ao uso da softmax functions que tem outputs de predicao entre 1 e 0. 
#A gente precisa se utiliza de cross entropy para verificar o quao bem a neural se encaixa na data
#CROSS ENTROPY
#Tem uma formula geral e e relativamente simples, seria:
#Cross entropy= -log com base euler(valor do output pos softmax ou a predicted probability)
#A formula verdadeira tem uma somataria, de todos outputs. No entanto as probabilidades das outras conclusoes sao nulificadas
#Para calcular o total error vc pega todas as formulas de todas as classes(n de outputs) e adiciona
#Se voce comparar as funcoes ssr e cross entropy, vc comeca a entender pq se utiliza ela
#Na sum squared residuals, a funcao entre e 0 e 1 e basicamente igual, e uma linha reta
#Enquanto na cross entropy, ela sobe radicalmente perto do 0 e vira linha reta no 1
#Entao quando a gente se utilizar de gradiente descente, a reta de avaliacao ssr vai ser bem retinha. Fazendo com que os step sizes
#Sejam muito pequenos
#Enquanto no da cross entropy eles vai ser muito maiores
#Mas e agora como fazer o backpropagation
#BACKPROPAGATION COM CROSS ENTROPY
#Para poder otimizar o bias a ser melhorado a gente precisa do gradiente descente o que faz com que a gente precise calcular a derivada
#Dele de acordo com a cross entropy
#Formula d CE output/d bias a ser melhorado
#Tendo em consideracao que a cross entropy e interligado a bias final.Pelo output predito de uma das classes e a funcao
#Final achada, a gente pode se usar da chain rule para montar a associacao
#Que seria: d CEoutput/d bias a ser melhorado = d CEoutput/d probabilidade do output * d probabilidade do output/d funcao final(raw output) do output * d funcao final do output / d bias a ser melhorado
# O que resulta , probabilidade do output -1
#Vale lembrar que a probabilidade de output seguida tem a mesma funcao final do que a inicial, logo
#d CEoutpu segundo/ d bias da funcao (nesse caso o mesmo da de cima)=d CE probabilidade secundaria/ d predicao do output secundaria * d predicao do output secundaria/d funcao final * d Funcao final/d bias a ser melhorado
#o que resulta, na prababilidade do output original
#Todos os output das classe sucessoras, vao ser a probabilidade do output da original
#obviamente sendo definido pelo bias a ser melhorado
#Logo se for o bias da primeira classe, ele vai ter a prababilidade do output -1 e o resto vai ter a prababilidade do output como derivada
#Se for da segunda, ele e o que vai ter a probabilidade do output - 1 e o resto vai ter a prababilidade do output como derivada
#E assim vai
#Voltando ao tema, se nos fizermos a funcao em que eixo x seria um monte de bias a serem melhorados random e o eixo y a total cross entropy(soma das entropias)
#Se a gente achar o valor minimo deste grafico, ou em outras palavras o b a ser melhorado minimo
#O que a gente acha por gradiente descente, fazendo entao a formula
#d Cross entropy/ d Do b final a ser achado =dCE(Pred1)/ db a ser melhorado + dCE(Pred1)/ db a ser melhorado... e assim vai
#Depende dos numeros de classes, tendo como final a inclinacao do ponto. Que ao por na formula
#Step size= inclinacao * learning rate achamos o step size.
#Novo b a ser melhorado = velho b a ser melhorado - step size
#E repete o processo de gradiencia, ate as predicoes nao melhorarem mais
#CLASSIFICACAO DE IMAGENS USANDO NEURAL
#Para identificar imagens voce precisa se utilizar de uma convulutional neural network
#Imagens sao nada mais alem de um monte de pixels
#Para voce analisar uma imagens tudo que voce tem que fazer e por todos os pixels num formoto de coluna e considerar eles como nodes
#No entanto esse metodo e muito pesado, alem do mais ele n tolera pequenas mudancas no angulo da imagen e nao tira vantagem das imagens complexas
#Logo se utilizarmos de uma convolutional network que evita todos esses problemas a gente fica mais suave
#A primeira coisa que ela faz e filtrar a imagen, e pegar um conjunto pequeno de pixels randomicos
#Depois disso ela treina e reajeita os filtros atraves de backpropagation
#Depois disso ela aplica o filtro da imagen e multiplica cada pixel da imagen por cada pixel do filtro,
# e adiciona todos eles no final. O nome disso e dot product
#Ao calcularmos o dot product, entre a imagen e o filtro. A gente pode dizer que o filtro e convolved com o input
#E isso que origina o nome rede neural convolucional
#Depois do processo de dot product, voce move um ou mais pixels para lado e fica calculando o dot product entre os pixels
#Que se sobresaiem na imagen, adiciona um bias,e no final poim o valor num feature map
#Isso faz com que a gente se aproveite de qualquer relacao que pode haver entre os pixels das imagen
#Depois disso a gente aplica uma funcao de ativacao no feature map o que zera todos os valores negativos
#Ae a gente aplica um mesmo filtro, no entanto dessa vez ele so pega o maior valor
#Esse filtro so pega valores maiores, e ele vai analisando cada parte do feature map em conjuntos de regioes
#O nome disso e max pooling, ele basicamente ta analisando qual parte do filtro inicial se encaixo melhor na imagen inserida
#Existe uma alternativa por max pooling, isso seria average pooling ao inves de ele pegar o maximo
#Ele pega o numero de feature maps pos funcao de ativacao e usa eles como denominador dos valores dentro do feature
#Depois disso a gente pega o array resultante. E poim na neural normal
#Obviamente, convulutiona neural ajudam a diminuir o numero de inputs necessarios.
