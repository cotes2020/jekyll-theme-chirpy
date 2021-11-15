---
layout: post
title:  "Detecção de objetos com haarcascade"
date:   2020-12-04 08:25:34 -0300
categories: haarcascade detecção classificador
---

Nesse capítulo você irá aprender uma maneira rápida e direta de como criar um classificador Haar Cascade. Além disso, no final do capítulo será disponibilizado um código para detecção de objetos com o classificador criado, que com pequenas alterações, pode se adequar a qualquer situação.

Nesse contexto, o método Haar Cascade, é um método de detecção de objetos proposto por Paul Viola e Michael Jones. É uma abordagem baseada em Machine Learning, em que uma função cascade é treinada com muitas imagens positivas e negativas. Logo, é usado para  detectar objetos em outras imagens.

## Preparando o ambiente

Para esse projeto é necessário que você tenha instalado em sua máquina apenas 3 itens.

1 - Editor de textos de sua preferência, eu particularmente uso o Visual Studio Code.

2 - Alguma versão Python de sua preferência, eu particularmente uso a  3.8.5.

3 - Biblioteca OpenCV .

Aqui não irei explicar como você pode fazer o download e instalação desses itens, pois na internet existem diversos tutorias detalhados de como fazer isso.

## Passos

A criação de um classficador usando o HaarCascade pode ser descrita em um conjunto de 5 passos.

1 - Escolher o objeto.


2 - Selecionar imagens negativas.


3 - Selecionar imagens positivas.


4 - Gerar o vetor de positivas.


5 - Treinar o classificador.


### 1 - Escolher o objeto.

O primeiro passo é escolher o objeto que será identificado, para isso você deverá pensar nos seguintes aspectos:

    * Serão objetos rígidos como uma logo (nike) ou com variações (cadeira,copo)?

    * Objetos rigidos são mais eficientes e mais rápidos.

    * Ao treinar muitas variações pode ser que o classificador fique fraco, portanto, fique atento a isso.

    * Objetos que a cor é fundamental não são recomendados, pois as imagens serão passadas para a escala de cinza.

Para esse projeto, escolhi o objeto faca para ser detectado.

### 2 - Selecionar imagens negativas.

Para selecionar as imagens negativas, você deve ficar atento aos seguintes aspectos:

    * Podem ser qualquer coisa, menos o objeto.

    * Devem ser maiores que as positivas, pois a openCV vai colocar as imagens positivas dentro das negativas.

    * Se possível usar fotos de prováveis fundos onde o objeto é encontrado.

        *Ex: Objeto = carro
         Usar imagens de asfalto e ruas vazias.

Logo, você deve ficar atento as imagens escolhidas, pois como dito elas podem ser qualquer coisa, exceto o objeto escolhido, como escolhemos facas como objeto de detecção devemos, iremos precisar de imagens que não tenham facas.

Quantas imagens negativas?
É relativo, para esse projeto eu conto com 3000 imagens negativas, com diversas variações de fundo. Entretanto, isso vai depender dos resultados obtidos no treinamento, pode ser que eu precise de mais imagens ou não, isso será explicado mais a frente com mais detalhes.

Exemplos de imagens negativas:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/86.jpg" width="100" height="100"/>
</div>
<p align="center"> <b>Figura 1</b></p>

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/2833.jpg" width="100" height="100"/>
</div>
<p align="center"> <b>Figura 2</b></p>

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/43.jpg" width="100" height="100"/>
</div>
<p align="center"> <b>Figura 3</b></p>


OBS: Todas essas imagens tem dimensões 100x100, essa informação será importante para futuras explicações.

Aqui é importante mencionar que você deve criar uma pasta (ex: projeto) onde estará outra pasta com as imagens negativas, na pasta projeto também deve estar as imagens positivas.

Exemplo:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/1.jpg" width="1000" height="200"/>
</div>
<p align="center"> <b>Pasta</b></p>

Na pasta das imagens negativas você deve colocar esse arquivo:

https://github.com/luis131313/cookbook/blob/master/imagens/cap2/criar_lista.bat

E deve executá-lo ao final da escolha das imagens negativas, pois ele vai gerar uma lista com as imagens negativas.

### 3 -Selecionar imagens positivas.

Para selecionar as imagens positivas, você deve ficar atento aos seguintes aspectos:

    * Apenas o objeto.

    * Quantas imagens?

        * Depende da: Qualidade da imagem, tipo do objeto, poder computacional disponível.

    * As imagens devem ter o mesmo tamanho e a proporção precisa ser a mesma, caso contrário a openCV faz isso 
    automaticamente e gera problemas de distorção do objeto.

        * Ex: Uma imagem 100x50, passada pra 25x25, vai ter o objeto descaracterizado.

    * Imagens grandes podem gerar problemas, fazendo o treinamento durar até meses.

    * Sempre que possível usar imagens com fundo branco.

Como dito no primeiro passo, você deve tomar cuidado com as variações do objeto, caso você queira realizar a detecção de um objeto em diferentes ângulos é sugerido que você faça diferentes classificadores. Para exemplificar isso, cito o classificador haarcascade frontal face, que realiza a detecção frontal da face (esse classificador inclusive é de fácil acesso, até a openCV disponibiliza ele pra você) e caso você queira a detecção lateral da face, você deve usar outro classificador, esses cuidados devem ser tomados para que você tenha um bom classificador.

Em relação a quantidade, alguns estudos sugerem que um bom classificador deve ter no mínimo 5000 mil imagens como entrada para o treinamento.

Exemplos de imagens positivas que irei usar para o treinamento do classificador:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/faca_9.png" width="100" height="50"/>
</div>
<p align="center"> <b>Figura 1</b></p>


<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/faca_4.png" width="100" height="50"/>
</div>
<p align="center"> <b>Figura 2</b></p>

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/faca_10.png" width="100" height="50"/>
</div>
<p align="center"> <b>Figura 3</b></p>


OBS: Todas essas imagens tem dimensões 100x50, essa informação será importante para explicações futuras.

Aqui temos o pulo do gato, é possível criar mais imagens positivas a partir das imagens que você já tem, para isso baixe esse arquivo:

https://github.com/luis131313/cookbook/blob/master/imagens/cap2/opencv_createsamples.exe

Coloque ele junto com as imagens positivas, depois basta abrir o CMD, entrar na pasta onde estão suas imagens positivas e digitar esse comando:

```
opencv_createsamples -img faca_1.png -bg negativas/bg.txt -info positivas/positivas.lst -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 300 -bgcolor 255 -bgthresh 10
```

Parâmetros:

-img = Nome da imagem base.

-bg = Nome da pasta / nome do arquivo .txt com as informações das imagens negativas.

-info = Nome da pasta / Nome do arquivo .lst (sempre altere esse parâmetro quando usar uma nova imagem (Ex: positivas2/positivas2.lst, positivas3/positivas3.lst)).

-maxangle (x,y,z) =  Variação de rotação que a imagem terá.

-num = Número de imagens que serão criadas.

-bgtresh = parâmetro que permite a retirada do fundo da imagem, deixando apenas o objeto de interesse (aqui se justifica o fundo branco). Esse parâmetro deve ser analisado com cuidado, pois:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/8.jpg" width="100" height="50"/>
</div>
<p align="center"> <b>bgtresh 10</b></p>

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/0001_0010_0010_0060_0060.jpg" width="100" height="50">
</div>
<p align="center"> <b>bgtresh 100</b></p>


Como resultado você terá a quantidade de imagens mencionada em uma pasta de acordo com o nome que você escolheu e um arquivo .lst que terá informações sobre essas imagens, esse arquivo é de extrema importância e é a partir dele que iremos criar o vetor de imagens.

Nesse caso, usei 10 imagens e criei um total de 3000 imagens a partir desse comando, logo, você terá as pastas positivas1, positivas2 e etc...

Ex:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/2.jpg" width="200" height="200">
</div>
<p align="center"> <b>Pasta</b></p>

## 4 - Gerar o vetor de positivas.

Aqui você deverá criar um vetor para cada pasta com as imagens positiva (positivas1, positivas2, etc...) e depois juntar esses vetores em apenas um vetor.
Para isso, digite esse comando no CMD:

```
opencv_createsamples -info positivas1/positivas1.lst -num 2000 -w 50 -h 25 -vec vetor1.vec
```

Parâmetros:

-info = Nome da pasta que contém as imagens / arquivo .lst (Como criamos 3000 imagens a partir de 10 imagens, temos 10 pastas com 300 imagens cada, logo, teremos que repetir esse comando 10 vezes, alterando o nome da pasta e do arquivo .lst, para positivas2 / positivas2.lst, etc...).

- w e -h = são as dimensões, como nossas imagens eram 100 x 50, eu reduzi para 50 x 25, para reduzir o tamanho do arquivo, até porque para treinar o classificador com as imagens em 100 x 50 eu deveria ter um super computador.

- vec = Nome do vetor (Aqui você também tem que alterar, colocando vetor1, vetor2, etc...).

Após isso, devemos unir todos esses vetores em apenas um, para isso crie uma pasta chamada "vec" e coloque todos os vetores nela.

Depois baixe esse arquivo:

https://github.com/luis131313/cookbook/blob/master/imagens/cap2/mergevec.py

e coloque ele na pasta do seu projeto.

após isso, digite no CMD:

```
python mergevec.py -v vec/ -o vetor_final.vec
```

Após a conclusão você terá um arquivo chamado vetor_final.vec, que é o vetor que iremos utilizar.

Nesse momento a pasta do seu projeto estará assim:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/projeto.png" width="807" height="323">
</div>
<p align="center"> <b>Pasta “Projeto”</b></p>


## 5 - Treinar o classificador

Para o passo final, você deve primeiramente baixar esses arquivos:

https://github.com/luis131313/cookbook/blob/master/imagens/cap2/opencv_traincascade.exe

https://mega.nz/file/09YnVKQb#LdE1iz05i9OeoMqoZtuC3lVn4teeA7gqozVS-N1hG2U

Após isso, você deve abrir a pasta negativas e colocar esses dois arquivos e o vetor final nela, e criar uma pasta chamada “classificador”.

Dessa maneira:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/negativas3.png" width="807" height="323"/>
</div>
<p align="center"> <b>Pasta “classificador”</b></p>

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/negativas2.png" width="807" height="323"/>
</div>
<p align="center"> <b>Arquivos</b></p>


Após isso, abra o CMD na pasta negativas e digite o seguinte comando:

```
opencv_traincascade -data classificador -vec vetor_final.vec -bg bg.txt -numPos 2400 -numNeg 1200 -numStages 15 -w 30 -h 15 -precalcBufSize 1024 -precalcIdxBufSize 1024
```

Parâmetros

-data = Nome da pasta que os arquivos de treinamento serão armazenados.

-vec = Nome do vetor.

-bg = informações das imagens negativas.

-numPos = Número de imagens positivas.

-numNeg = Número de imagens negativas.

-numStages = Número de estágios.

-w e -h = dimensões das imagens.

-precalcBufSize e -precalcIdxBufSize = memória utilizada para o treinamento.

Após o treinamento, na pasta classificador você terá esses arquivos:

<div class="image-container" style="display: flex; justify-content: center;">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/cascade.png" width="798" height="217"/>
</div>
<p align="center"> <b>Arquivos</b></p>

O arquivo cascade.xml é o nosso classificador.

O arquivo params.xml são os parâmetros usados no treinamento.

E os outros arquivos, são os resultados de cada estágio do treinamento.

Sobre o uso dos parâmetros:

* É indicado que você use metade do número de imagens positivas para as negativas no primeiro treinamento.

* Alguns estudos sugerem que um bom classificador deve ter no mínimo 5000 imagens positivas.

* Após o primeiro treinamento, se você notar que está tendo muitos falsos positivos, aumente o número de imagens negativas, se notar que não está realizando a detecção, aumente o número de imagens positivas, faça novos treinamentos até ter bons resultados.

* Para melhorar os resultados você também pode aumentar o número de estágios.

* Não se esqueça que a soma dos parâmetros -precalcBufSize e -precalcIdxBufSize não pode ser maior que a memória disponível.

* Quanto mais imagens negativas, positivas, estágios de treinamento e dimensão das imagens, mais o treinamento vai demorar, podendo fazer o treinamento durar até meses.


## Código para detecção

Agora irei apresentar um código que irá realizar a detecção de facas.

Para isso baixe esse classificador que eu criei, ele ainda não está pronto, logo não irá apresentar resultados excelentes.

https://github.com/luis131313/cookbook/blob/master/imagens/cap2/cascade_facas.xml

```python

import cv2

#variável que armazena a imagem

imagem1 = 'teste1.png'

#variável que armazena o arquivo xml

cascade_path1 = 'cascade_facas.xml' 

#cria o classificador

clf1 = cv2.CascadeClassifier(cascade_path1)

#lê a imagem

img1 = cv2.imread(imagem1)

#converte para cinza

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#Função da detecção

deteccoes1 = clf1.detectMultiScale(gray1, scaleFactor=1.01, minNeighbors=5, minSize=(1,1))

#desenha o retângulo com as coordenadas obtidas

for(x,y,w,h) in deteccoes1:
    img1 = cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255), 2)

#para visualizar a imagem

cv2.imshow('Classificador 1', img1)

#mantém a janela aberta até que eu digite uma tecla

cv2.waitKey(0)

#destrói a janela

cv2.destroyAllWindows()

```

Ao executar o código com um exemplo, teremos essa detecção:

<div align="center">
    <p align="center">
    <img src="{{site.baseurl}}/assets/img/imagens/cap2/resultado.png" width="1193" height="667"/>
    </p>
    <p> <b> Imagem retirada do Google</b> </p>
</div>

Podemos notar alguns falsos positivos, o que indica que seria interessante realizar um novo treinamento com mais imagens negativas.

## Considerações finais

Vários dos arquivos apresentados foram cedidos pela www.iaexpert.academy, agradeço imensamente pela generosidade.

Fiz esse tutorial com muito carinho e espero que seja útil para você, a intenção aqui foi realizar um pequeno projeto usando o método HaarCascade, ainda existe muito há aprender sobre esse método, mas a minha intenção é apenas introduzir esse assunto.

Desejo bons estudos e bons trabalhos.

Atenciosamente,

Luis Fernando Santos Ferreira, Aluno do curso de Ciência da Computação na Universidade Federal de Lavras.

Linkedin: https://www.linkedin.com/in/luis-ferreira-3b02131a8/
