---
title: Como contribuir neste Blog
author:
  name: Natalia Amorim
  link: https://github.com/NataliaCarvalho03
date: 2022-01-23 16:42:34 -0300
categories: [contribuicao, artigos]
tags: [getting started]
pin: true
---

Aprenda como enviar o seu tutorial em forma de artigo para o Blog oficial do Grupo OpenCV Brasil (apelidado carinhosamente de OpenCVismo Brasil).

## 1. Fork o repositório do Blog

[Acesse o repositório do nosso blog](https://github.com/Grupo-OpenCV-BR/Grupo-OpenCV-BR.github.io) e dê um fork. 

## 2. Escreva seu artigo em um arquivo markdown

### Antes de escrever o conteúdo

Antes de começar a escrever o conteúdo propriamente dito, lembre-se de colocar o cabeçalho que todos os nossos artigos devem ter, um exemplo:

```
---
title: O título do seu Artigo
author:
  name: Seu Nome
  link: link para seu github ou linkedin
date: 2022-01-23 16:42:34 -0300
categories: [categaria 1, categoria2]
tags: [tag1 tag2]
pin: false
---
```

- Escreva seu título entre aspas duplas (sim, é uma string);

- Escreva seu nome e coloque um link para uma rede social sua: Pode ser seu blog, linkedin ou github. O importante é o leitor te encontrar!

- Modifique a data para a data em que você escreveu seu artigo, o formato da data é ```aaaa-mm-dd``` (ano-mês-dia);

- Escreva em quais categorias seu conteúdo se encaixa (Ex: filtros deepLearning Yolo). As categorias devem ser separadas por espaço e cada artigo deve ter no máximo três categorias. O mesmo vale para as tags


### Inserindo Imagens

Imagens podem ser inseridas no artigo para melhorar o entendimento do leitor. Você pode utilizar a própria sintaxe da linguagem markdown para fazer isso de forma simples e rápida. Um exmeplo:

```md
![image info](link-para-a-sua-imagem)
```

Se você produziu imagens para inserir no seu tutorial, crie uma pasta dentro de ```/assets/img/imagens/nome-da-sua-pasta``` e coloque as imagens que você produziu dentro do seu diretório. Assim, você pode chamar essas imagens em seu artigo da seguinte forma:

{% raw %}
```
![image info]({{ site.baseurl }}/assets/img/imagens/nome-da-sua-pasta/sua-imagem.png)
```
{% endraw %}


### Inserindo código-fonte



Este não deveria precisar de maiores explicações, não é? Para inserir seu código fonte, basta usar a sintaxe padrão do markdown para código: Conteúdo do código entre uma par de três crases. Um exemplo:


![image info]({{site.baseurl}}/assets/img/imagens/como-contribuir/exemplo-codigo.png)


### Deixe sua assinatura no artigo

No final do artigo você pode colocar uma assinatura para deixar as pessoas saberem que você é o autor. Olhe como os outros membros fizeram suas assinaturas e construa a sua.


### Salvando o artigo


Seu artigo deve estar em um arquivo markdown (.md) e deve ser colocado dentro do diretório _posts. Fique atento para dar o nome correto ao seu arquivo, perceba que o nome do arquivo .md tem uma padronização, por exemplo:

```
2022-01-23-como_contribuir.md
```
- Primeiro insira a data que você criou este conteúdo no formato AAAA-MM-DD (ano-mês-dia);

- Depois coloque um título resumido do seu artigo (com no máximo três palavras separadas por underline);

- O nome do seu artigo deve ficar com o mesmo padrão do exemplo acima.


## 3. Deploy local

Você pode testar o blog localmente em sua máquina, assim você testa se o site está funcionando com seu artigo antes de enviar a pull request e quebrar o site blog. As instruções sobre o que você precisa instalar e como rodar subir o site localmente podem ser encontradas [aqui](https://github.com/cotes2020/jekyll-theme-chirpy) . Não seja preguiçoso(a), leia! :)


## 4. Envie uma Pull Request

Agora que você já escreveu, se atentou para os padrões necessários, conferiu se o site não quebrou com as suas atualizações, é hora de enviar uma PR para qe seu artigo entre para o blog oficial:

- Abra uma PR e marque Natália Carvalho (NataliaCarvalho03) como revisora;

- Espere que o processo de revisão termine, caso haja alguma coisa a corrigir, você será avisado.


## Conclusão

Então, garotos e garotas, é isso! Sintam-se a vontade para contribuir com novos artigos e qualquer dúvida, basta nos contatar nos grupos do Telegram ou Discord!


Com muito carinho e um pouco de agressão...<br/><br/>

Natália C. de Amorim<br/>
Engenheira em Visão Computacional e fundadora do Grupo OpenCV Brasil