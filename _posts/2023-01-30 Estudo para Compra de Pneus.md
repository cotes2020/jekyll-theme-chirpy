---
title: "Compra de Pneus"
author: "Ramon Roldan"
date: "2023-01-30"
categories: [Data Science, Tydeverse]
tags: [Tydeverse, Analitycs, Dashboard, Data Science, ETL]
---

## Principal Objetivo

Este estudo busca utilizar técnicas de Data Science para sugerir a melhor opção de compra de pneus conforme produtos atualmente disponíveis no mercado de varejo.

## Motivação do Estudo

Recentemente tive a felicidade de comprar meu primeiro carro, um lindo Ford Ka branco modelo 2017, mas como todo carro de segunda mão sempre precisamos arrumar alguma coisa.

Orientado para cultura de Data Driven, e muita experiencia na área de suprimentos, decidi realizar uma pesquisa de mercado.

## Organização do Trabalho

Premissas: Este trabalho foi realizado com a linguagem R, IDE Rstudio, com Quarto, e sistema operacional Linux Mint. Foram utilizados conhecimentos de data science e metodologias ágeis. Seguindo as boas práticas do mercado demos preferencia para bibliotecas do tidyverse.

Principais Etapas:

-   Definição do objetivo do trabalho;

-   Versionar trabalho no GitHub;

-   Utilizar a ferramenta Kanban para organizar projeto no formato de metodologia ágil;

-   Coleta dos dados;

-   Realizar analise exploratória;

-   Limpeza e tratamento dos dados;

-   Salvar modelo treinado em arquivo r para posteriormente aplicar no framework tidymodels;

-   Desenvolver um dashboard dinâmico com os pacotes flexdashboard, shiny e ploty;

-   Realizar deploy do modelo and uploud in the shinyapp.io

## Versionar trabalho no GitHub

Foi criado um repositorio para ajudar no armazentamento de arquivos e versionamento de todo o projeto.

!\[Imagem do Repositório\](img/print_github.png)

## Site do Imetro com Dataset

<https://dados.gov.br/dados/conjuntos-dados/programa-brasileiro-de-etiquetagem-pbe>

## Dataset

## Web Scrapping PneuStore


```r
library(tidyverse)
library(rvest)
```

```
## 
## Attaching package: 'rvest'
```

```
## The following object is masked from 'package:readr':
## 
##     guess_encoding
```

```r
url <- 'https://www.pneustore.com.br/categorias/pneus-de-carro/pneus-175-65r14/produto/pneu-firestone-aro-14-f-600-175-65r14-82t-10100082'

base <- read_html(url) %>%
  html_node("table") %>%
  html_table() %>%
  as_tibble() 

base %>% rename('variavel'='X1','resultado'='X2')
```

```
## # A tibble: 25 × 2
##    variavel             resultado                                             
##    <chr>                <chr>                                                 
##  1 Marca                "FIRESTONE"                                           
##  2 Fabricante           "BRIDGESTONE"                                         
##  3 Modelo               "F-600"                                               
##  4 Medida               "175/65R14"                                           
##  5 Largura              "175mm"                                               
##  6 Perfil               "65%"                                                 
##  7 Aro                  "14"                                                  
##  8 Diâmetro total em mm "583.1"                                               
##  9 Índice de peso       "82 - 475 kg\n\t\t\t\t\t\t\t,\n\t\t\t\t\t\t\t82"      
## 10 Índice de velocidade "T - 190 km/h\n\t\t\t\t\t\t\t\t\t,\n\t\t\t\t\t\t\t\t\…
## # … with 15 more rows
```
