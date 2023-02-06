---
title: "Compra de Pneus"
author: "Ramon Roldan"
date: "2023-01-30"
categories: [Data Science, Tydeverse]
tags: [Tydeverse, Analitycs, Dashboard, Data Science, ETL]
---

## Principal Objetivo

Este estudo busca utilizar técnicas de Data Science para sugerir a melhor opção de compra de pneus conforme produtos atualmente disponíveis no mercado varejista online.

O produto que procuramos são 4 unidades de um pneu aro 14, e escolheremos conforme melhor desempenho no indicador aqui criado: "Nota_Conceitual".

**Nota_Conceitual** = Nota de inmetro (peso 70%) + Menor Preço (Peso 20%) + Fornecedores Reconhecidos (Peso 10%)

## Motivação do Estudo

Recentemente tive a felicidade de comprar meu primeiro carro, um lindo Ford Ka branco modelo 2017, mas como todo carro de segunda mão sempre precisamos arrumar alguma coisa.

Orientado para cultura de Data Driven, e muita experiencia na área de suprimentos, decidi realizar uma pesquisa de mercado.

## Organização do Trabalho

Buscando eficiência divideremos o trabalho em duas partes:

1- Realizar uma consulta no site do inmetro, baixando os dados de pneus, e analisar quais foram as avaliações da organização reguladora.

Esta etapa visa entender quem são os principais players e principais produtos, para não deixarmos de fora algum eventual produto e/ou fornecedor muito bom mas com pouca relevancia no mercado varejista online e principalmente garantirmos a qualidade.

2- Realizar uma consulta de mercado em todos os produtos disponíveis no site pneus store, que se enquadrem nas caractéristicas do produto que procuramos.

Esta etapa visa entender os preços por meio de cotação a mercado em tempo real para garantirmos o melhor preço na hora da compra.

**Premissas**: Este trabalho foi realizado com a linguagem R, IDE Rstudio, com Quarto, e sistema operacional Linux Mint. Foram utilizados conhecimentos de data science e metodologias ágeis. Seguindo as boas práticas do mercado demos preferencia para bibliotecas do tidyverse.

**Principais Etapas**:

-   Definição do principal objetivo a ser alcançado;
-   Detalhes sobre a Motivação deste Estudo;
-   Versionar Projeto no GitHub;
-   Detalhes sobre as fontes de dados;
-   Coleta dos dados;
-   Realizar analise Inferencial

## Versionar projeto no GitHub

Foi criado um repositorio para ajudar no armazenamento de arquivos e versionamento de todo o projeto.

![Imagem do Repositório](assets/img/compra_pneus/print_github.png)

E adotada a ferramenta Kanban para organizar projeto no formato de metodologia ágil:

![Imagem do Kanban](assets/img/compra_pneus/kanbam.png)

## Detalhes sobre a Fonte de Dados

### Informações do Inmetro

O Instituto Nacional de Metrologia, Qualidade e Tecnologia [Inmetro](https://dados.gov.br/dados/conjuntos-dados/programa-brasileiro-de-etiquetagem-pbe) é uma autarquia federal vinculada ao Ministério da Economia. Sua missão institucional é prover confiança à sociedade brasileira nas medições e na qualidade dos produtos, por meio da Metrologia e da Avaliação da Conformidade, promovendo a harmonização das relações de consumo, a inovação e a competitividade do País.

Neste trabalho utilizaremos os dados abertos do Programa Brasileiro de Etiquetagem (PBE), disponibilizados pelo governo federal, para auxilar na escolha pautada em dados:

![Programa Brasileiro de Etiquetagem (PBE)](assets/img/compra_pneus/inmetro.png){alt="Programa Brasileiro de Etiquetagem (PBE)"}

### Loja Pneu Store

A [Pneu Store](https://www.pneustore.com.br/) é uma loja online especializada na comercialização de pneus para veiculos em todo o territorio Brasileiro.

Este site é referência de mercado por apresentar detalhes técnicos dos produtos, bem como preços competitivos de marcas renomeadas, e boas avaliações no reclame aqui.

Vale a pena destacar que tive indicações tanto de alguns profissinais da área, mecânicos, quanto de amigos meus, consumidores finais, que compram e ainda não tiveram problema.

![Imagem do site Pneu Store](assets/img/compra_pneus/print_pneustore.png)

## Coleta dos Dados

### Dataset Inmetro

```r
#Utilizaremos a biblioteca tidyverse para o tratamento, manipulação e vizualização de dados.
library(tidyverse)
#A biblioteca janitor auxiliara na limpeza e tratamento de dados brutos
library(janitor)
```



```
## ℹ Using "','" as decimal and "'.'" as grouping mark. Use `read_delim()` for more control.
```

```
## Rows: 343071 Columns: 33
## ── Column specification ────────────────────
## Delimiter: ";"
## chr (29): NumeroRegistro, Status, Restri...
## dbl  (1): Classificação de Nível de Emis...
## lgl  (3): Coeficiente de Aderência em Pi...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

```
## Rows: 343,071
## Columns: 33
## $ numero_registro                             <chr> …
## $ status                                      <chr> …
## $ restricao                                   <chr> …
## $ data_concessao                              <chr> …
## $ data_validade                               <chr> …
## $ data_cancelamento                           <chr> …
## $ razao_social                                <chr> …
## $ cnpj_cpf                                    <chr> …
## $ telefone                                    <chr> …
## $ logradouro                                  <chr> …
## $ numero                                      <chr> …
## $ complemento                                 <chr> …
## $ cep                                         <chr> …
## $ bairro                                      <chr> …
## $ municipio                                   <chr> …
## $ uf                                          <chr> …
## $ email                                       <chr> …
## $ pac                                         <chr> …
## $ portaria_inmetro                            <chr> …
## $ familia                                     <chr> …
## $ certificado                                 <chr> …
## $ item_marca                                  <chr> …
## $ item_modelo                                 <chr> …
## $ item_descricao                              <chr> …
## $ item_status                                 <chr> …
## $ item_data_alteracao                         <chr> …
## $ item_codigode_barra                         <chr> …
## $ classificacao_de_aderencia_em_pista_molhada <chr> …
## $ classificacao_de_eficiencia_energetica      <chr> …
## $ classificacao_de_nivel_de_emissao_de_ruido  <dbl> …
## $ coeficiente_de_aderencia_em_pista_molhada   <lgl> …
## $ coeficiente_de_resistencia_ao_rolamento     <lgl> …
## $ nivel_de_emissao_de_ruido                   <lgl> …
```


### Dados do site Pneus Store

```r
#Utilizaremos a biblioteca tidyverse para o tratamento, manipulação e vizualização de dados.
library(tidyverse)
#A biblioteca rvest será utilizada para realizar web scrapping das informações disponíveis.
library(rvest)
```



```r
#Reaização do Web scrapping para coleta de dados

url <- 'https://www.pneustore.com.br/categorias/pneus-de-carro/pneus-175-65r14/produto/pneu-firestone-aro-14-f-600-175-65r14-82t-10100082'

base_pneus_store <- read_html(url) %>%
  html_node("table") %>%
  html_table() %>%
  as_tibble() 

base_pneus_store <- base_pneus_store %>% rename('variavel'='X1','resultado'='X2')
base_pneus_store
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
##  9 Índice de peso       "82 - 475 kg\n\t\t\…
## 10 Índice de velocidade "T - 190 km/h\n\t\t…
## # … with 15 more rows
```
