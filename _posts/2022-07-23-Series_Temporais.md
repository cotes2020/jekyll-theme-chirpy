---
title: "Series Temporais"
author: "Ramon Roldan"
date: "2022-06-29"
categories: [Data Science, Tydeverse]
tags: [Machine Learning, Tydeverse, Times Series, IBOV, Data Science, ETL, Mercado Financeiro, API, DataViz]
output:
  md_document:
    variant: gfm
    preserve_yaml: TRUE
knit: (function(inputFile, encoding) {
  rmarkdown::render(inputFile, encoding = encoding, output_dir = "/home/ramon_de_lara/Ramon/roldanramon.github.io/_posts") })
---

## Objetivo

O trabalho busca desenvolver um dashboard que rode um modelo de machine learning capaz de prever o movimento do dia seguinte no indice ibovespa, baseado em dados históricos disponibilizados via API no yahoo finace, e para isso serão aplicadas técnicas estatisticas de regressão logística e regressão linear em series temporáis.

O resultado final esta disponível em: https://sfefqj-ramon-roldan.shinyapps.io/Financial_Market_Analysis/

## Organização do trabalho
* Definição do objetivo do trabalho;
* Coleta dos dados por meio da API;
* Limpeza e tratamento dos dados;
* Start an exploratory analysis;
* Aplicar framework do tidymodels para desenvolver modelo;
* Avaliar resultados do modelo;
* Criar um dashboard com o flexdashboard, shiny e ploty;
* Realizar deploy do modelo and uploud in the shinyapp.io

### Carregar Bibliotecas necessárias
```{r}
knitr::opts_chunk$set(echo = TRUE)
library(flexdashboard)
library(shiny)
library(tidymodels)
library(readr)
library(janitor)
library(stringr)
library(lubridate)
library(ggplot2)
library(plotly)
library(patchwork)
library(knitr)
library(kableExtra)
library(quantmod)
```

### Coleta e Tratamento dos dados
```{r}
modelo <- read_rds('win_model.rds')

#Carrega Base
base <- getSymbols(Symbols = '^BVSP')
base <- BVSP %>% data.frame() %>% rownames_to_column() %>% clean_names() %>%
  rename("data"='rowname',"ultimo"='bvsp_close',"abertura"='bvsp_open', "maxima"='bvsp_high',"minima"='bvsp_low', 'vol'='bvsp_volume') %>%
  mutate(data = as_date(data),
         vol=as.character(vol),
         var_percent= as.character(paste0(round(((ultimo/lag(ultimo))-1)*100,2),'%')),
         meta = if_else(var_percent > 0,1,0) %>% as.factor(),
         abertura= format(abertura,big.mark='.',decimal.mark=',') %>% as.numeric(),
         maxima= format(maxima,big.mark='.',decimal.mark=',') %>% as.numeric(),
         minima= format(minima,big.mark='.',decimal.mark=',') %>% as.numeric(),
         ultimo= format(ultimo,big.mark='.',decimal.mark=',') %>% as.numeric(),
         greenRed=ifelse(abertura-ultimo>0,"Red","Green")) %>%
  select(-bvsp_adjusted) %>% filter(!is.na(ultimo))

shiny::dateRangeInput(inputId = 'periodo',label = 'Período',start = max(base$data)-60,end = max(base$data),language = 'pt')

downloadHandler(
    filename = function() { 
      paste("dataset-", Sys.Date(), ".csv", sep="")
    },
    content = function(file) {
      readr::write_csv(base %>% select(-c(var_percent,meta,greenRed)) %>% 
                         filter(data>= input$periodo[1] & data <= input$periodo[2]), file)
    })

novo_dado <- base %>% filter(data == max(data))

#Aplica o modelo para obter a probabilidade de movimento do dia seguinte
resultado_valor <- round(if_else(predict(object = modelo, new_data = novo_dado)==0,
        predict(object = modelo, new_data = novo_dado,type = 'prob') %>% pull(1),
        predict(object = modelo, new_data = novo_dado,type = 'prob') %>% pull(2))*100, 2)

resultado_label=if_else(predict(object = modelo, new_data = novo_dado)==0,'Baixa','Alta')
```


### Probabilidade e direção do Movimento
```{r}
renderGauge({
gauge(value = resultado_valor,label = resultado_label, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0, 39)))
})
```

### Concentração Dos Dados
```{r}
renderPlot({
p1 <- ggplot(base %>% filter(data>= input$periodo[1] & data <= input$periodo[2]))+
  geom_density(aes(x =ultimo),fill='blue',alpha=.25)+
  geom_vline(xintercept = novo_dado %>% pull(ultimo) ,color='orange')+
  ylab('')+xlab('')+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

p2 <- ggplot(base %>% filter(data>= input$periodo[1] & data <= input$periodo[2]))+
  geom_boxplot(aes(x =ultimo,y = 1),fill='blue',alpha=.25)+
  geom_vline(xintercept = novo_dado %>% pull(ultimo) ,color='orange')+
  ylab('')+xlab('')+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

p1 / p2
})
```

### Comportamento Histórico
```{r}
renderPlotly({
  ggplot(data = base %>% filter(data>= input$periodo[1] & data <= input$periodo[2]))+
    geom_segment(aes(x = data,
                     xend=data,
                     y =abertura,
                     yend =ultimo,
                     colour=greenRed),
                 size=3)+
    geom_segment(aes(x = data,
                     xend=data,
                     y =maxima,
                     yend =minima,
                     colour=greenRed))+
    scale_color_manual(values=c("Forest Green","Red"))+
    theme(legend.position ="none",
          axis.title.y = element_blank(),
          axis.title.x=element_blank(),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
          plot.title= element_text(hjust=0.5))
  })
```

### Previsão com Regressão Linear
```{r}
renderPlotly({
  ggplot(data = base %>% filter(data>= input$periodo[1] & data <= input$periodo[2]))+
  geom_smooth(aes(x =data, y = ultimo))+
  ylab('')+xlab('')
})
```

