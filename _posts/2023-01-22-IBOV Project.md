---
title: "IBOV Project"
author: "Ramon Roldan"
date: "2022-06-30"
categories: [Data Science, Tydeverse]
tags: [Machine Learning, Tydeverse, Times Series, IBOV, Data Science, ETL]
---

## Organização do trabalho

* Definição do objetivo do trabalho;
* Versionar trabalho no GitHub;
* Utilizar a ferramenta Kanban para organizar projeto no formato de metodologia ágil;
* Coleta dos dados;
* Start an exploratory analysis;
* Limpeza e tratamento dos dados;
* Salvar modelo treinado em arquivo r para posteriormente aplicar no framework tidymodels;
* Aplicar framework do tidymodels para desenvolver modelo;
* Avaliar resultados do modelo e necessidade de tunar os hiperparametros;
* Desenvolver um dashboard dinâmico com os pacotes flexdashboard, shiny e ploty;
* Realizar deploy do modelo and uploud in the shinyapp.io

## Definição do objetivo do trabalho

O trabalho busca desenvolver um dashboard que rode um modelo de machine learning capaz de prever o movimento do dia seguinte no indice ibovespa, baseado em dados históricos disponibilizados via API no yahoo finace, e para isso serão aplicadas técnicas estatisticas de regressão logística e regressão linear em series temporáis.

Para organização do projeto foi utilizada a ferramenta KANBAN, visando framework de metodoligas ágeis, e versionamento de código no GitHub.

O resultado final esta disponível no dashboard que pode ser acessado em:
https://sfefqj-ramon-roldan.shinyapps.io/Financial_Market_Analysis/

## Versionar trabalho no GitHub

![Repositório criado para ajudar no versionamento do projeto](https://github.com/RoldanRamon/roldanramon.github.io/blob/main/assets/img/github_versionamento.png)


## Utilizar a ferramenta Kanban para organizar projeto no formato de metodologia ágil
![Kanban no Github para acompanhar evolução](roldanramon.github.io/assets/img/IBOV_PROJECT.png)


## Coleta dos dados

Para construir uma primeira versão do modelo foram usados os dados históricos do índice Ibovespa disponíbilizados pelo jornal investing.

![Investing PrintScreeng](/home/ramon_de_lara/Ramon/roldanramon.github.io/assets/img/ibov investing.png)


```r
rm(list = ls())
library(tidymodels)
library(readr)
library(janitor)
library(stringr)
library(lubridate)
library(ggplot2)
library(plotly)
library(DataExplorer)
```

```
## Error in library(DataExplorer): there is no package called 'DataExplorer'
```

```r
#library(quantmod)

#carregando Base extraida do site investing
base <- read_csv('Futuros Ibovespa - Dados Históricos.csv') %>% clean_names() %>% 
  mutate(data = lubridate::dmy(data),
         meta = if_else(var_percent > 0,1,0) %>% as.factor()) %>% 
  #select(-c(var_percent,vol)) %>%
  arrange(data)
```

```
## Rows: 361 Columns: 7
## ── Column specification ────────────────────
## Delimiter: ","
## chr (3): Data, Vol., Var%
## dbl (4): Último, Abertura, Máxima, Mínima
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

```r
#base do yahoo finance
#base_yahoo <- quantmod::getSymbols(Symbols = '^BVSP') %>% clean_names()

#Avaliando tipos de dados e verificando dados faltantes
DataExplorer::plot_intro(base)
```

```
## Error in loadNamespace(x): there is no package called 'DataExplorer'
```

```r
#Analisando gráfico da seríe temporal
ggplotly(
ggplot(base,aes(x = data,y = ultimo))+
  geom_line()+
  geom_point(color='blue',size=1)+
  ggtitle('Gráfico Cotação diária')+
  scale_x_date(date_breaks = "1 month", date_labels = "%b %d")
)
```

```
## PhantomJS not found. You can install it with webshot::install_phantomjs(). If it is installed, please make sure the phantomjs executable can be found via the PATH variable.
```

```
## Error in path.expand(path): invalid 'path' argument
```

```r
#Dividindo entre treino e teste
split_base <- initial_split(base,prop = .8)
train_base <- training(split_base)
test_base <- testing(split_base)


# Criando modelo ----------------------------------------------------------
lr_model <- logistic_reg() %>% 
  set_mode('classification') %>% 
  set_engine('glm')


# Criando Recipe ----------------------------------------------------------
lr_recipe <- recipe(meta ~ .,data = train_base) %>% 
  step_rm(var_percent,vol) %>% prep()


# Criando Workflow --------------------------------------------------------
wkf_model <- workflow() %>% 
  add_model(lr_model) %>% 
  add_recipe(lr_recipe)


# Treinando Modelo --------------------------------------------------------
lr_result <- last_fit(wkf_model,split = split_base)


# Avaliando Resultado -----------------------------------------------------
#Accuracy and roc_auc
lr_result %>% collect_metrics()
```

```
## # A tibble: 2 × 4
##   .metric  .estimator .estimate .config     
##   <chr>    <chr>          <dbl> <chr>       
## 1 accuracy binary         0.890 Preprocesso…
## 2 roc_auc  binary         0.957 Preprocesso…
```

```r
#Matriz de Confusão
lr_result %>% unnest(.predictions) %>% conf_mat(truth = meta, estimate = .pred_class) %>% autoplot(type='heatmap')
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1-1.png)

```r
# Salvando modelo Final ---------------------------------------------------
final_lr_result <- fit(wkf_model,base)
saveRDS(object = final_lr_result,file = 'win_model.rds')
```



### Carregar Bibliotecas necessárias

```r
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



### Carrega Modelo Salvo

```r
Filtros {.sidebar}
--------------------------------------------------

#Carrega Modelo
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

#Botão para Download da série
downloadHandler(
    filename = function() { 
      paste("dataset-", Sys.Date(), ".csv", sep="")
    },
    content = function(file) {
      readr::write_csv(base %>% select(-c(var_percent,meta,greenRed)) %>% 
                         filter(data>= input$periodo[1] & data <= input$periodo[2]), file)
    })

#Filtra último dia do banco de dados
novo_dado <- base %>% filter(data == max(data))

#Aplica o modelo para obter a probabilidade de movimento do dia seguinte
resultado_valor <- round(if_else(predict(object = modelo, new_data = novo_dado)==0,
        predict(object = modelo, new_data = novo_dado,type = 'prob') %>% pull(1),
        predict(object = modelo, new_data = novo_dado,type = 'prob') %>% pull(2))*100, 2)

resultado_label=if_else(predict(object = modelo, new_data = novo_dado)==0,'Baixa','Alta')
```

```
## Error: <text>:1:9: unexpected '{'
## 1: Filtros {
##             ^
```

Row {data-height=300}
-----------------------------------------------------------------------

### Probabilidade e direção do Movimento


```r
#Informa dentro do dashboard
renderGauge({
gauge(value = resultado_valor,label = resultado_label, min = 0, max = 100, symbol = '%', gaugeSectors(
  success = c(80, 100), warning = c(40, 79), danger = c(0, 39)))
})
```

<!--html_preserve--><div class="gauge html-widget html-widget-output shiny-report-size html-fill-item-overflow-hidden html-fill-item" id="outbef211af7c61cd7e" style="width:100%;height:200px;"></div><!--/html_preserve-->

### Concentração Dos Dados


```r
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

<!--html_preserve--><div class="shiny-plot-output html-fill-item" id="outde96ecc69439d376" style="width:100%;height:400px;"></div><!--/html_preserve-->

Row {data-height=700}
-----------------------------------------------------------------------

### Comportamento Histórico


```r
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

<!--html_preserve--><div class="plotly html-widget html-widget-output shiny-report-size shiny-report-theme html-fill-item-overflow-hidden html-fill-item" id="out5a713328c4b33554" style="width:100%;height:400px;"></div><!--/html_preserve-->

### Previsão com Regressão Linear


```r
renderPlotly({
  ggplot(data = base %>% filter(data>= input$periodo[1] & data <= input$periodo[2]))+
  geom_smooth(aes(x =data, y = ultimo))+
  ylab('')+xlab('')
})
```

<!--html_preserve--><div class="plotly html-widget html-widget-output shiny-report-size shiny-report-theme html-fill-item-overflow-hidden html-fill-item" id="outa9d1abb694ad0c89" style="width:100%;height:400px;"></div><!--/html_preserve-->

