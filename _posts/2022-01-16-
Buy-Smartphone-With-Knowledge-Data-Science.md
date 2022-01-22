---
title: "Buy Smartphone With Knowledge Data Science"
author: "Ramon Roldan"
date: "2022-01-16"
categories: [Text Mining, Web Scrapping, Tydeverse, Data Science]
tags: [Text Mining, Web Scrapping, Tydeverse, Data Science]
---
This post has the main objetive to delivery a ideia about how buy the smartphone using the data science techniques.

In the begining i 


Load raw URL smartphone’s in a tibble

``` r
aparelhos<- tibble::tibble(url = unique(paste('https://www.tudocelular.com', rvest::read_html('https://www.tudocelular.com/celulares/fichas-tecnicas.html') %>% rvest::html_elements(xpath = '//*[@id="cellphones_list"]/article') %>% rvest::html_nodes('a') %>% rvest::html_attr('href'),sep = ''))) %>%
  dplyr::filter(stringr::str_detect(string = url, pattern = 'ficha')) %>% dplyr::mutate(url = as.character(url))
```

Developing of function to obtain the raw data of each smartphone

``` r
fun_to_get_info <- function(phone) {
  
  url_link<- rvest::read_html(as.character(phone))
  
  telefone <-tibble::tibble(
    nomes = url_link %>% rvest::html_elements(xpath = '//*[@id="controles_titles"]') %>% rvest::html_nodes('li') %>% rvest::html_text() %>% readr::parse_character(),
    atributos = ifelse(
      !is.na(url_link %>% rvest::html_elements(xpath = '//*[@id="phone_columns"]') %>% rvest::html_nodes('li') %>% rvest::html_text() %>% readr::parse_character()),
      url_link %>% rvest::html_elements(xpath = '//*[@id="phone_columns"]') %>% rvest::html_nodes('li') %>% rvest::html_text() %>% readr::parse_character(),
      url_link %>% rvest::html_elements(xpath = '//*[@id="phone_columns"]') %>% rvest::html_nodes('li') %>% rvest::html_node('i') %>% rvest::html_attr('class') %>% readr::parse_character()
    )
  ) %>% 
    dplyr::mutate('Nome do Aparelho' = url_link %>% rvest::html_elements(xpath = '//*[@id="fwide_column"]/h2') %>% rvest::html_text())
  return(telefone)
}
```

Presentation by top ranking

``` r
htop<- purrr::map_dfr(.x = aparelhos$url, .f = fun_to_get_info) %>% 
  tidyr::pivot_wider(names_from = 'Nome do Aparelho', values_from = atributos)
```

    ## Warning: Values are not uniquely identified; output will contain list-cols.
    ## * Use `values_fn = list` to suppress this warning.
    ## * Use `values_fn = length` to identify where the duplicates arise
    ## * Use `values_fn = {summary_fun}` to summarise duplicates

Resultados
    ## # A tibble: 88 × 52
    ##    nomes         `vivo Y10 T1` `vivo Y10` `Motorola Moto Tab … `Blackview BV880…
    ##    <chr>         <list>        <list>     <list>               <list>           
    ##  1 Sistema Oper… <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  2 Disponibilid… <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  3 Dimensões     <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  4 Peso          <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  5 Hardware      <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  6 - Tela        <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  7 - Câmera      <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  8 - Desempenho  <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ##  9 Sim Card      <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ## 10 Dual Sim      <chr [1]>     <chr [1]>  <chr [1]>            <chr [1]>        
    ## # … with 78 more rows, and 47 more variables: OnePlus 10 Pro <list>,
    ## #   vivo Y33T <list>, Oppo A36 <list>, Realme 9i <list>, Honor Magic V <list>,
    ## #   Xiaomi 11i HyperCharge <list>, Xiaomi 11i 5G <list>,
    ## #   vivo IQOO 9 Pro <list>, vivo IQOO 9 <list>, vivo V23 Pro <list>,
    ## #   vivo V23 5G <list>, Oppo A96 5G <list>, Realme GT 2 Pro <list>,
    ## #   Realme GT 2 <list>, Samsung Galaxy S21 FE 5G <list>, vivo Y21T <list>,
    ## #   Xiaomi 12 Pro <list>, Xiaomi 12 <list>, Xiaomi 12X <list>, …
