---
title: "Buy Smartphone With Knowledge Data Science"
author: "Ramon Roldan"
date: "2022-01-16"
categories: [Text Mining, Web Scrapping, Tydeverse, Data Science]
tags: [Text Mining, Web Scrapping, Tydeverse, Data Science]
---
This post has the main objetive to delivery a ideia about how buy the smartphone using the data science techniques.

In jan/2021 i need buy a new smartphone, then I thinking with me: Why not develop a machine learning model to help me?


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

