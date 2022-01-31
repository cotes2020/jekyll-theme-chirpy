---
title: "Buy Smartphone with Data Science"
author: "Ramon Roldan"
date: "2022-01-29"
categories: [Data Science, Tydeverse]
tags: [Text Mining, Web Scraping, Tydeverse, Data Science, ETL, Strategic Sourcing, Procurement, Supply Chain]
output:
  md_document:
    variant: gfm
    preserve_yaml: TRUE
knit: (function(inputFile, encoding) {
  rmarkdown::render(inputFile, encoding = encoding, output_dir = "/home/ramon_de_lara/Ramon/roldanramon.github.io/_posts") })
---

This post has the main objective to sharing my experience with a
business case, where i pretend tell how i buyed a new smartphone using
the data science and strategic sourcing techniques.

In jan/2021 i needed buy a new smartphone, then I thinking with me: Why
not using data science to develop a machine learning model to help me?

<u>This project is divide in 5 parts:</u>

1.  Getting the data by web scraping;
2.  Cleaning the data;
3.  Start an exploratory analysis in the data;
4.  Setting the “weights” and create a model to help filter some
    devices;
5.  Conclusion: choosing “the best” device.

### 1. Getting the data by web scraping

Reading the main URL to obtain data about html page and then start web
scraping process:

``` r
aparelhos<- tibble::tibble(url = unique(paste('https://www.tudocelular.com', rvest::read_html('https://www.tudocelular.com/celulares/fichas-tecnicas.html') %>% rvest::html_elements(xpath = '//*[@id="cellphones_list"]/article') %>% rvest::html_nodes('a') %>% rvest::html_attr('href'),sep = ''))) %>%
  dplyr::filter(stringr::str_detect(string = url, pattern = 'ficha')) %>% dplyr::mutate(url = as.character(url))
slice_head(aparelhos,n = 5)
```

    ## # A tibble: 5 × 1
    ##   url                                                                           
    ##   <chr>                                                                         
    ## 1 https://www.tudocelular.com/Gionee/fichas-tecnicas/n7583/Gionee-G13-Pro.html  
    ## 2 https://www.tudocelular.com/vivo/fichas-tecnicas/n7581/vivo-Y75-5G.html       
    ## 3 https://www.tudocelular.com/TECNO/fichas-tecnicas/n7580/TECNO-Pop-5X.html     
    ## 4 https://www.tudocelular.com/Redmi/fichas-tecnicas/n7579/Redmi-Note-11S.html   
    ## 5 https://www.tudocelular.com/Redmi/fichas-tecnicas/n7578/Redmi-Note-11-Pro.html

Developing of function to do web scraping and then obtain the data about
each device:

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

Apply function to do download the data and presentation on the beautiful
table:

``` r
htop<- purrr::map_dfr(.x = aparelhos$url, .f = fun_to_get_info) %>% 
  tidyr::pivot_wider(names_from = 'nomes', values_from = atributos, values_fn = list) %>% janitor::clean_names() %>% as_tibble()

knitr::kable(x = slice_head(htop,n = 5),
             caption = "The top Devices")
```

| nome\_do\_aparelho | sistema\_operacional         | disponibilidade | dimensoes               | peso       | hardware | tela     | camera   | desempenho | sim\_card | dual\_sim     | gsm                           | hspa | lte | velocidade\_maxima\_de\_download | velocidade\_maxima\_de\_upload | processador                                    | chipset                       | x64\_bit | gpu            | ram  | memoria\_max | memoria\_expansivel                            | polegadas | resolucao                              | densidade\_de\_pixels | tipo           | fps    | cores      | megapixel                   | estabilizacao | autofoco | foco\_por\_toque | flash    | localizacao | deteccao\_facial | camera\_frontal | resolucao\_da\_gravacao | auto\_focagem\_de\_video | fps\_da\_gravacao | opcoes\_da\_camera\_frontal | wi\_fi            | bluetooth       | usb           | nfc   | gps                          | acelerometro | proximidade | giroscopio | bussola | tv    | vibracao | viva\_voz | ampere   | x5g  | aperture\_size                | hdr  | dual\_shot | estabilizacao\_de\_video | dual\_rec | video\_camera\_frontal | impressao\_digital | radio\_fm | outros                           | protecao        | tamanho\_do\_sensor | angulo\_maximo | irda | slow\_motion | custo\_beneficio | blink\_detection | autonomia\_conversacao | autonomia\_em\_standby | resistencia\_a\_agua | barometro | zoom\_otico | video\_hdr | stereo\_sound\_rec | foto\_em\_video | segundo\_display | mic\_de\_reducao\_de\_ruido | sar\_eu | melhor\_preco | preco\_amazon | faixa\_de\_preco | deteccao\_de\_sorriso | preco\_submarino | gesto |
|:-------------------|:-----------------------------|:----------------|:------------------------|:-----------|:---------|:---------|:---------|:-----------|:----------|:--------------|:------------------------------|:-----|:----|:---------------------------------|:-------------------------------|:-----------------------------------------------|:------------------------------|:---------|:---------------|:-----|:-------------|:-----------------------------------------------|:----------|:---------------------------------------|:----------------------|:---------------|:-------|:-----------|:----------------------------|:--------------|:---------|:-----------------|:---------|:------------|:-----------------|:----------------|:------------------------|:-------------------------|:------------------|:----------------------------|:------------------|:----------------|:--------------|:------|:-----------------------------|:-------------|:------------|:-----------|:--------|:------|:---------|:----------|:---------|:-----|:------------------------------|:-----|:-----------|:-------------------------|:----------|:-----------------------|:-------------------|:----------|:---------------------------------|:----------------|:--------------------|:---------------|:-----|:-------------|:-----------------|:-----------------|:-----------------------|:-----------------------|:---------------------|:----------|:------------|:-----------|:-------------------|:----------------|:-----------------|:----------------------------|:--------|:--------------|:--------------|:-----------------|:----------------------|:-----------------|:------|
| Gionee G13 Pro     | HarmonyOS                    | 2022/1          | 158 x 76 x 9.2 mm       | 195 gramas | 0 / 10   | 0 / 10   | 0 / 10   | 0 / 10     | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 300 Mbps                         | 100 Mbps                       | 1x 2.0 GHz Cortex-A75 + 3x 1.8 GHz Cortex-A55  | T310 Unisoc                   | ok       | PowerVR GE8300 | 4 GB | 32 GB        | wrong                                          | 6.26      | 720 x 1600 pixel , 4163 x 3122 pixel   | 280 ppi               | IPS LCD, Litio | 60 Hz  | 16 milhões | 13 Mp + 2 Mp                | Digital       | ok       | ok               | LED      | ok          | ok               | 5 Mp            | Full HD                 | ok                       | 30 fps            | Face Detection              | ok                | ok              | Type-C 2.0    | wrong | A-GPS                        | ok           | ok          | wrong      | wrong   | wrong | ok       | ok        | 3500 mAh | NULL | NULL                          | NULL | NULL       | NULL                     | NULL      | NULL                   | NULL               | NULL      | NULL                             | NULL            | NULL                | NULL           | NULL | NULL         | NULL             | NULL             | NULL                   | NULL                   | NULL                 | NULL      | NULL        | NULL       | NULL               | NULL            | NULL             | NULL                        | NULL    | NULL          | NULL          | NULL             | NULL                  | NULL             | NULL  |
| vivo Y75 5G        | Android 11 Funtouch 12       | 2022/1          | 164 x 75.84 x 8.25 mm   | 188 gramas | 0 / 10   | 0 / 10   | 0 / 10   | 0 / 10     | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 2770 Mbps                        | \-                             | 2x 2.2 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55  | Dimensity 700 MediaTek MT6833 | ok       | Mali-G57 MC2   | 8 GB | 128 GB       | Slot híbrido SIM/MicroSD MicroSDXC atè 1024 GB | 6.58      | 1080 x 2408 pixel, 8165 x 6124 pixel   | 401 ppi               | IPS LCD, LiPo  | 60 Hz  | 16 milhões | 50 Mp + 2 Mp + 2 Mp         | Digital       | ok       | ok               | Dual LED | ok          | ok               | 16 Mp F 2       | Full HD                 | ok                       | 30 fps            | HDR/Face Detection          | 802.11 a/b/g/n/ac | 5.1 com A2DP/LE | Type-C        | wrong | A-GPS/GLONASS/BeiDou/Galileo | ok           | ok          | ok         | ok      | wrong | ok       | ok        | 5000 mAh | ok   | F 1.8 + F 2.4 + F 2.4         | ok   | ok         | ok                       | ok        | Full HD, 30fps         | ok                 | ok        | Wi-Fi DirectWi-Fi hotspotUSB OTG | NULL            | NULL                | NULL           | NULL | NULL         | NULL             | NULL             | NULL                   | NULL                   | NULL                 | NULL      | NULL        | NULL       | NULL               | NULL            | NULL             | NULL                        | NULL    | NULL          | NULL          | NULL             | NULL                  | NULL             | NULL  |
| TECNO Pop 5X       | Android 10 (Go Edition) HiOS | 2022/1          | 166 x 75.9 x 8.5 mm     | 150 gramas | 0 / 10   | 0 / 10   | 0 / 10   | 0 / 10     | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 150 Mbps                         | 50 Mbps                        | 1.4 GHz Quad Core                              | Unisoc SC9832E                | wrong    | Mali-T820 MP1  | 2 GB | 32 GB        | MicroSDXC                                      | 6.52      | 720 x 1600 pixel , 3266 x 2449 pixel   | 269 ppi               | IPS LCD, LiPo  | 60 Hz  | 16 milhões | 8 Mp + 0.07 Mp + 0.07 Mp    | Digital       | ok       | ok               | LED      | ok          | ok               | 5 Mp F 2        | Full HD                 | ok                       | 30 fps            | Face Detection              | ok                | ok              | Micro USB 2.0 | wrong | A-GPS                        | ok           | ok          | wrong      | wrong   | wrong | ok       | ok        | 4000 mAh | NULL | NULL                          | ok   | NULL       | NULL                     | NULL      | NULL                   | ok                 | ok        | Wi-Fi hotspot                    | NULL            | NULL                | NULL           | NULL | NULL         | NULL             | NULL             | NULL                   | NULL                   | NULL                 | NULL      | NULL        | NULL       | NULL               | NULL            | NULL             | NULL                        | NULL    | NULL          | NULL          | NULL             | NULL                  | NULL             | NULL  |
| Redmi Note 11S     | Android 11 MIUI 13           | 2022/1          | 159.9 x 73.9 x 8.1 mm   | 179 gramas | 7.9 / 10 | 8.5 / 10 | 8.5 / 10 | 4.8 / 10   | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 390 Mbps                         | 150 Mbps                       | 2x 2.05 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55 | Helio G96 MediaTek            | ok       | Mali-G57 MC2   | 6 GB | 128 GB       | MicroSDXC                                      | 6.43      | 1080 x 2400 pixel , 12000 x 9000 pixel | 409 ppi               | AMOLED, LiPo   | 90 Hz  | 16 milhões | 108 Mp + 8 Mp + 2 Mp + 2 Mp | Digital       | ok       | ok               | LED      | ok          | ok               | 16 Mp F 2.4     | Full HD                 | ok                       | 30 fps            | Face Detection              | 802.11 a/b/g/n/ac | 5.0 com A2DP/LE | Type-C 2.0    | ok    | A-GPS/GLONASS/BeiDou/Galileo | ok           | ok          | ok         | ok      | wrong | ok       | ok        | 5000 mAh | NULL | F 1.9 + F 2.2 + F 2.4 + F 2.4 | ok   | NULL       | NULL                     | NULL      | Full HD, 30fps         | ok                 | NULL      | Wi-Fi DirectWi-Fi hotspotUSB OTG | Gorilla Glass 3 | 1/1.52 "            | 118 °          | ok   | NULL         | NULL             | NULL             | NULL                   | NULL                   | NULL                 | NULL      | NULL        | NULL       | NULL               | NULL            | NULL             | NULL                        | NULL    | NULL          | NULL          | NULL             | NULL                  | NULL             | NULL  |
| Redmi Note 11 Pro  | Android 11 MIUI 13           | 2022/1          | 164.19 x 76.1 x 8.12 mm | 202 gramas | 7.9 / 10 | 8.4 / 10 | 8.5 / 10 | 4.8 / 10   | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 390 Mbps                         | 150 Mbps                       | 2x 2.05 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55 | Helio G96 MediaTek            | ok       | Mali-G57 MC2   | 6 GB | 128 GB       | Slot híbrido SIM/MicroSD MicroSDXC atè 1024 GB | 6.67      | 1080 x 2400 pixel , 12000 x 9000 pixel | 395 ppi               | AMOLED, LiPo   | 120 Hz | 16 milhões | 108 Mp + 8 Mp + 2 Mp + 2 Mp | Digital       | ok       | ok               | LED      | ok          | ok               | 16 Mp F 2.4     | Full HD                 | ok                       | 30 fps            | Face Detection              | 802.11 b/g/n/ac   | 5.1 com A2DP/LE | Type-C 2.0    | ok    | A-GPS/GLONASS/BeiDou/Galileo | ok           | ok          | ok         | ok      | wrong | ok       | ok        | 5000 mAh | NULL | F 1.9 + F 2.2 + F 2.4 + F 2.4 | ok   | NULL       | NULL                     | NULL      | Full HD, 30fps         | ok                 | NULL      | Wi-Fi DirectWi-Fi hotspotUSB OTG | Gorilla Glass 5 | 1/1.52 "            | 118 °          | ok   | 120 fps      | NULL             | NULL             | NULL                   | NULL                   | NULL                 | NULL      | NULL        | NULL       | NULL               | NULL            | NULL             | NULL                        | NULL    | NULL          | NULL          | NULL             | NULL                  | NULL             | NULL  |

The top Devices

### 2. Cleaning the data

### 3. Start an exploratory analysis in the data

### 4. Setting the “weights” and create a model to help filter some devices

### 5. Conclusion: choosing “the best” device
