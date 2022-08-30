---
title: "Buy Smartphone with Data Science"
author: "Ramon Roldan"
date: "2022-01-28"
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
    ## 1 https://www.tudocelular.com/vivo/fichas-tecnicas/n8042/vivo-V25e.html         
    ## 2 https://www.tudocelular.com/vivo/fichas-tecnicas/n8106/vivo-Y16.html          
    ## 3 https://www.tudocelular.com/vivo/fichas-tecnicas/n8100/vivo-iQOO-Z6x.html     
    ## 4 https://www.tudocelular.com/vivo/fichas-tecnicas/n8091/vivo-iQOO-Z6-China.html
    ## 5 https://www.tudocelular.com/Redmi/fichas-tecnicas/n8097/Redmi-Note-11-SE-ndia…

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

| nome\_do\_aparelho       | sistema\_operacional      | disponibilidade | dimensoes                | peso         | hardware | tela     | camera   | desempenho | sim\_card | dual\_sim     | gsm                           | hspa | lte | velocidade\_maxima\_de\_download | velocidade\_maxima\_de\_upload | processador                                                           | chipset                                 | x64\_bit | gpu            | ram  | memoria\_max | memoria\_expansivel                            | polegadas | resolucao                            | densidade\_de\_pixels | tipo               | fps    | cores      | megapixel                  | aperture\_size                | estabilizacao | autofoco | foco\_por\_toque | flash    | hdr  | dual\_shot | localizacao | deteccao\_facial | camera\_frontal | resolucao\_da\_gravacao | auto\_focagem\_de\_video | slow\_motion | dual\_rec | opcoes\_da\_camera\_frontal | wi\_fi              | bluetooth               | usb        | nfc   | gps                               | acelerometro | proximidade | giroscopio | bussola | impressao\_digital | tv    | vibracao | viva\_voz | outros                           | ampere   | radio\_fm | x5g  | fps\_da\_gravacao | video\_camera\_frontal | autonomia\_conversacao | autonomia\_em\_standby | estabilizacao\_de\_video | protecao        | tamanho\_do\_sensor | angulo\_maximo | foto\_em\_video | irda | stereo\_sound\_rec | barometro | mic\_de\_reducao\_de\_ruido | sar\_eu | melhor\_preco | preco\_extra | faixa\_de\_preco | custo\_beneficio | video\_hdr | deteccao\_de\_sorriso | segundo\_display | zoom\_otico | foto\_3d | gesto | resistencia\_a\_agua |
|:-------------------------|:--------------------------|:----------------|:-------------------------|:-------------|:---------|:---------|:---------|:-----------|:----------|:--------------|:------------------------------|:-----|:----|:---------------------------------|:-------------------------------|:----------------------------------------------------------------------|:----------------------------------------|:---------|:---------------|:-----|:-------------|:-----------------------------------------------|:----------|:-------------------------------------|:----------------------|:-------------------|:-------|:-----------|:---------------------------|:------------------------------|:--------------|:---------|:-----------------|:---------|:-----|:-----------|:------------|:-----------------|:----------------|:------------------------|:-------------------------|:-------------|:----------|:----------------------------|:--------------------|:------------------------|:-----------|:------|:----------------------------------|:-------------|:------------|:-----------|:--------|:-------------------|:------|:---------|:----------|:---------------------------------|:---------|:----------|:-----|:------------------|:-----------------------|:-----------------------|:-----------------------|:-------------------------|:----------------|:--------------------|:---------------|:----------------|:-----|:-------------------|:----------|:----------------------------|:--------|:--------------|:-------------|:-----------------|:-----------------|:-----------|:----------------------|:-----------------|:------------|:---------|:------|:---------------------|
| vivo V25e                | Android 12 Funtouch 12    | 2022/3          | 159.2 x 74.2 x 7.79 mm   | 183 gramas   | 6.6 / 10 | 8.5 / 10 | 7.7 / 10 | 5 / 10     | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 390 Mbps                         | 150 Mbps                       | 2x 2.2 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55                         | Helio G99 MediaTek                      | ok       | Mali-G57 MC2   | 8 GB | 256 GB       | Slot híbrido SIM/MicroSD MicroSDXC atè 1024 GB | 6.44      | 1080 x 2404 pixel, 9238 x 6928 pixel | 409 ppi               | AMOLED, LiPo       | 60 Hz  | 16 milhões | 64 Mp + 2 Mp + 2 Mp        | F 1.79 + F 2.4 + F 2.4        | Ótica         | ok       | ok               | LED      | ok   | ok         | ok          | ok               | 32 Mp F 2       | wrong                   | ok                       | ok           | ok        | Face Detection              | 802.11 a/b/g/n/ac   | 5.2 com A2DP/LE         | Type-C 2.0 | wrong | A-GPS/GLONASS/BeiDou/Galileo/QZSS | ok           | ok          | ok         | ok      | ok                 | wrong | ok       | ok        | Wi-Fi DirectWi-Fi hotspotUSB OTG | 4500 mAh | NULL      | NULL | NULL              | NULL                   | NULL                   | NULL                   | NULL                     | NULL            | NULL                | NULL           | NULL            | NULL | NULL               | NULL      | NULL                        | NULL    | NULL          | NULL         | NULL             | NULL             | NULL       | NULL                  | NULL             | NULL        | NULL     | NULL  | NULL                 |
| vivo Y16                 | Android 12 Funtouch 12    | 2022/3          | 163.95 x 75.55 x 8.19 mm | 183 gramas   | 5.3 / 10 | 5.3 / 10 | 5.7 / 10 | 4.7 / 10   | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 300 Mbps                         | 150 Mbps                       | 4x 2.3 GHz Cortex-A53 + 4x 1.8 GHz Cortex-A53                         | Helio P35 MediaTek MT6765               | ok       | PowerVR GE8320 | 3 GB | 32 GB        | MicroSDXC                                      | 6.51      | 720 x 1600 pixel , 4163 x 3122 pixel | 270 ppi               | IPS LCD, LiPo      | 60 Hz  | 16 milhões | 13 Mp + 2 Mp               | F 2.2 + F 2.4                 | Digital       | ok       | ok               | LED      | NULL | NULL       | ok          | ok               | 5 Mp F 2.2      | wrong                   | ok                       | NULL         | NULL      | NULL                        | 802.11 a/b/g/n/ac   | 5.0 com A2DP/LE         | Type-C 2.0 | wrong | A-GPS/GLONASS/BeiDou/Galileo      | ok           | ok          | ok         | ok      | ok                 | wrong | ok       | ok        | Wi-Fi hotspotUSB OTG             | 5000 mAh | ok        | NULL | NULL              | NULL                   | NULL                   | NULL                   | NULL                     | NULL            | NULL                | NULL           | NULL            | NULL | NULL               | NULL      | NULL                        | NULL    | NULL          | NULL         | NULL             | NULL             | NULL       | NULL                  | NULL             | NULL        | NULL     | NULL  | NULL                 |
| vivo iQOO Z6x            | Android 11 OriginOS Ocean | 2022/3          | 163.87 x 75.33 x 9.27 mm | 204 gramas   | 7.8 / 10 | 8.4 / 10 | 7.6 / 10 | 5 / 10     | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 2770 Mbps                        | \-                             | 2x 2.4 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55                         | Dimensity 810 MediaTek                  | ok       | Mali-G57 MC2   | 6 GB | 128 GB       | wrong                                          | 6.58      | 1080 x 2408 pixel, 8165 x 6124 pixel | 401 ppi               | IPS LCD, LiPo      | 60 Hz  | 16 milhões | 50 Mp + 2 Mp               | F 1.8 + F 2.4                 | Digital       | ok       | ok               | Dual LED | ok   | ok         | ok          | ok               | 8 Mp F 2        | Full HD                 | ok                       | NULL         | ok        | Face Detection              | 802.11 a/b/g/n/ac   | 5.1 com A2DP/LE/aptX HD | Type-C 2.0 | wrong | A-GPS/GLONASS/BeiDou/Galileo/QZSS | ok           | ok          | wrong      | ok      | ok                 | wrong | ok       | ok        | Wi-Fi DirectWi-Fi hotspotUSB OTG | 6000 mAh | wrong     | ok   | 30 fps            | Full HD, 30fps         | 1080 minutos           | 849 horas              | NULL                     | NULL            | NULL                | NULL           | NULL            | NULL | NULL               | NULL      | NULL                        | NULL    | NULL          | NULL         | NULL             | NULL             | NULL       | NULL                  | NULL             | NULL        | NULL     | NULL  | NULL                 |
| vivo iQOO Z6 (China)     | Android 12 OriginOS Ocean | 2022/3          | 164.17 x 75.8 x 8.59 mm  | 194.6 gramas | 7.3 / 10 | 8.4 / 10 | 8.3 / 10 | 6.9 / 10   | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 3700 Mbps                        | 1600 Mbps                      | 1x 2.5 GHz Cortex-A78 + 3x 2.4 GHz Cortex-A78 + 4x 1.8 GHz Cortex-A55 | Snapdragon 778G Plus Qualcomm SM7325-AE | ok       | Adreno 642L    | 8 GB | 256 GB       | wrong                                          | 6.64      | 1080 x 2388 pixel, 9238 x 6928 pixel | 395 ppi               | IPS LCD, LiPo      | 120 Hz | 16 milhões | 64 Mp + 2 Mp + 2 Mp        | F 1.79 + F 2.4 + F 2.4        | Ótica         | ok       | ok               | Dual LED | ok   | ok         | ok          | ok               | 8 Mp F 2        | 4K (2160p)              | ok                       | ok           | ok        | Face Detection              | 802.11 a/b/g/n/ac/6 | 5.2 com A2DP/LE/aptX HD | Type-C 2.0 | ok    | A-GPS/GLONASS/BeiDou/Galileo/QZSS | ok           | ok          | ok         | ok      | ok                 | wrong | ok       | ok        | Wi-Fi DirectWi-Fi hotspotUSB OTG | 4500 mAh | wrong     | ok   | 30 fps            | Full HD, 30fps         | 780 minutos            | 429 horas              | ok                       | NULL            | NULL                | NULL           | NULL            | NULL | NULL               | NULL      | NULL                        | NULL    | NULL          | NULL         | NULL             | NULL             | NULL       | NULL                  | NULL             | NULL        | NULL     | NULL  | NULL                 |
| Redmi Note 11 SE (Índia) | Android 11 MIUI 12.5      | 2022/3          | 160.46 x 74.5 x 8.29 mm  | 178.8 gramas | 7.8 / 10 | 8.5 / 10 | 8.1 / 10 | 4.8 / 10   | Nano      | Dual stand-by | Quad Band (850/900/1800/1900) | ok   | ok  | 600 Mbps                         | 150 Mbps                       | 2x 2.05 GHz Cortex-A76 + 6x 2.0 GHz Cortex-A55                        | Helio G95 MediaTek                      | ok       | Mali-G76 MC4   | 6 GB | 128 GB       | MicroSDXC                                      | 6.43      | 1080 x 2400 pixel, 9238 x 6928 pixel | 409 ppi               | Super AMOLED, LiPo | 60 Hz  | 16 milhões | 64 Mp + 8 Mp + 2 Mp + 2 Mp | F 1.9 + F 2.2 + F 2.4 + F 2.4 | Digital       | ok       | ok               | LED      | ok   | NULL       | ok          | ok               | 13 Mp F 2.45    | 4K (2160p)              | ok                       | ok           | NULL      | Face Detection              | 802.11 a/b/g/n/ac   | 5.0 com A2DP/LE         | Type-C 2.0 | ok    | A-GPS/GLONASS/BeiDou/Galileo      | ok           | ok          | ok         | ok      | ok                 | wrong | ok       | ok        | Wi-Fi DirectWi-Fi hotspot        | 5000 mAh | ok        | NULL | 30 fps            | Full HD, 30fps         | NULL                   | NULL                   | NULL                     | Gorilla Glass 3 | 1/1.97 " + 1/4.0 "  | 118 °          | ok              | ok   | NULL               | NULL      | NULL                        | NULL    | NULL          | NULL         | NULL             | NULL             | NULL       | NULL                  | NULL             | NULL        | NULL     | NULL  | NULL                 |

The top Devices

### 2. Cleaning the data

``` r
teste<-htop %>% keep( ~!is.null(.))

teste<-htop %>% filter(complete.cases(across(.cols = nome_do_aparelho:estabilizacao_de_video,.fns = ~. == 'NULL')))
```

### 3. Start an exploratory analysis in the data

### 4. Setting the “weights” and create a model to help filter some devices

### 5. Conclusion: choosing “the best” device
