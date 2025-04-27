---
title: Negev Rain EDA
date: 2025-04-17 10:00:00 +0300
categories: [Data Analysis, EDA]
tags: [jupyter-notebook, jupyter-slideshow, pandas, numpy, tableau, rast-api]
render_with_liquid: true
description: Negev Rain Annual EDA
image:
  path: "/assets/img/images/desert n.webp"
  lqip: data:image/webp;base64,UklGRpwAAABXRUJQVlA4IJAAAAAQBACdASoUAAsAPm0skkWkIqGYBABABsSzAE6ZQYwAJz6MRM3gaI9K6ADz3vhxvYQG4cBtBWinCjdfsluHofmBYWfCSSXjugaAku5HFeIGE//FofQ9e74Vun/VfrTPQKTR3M4hJUrIBkgUiUV5IcWLs76Uga5zTpz2P0a6JvtJV7oEPfJPukvVFrncjcKAAAA=
  alt: desert negev eda
---



# 🌧️ Explore Annual Rainfall Patterns in Southern Israel

[![IMS Logo](https://ims.gov.il/themes/imst/ims/images/logo.jpg)](https://ims.gov.il/en)<br>

## 🔍 Overview
This project explores the annual rainfall patterns across twenty-six meteorological stations in Southern Israel, analyzing data from over 1,800 measurements. The dataset was obtained from the [Israel Meteorological Service (IMS)](https://ims.gov.il/he/data_gov) and processed using Python along with various libraries for statistical analysis and data visualization.

## 📊 Data and Web API
The project leverages rainfall data which can be explored through the provided Jupyter Notebook. You can also access the data and findings via the [Web API](https://ims.gov.il/he/ObservationDataAPI), but you will need a valid token to access it.

## 💬 Discussion
Analyzing rainfall data proves challenging when relying on a single trendline due to substantial correlation within the dataset. The data does not fit neatly into a linear trend, and different subsets of years lead to varying slopes in trendlines. This is visually demonstrated, as selecting different ranges of years can show either a positive or negative slope.

This study challenges the assumption that rainfall patterns are solely influenced by the time period chosen for measurements. By calculating all possible trendlines, the project captures the respective slopes over different time periods, allowing for a more comprehensive analysis. The heatmaps presented in the project visually represent these fluctuating slopes, revealing that rainfall amounts in southern Israel have generally declined over time.

While examining a 70-year span, the slope of the trendline varies significantly, but by focusing on periods of at least 10 years, a consistent downward trend emerges, marked by a predominant red hue, particularly in the upper corner. This suggests a clear and ongoing decrease in rainfall over time.

## 🎞️ slideshow
{% include embed/webpage.html url="https://nisanman.github.io/NegevRainAnnual/#/1/1" height="700"%}

[slideshows link](https://nisanman.github.io/NegevRainAnnual/#/)
This is a data presentation project based on rainfall analysis using Python and Jupyter Notebook.

## 📈 [Tableau Visualization Link](https://public.tableau.com/shared/GN7J29MRK?:display_count=n&:origin=viz_share_link)
