---
title: "Portfolio Analysis: a beginning"
date: 2023-11-25 13:17:00 -0500
categories: [Analytics, R programming]
tags: [Finance]
render_with_liquid: false
---

This document is an exploration in R coding, applied to analyzing my financial portfolio. We will attempt to scrape ETF stock data from online sources, calculate indicators, and produce tables, graphs and charts.

## Librairies

The following libraries are used:

The **rvest** package in R is a popular and powerful tool designed for web scraping. It allows users to easily read and manipulate the data from web pages. The **tydiverse** package is a collection of packages useful for datascience, including *ggplot2* and *dyplr* which are necessary for the code used here. The package **flextable** is used to produce nice looking static tables.

````r
    library(rvest)
    library(tidyverse)
    library(flextable)
````

## Get financial data from chosen tickets

The following code uses tools of the **rvest** library to scrape data from finance.yahoo.com. We are creating a function This requires getting the XPATH to specific data in the web page. If the web page changes, it will break the script. It could be better in the future to change this for getting info from a database (via API or other means).

````r
  # is a function that takes one argument, ticker. The function constructs a URL for the Yahoo Finance page of the given ticker, then uses read_html to download and parse the HTML content of that page. 
    get_financials <- function(ticker) {
      url <- paste0("https://finance.yahoo.com/quote/", ticker)
      page <- read_html(url)
      
      pe_ratio <- page %>%
        html_nodes(xpath = '/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div[2]/table/tbody/tr[3]/td[2]') %>%
        html_text() %>%
        as.numeric()
      
    #Earnings yield is the inveser of the P/E ratio. As such we can simply calculate it here.
      earnings_yield <- round(1 / pe_ratio, 4)
      

      expense_ratio_text <- page %>%
        html_nodes(xpath = '/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div[2]/table/tbody/tr[7]/td[2]') %>%
        html_text()
      
      # Remove the '%' sign and convert to numeric
      expense_ratio <- as.numeric(gsub("%", "", expense_ratio_text))
      
      
      return(data.frame(Ticker = ticker, ExpenseRatio = expense_ratio, PE = pe_ratio, EarningsYield = earnings_yield))
    }

    tickers <- c("VUN.TO", "VCN.TO", "XEF.TO", "AVUV", "AVDV", "XEC.TO", "AVES")  # The tickers part of the portfolio

    # Define the weights for each ticker in the portfolio. This will be used to calculate the weighted averages.
    portfolio_weights <- c("VUN.TO" = 0.315, "VCN.TO" = 0.23, "XEF.TO" = 0.165, 
                           "AVUV" = 0.115, "AVDV" = 0.075, "XEC.TO" = 0.05, "AVES" = 0.05)

    # lapply is a function in R that applies a function over a list or vector. In this case, the function get_financials is applied to each element of the tickers vector. bind_rows() combines multiple data frames into one by binding them row-wise. This means that it takes data frames and stacks them on top of each other.
    financial_data <- lapply(tickers, get_financials) %>%
      bind_rows()
````

The following code adjusts for the missing values on the yahoo site,
since the Vanguard and Blackrock tickers are missing. I will eventually
add code to scrape the data from their respective websites.

````r
    # Manually adjust 
    correct_expense_ratios <- c("VUN.TO" = 0.17, "VCN.TO" = 0.05, "XEF.TO" = 0.22, "XEC.TO" = 0.28)
    financial_data$ExpenseRatio <- ifelse(financial_data$ExpenseRatio == 0,
                                          correct_expense_ratios[financial_data$Ticker],
                                          financial_data$ExpenseRatio)
````

End.

<div class="tabwid"><style>.cl-bb66bc5a{}.cl-bb6042a8{font-family:'Arial';font-size:11pt;font-weight:normal;font-style:normal;text-decoration:none;color:rgba(0, 0, 0, 1.00);background-color:transparent;}.cl-bb62d130{margin:0;text-align:left;border-bottom: 0 solid rgba(0, 0, 0, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);padding-bottom:5pt;padding-top:5pt;padding-left:5pt;padding-right:5pt;line-height: 1;background-color:transparent;}.cl-bb62d13a{margin:0;text-align:right;border-bottom: 0 solid rgba(0, 0, 0, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);padding-bottom:5pt;padding-top:5pt;padding-left:5pt;padding-right:5pt;line-height: 1;background-color:transparent;}.cl-bb62e3e6{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 1.5pt solid rgba(102, 102, 102, 1.00);border-top: 1.5pt solid rgba(102, 102, 102, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}.cl-bb62e3f0{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 1.5pt solid rgba(102, 102, 102, 1.00);border-top: 1.5pt solid rgba(102, 102, 102, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}.cl-bb62e3f1{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 0 solid rgba(0, 0, 0, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}.cl-bb62e3fa{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 0 solid rgba(0, 0, 0, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}.cl-bb62e3fb{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 1.5pt solid rgba(102, 102, 102, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}.cl-bb62e3fc{width:0.75in;background-color:transparent;vertical-align: middle;border-bottom: 1.5pt solid rgba(102, 102, 102, 1.00);border-top: 0 solid rgba(0, 0, 0, 1.00);border-left: 0 solid rgba(0, 0, 0, 1.00);border-right: 0 solid rgba(0, 0, 0, 1.00);margin-bottom:0;margin-top:0;margin-left:0;margin-right:0;}</style><table data-quarto-disable-processing='true' class='cl-bb66bc5a'><thead><tr style="overflow-wrap:break-word;"><th class="cl-bb62e3e6"><p class="cl-bb62d130"><span class="cl-bb6042a8">Metric</span></p></th><th class="cl-bb62e3f0"><p class="cl-bb62d13a"><span class="cl-bb6042a8">Value</span></p></th></tr></thead><tbody><tr style="overflow-wrap:break-word;"><td class="cl-bb62e3f1"><p class="cl-bb62d130"><span class="cl-bb6042a8">Weighted Average PE</span></p></td><td class="cl-bb62e3fa"><p class="cl-bb62d13a"><span class="cl-bb6042a8">14.22</span></p></td></tr><tr style="overflow-wrap:break-word;"><td class="cl-bb62e3f1"><p class="cl-bb62d130"><span class="cl-bb6042a8">Weighted Average ER</span></p></td><td class="cl-bb62e3fa"><p class="cl-bb62d13a"><span class="cl-bb6042a8">0.19</span></p></td></tr><tr style="overflow-wrap:break-word;"><td class="cl-bb62e3fb"><p class="cl-bb62d130"><span class="cl-bb6042a8">Weighted Average EY</span></p></td><td class="cl-bb62e3fc"><p class="cl-bb62d13a"><span class="cl-bb6042a8">8.29</span></p></td></tr></tbody></table></div>
