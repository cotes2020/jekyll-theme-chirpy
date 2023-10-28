---
title: Olist E-Commerce Reporting Dashboard
date: 2023-01-26 00:00:00 +0800
categories: [Projects Portfolio, Business Intelligence]
tags: [Microsoft Power BI, DAX, API, Power Query, Data Visualization]
render_with_liquid: false
pin: false
---
This article is a component of my Power BI project, which centers around OLIST, a fictitious Brazilian e-commerce platform. It utilizes Microsoft Power BI to create dashboards covering various aspects of OLIST's Brazilian e-commerce operations, including Executive Insights, Exploratory Analysis, Customer Investigation and Satisfaction, Delivery Analysis, and Forecasting.

## Executive Summary
-   **Data synopsis:** Brazilian E-commerce Public Dataset: Retail
    datasets of 100k orders placed on Olist spanning between
    October'2016 and September'2018 across several states. Information
    is trickled with price, orders, order status, payment, freight and
    user review along with many other parameters.


<iframe title="Report Section" width="800" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiMTAxNzk2NzYtNjNiNi00NGNiLWJkNjYtMzhmOWViNTUyNzA3IiwidCI6IjBlMGRiMmFkLWM0MTYtNDdjNy04OGVjLWNlYWM0ZWU3Njc2NyIsImMiOjEwfQ%3D%3D&pageName=ReportSectioncc096a47c2520501357d" frameborder="0" allowFullScreen="true"></iframe>


## Introduction

<div align="justify">
    Businesses have always tried to keep their customers base engaged and
satisfied with the services provided by them. For remaining relevant in
the industry, they need to incorporate the latest technological advances
into their services. More than a decade back, it was the internet which
was completely new and various industries tried to leverage the
capabilities of this technology that effortlessly acted as a medium of
communication between various businesses and their customers. In this
decade, industries have started to provide services that are catered
towards each client's individual needs. For such services, they are
required to leverage the power of artificial intelligence.
    </div>

### Company Background

<p align="justify">The <a href="https://www.olist.com/pt-br" target="_blank">Olist store</a> is an e-commerce business headquartered in Sao Paulo,
Brazil. This firm acts as a single point of contact between various
small businesses and the customers who wish to buy their products.
Recently, they uploaded a <a href="https://www.kaggle.com/olistbr/brazilian-ecommerce" target="_blank">dataset on Kaggle</a> that
contains information about 100k orders made at multiple marketplaces
between 2016 to 2018. What we purchase on e-commerce websites is
affected by the reviews which we read about the product posted on that
website. This firm can certainly leverage these reviews to remove those
products which consistently receive negative reviews. It could also
advertise those items which are popular amongst the customers.
</p>

### Formulation of business problem

<div align="justify">
    In this project, with a use of Microsoft Power BI
tool, a presented dashboards that summarizes the overall satisfaction of
the customers with the products which he or she had just purchased. As
well as descriptive, deliveries analysis and forecasting analysis.
    </div>

### Organizational structure (Stakeholder)

<div align="justify">
    Olist organizational structure data is layered. The CEO leads the
company and is sometimes the chairman or owner. He has the most invested
in the company and makes significant decisions, including as analyzing
and approving team budgets and assuring subordinates can do their jobs
well.
    </div>


<div align="justify">
Second is the marketing team; whose role is to market the company\'s
products using cutting-edge technology to compete commercially. The
financial team helps the company manage its accounts, income before
interest, taxes, depreciation, debt, loans, and cash according to the
budget. This group will also provide clients conditional vouchers to
increase sales or any marketing budget required to get this project up
and running.
    </div>


<div align="justify">
Information technology (IT) team is responsible for overseeing every
aspect of the company\'s system, including its hardware and software.
This team supports all other teams inside the organization. The
e-commerce team, which comes in at number seven, is crucial to the
online sales process. Customers will be dealt with personally by these
personnel, who will also guarantee their happiness with every
transaction There is more department that exist in this company however
for our issue and project, these are the most impacted and involved in
the project.
    </div>

## Dataset

<div align="justify">
I am going to make use of the dataset that was so generously provided by
Olist, which is the largest department store in Brazilian marketplaces.
Olist eliminates administrative burdens and consolidates legal
obligations by linking small enterprises located anywhere in Brazil to
various distribution channels. These retailers are allowed to sell their
wares through the Olist Store and have Olist\'s logistics partners
transport the products straight to the buyers after the sale has been
completed.
    </div>

<p align="justify">
The data consist of almost 100 000 customer id and order id. To
summarize, the data consist of order detail, customer detail, product
detail, seller detail, payment detail, geolocation detail and review
detail. The entity-relationship model is shown in Figure 1 to help
understand how the data interrelate with each other.
<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank">Kaggle</a>
</p>
  
![Figure 1](Olist/img1.jpg)
_**Figure 1:**  Entitiy-relationship model diagram of Olist Dataset_



| **Table**                   | **Description**                                      |
| ---------------------------- | ---------------------------------------------------- |
| olist_orders_dataset         | Connected to 4 other tables; used for order details.|
| olist_order_items_dataset    | Contains item details: shipping date, price, etc.    |
| olist_order_reviews_dataset  | Contains customer reviews on purchased products.    |
| olist_products_dataset       | Product info: ID, category, measurements.           |
| olist_order_payments_dataset | Payment details related to orders.                   |
| olist_customers_dataset      | Customer base information for the firm.             |
| olist_sellers_dataset        | Information on registered sellers.                   |
| olist_geolocation_dataset    | Geographical data for sellers and customers.        |



## Objectives of data stories
1. How many customers, orders, and orders per customer does the
company have?
2. What is the number of customers by state?
3. What is the number of orders by month?
4. What are the top and bottom 5 product categories?
5. Visualize the company’s customers’ demographics, sales trend, orders
by categories, orders changes by year

## Visualization Analytics

### Data model

<div align="justify">
Once I have downloaded the dataset from
Kaggle, it is possible to load it into Power BI. The data will have to
be pre-processed in order to obtain relevant analytics as it only has
the tables and keys referring to each file of the dataset. At first we
obtain the data model visible in the following image.
    </div>
 
 
![Figure 2](Olist/img2.jpg)
_**Figure 2:** Model Before_

<div align="justify">
Thus, it needs to make links between tables depending on how they
interact with each other. The olist_geolocation and product_category
tables are not linked to the model at all which means it is not possible
to leverage this data. The temporal is also not explicitely indicated as
Power BI needs rigorous time management to apply it to the other tables.
Moreover, the variables have a type depending on whether they are
numerical or textual data. Indeed, we can see that it contains data
about transportation logistics, customers, sellers and their different
products each with a different type.
The following table describes the entity relationships used in this analysis. Eventually, we can see the result
on the following image in Figure 3.
    </div>


| Entity        | Relationship Type | Related Entity     | Key                    |
|---------------|-------------------|--------------------|------------------------|
| Sellers       | One-to-Many       | Geolocation        | geolocation_zip_code_prefix |
| Sellers       | One-to-Many       | Orders             | orderID                |
| Sellers       | Many-to-One       | Brazil State       | StateID                |
| Brazil State  | One-to-Many       | Customers & sellers | StateID                |
| Geolocation   | Many-to-One       | Sellers             | zip code prefix        |
| Geolocation   | Many-to-Many      | Customers           | zip code prefix        |
| Customers     | One-to-One        | Order Dates         | customerID             |
| Products      | Many-to-One       | Product Category    | ProductID              |
| Products      | One-to-Many       | Orders             | ProductID              |
| Orders        | Many-to-One       | Order Dates         | OrderID                |
| Order Dates   | One-to-Many       | Payments            | OrderID                |
| Order Dates   | One-to-Many       | Reviews             | ReviewID               |
| Order Dates   | Many-to-One       | Weekday             | Date                   |



  
  
![Figure 3](Olist/img3.jpg)
_**Figure 3:** Model After_


### Table analysis
I will dive into some of these tables that were added, I have renamed them to
make it easier catch and this will obtain as the final data model that allows us to make
analytics.

1. Orders Table is at the center of the data model. It represents the most
important part of the dataset describing best what can be obtained from
the data. Thus, in this case, it about the orders made by a customer
buying a specific product. This table contains only primary keys that
are necessary to infer knowledge about the data. I have created two
formula expression using DAX through this table namely “% of sales” to divide the price of the total
price and shows a percentage as a profit ratio, and “qt ordered” where
this counts the unique orders.

```
% of sales = DIVIDE([price],SUM([price]),0)
```

```
qt order id ordered = CALCULATE(COUNT(orders[order_id]),ALLEXCEPT(orders,orders[order_id]))
```
2. The geolocation table is the Space table represented by geolocation that
is now connected to the data model in order to use the information it
contains for the two other tables: olist_customer and olist_seller.
Indeed, with the olist_geolocation table, we are able to create secondary
keys about customers and sellers to obtain knowledge about their postal
code, city or state that were previously impossible to understand.
Moreover, we created a hierarchy in order to go from the country and
drill down to the state then the city

3. Order dates have a precise temporal relationship with the data time.
Thanks to this table, it is possible to gain insights about temporal data by 
classifying it chronologically. In the "orders_date" table, we can distinguish 
the time difference in days between orders, allowing us to group them by the order 
date and obtain indicators such as deliveries or purchases. Furthermore, this 
table enables us to drill down from the year to the quarter, month, and day.

I have created a "delivery_days" expression that calculates the difference 
between the estimated date and the delivered date in days.

```
delivery_days = DATEDIFF('orders_dates'[order_estimated_DATE],'orders_dates'[order_delivered_DATE],DAY)
```
Additionally, there is a "delivery indicator" expression formula that indicates whether the delivery occurred before or after the estimated date. By utilizing both the "delivery_days" and "delivery indicator" expressions, we can determine how many days it takes for a delivery and how many deliveries took more than 100 days, for example, using the following DAX formula:

```
delivery_indicator = IF('orders_dates'[delivery_days]>0,"In advance", 
    IF('orders_dates'[delivery_days]=0,
    "On time",
    "Late"
    ))
```
Another clever set of expressions includes "time_day" and "time_hour," which serve to differentiate the time between approval, delivery, and orders in days and hours, respectively.

```
time_day (approved vs delivered) = (DATEDIFF(orders_dates[order_approved_at],orders_dates[order_delivered_customer_date],DAY))
```

4. In the review table, the goal was to generate meaningful indicators that could be analyzed to derive insights about the products. Therefore, I have introduced the "review_indicator," which determines whether a comment has been made after a purchase or not. This indicator will provide us with information about how customers feel about the product, whether they liked it or not, and if they are inclined to share their experiences with others.
The "review_indicator" categorizes the reviews based on recurrent keywords. This will enable us to identify the most descriptive aspects of the products when customers share similar opinions.

```
review_indicator = IF(reviews[review_comment_message]=="" || reviews[review_comment_message]=="-","No Comment","With Comment")
```
5. The Brazil state table displays the state ID and city names and is modeled in a one-to-many relationship with both the sellers and customers tables. 
By utilizing this distinct table, we can perform analyses exclusively within Brazil, leveraging its extensive dataset, while excluding the rest of the world for descriptive analysis.



### Dashboards

![Figure 4](Olist/img4.jpg)
_**Figure 4:** Header_

![Figure 5](Olist/img5.png)
_**Figure 5:** Executive Insights by Decisive Data_

Frequently, the question "How are we performing?" can lead to a cascade of further questions, spinoffs, and investigative research. This is especially true for globally-oriented companies. I aimed to create a report that proactively addresses this kind of exploration. The purpose of this report is to facilitate data-driven decision-making while emphasizing user flexibility and visual analysis. As a result, this dashboard can adapt to the evolving needs of the global business.

The Executive Insights page highlights the strong focus of this dashboard on sales and customers, with the goal of fulfilling objectives, increasing customer satisfaction, and boosting sales by uncovering insights from the dashboard. This is achieved after creating three unique formulas to serve in this dashboard:

```
Total sales = SUM('orders'[price]) + SUM('orders'[freight_value])
```
```
count customer(unique ) = DISTINCTCOUNT(customers[customer_unique_id])
```
```
count orders = COUNT(orders[order_id])
```
Upon revisiting the figures, it becomes clear that Sao Paulo consistently leads in total sales. Despite a rapid increase in sales over the past three years, the waterfall graph does not indicate a decrease in values.

It is evident that a significant number of both late and early deliveries can be observed in the data. As illustrated, there are five orders that took more than 100 days to deliver, which undoubtedly had an impact on customer satisfaction.

![Figure 6](Olist/img6.jpg)
_**Figure 6:** Descriptive Analytics_

Next, let's delve deeper into descriptive analytics. We have multiple options at our disposal, such as selecting a specific day, month, or year from the slicer at the top of the page.

The table graph breaks down the product categories based on the features of their average price, the sum of prices (revenue), the profit ratio, and the number of quantities customers ordered in each category. The top-performing category is "health and beauty," which has received the most orders. This isn't surprising, given that females are known to spend more on fashion. This category has generated $772,238 in revenue with a profit ratio of 6%.

On the descriptive analysis page, we can observe that we've achieved approximately $16 million in sales, served 94,000 customers, and processed more than 100,000 orders.

![Figure 7](Olist/img7.jpg)
_**Figure 6:** Customer Investigation_

Every business has wondered about the recent additions to their customer base. Customers are the driving force behind organizational growth. With their support, they can increase revenue, and without them, sustaining growth becomes a challenge. This is why conducting customer investigations is of paramount importance. This page illustrates that among nearly 100,000 customers.

Returning to the visuals in Figure 7, it's observed that the majority of new customers tend to join between May and August over the course of three years. The preferred payment methods are credit card and boleto payment.

The top three customers have made purchases of more than 21 items, amounting to approximately four thousand dollars. On Mondays, the highest order quantities are observed, while on Sundays, the highest average spending per order is recorded. This pattern aligns with the common tendency for people to spend more on weekends.

![Figure 8](Olist/img8.jpg)
_**Figure 7:** Customer Satisfaction_

When we examine the charts in Figure 8 above, it becomes evident that the overall average rating appears quite positive, with an average of 4 stars from approximately 99.5k customers. However, there is a notable decrease in the number of quantities ordered in December, following a slight increase three months prior.

Among the top five selling categories, there are approximately 40k orders out of the total 99k, while the bottom five categories haven't exceeded 60 orders. This disparity underscores the variation in product popularity.

Furthermore, the ratio of recurrent customers has decreased by 1% compared to the two years prior. This may be attributed to the significant increase in orders during the last year, and there is no conclusive evidence to suggest that customers have become less loyal. The formula used for calculating the recurrent customers ratio is provided below:

```
recurrent customers ratio = CALCULATE(DIVIDE([recurrent count], [count customer(unique )]))
```

![Figure 9](Olist/img9.jpg)
_**Figure 8:** Delivery Days_

The delivery page in Figure 8, provides insight into the process of delivering products from the seller's location to the customer's desired destination. The freight value appears acceptable, and there were 96.4 orders delivered out of the total orders, even considering the canceled orders.

We can observe that the average review score is not very high during the first quarter. This could be attributed to longer delivery times during this period. In contrast, the third quarter exhibits higher review scores despite a relatively shorter delivery duration.

![Figure 10](Olist/img10.jpg)
_**Figure 9:** Forecast_
The question on the minds of Olist's leaders is about their expected annual growth in the upcoming years. In any case, the annual growth appears promising, as indicated by the highest order quantities in all three years and the increased predictions for new customer acquisitions.


## Insights


<div align="justify">
    Olist has a delivery success rate of approximately 85%. This may
indicate that the company is facing some challenges with its delivery
process, and it may be worthwhile to investigate the causes of the
undelivered orders and take steps to improve the delivery success rate.
Some potential areas to look into could include the efficiency of the
company's logistics and fulfillment processes, the reliability of its
transportation partners, and any potential issues with the quality or
accuracy of the orders being placed.
    </div>
 

<div align="justify">
    Olist company has a high level of customer satisfaction overall, with a
significant number of positive reviews and scores. However, the fact
that the lowest-rated product category is "Security and Services"
suggests that this type of product may need improvement.
    </div>

## Recommendations


-   Monitor and analyze customer reviews regularly to identify trends
    and areas for improvement. This could involve using data analysis
    tools to identify common themes in customer feedback and using this
    information to make changes and improve the customer experience.


-   Investigate the causes of undelivered orders. This could involve
    analyzing the undelivered orders to identify common themes or
    factors that may be contributing to the problem. For example, are
    certain regions or customer demographics more likely to have
    undelivered orders? Are there particular products or types of orders
    that are more likely to be undelivered? 


-   Communicate with customers about the delivery process. Olist should
    be transparent with customers about the delivery process and provide
    them with regular updates on the status of their orders. This will
    help to build trust and create a positive customer experience. It
    will also give customers the opportunity to provide feedback on
    their experiences with the delivery process, which can be used to
    identify areas for improvement.


-   Overall, the key is to continue providing high-quality products and
    services, while also being responsive to customer feedback and
    working to improve areas that may need attention. Also regularly
    monitor and review the delivery success rate and communicate with
    customers about the process.


## Suggestions:

-   Special offerings to boost overall sales on low sales period 

-   Improve bottom selling categories by providing advertisements or promotions

-   Outsourcing drivers for delivery during Sales or Festival periods

-   Investigate and Review the partner company with low review score

-   analysis customers comments and reviews provided in the dataset with NLP or
    any kind of language processing models

