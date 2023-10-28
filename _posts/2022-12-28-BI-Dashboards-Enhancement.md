---
title: Strategic Business Intelligence Dashboard Enhancement
date: 2022-12-16 00:00:00 +0800
categories: [Projects Portfolio, Business Intelligence]
tags: [Microsoft Power BI, DAX, API, Power Query, Data Visualization]
render_with_liquid: false
pin: false
---

This article is a component of my Power BI project, which revolves around rectifying incorrect visualizations. Our primary goal is to gain a thorough understanding of the data and implement appropriate visualizations to improve accuracy and effectiveness.


<iframe title="Customer segmentation and Sales dashboard draft" width="800" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiZjYwZjIzODktZGY0Ni00Y2UyLTg0YTItODMzOTZhOTIxYTlmIiwidCI6IjBlMGRiMmFkLWM0MTYtNDdjNy04OGVjLWNlYWM0ZWU3Njc2NyIsImMiOjEwfQ%3D%3D&pageName=ReportSectionf41a13297f1ea9120f47" frameborder="0" allowFullScreen="true"></iframe>



## Introduction
In today's fast-paced and data-driven business environment, it is essential for
companies to have access to accurate and up-to-date information about their
operations. One effective way to do this is by creating dashboards that allow managers
to quickly and easily visualize key performance indicators and trends. In this project,
we will be creating dashboards for product sales, sales analysis, and customer
segmentation to help our company better understand its performance and identify
opportunities for improvement.



Having these dashboards will provide a number of benefits for our company.
For example, the product sales dashboard will allow us to see which products are
performing well and which ones may need more attention. The sales analysis
dashboard will give us insights into overall sales trends and help us identify
opportunities for growth. And the customer segmentation dashboard will help us better
understand the needs and preferences of our different customer segments, which will
allow us to tailor our marketing and sales efforts to their specific needs.

## Discussion on the Given Dashboards

### 1. Customer segmentation dashboard

![Figure 1](BI1/img1.jpg)
_**Figure 1:** Customer segmentation dashboard_


A customer segmentation dashboard is a visualization tool that is used to
display key metrics and data related to customer segments. Customer segmentation is
the process of dividing a customer base into smaller groups based on common
characteristics, such as demographics, purchasing habits, or geographic location. The
main function of a customer segmentation dashboard is to provide a comprehensive
view of the customer base, broken down by customer segment. This can help
businesses to understand the specific needs and preferences of different customer
segments, and to tailor their marketing, sales, and customer service efforts accordingly.
A customer segmentation dashboard can be used to track and analyze customer data
over time, identify trends or patterns in customer behavior, and identify areas for
improvement in the customer experience. It can also be used to track the performance
of different customer segments or to identify opportunities for upselling or cross-
selling to existing customers.

### 2. Sales analysis dashboard

![Figure 2](BI1/img2.jpg)
_**Figure 2:** Sales analysis dashboard_

The function of a sales analysis dashboard is to provide a comprehensive
overview of the performance of a company's sales activities. It typically includes
metrics such as sales revenue, sales volume, average transaction value, and conversion
rate, as well as key drivers of sales such as product or customer segments. A sales
analysis dashboard can help companies understand how their sales are performing,
identify trends and patterns, and make informed decisions about how to optimize their
sales efforts. In this dashboard, the value of discount, sales, profit, order quantity and
total cost are placed at the top of the dashboard as a card which generally used as an
overview for specific value and parameter. At the bottom, we can observe sales, by
different customer segment, discount by product category and profit by product name
and city.

### 3. Product sales dashboard

![Figure 3](BI1/img3.jpg)
_**Figure 3:** Product sales dashboard_

A product sales dashboard is used to reflect product sales and profitability for
the past period. Displaying information, such as trends and key indicators, enables
managers to understand real product sales information to guide decision-making and
achieve the purpose of data-driven decision-making. Sales and profit totals in this
dashboard provide an overall view to give viewers a general perception of sales. The
pie chart of profit by product name and product category shows the profit share of each
product. Based on the profit pie chart, management can understand which products are
the most profitable and which are barely profitable and thus adjust the product mix.
Similarly, the treemap of profit by city and product name shows profits by city and
product dimension, allowing us to understand the sales and profits of various products
between different cities. Aggregating by region gives management insight into product
preferences across cities, how sales strategies are being executed across cities and a
host of other issues. At last, the stacked bar chart of quantity ordered new by month
and product name represents the order volume of each product summarized by month.
This chart reflects the order trend from January to June and can be broken down for
each product. In summary, the product sales dashboard reflects the basic situation of
product sales and provides the data basis for sales decisions, i.e. data-driven decisions.

## Issues

### 1. Wrong Chart Implementation

Using the correct type of chart is important because it helps to clearly and
effectively communicate the data and insights being presented. Using the wrong type
of chart can lead to confusion or misinterpretation of the data, which can lead to
incorrect decisions or actions being taken. It is important to choose the appropriate
chart type based on the nature of the data and the message being conveyed, in order to
effectively communicate the information to the viewer.
<br>
In customer segmentation dashboard for instance, waterfall chart as shown in
Figure 4 below. Is used to show how an initial value is affected by a series of intermediate
positive or negative values, resulting in a final value. This type of chart is often used
to show the breakdown of a whole into its component parts, or to show the
contributions of different factors to an overall change. In the context of a customer
segmentation dashboard, a waterfall chart would not be an appropriate visualization
because the focus of the dashboard is on understanding and analyzing the
characteristics and performance of different customer segments, rather than on
showing the breakdown of a whole into its component parts or the contributions of
different factors to an overall change. A more suitable chart type for this purpose might
be a bar chart or a pie chart, depending on the specific data and insights being
presented.

![Figure 4](BI1/img4.jpg)
_**Figure 4:** Waterfall chart from customer segmentation dashboard_

This case also happened in product sales dashboard, using a Treemap for a
product sales dashboard may not be the most effective visualization choice. While
Treemaps can effectively represent the relative sizes of different categories, they can
be difficult to compare and may not accurately display smaller categories. Figure 5.
illustrate Treemap from product sales dashboard. It may be more effective to use a
different visualization method, such as a bar chart or pie chart, to effectively
communicate the sales data for each product

![Figure 5](BI1/img5.jpg)
_**Figure 5:** Treemap from product sales dashboard_


### 2. Unnecessary Volume of Data

Too much data is displayed in charts. Showing too much data is a problem that
many producers often face. This problem is because the producer does not understand
the problem from the user’s point of view but tries to show as much information as
possible on the dashboard. This manner will cause viewers to get lost instead of giving
them more helpful information. Take the pie chart in product sales dashboard as an
example which is shown in Figure 6 below, it shows so many categories that it is
impossible to see how many categories there are and the percentage of each category.
The same is true for the other two charts in this dashboard

![Figure 6](BI1/img6.jpg)
_**Figure 6:** Pie chart from product sales dashboard_

### 3. Indistinguishable Colour Scheme

The use of color scheme in a dashboard is important because it helps the user
to quickly and easily understand the data being presented. By using a consistent color
scheme throughout the dashboard, the user is able to easily identify different data
points and trends. Additionally, using contrasting colors can help to highlight
important data points and draw the user's attention to specific areas of the dashboard.
It is also important to use colors that are easy on the eyes and do not strain the user's
vision, as this can make it difficult for the user to effectively interpret the data.


As can be seen from the product sales dashboard, as demonstrated in Figure 7 below.
The stacked bar chart and treemap use colour to distinguish different products.
However, it can be seen that some of the products adjacent to each other in the figure
use colours that are similar. Such a colour scheme makes it impossible to understand
data well in charts and even causes visual errors.

![Figure 7](BI1/img7.jpg)
_**Figure 7:** Stacked bar chart from product sales dashboard_


## Enhanced Dashboards

### 1. Customer Segmentation Dashboard

![Figure 8](BI1/img8.jpg)
_**Figure 8:** Revised customer segmentation dashboard_

In hindsight of the customer segmentation dashboard's objective, it is essential to ensure that the dashboard effectively achieves this goal. To do so, it must provide answers to general questions, such as assessing the performance of each customer segment and determining the current value of these segments. Once we have these questions addressed, we can gain insights into specific customer segments and develop optimized marketing and sales strategies. Figure 8 depicts the new customer segmentation dashboard, and below are descriptions of some of the visuals within it.

- Cards
Cards are essential for displaying and monitoring numbers, and they are placed on the upper left part of the dashboard to provide an overview of key statistics. These cards include total sales and total profit, offering quick access to specific parameters. The numbers on these cards can be adjusted according to the selected filters, offering flexibility in data presentation. An example of this flexibility is demonstrated by the use of slicers for states and cities, positioned just below the card items. Slicers allow users to filter data based on their parameters, affecting the entire dashboard.

- Scatter Chart
The scatter plot showing profit and order value by customer segment helps us analyze how different customer segments perform. It reveals which segments generate higher profits and order values and highlights less profitable ones. This information guides marketing strategies and operational improvements. Additionally, the plot can uncover correlations and trends between profit and order value, aiding in identifying growth and optimization opportunities.

- Table
The table in the dashboard displays city data along with sales, order count, and order value metrics. It serves the purpose of presenting a clear and concise overview of this information, enabling users to compare and analyze different city data points. The table can help identify trends and patterns, and when combined with charts or graphs, it provides additional context and insights. Analyzing this table allows users to gain a deeper understanding of city performance, aiding in informed decisions about optimizing sales and operations in various locations.

- Slicer
The slicer by State and City, is a useful visual element had to be in our
customer segmentation dashboard, it acts as a canvas visual filter where it enables a
user to sort and filter with a city of interest. To see the customer performance or
analysis in specific province.

### 2. Sales Analysis Dashboard

![Figure 9](BI1/img9.jpg)
_**Figure 9:** Revised Sales Analysis dashboard_

Referring to the design presented in Figure 9, it is a sales analysis dashboard that offers a wide range of information. This interface provides comprehensive sales analysis, including data on sales, profit, quantity, and product categories over a specific period.

Furthermore, it offers insights into the company's products, customer types, and location, including regions. The dashboard presents sales analytics for various customer segments, such as corporate, small business, home office, and consumer customers. It also displays the current discount percentages based on product categories like furniture, office supplies, and technology. Additionally, it showcases the volume of products sold and the corresponding revenue for each location within the active area.

To enhance data analysis, a visual slicer display is employed, allowing decision-makers to efficiently analyze data by order month and date. In total, seven different types of elements have been incorporated to improve the functionality of our dashboard.

- Card
We created three different cars that seek to show a number, such as the amount
of sales, profit and discounts given. A Power BI card is a sort of visualization that is
excellent for displaying such figures. Type of visual cards is used in card visualization,
a participative technique for gathering data that enables groups to exchange and
brainstorm ideas.

- Slicer
When we want to observe the dashboard's overall visual display, the slicer is a
good option. For quicker access, place frequently used or significant filters on the
report canvas. Make it simpler to view the current condition of the filters without
opening another list. Use the data table's hidden and unused columns as filters. To
provide more thorough date filtering in this modification, we used two types of
portable slicers. The first one is by order date in days within the given period in the
dataset between 1/1/2015 to 30/6/2015, The second one is by Weekend and Weekday,
we came to this after creating two columns, one for Day of Week for each order date
by using the WEEKDAY function built in Power BI, and the second column is
Weekday/Weekend by using the IF Condition function. We should utilize the slicer
tool since our goal is to have a stronger impact on the selective filtering of the data in
the visualization during the period.

- Clustered bar chart
Next, we employ a clustered bar chart, which shows values or measurements
with bars that are proportional to the data. Product category data are shown in this
clustered bar chart broken down by city. Bar charts with clusters are useful for
graphically illustrating (visualizing) our data. In addition to statistical indications, it is
utilized. Multiple data series are shown in clustered horizontal columns in clustered
bar charts. The horizontal bars are organized by city because each data series has the
same axis name. Multiple series are directly compared within a particular category
using clustered bars. The beginning of the chart is Washington have the high profit by
city and product category.

- Pie Chart
The pie chart type is the last but certainly not least of the charts in our sales
analysis dashboard. All versions of Power BI provide built-in chart visualizations
called pie charts. Depending on the value of each data label, each set of categorical
data is displayed in a pie form in a circular pie chart.
Pie charts can be used to indicate percentages at specific moments in time as
well as to display general percentages. Pie charts do not depict changes over time, in
contrast to bar graphs and line graphs. a display of information regarding discounts
based on product categories, for instance. As much as 55.14% of the second office
supply category is visible, followed by 24.02% of the technology category and 20.84%
of the kind of furniture product category

### 3. Product Sales Dashboard

![Figure 10](BI1/img10.jpg)
_**Figure 10:** Revised Product Sales dashboard_

First, we adjusted the spacing between components in the new dashboard. As
we can see from the summary data area in Figure 10, all components are bordered, and
the negative space is unified, making the overall layout more aesthetically pleasing
without being crowded or sparse.

- Line and Stacked Column Chart
Line and stacked column chart can show the number of quantity ordered of the
products name by months from January to June. By looking at this visualization, it’s
very easy to notice the interesting products by the customers and see the trending
period of the products.

- Clustered Bar Chart
Next, we replaced the Treemap with a Bar Chart in the new dashboard because
it is easier to make comparison. Treemap tastes confusing because it is difficult to
extract useful information from it. The Bar Chart which shows profits with bars of
product names and their categories. The horizontal bars are organized by product
names as shown in this clustered bar chart listed as a descending manner by the profit,
and broken down by product categories with different colors described as a legend. In
addition to statistical indications, the quantities ordered and the remaining cards show
more specific information about these products separately

- Pie chart
Lastly, we used a pie chart to shows the sales and profits of different product
categories type, with the legend as product category and the profit as values, while we
added the sales data as a tooltip.
In response to the visual hierarchy, the style of summary data area has been
changed to highlight the importance so that the viewer’s attention will fall on this area
during the initial viewing. Guides viewers to explore the dashboard in order from
overall to partial.
The overall layout of the dashboard has been adjusted, resulting in a more
aesthetically pleasing instrument panel and a more balanced placement of components.
People have a special obsession with symmetry, and asymmetry most of the time
means unattractive. We Filled the background colour for the title area to make the title
stand out and changed the colour of the title text to a more eye-catching white.
Viewers’ attention is first drawn to the title when viewing the chart so that they are
informed of the intended theme of the chart before viewing the data.

Changed the colour scheme. Using more contrasting colours in the new colour
scheme makes it easier for viewers to distinguish between different data areas.


