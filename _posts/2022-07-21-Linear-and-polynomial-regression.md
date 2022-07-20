---
title: Linear and Polynomial Regression
author: RealJuney
date: 2022-07-21
categories: [Study]
tags: []
---

## Supervised Learning

Supervised Learning is a type of machine learning. It is about learning a function that maps an input to an output based on example input-output data. There are other types of machine learning such as unspervised learning which uses data with no labels and reinforcement learning which uses an agent that learn from its own action, but these will be covered later in the study.

This below figure shows what supervied learning does.
![](/assets/img/regression/img1.png)





## Linear Regression

Linear Regression or Linear Fitting is a perfect example of supervised learning.

Linear Regression: Modeling a relationship between a dependent variable and one or more independent variable as a linear function.
![](/assets/img/regression/img2.png)


### Simple Linear Regression
Let there be only one independent variable. Then the linear regression would look like this.
![](/assets/img/regression/img3.png)

The loss function would look somewhat like below.
![](/assets/img/regression/img4.png)

The point where gradient of the loss function is equal to 0 is the point where the loss function is the smallest. We can get the following equation by using this information. (In this case, N = 4)
![](/assets/img/regression/img5.png)


## Polynomial Regression
Polynomial Regression is similar but models a relationship as a polynomial function.
We can calculate the parameters similarly, but instead of matrix X we use a matrix called 'feature matrix'. Example below.
![](/assets/img/regression/img6.png)