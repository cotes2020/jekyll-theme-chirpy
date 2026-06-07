---
title: Understanding Linear Regression from Scratch
date: 2026-06-07 12:00:00 +0100
categories: [Machine Learning, Fundamentals]
tags: [math, python, regression]
math: true
---

Linear regression is one of the most fundamental concepts in machine learning. It is used to predict a quantitative outcome (the **dependent variable**, usually denoted as $y$) based on one or more predictor variables (the **independent variables**, usually denoted as $x$).

## 1. The Core Idea: The Line of Best Fit

The goal is to find a straight line that best represents the relationship between your data points. Mathematically, for a simple linear regression (one predictor), this is represented by the equation:

$$y = wx + b$$

Where:
- $y$: The predicted value (output).
- $x$: The input feature.
- $w$ (**Weight/Slope**): How much $y$ changes for every unit change in $x$.
- $b$ (**Bias/Intercept**): The value of $y$ when $x$ is zero.

The vertical dashed lines below represent the **residuals**—the distance between the actual data points (red dots) and our model's predictions (the blue line). Our goal is to minimize this distance!

![Linear Regression Line](https://miro.medium.com/1*pSFdOyWKLK-1DegCoSvBNQ.png)
_Figure 1: The line of best fit minimizing the distance to real data points._

## 2. How the Model "Learns"

When you start training a model, it doesn't know the correct $w$ and $b$. It makes a random guess, and then it measures how wrong that guess is using a **Loss Function**.

### The Loss Function: Mean Squared Error (MSE)

The most common way to measure error is **Mean Squared Error (MSE)**. It calculates the difference between the actual value ($y_i$) and the predicted value ($\hat{y}_i$), squares it to eliminate negative signs, and averages it across all data points:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Optimization: Gradient Descent

To minimize this error, the model uses an algorithm called **Gradient Descent**. Think of it like walking down a mountain in a thick fog; you feel the slope beneath your feet (the gradient) and take small steps down until you reach the lowest point (the minimum error).

We calculate the partial derivatives of our loss function with respect to $w$ and $b$ to update our parameters:

$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

Using these gradients, we update our parameters iteratively using a tuning parameter called the **learning rate** ($\alpha$):

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

## 3. Key Assumptions of Linear Regression

For linear regression to work effectively, a few assumptions should ideally hold true:

1. **Linearity**: The relationship between $x$ and $y$ is a straight line.
2. **Independence**: The observations are independent of each other.
3. **Homoscedasticity**: The variance of residual errors is constant across all levels of $x$.
4. **Normality**: The residual errors are normally distributed.

> **Why this matters for your portfolio:** Implementing this model from scratch using only `NumPy` is an excellent way to show you understand vectorization and the mathematical core of AI, rather than just importing `scikit-learn`.
{: .prompt-info }
