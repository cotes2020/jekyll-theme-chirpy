---
title: Strategy Pattern
date: 2018-01-02T15:17:42+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: ['C#', software architecture]
---
The Strategy pattern is One of the simpler design patterns and probably a good one to get started with design patterns. Additionally, it is also very practical and can help to clean up the code.

## Goals

  * encapsulate related algorithms so they are callable through a common interface
  * let the algorithm vary from the class using it
  * allow a class to maintain a single purpose

## When to use it

The most obvious sign that you might want to use the strategy pattern are switch statements. In my example further down, I will also show how a switch statement can be replaced by the Strategy pattern.

Another hint that you should use the Strategy pattern is when you want to add a new calculation but in doing so you have to modify your class which violates the open-closed principle.

## UML Diagram

[<img loading="lazy" class="wp-image-498 aligncenter" src="/assets/img/posts/2018/01/Strategy.jpg" alt="Strategy Pattern UML diagram" width="600" height="416" />](/assets/img/posts/2018/01/Strategy.jpg)

The context class does the work. It takes the desired strategy in the constructor (the strategy can also be passed as parameter in the method as you will see later). The strategy interface declares a method which is called by the context class to perform the calculation I want on a concrete strategy.

## Implementation without the Strategy pattern

There are many examples of implementing the Strategy pattern on the internet. Popular ones are implementing different search algorithm or different calculations for products or orders. In my example, I will implement a cost calculator for products. The costs depend on the country in which the product is produced.

My products have some basics properties like price or name and also the production country. The ProductionCostCalculatorService class implements a calculate Methode in which it has a switch Statement. Depending on the production Country, the production costs are calculated differently.

<div id="attachment_458" style="width: 420px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/01/Calculating-production-costs-using-a-switch-statement.jpg"><img aria-describedby="caption-attachment-458" loading="lazy" class="size-full wp-image-458" src="/assets/img/posts/2018/01/Calculating-production-costs-using-a-switch-statement.jpg" alt="Calculating production costs using a switch statement" width="410" height="311" /></a>
  
  <p id="caption-attachment-458" class="wp-caption-text">
    Calculating production costs using a switch statement
  </p>
</div>

The production countries are China, Australia and the USA. If a different production Country is passed in the product, a UnknownProductionCountryException is thrown.

This approach has some flaws. The biggest problem is that it is not easy to add a new country. To achieve that, a new case has to be added to the switch. This doesn&#8217;t sound like a problem but it violates the open-closed principle and is also a problem if you have to add a new country every week. You will end up with a huge switch Statement. Another design flaw is that the product doesn&#8217;t have to know where it is produced. Therefore the algorithm shouldn&#8217;t rely on the information from the product.

You can find the source code on [GitHub](https://github.com/WolfgangOfner/WithoutStrategyPattern). The solution to the problems is the Strategy pattern.

## Implementation of the Strategy pattern

To implement the strategy pattern, I have to implement a new class for every strategy. As a result, I will have the classes ChinaProductionCostStrategy, AustraliaProductionCostStrategy and UsaProductionCostStrategy. These classes implement the new Interface IProductionCostCalculatorService which has one method. The method is Calculate and takes a product as parameter.

After these changes, I can modify the ProductionCostCalculatorService class. First, I inject the IProdctionCostCalculatorService in the constructor. Then, I change the CalculateProductionCost method to return the return value of the Calculate method from the Interface.

<div id="attachment_459" style="width: 784px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/01/ProductionCostCalculatorService.jpg"><img aria-describedby="caption-attachment-459" loading="lazy" class="size-full wp-image-459" src="/assets/img/posts/2018/01/ProductionCostCalculatorService.jpg" alt="ProductionCostCalculatorService" width="774" height="247" /></a>
  
  <p id="caption-attachment-459" class="wp-caption-text">
    ProductionCostCalculatorService with the Strategy pattern
  </p>
</div>

The last step is to modify the call of the calculation. For every strategy, I need an object, which I inject into the constructor of the ProductionCostCalculatorService.

<div id="attachment_461" style="width: 983px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/01/Calculating-production-costs-using-strategies-and-print-out.jpg"><img aria-describedby="caption-attachment-461" loading="lazy" class="size-full wp-image-461" src="/assets/img/posts/2018/01/Calculating-production-costs-using-strategies-and-print-out.jpg" alt="Calculating production costs using strategies and print out" width="973" height="207" /></a>
  
  <p id="caption-attachment-461" class="wp-caption-text">
    Calculating production costs using strategies and print out
  </p>
</div>

I have to admit that These calls look a bit messy and I am not a big fan of them. Therefore it is possible to change the ProductionCostCalculatorService, so it takes the strategy as parameter in the Calculate method instead of the constructor.

<div id="attachment_462" style="width: 892px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/01/Method-call-with-strategy.jpg"><img aria-describedby="caption-attachment-462" loading="lazy" class="size-full wp-image-462" src="/assets/img/posts/2018/01/Method-call-with-strategy.jpg" alt="Method call with strategy" width="882" height="128" /></a>
  
  <p id="caption-attachment-462" class="wp-caption-text">
    Method call with strategy as parameter
  </p>
</div>

This change also tidies up the ProductionCostCalculatorService, which makes it even easier to read. With this changes implemented, I can now change the call of the calculation. It is not necessary anymore to create a new ProductionCostCalculatorService object for every different production country. Instead, I pass the country as parameter in the CalculateProductionCost method.

<div id="attachment_463" style="width: 1251px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/01/Calculating-production-costs-using-the-strategy-as-paramater.jpg"><img aria-describedby="caption-attachment-463" loading="lazy" class="size-full wp-image-463" src="/assets/img/posts/2018/01/Calculating-production-costs-using-the-strategy-as-paramater.jpg" alt="Calculating production costs using the strategy as paramater" width="1241" height="147" /></a>
  
  <p id="caption-attachment-463" class="wp-caption-text">
    Calculating production costs using the strategy as parameter
  </p>
</div>

You can find the source code on <a href="https://github.com/WolfgangOfner/StrategyPattern" target="_blank" rel="noopener">GitHub</a>.

## Conclusion

In this example, I showed that the Strategy pattern is a simple pattern which helps to tidy up our code. While programming, look out for switch statements since these often indicate that the Strategy pattern could be used. As a result of the Strategy pattern, decoupling between the algorithm and other classes can be achieved.

Additionally, I showed a second variation on how to implement the pattern which leads to an even cleaner implementation.