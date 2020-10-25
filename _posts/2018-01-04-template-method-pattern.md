---
title: Template Method Pattern
date: 2018-01-04T21:01:01+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: ['C#',software architecture]
---
The Template Method pattern helps to create the skeleton of an algorithm. This skeleton provides one or many methods which can be altered by subclasses but which don&#8217;t change the algorithm&#8217;s structure.

## UML Diagram

[<img loading="lazy" class="aligncenter wp-image-502" src="/assets/img/posts/2018/01/Template-Method-UML-diagram.jpg" alt="Template Method UML diagram" width="326" height="500" />](/assets/img/posts/2018/01/Template-Method-UML-diagram.jpg)

The UML diagram for the Template Method is pretty simple. It has one abstract class with the TemplateMethod and one or many sub methods which can be overridden by subclasses. These sub methods can be either abstract or virtual. If they are abstract, they have to be implemented. If they are virtual, they can be overridden to alter the behavior of this step.

## Implementation of the Template Method pattern

To be honest, the Template Method was pretty confusing to me in the beginning but after I tried to implement it, it became clear how it works. My implementation is pretty simple but it helped me to understand the pattern and I hope it helps you too.

I want to do some calculation and then save the result somewhere. The algorithm has three steps whereas step two and three can be overridden by a subclass. The class Calculator offers the TemplateMethod. This method contains the three steps.

[<img loading="lazy" class="aligncenter size-full wp-image-471" src="/assets/img/posts/2018/01/TemplateMethod.jpg" alt="TemplateMethod" width="237" height="160" />](/assets/img/posts/2018/01/TemplateMethod.jpg)

The first step, BeforeCalculation can&#8217;t be overridden by the subclasses and therefore will always be executed. The second step of the algorithm, CalculateSomething can be overridden. The classes CalculatorOracle and CalculatorSqlAzure override this method and do their own calculations. The CalculatorSqlAzure also overrides the property Result.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Implementation-of-CalculateSomething-in-the-CalculatorSqlAzure-class.jpg"><img aria-describedby="caption-attachment-472" loading="lazy" class="size-full wp-image-472" src="/assets/img/posts/2018/01/Implementation-of-CalculateSomething-in-the-CalculatorSqlAzure-class.jpg" alt="Implementation of CalculateSomething in the CalculatorSqlAzure class" /></a>
  
  <p>
    Implementation of CalculateSomething in the CalculatorSqlAzure class
  </p>
</div>

The CalculatorOracle class only overrides the CalculateSomething method.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Implementation-of-CalculateSomething-in-the-CalculatorOracle-class.jpg"><img aria-describedby="caption-attachment-473" loading="lazy" class="size-full wp-image-473" src="/assets/img/posts/2018/01/Implementation-of-CalculateSomething-in-the-CalculatorOracle-class.jpg" alt="Implementation of CalculateSomething in the CalculatorOracle class" /></a>
  
  <p>
    Implementation of CalculateSomething in the CalculatorOracle class
  </p>
</div>

The third and last step, SaveResult is only overridden by the CalculatorSqlAzure class. This means that the CalculatorOracle class uses the method provided by the Calculator.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Overriden-SaveResult-method-in-the-CalculatorSqlAzure-class.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2018/01/Overriden-SaveResult-method-in-the-CalculatorSqlAzure-class.jpg" alt="Overridden SaveResult method in the CalculatorSqlAzure class" /></a>
  
  <p>
    Overridden SaveResult method in the CalculatorSqlAzure class
  </p>
</div>

I know that this example is not really what you will see in a real-world project but I hope that it helped you to understand the Template Method pattern. You can find the source code on <a href="https://github.com/WolfgangOfner/TemplateMethodPattern" target="_blank" rel="noopener">GitHub</a>.

## Consequences

The Template Method pattern to achieve a clean design. The algorithm provided by the base class is closed for modification but is open for extension by subclasses. As a result, your design satisfies the open-closed principle. The pattern also helps to implement the Hollywood principle (&#8220;Don&#8217;t call us, we call you&#8221;).

One downside is that the steps of the algorithm must be already known when the pattern is applied. Therefore the Template Method pattern is great for reuse but it is not as flexible as for example the <a href="/strategy-pattern/" target="_blank" rel="noopener">Strategy pattern</a>.

## Related patterns

<a href="/strategy-pattern/" target="_blank" rel="noopener">Strategy</a>: Inject a complete algorithm implementation into another module

Decorator: Compose an algorithm or behavior from several sub-parts

Factory: Define a common interface for creating new instances of types with many implementations

## Conclusion

I showed how the Template Method pattern can be used to provide a skeleton for an algorithm. One or many parts of this algorithm can be overridden by subclasses. This behavior helps achieving the open-closed principle and therefore a clean overall design. I also implemented a simple example to show how the pattern works.