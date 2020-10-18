---
title: Visitor Pattern
date: 2017-12-17T18:55:46+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags:  ['C#', software architecture]
---
Today I want to talk about the visitor pattern. It is a powerful pattern and I think it is used too little. Maybe because it looks complex but once you got how it works, it is pretty easy and powerful. The visitor pattern belongs to the behavioral patterns.

## Definition

The <a href="http://www.dofactory.com/net/visitor-design-pattern" target="_blank" rel="noopener">Gang of Four</a> defines the visitor pattern as followed: &#8220;Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.&#8221;

## Goals

  * Remove duplication of code
  * Separates an algorithm from an object structure by moving the hierarchy of methods into one object.
  * Helps to ensure the SRP
  * Ensures that we can add new operations to an object without modifying it

## UML Digram

As already mentioned, the visitor pattern might look complex when you only look at the UML diagram. So bear with me. At the end of this post, you will be able to understand and to implement it.

[<img loading="lazy" class="aligncenter wp-image-492" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram.jpg" alt="Visitor pattern UML diagram" width="700" height="482" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram.jpg 2296w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram-300x206.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram-768x529.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram-1024x705.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Visitor-pattern-UML-diagram.jpg)

&nbsp;

On the UML diagram, you can see the elements of the visitor pattern. The important parts are the Visitor interface which is implemented by the ConcreteVisitor. This visitor has a Visit Method which takes a ConcreteElement as parameter. You need a Visit method for every ConcreteElement you want to work on.

On the other side, you have an interface which implements the Accept method with IVisitor as parameter.

Now the client can make a list of IElement and call Accept and each element. The Accept method then calls the Visit method of the ConcreteVisitor. The ConcreteVisitor does the calculations and stores the result in a public property. After the calculation is done, the client can access the result by using the property of the ConcreteVisitor.

If you want to add a new ConcreteClass, you only have to add a new Visit method to the IVisitor and implement it in the ConcreteVisitor. So you don&#8217;t have to modify any other classes.

If this sounds too complicated don&#8217;t worry. Following I will show how an implementation works without the visitor pattern and then I will refactor it to use the pattern. This will point out the advantages.

## Implementation without the Visitor Pattern

I want to create an application which calculates the pay of a person. The pay consists of a salary class which has the salary before and after tax as properties. Also added is a bonus which will be calculated in the Bonus class depending on the revenue. Lastly, I subtract the costs of the marketing from the pay. The person class has a list for the bonus, marketing and salary.

After all the classes are set up, I can add elements to the lists of the person.

<div id="attachment_370" style="width: 483px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Set-up-person.jpg"><img aria-describedby="caption-attachment-370" loading="lazy" class="size-full wp-image-370" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Set-up-person.jpg" alt="Set up person" width="473" height="129" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Set-up-person.jpg 473w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Set-up-person-300x82.jpg 300w" sizes="(max-width: 473px) 100vw, 473px" /></a>
  
  <p id="caption-attachment-370" class="wp-caption-text">
    Set up person
  </p>
</div>

To calculate the pay of the person, I have to create a foreach loop for Salary, Bonus and Marketing.

<div id="attachment_371" style="width: 541px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Calculate-pay-of-the-person.jpg"><img aria-describedby="caption-attachment-371" loading="lazy" class="size-full wp-image-371" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Calculate-pay-of-the-person.jpg" alt="Calculate pay of the person" width="531" height="305" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Calculate-pay-of-the-person.jpg 531w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Calculate-pay-of-the-person-300x172.jpg 300w" sizes="(max-width: 531px) 100vw, 531px" /></a>
  
  <p id="caption-attachment-371" class="wp-caption-text">
    Calculate pay of the person
  </p>
</div>

The program calculates the pay and then prints it with the name of the person to the console. This works fine. But what if I want to add another income source, for example, business expenses. In this case, I have to add a new class BusinessExpenses, add a new list for the BusinessExpenses to the person class and I also have to add a new foreach loop to the calculation of the total pay. Basically, all classes have to change to implement the business expenses. This violates the Single Responsible Principle.

That’s where the visitor comes into play. You can find the solution on [GitHub](https://github.com/WolfgangOfner/WithoutVisitorPattern).

## Implementation of the Visitor Pattern

The first step is to implement the IVisitor interface, containing three Visit methods with Salary, Marketing and Bonus as parameter.

<div id="attachment_372" style="width: 280px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IVisitor.jpg"><img aria-describedby="caption-attachment-372" loading="lazy" class="size-full wp-image-372" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IVisitor.jpg" alt="IVisitor interface" width="270" height="176" /></a>
  
  <p id="caption-attachment-372" class="wp-caption-text">
    IVisitor interface
  </p>
</div>

If I want to expand the functionality of my program, I only have to add a new Visit method to the interface. Next, I implement another interface. I call this interface IAsset. The interface has only one method, Accept with IVisitor as parameter.

<div id="attachment_373" style="width: 348px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IAsset.jpg"><img aria-describedby="caption-attachment-373" loading="lazy" class="size-full wp-image-373" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IAsset.jpg" alt="IAsset interface" width="338" height="87" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IAsset.jpg 338w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/IAsset-300x77.jpg 300w" sizes="(max-width: 338px) 100vw, 338px" /></a>
  
  <p id="caption-attachment-373" class="wp-caption-text">
    IAsset interface
  </p>
</div>

The Salary, Bonus, Marketing and Person class implement the IAsset interface. All classes implement the Accept method the same way. The visitor calls visit with this as parameter.  There is a slight difference of the implementation in the Person class which I will talk in a second.

<div id="attachment_374" style="width: 286px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept.jpg"><img aria-describedby="caption-attachment-374" loading="lazy" class="size-full wp-image-374" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept.jpg" alt="Implementation of Accept" width="276" height="72" /></a>
  
  <p id="caption-attachment-374" class="wp-caption-text">
    Implementation of Accept
  </p>
</div>

With this change, the lists for the salary, bonus and marketing are not needed any longer in the Person class. I replace these three lists with a list of the type IAsset called Assets. This list contains all assets which are needed to calculate the pay of a person. The Accept method iterates through the Assets list and calls visit on every item of the list.

<div id="attachment_375" style="width: 316px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept-in-Person-class.jpg"><img aria-describedby="caption-attachment-375" loading="lazy" class="size-full wp-image-375" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept-in-Person-class.jpg" alt="Implementation of Accept in Person class" width="306" height="183" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept-in-Person-class.jpg 306w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept-in-Person-class-300x180.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-Accept-in-Person-class-124x74.jpg 124w" sizes="(max-width: 306px) 100vw, 306px" /></a>
  
  <p id="caption-attachment-375" class="wp-caption-text">
    Implementation of Accept in Person class
  </p>
</div>

The last step is to implement a class which contains the logic for the calculation of the salary. This class is the ConcreteVisitor from the UML diagram. I call it TotalSalaryVisitor. The TotalSalaryVisitor implements the IVisitor and therefore also implements all the Visit methods. In these methods the actual calculation takes place. The result will be stored in a public property called TotalSalary. This means that the Visit methods for Salary and Bonus add the SalaryAfterTax and BonusAfterTax to the TotalSalary. The Visit method for the Marketing subtracts the MarketingCosts from the TotalSalary.

<div id="attachment_376" style="width: 372px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/TotalSalaryVisitor.jpg"><img aria-describedby="caption-attachment-376" loading="lazy" class="size-full wp-image-376" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/TotalSalaryVisitor.jpg" alt="TotalSalaryVisitor implementation" width="362" height="376" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/TotalSalaryVisitor.jpg 362w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/TotalSalaryVisitor-289x300.jpg 289w" sizes="(max-width: 362px) 100vw, 362px" /></a>
  
  <p id="caption-attachment-376" class="wp-caption-text">
    TotalSalaryVisitor implementation
  </p>
</div>

### Executing the calculation

With everything set up, I can remove the logic from the Main method. I also add the values for Salary, Bonus and Marketing to the Assets list. To calculate the salary of a person I call the Accept method of the person with the TotalSalaryVisitor as parameter. Lastly, I print the total pay by accessing the TotalSalary property of the visitor.

<div id="attachment_378" style="width: 704px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Executing-the-calculation-of-the-pay-of-a-person.jpg"><img aria-describedby="caption-attachment-378" loading="lazy" class="size-full wp-image-378" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Executing-the-calculation-of-the-pay-of-a-person.jpg" alt="Executing the calculation of the pay of a person" width="694" height="240" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Executing-the-calculation-of-the-pay-of-a-person.jpg 694w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Executing-the-calculation-of-the-pay-of-a-person-300x104.jpg 300w" sizes="(max-width: 694px) 100vw, 694px" /></a>
  
  <p id="caption-attachment-378" class="wp-caption-text">
    Executing the calculation of the pay of a person
  </p>
</div>

### Adding new Assets

If I want to add a new asset, let&#8217;s say business expenses, I only have to add the new class BusinessExpenses. This class then implements IAsset. In the Visitor, I add a new Visit method which adds or subtracts the business expenses from the TotalSalary property. With the visitor pattern, I was able to extend the application without changing the call of the calculation.

### Adding new calculations

With the visitor pattern, it is also possible to add new calculations without changing the program. For example, I want to add a calculation to see the amount of taxes a person pays. To achieve this I only have to add a new Visitor, called TaxVisitor and call the Accept method of the person with this new Visitor.

&nbsp;

<div id="attachment_379" style="width: 678px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Added-the-TaxVisitor.jpg"><img aria-describedby="caption-attachment-379" loading="lazy" class="size-full wp-image-379" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Added-the-TaxVisitor.jpg" alt="Added the TaxVisitor" width="668" height="146" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Added-the-TaxVisitor.jpg 668w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Added-the-TaxVisitor-300x66.jpg 300w" sizes="(max-width: 668px) 100vw, 668px" /></a>
  
  <p id="caption-attachment-379" class="wp-caption-text">
    Added the TaxVisitor
  </p>
</div>

You can find the source code of the implementation on <a href="https://github.com/WolfgangOfner/VisitorPattern" target="_blank" rel="noopener">GitHub</a>.

## Conclusion

I showed that the visitor pattern is great for decoupling data and the calculation. This decoupling helps to extend the functionality of the application without changing the existing code.