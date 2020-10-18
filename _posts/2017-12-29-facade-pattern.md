---
title: Facade Pattern
date: 2017-12-29T11:41:38+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: ['C#', software architecture]
---
The Facade pattern is often used without the programmer knowing that he uses it. In this post, I want to give the thing a developer often does automatically a name.

## Goals

  * Simplify complex code to make it easier to consume
  * Expose a simple interface which is built on complex code or several interfaces
  * Expose only needed methods of a complex API or library
  * Hide poorly designed APIs or legacy code behind well designed facades

## Downsides

  * Only selected interfaces of an API will be exposed
  * The facade needs to be updated to offer more functionality from the underlying system

## UML Diagram

[<img loading="lazy" class="aligncenter wp-image-489" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram.jpg" alt="Facade pattern UML diagram" width="700" height="349" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram.jpg 1420w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram-300x150.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram-768x383.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram-1024x511.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Facade-pattern-UML-diagram.jpg)

The UML diagram for the facade pattern is pretty empty. It only shows the facade which is called by a client and that it calls methods of subsystems. These subsystems can be classes within your own system but also third party libraries or calls to web services.

## Implementation without the Facade Pattern

In this example, I am implementing a fake API which provides information for books. The API has three methods: LookUpAuthor, LookUpPublisher and LookUpTitle. All three methods take the ISBN as string Parameter and also return a string. To get the information for a specific ISBN, every method has to be called.

<div id="attachment_447" style="width: 538px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/WithoutFacadePattern.jpg"><img aria-describedby="caption-attachment-447" loading="lazy" class="size-full wp-image-447" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/WithoutFacadePattern.jpg" alt="Implementation of API calls without the Facade Pattern" width="528" height="308" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/WithoutFacadePattern.jpg 528w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/WithoutFacadePattern-300x175.jpg 300w" sizes="(max-width: 528px) 100vw, 528px" /></a>
  
  <p id="caption-attachment-447" class="wp-caption-text">
    Implementation of API calls without the Facade Pattern
  </p>
</div>

This is ok when you have three methods but what if you also want to get information about the year of publication, related books from the author or an excerpt of the book? This would increase the methods very quickly and as a result, would bloat the code. Imagine this code with 10 services for every attribute of a book. This would be nasty to use. The solution to this problem is to implement a facade.

You can find the source code of this example on [GitHub](https://github.com/WolfgangOfner/WithoutFacadePattern).

## Implementation of the Facade Pattern

To hide all the service calls, I implement a facade which takes the ISBN as Parameter and returns all values from the service calls. Additionally to the facade, I implement a Book class, which holds all the information about a book and which is returned by the facade. It is quite common that facades have helper classes which contain all the information from different method calls.

Implementing a facade is pretty simple. I move all the service calls into a separate class called Book Service. This class exposes only one method, LookUpBookInformation which takes the ISBN as a parameter and returns a book object. Inside the LookUpBookInformation method, I implement all the service calls which were in the main method before. The return value of every service call is mapped to a property of the book object.

<div id="attachment_448" style="width: 468px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-the-BookService-class.jpg"><img aria-describedby="caption-attachment-448" loading="lazy" class="size-full wp-image-448" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-the-BookService-class.jpg" alt="Implementation of the BookService class" width="458" height="287" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-the-BookService-class.jpg 458w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Implementation-of-the-BookService-class-300x188.jpg 300w" sizes="(max-width: 458px) 100vw, 458px" /></a>
  
  <p id="caption-attachment-448" class="wp-caption-text">
    Implementation of the BookService class
  </p>
</div>

With this implementation, i can tidy up the main method and end up with only one method call.

<div id="attachment_449" style="width: 562px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Invoking-the-BookService.jpg"><img aria-describedby="caption-attachment-449" loading="lazy" class="size-full wp-image-449" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Invoking-the-BookService.jpg" alt="Invoking the BookService" width="552" height="213" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Invoking-the-BookService.jpg 552w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/12/Invoking-the-BookService-300x116.jpg 300w" sizes="(max-width: 552px) 100vw, 552px" /></a>
  
  <p id="caption-attachment-449" class="wp-caption-text">
    Invoking the BookService
  </p>
</div>

You can find the source code of this example on <a href="https://github.com/WolfgangOfner/FacadePattern" target="_blank" rel="noopener">GitHub</a>.

## Conclusion

In this post, I showed the Advantages of using the Facade pattern. I also presented an example without the pattern and how to refactor the code to tidy it up.