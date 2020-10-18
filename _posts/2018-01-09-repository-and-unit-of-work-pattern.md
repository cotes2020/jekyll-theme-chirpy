---
title: Repository and Unit of Work Pattern
date: 2018-01-09T19:32:51+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: ['C#',entity framework, software architecture]
---
The Repository pattern and Unit of Work pattern are used together most of the time. Therefore I will combine them in this post and show how to implement them both.

## Definition Repository

The Repository mediates between the domain and data mapping layers, acting like an in-memory collection of domain objects. (<a href="http://amzn.to/2CGvtzn" target="_blank" rel="noopener">&#8220;Patterns of Enterprise Application Architecture&#8221;</a> by Martin Fowler)

## Repository Pattern Goals

  * Decouple Business code from data Access. As a result, the persistence Framework can be changed without a great effort
  * Separation of Concerns
  * Minimize duplicate query logic
  * Testability

## Introduction

The Repository pattern is often used when an application performs data access operations. These operations can be on a database, Web Service or file storage. The repository encapsulates These operations so that it doesn&#8217;t matter to the business logic where the operations are performed. For example, the business logic performs the method GetAllCustomers() and expects to get all available customers. The application doesn&#8217;t care whether they are loaded from a database or web service.

The repository should look like an in-memory collection and should have generic methods like Add, Remove or FindById. With such generic methods, the repository can be easily reused in different applications.

Additionally to the generic repository, one or more specific repositories, which inherit from the generic repository, are implemented. These specialized repositories have methods which are needed by the application. For example, if the application is working with customers the CustomerRepository might have a method GetCustomersWithHighestRevenue.

With the data access set up, I need a way to keep track of the changes. Therefore I use the Unit of Work pattern.

## Definition Unit of Work

Maintains a list of objects affected by a business transaction and coordinates the writing out of changes.  (<a href="http://amzn.to/2CGvtzn" target="_blank" rel="noopener">&#8220;Patterns of Enterprise Application Architecture&#8221;</a> by Martin Fowler)

## Consequences of the Unit of Work Pattern

  * Increases the level of abstraction and keep business logic free of data access code
  * Increased maintainability, flexibility and testability
  * More classes and interfaces but less duplicated code
  * The business logic is further away from the data because the repository abstracts the infrastructure. This has the effect that it might be harder to optimize certain operations which are performed against the data source.

## Does Entity Framework implement the Repository and Unit of Work Pattern?

Entity Framework has a DbSet class which has Add and Remove method and therefore looks like a repository.  the DbContext class has the method SaveChanges and so looks like the unit of work. Therefore I thought that it is possible to use entity framework and have all the Advantages of the Repository and Unit of Work pattern out of the box. After taking a deeper look, I realized that that&#8217;s not the case.

The problem with DbSet is that its Linq statements are often big queries which are repeated all over the code. This makes to code harder to read and harder to change. Therefore it does not replace a repository.

The problem when using DbContext is that the code is tightly coupled to entity framework and therefore is really hard, if not impossible to replace it if needed.

## Repository Pattern and Unit of Work Pattern UML Diagram

<div id="attachment_529" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram.jpg"><img aria-describedby="caption-attachment-529" loading="lazy" class="wp-image-529" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram.jpg" alt="Repository pattern UML diagram" width="700" height="484" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram.jpg 1156w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram-300x208.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram-768x531.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-pattern-UML-diagram-1024x709.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-529" class="wp-caption-text">
    Repository pattern UML diagram
  </p>
</div>

The Repository pattern consists of one IRepository which contains all generic operations like Add or Remove. It is implemented by the Repository and by all IConcreteRepository interfaces. Every IConcreteRepository interface is implemented by one ConcreteRepository class which also derives from the Repository class. With this implementation, the ConcreteRepositoy has all generic methods and also the methods for the specific class. As an example: the CustomerRepository could implement a method which is called GetAllSeniorCustomers or GetBestCustomersByRevenue.

[<img loading="lazy" class="aligncenter wp-image-530" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Unit-of-Work-pattern-UML-diagram.jpg" alt="Unit of Work pattern UML diagram" width="292" height="400" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Unit-of-Work-pattern-UML-diagram.jpg 560w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Unit-of-Work-pattern-UML-diagram-219x300.jpg 219w" sizes="(max-width: 292px) 100vw, 292px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Unit-of-Work-pattern-UML-diagram.jpg)

The unit of work provides the ability to save the changes to the storage (whatever this storage is). The IUnitOfWork interface has a method for saving which is often called Complete and every concrete repository as property. For example, if I have the repository ICustomerRepository then the IUnitOfWork has an ICustomerRepositry property with a getter only. Additionally, IUnitOfWork inherits from IDisposable.

The UnitOfWork class implements the Complete method, in which the data get saved to the data storage. The advantage of this implementation is that wherever you want to save something you only have to call the Complete method from UnitOfWork and don&#8217;t care about where it gets saved.

## Implementation of the Repository and Unit of Work Pattern

For this example, I created a console project (RepositoryAndUnitOfWork) and a class library (RepositoryAndUnitOfWork.DataAccess). In the class library, I generate a database with a customer table.

## [<img loading="lazy" class="aligncenter size-full wp-image-528" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Customer-Table.jpg" alt="" width="393" height="228" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Customer-Table.jpg 393w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Customer-Table-300x174.jpg 300w" sizes="(max-width: 393px) 100vw, 393px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Customer-Table.jpg)

Next, I let Entity Framework generate the data model from the database. If you don&#8217;t know how to do that, check the <a href="https://msdn.microsoft.com/en-us/library/jj206878(v=vs.113).aspx" target="_blank" rel="noopener">documentation</a> for a step by step walkthrough.

### Implementing Repositories

After setting up the database, it&#8217;s time to implement the repository. To do that, I create a new folder, Repositories, in the class library project and add a new interface IRepositry. In this Interface, I add all generic methods I want to use later in my applications. These methods are, GetById, Add, AddRange, Remove or Find. To make the Interface usable for all classes I use the generic type parameter T, where T is a class.

[<img loading="lazy" class="aligncenter size-full wp-image-516" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IRepository.jpg" alt="IRepository" width="453" height="355" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IRepository.jpg 453w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IRepository-300x235.jpg 300w" sizes="(max-width: 453px) 100vw, 453px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IRepository.jpg)

After the generic repository, I also implement a specific repository for the customer. The ICustomerRepository inherits from IRepository and only implements one method.

[<img loading="lazy" class="aligncenter size-full wp-image-517" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/ICustomerRepository.jpg" alt="ICustomerRepository" width="471" height="87" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/ICustomerRepository.jpg 471w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/ICustomerRepository-300x55.jpg 300w" sizes="(max-width: 471px) 100vw, 471px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/ICustomerRepository.jpg)

After implementing all interfaces it&#8217;s time to implement concrete repository classes. First, I create a class Repository which inherits from IRepository. In this class, I implement all methods from the interface. Additionally to the methods, I have a constructor which takes a DbContext as Parameter. This DbContext instantiates a DbSet which will be used to get or add data.

[<img loading="lazy" class="aligncenter size-full wp-image-518" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository.jpg" alt="Repository" width="448" height="151" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository.jpg 448w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository-300x101.jpg 300w" sizes="(max-width: 448px) 100vw, 448px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Repository.jpg)

The implementations of the methods are pretty straight Forward. The only interesting one might be the Find method which takes an expression as parameter. In the implementation, I use Where to find all entries which fit the Expression of the parameter.

[<img loading="lazy" class="aligncenter size-full wp-image-519" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Find-method.jpg" alt="Find method" width="454" height="284" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Find-method.jpg 454w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Find-method-300x188.jpg 300w" sizes="(max-width: 454px) 100vw, 454px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Find-method.jpg)

The final step for the Repository pattern is to implement the CustomerReposiotry. This class derives from Repository and ICustomerRepository and implements the method from the interface. The constructor takes a CustomerDbEntities object as Parameter which is derived from DbContext and generated by Entity Framework.

[<img loading="lazy" class="aligncenter size-full wp-image-521" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/CustomerRepository.jpg" alt="CustomerRepository" width="797" height="252" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/CustomerRepository.jpg 797w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/CustomerRepository-300x95.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/CustomerRepository-768x243.jpg 768w" sizes="(max-width: 797px) 100vw, 797px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/CustomerRepository.jpg)

### Implementing Unit of Work

All repositories are created now, but I need a class which writes my data to the database, the unit of work. To implement this class, I first implement the IUnitOfWork interface in the repositories folder in the library project. This interface derives from IDisposable and has an ICustomerRepository property and the method Complete. This method is responsible for saving changes. The Name of the method could be Save, Finish or whatever you like best.

[<img loading="lazy" class="aligncenter size-full wp-image-523" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IUnitOfWork.jpg" alt="IUnitOfWork" width="312" height="144" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IUnitOfWork.jpg 312w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IUnitOfWork-300x138.jpg 300w" sizes="(max-width: 312px) 100vw, 312px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/IUnitOfWork.jpg)

Like before, I add the concrete implementation of IUnitOfWork to the repositories folder in the console application project. The constructor takes a CustomerDbEnties object as parameter and also initializes the ICustomerRepository. The Complete Method saves the context with SaveChanges and the Dispose method disposes changes.

[<img loading="lazy" class="aligncenter size-full wp-image-524" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/UnitOfWork.jpg" alt="UnitOfWork" width="391" height="408" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/UnitOfWork.jpg 391w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/UnitOfWork-288x300.jpg 288w" sizes="(max-width: 391px) 100vw, 391px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/UnitOfWork.jpg)

### Using the Repository and Unit of Work

The usage of the unit of work differs between a web application and a console application. In an MVC application, the unit of work gets injected into the constructor. In the console application, I have to use a using statement. I can use with unitOfWork.Customer.Method(), for example unitOfWork.GetBestCustomers(3). To save the changes use unitOfWork.Complete().

[<img loading="lazy" class="aligncenter wp-image-527" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Using-the-unit-of-work.jpg" alt="Using the unit of work" width="800" height="204" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Using-the-unit-of-work.jpg 1006w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Using-the-unit-of-work-300x77.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Using-the-unit-of-work-768x196.jpg 768w" sizes="(max-width: 800px) 100vw, 800px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/01/Using-the-unit-of-work.jpg)

You can find the source code on <a href="https://github.com/WolfgangOfner/RepositoryAndUnitOfWorkPattern" target="_blank" rel="noopener">GitHub</a>. If you want to try out the example, you have to change to connection string in the App.config to the location of the database on your computer

Note: In this example, I always talked about writing and reading data from the database. The storage location could also be a web service or file drive. If you want to try my example, you have to change the connection string for the database in the App.config file to the Location of the database on your computer.

## Conclusion

In this post, I showed how to implement the Repository and Unit of Work pattern. Implementing both patterns results in more classes but the advantages of abstraction increased testability and increased maintainability outweigh the disadvantages. I also talked about entity framework and that although it looks like an out of the box Repository and Unit of Work pattern, it comes at the cost of tight coupling to the framework and should not be used to replace the patterns.