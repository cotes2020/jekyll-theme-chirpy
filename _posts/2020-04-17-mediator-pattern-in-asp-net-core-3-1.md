---
title: Mediator Pattern in ASP .NET Core 3.1
date: 2020-04-17T14:05:38+02:00
author: Wolfgang Ofner
categories: [Design Pattern, ASP.NET]  
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, mediator, MediatR, microservice, RabbitMQ, Swagger]
---
The mediator pattern is a behavioral design pattern that helps to reduce chaotic dependencies between objects. The main goal is to disallow direct communication between the objects and instead force them to communicate only via the mediator.

## Problem

Services or classes often have several dependencies on other classes and quickly you end up with a big chaos of dependencies. The mediator pattern serves as an organizer and calls all needed services. No service has a dependency on another one, only on the mediator.

You can see the mediator pattern also in real life. Think about a big airport like JFK with many of arriving and departing planes. They all need to be coordinated to avoid crashes. It would be impossible for a plan to talk to all other planes. Instead, they call the tower, their mediator, and the tower talks to all planes and organizes who goes where.

## Advantages and Disadvantages of the Mediator Pattern

The mediator pattern brings a couple of advantages:

  * Less coupling: Since the classes don&#8217;t have dependencies on each other, they are less coupled.
  * Easier reuse: Fewer dependencies also helps to reuse classes.
  * Single Responsibility Principle: The services don&#8217;t have any logic to call other services, therefore they only do one thing.
  * Open/closed principle: Adding new mediators can be done without changing the existing code.

There is also one big disadvantage of the mediator pattern:

  * The mediator can become such a crucial factor in your application that it is called a &#8220;god class&#8221;.

## Implementation of the Mediator Pattern

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.

In my microservices, the controllers call all needed services and therefore work as the mediator. Additionally, I am using the MediatR NuGet package, which helps to call the services.

### Installing MediatR

I am installing the MediatR and the MediatR.Extensions.Microsoft.DependencyInjection in my Api project. In the Startup class, I registered my mediators using:

[code language=&#8221;CSharp&#8221;]  
services.AddMediatR(Assembly.GetExecutingAssembly());  
[/code]

I can do this because the controllers are in the same project. In the OrderApi, I am also using the ICustomerNameUpdateService interface as mediator. Therefore, I also have to register it.

[code language=&#8221;CSharp&#8221;]  
services.AddMediatR(Assembly.GetExecutingAssembly(), typeof(ICustomerNameUpdateService).Assembly);  
[/code]

Now, I can use the IMediator object with dependency injection in my controllers.

[code language=&#8221;CSharp&#8221;]  
public class OrderController : ControllerBase  
{  
private readonly IMapper _mapper;  
private readonly IMediator _mediator;

public OrderController(IMapper mapper, IMediator mediator)  
{  
_mapper = mapper;  
_mediator = mediator;  
}  
[/code]

### Using the Mediator pattern

Every call consists of a request and a handler. The request is sent to the handler which processes this request. A request could be a new object which should be saved in the database or an id of on object which should be retrieved. I am using [CQRS](https://www.programmingwithwolfgang.com/cqrs-in-asp-net-core-3-1), therefore my requests are either a query for read operations or a command for a write operation.

In the OrderController, I have the Order method which will create a new Order object. To create the Order, I create a CreateOrderCommand and map the Order from the post request to the Order of the CreateOrderCommandObject. Then I use the Send method of the mediator.

[code language=&#8221;CSharp&#8221;]  
[HttpPost]  
public async Task<ActionResult<Order>> Order([FromBody] OrderModel orderModel)  
{  
try  
{  
return await _mediator.Send(new CreateOrderCommand  
{  
Order = _mapper.Map<Order>(orderModel)  
});  
}  
catch (Exception ex)  
{  
return BadRequest(ex.Message);  
}  
}  
[/code]

The request (or query and command in my case) inherit from IRequest<T> interface which where T indicates the return value. If you don&#8217;t have a return value, then inherit from IRequest.

[code language=&#8221;CSharp&#8221;]  
public class CreateOrderCommand : IRequest<Order>  
{  
public Order Order { get; set; }  
}  
[/code]

The send method sends the object to the CreateOrderCommmandHandler. The handler inherits from IRequestHandler<TRequest, TResponse> and implements a Handle method. This Handle method processes the CreateOrderCommand. In this case, it calls the AddAsync method of the repository and passes the Order.

[code language=&#8221;CSharp&#8221;]  
public class CreateOrderCommandHandler : IRequestHandler<CreateOrderCommand, Order>  
{  
private readonly IOrderRepository _orderRepository;

public CreateOrderCommandHandler(IOrderRepository orderRepository)  
{  
_orderRepository = orderRepository;  
}

public async Task<Order> Handle(CreateOrderCommand request, CancellationToken cancellationToken)  
{  
return await _orderRepository.AddAsync(request.Order);  
}  
}  
[/code]

If you don&#8217;t have a return value, the handler inherits from IRequestHandler<TRequest>.

## Conclusion

The mediator pattern is a great pattern to reduce the dependencies within your application which helps you to reuse your components and also to keep the Single Responsible Principle. I showed I implemented it in my ASP .NET Core 3.1 microservices using the MediatR NuGet package.

<a href="/rabbitmq-in-an-asp-net-core-3-1-microservice" target="_blank" rel="noopener noreferrer">In my next post, I will implement RabbitMQ</a> which enables my microservices to exchange data in a decoupled asynchronous way.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.