---
title: Programming a Microservice with .NET Core 3.1
date: 2020-04-13T17:53:40+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, Swagger]
---
In [my last post](https://www.programmingwithwolfgang.com/microservices-getting-started/), I talked about the theory of a microservice. Today it is going to be more practical. I will create two microservices using ASP .NET Core 3.1. Over the next posts., I will extend the microservices using CQRS, docker and docker-compose, RabbitMQ and automatic builds and deployments.

## Create a Microservice using ASP .NET Core 3.1

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>. I will talk about the key aspects of the microservice but not about every detail. You will need at least a basic understanding of C# and ASP.NET Core.

## To-do list for the Microservice

Our two microservice should satisfy the following requirements:

  * Implement a Customer API with the following methods: create customer, update customer
  * Implement an Order API with the following methods: create order, pay order, get all orders which had already been paid
  * When a customer name is updated, it should also be updated in the Order API
  * The APIs should not directly call each other to update data
  * ASP .NET Core 3.1 Web API with DI/IoC
  * Communication between microservices should be implemented through some kind of queue
  * Use DDD and CQRS approaches with the Mediator and Repository Pattern

To keep it simple, I will use an in-memory database. During the implementation, I will point out what you have to change if you want to use a normal database. I will split up the full implementation. In this post, I will create the microservices with the needed features. In the following posts, I will implement <a href="/document-your-microservice-with-swagger" target="_blank" rel="noopener noreferrer">Swagger</a>, create a <a href="/dockerize-an-asp-net-core-microservice-and-rabbitmq/" target="_blank" rel="noopener noreferrer">Docker container</a>, set up <a href="/rabbitmq-in-an-asp-net-core-3-1-microservice" target="_blank" rel="noopener noreferrer">RabbitMQ</a> and explain [CQRS](https://www.programmingwithwolfgang.com/cqrs-in-asp-net-core-3-1/) and <a href="/mediator-pattern-in-asp-net-core-3-1/" target="_blank" rel="noopener noreferrer">Mediator</a>.

## Structure of the Microservice

I created a solution for each microservice. You can see the structure of the microservices on the following screenshot.

<div id="attachment_1867" style="width: 264px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Structure-of-the-Order-Microservice.jpg"><img aria-describedby="caption-attachment-1867" loading="lazy" class="wp-image-1867 size-full" title="Structure of the Order Microservice" src="/assets/img/posts/2020/04/Structure-of-the-Order-Microservice.jpg" alt="Structure of the Order Microservice" width="254" height="212" /></a>
  
  <p id="caption-attachment-1867" class="wp-caption-text">
    Structure of the Order Microservice
  </p>
</div>

Both microservice have exactly the same structure, except that the order solution has a Messaging.Receive project and the customer solution has a Messaging.Send project. I will use these projects later to send and receive data using RabbitMQ.

An important aspect of an API is that you don&#8217;t know who your consumers are and you should never break existing features. To implement versioning, I place everything like controllers or models in a v1 folder. If I want to extend my feature and it is not breaking the current behavior, I will extend it in the already existing classes. If my changes were to break the functionality, I will create a v2 folder and place the changes there. With this approach, no consumer is affected and they can implement the new features whenever they want or need them.

## The API Project

The API project is the heart of the application and contains the controllers, validators, and models as well as the startup class in which all dependencies are registered.

### Controllers in the API Project

I try to keep the controller methods as simple as possible. They only call different services and return a model or status to the client. They don&#8217;t do any business logic.

```csharp
[HttpPost]
public async Task<ActionResult<Customer>> Customer([FromBody] CreateCustomerModel createCustomerModel)
{
    try
    {
        return await _mediator.Send(new CreateCustomerCommand
        {
            Customer = _mapper.Map<Customer>(createCustomerModel)
        });
    }
    catch (Exception ex)
    {
        return BadRequest(ex.Message);
    }
} 
```

The _mediator.Send is used to call a service using CQRS and the Mediator pattern. I will explain that in a later post. For now, it is important to understand that a service is called and that a Customer is returned. In case of an exception, a bad request and an error message are returned.

My naming convention is that I use the name of the object, in that case, Customer. The HTTP verb will tell you what this action does. In this case, the post will create an object, whereas put would update an existing customer.

### Validators

To validate the user input, I use the NuGet FluentValidations and a validator per model. Your validator inherits from AbstractValidator<T> where T is the class of the model you want to validate. Then you can add rules in the constructor of your validator. The validator is not really important for me right now and so I try to keep it simple and only validate that the first and last name has at least two characters and that the age and birthday are between zero and 150 years. I don&#8217;t validate if the birthday and the age match. This should be changed in the future.

```csharp 
public class CreateCustomerModelValidator : AbstractValidator<CreateCustomerModel>
{
    public CreateCustomerModelValidator()
    {
        RuleFor(x => x.FirstName)
            .NotNull()
            .WithMessage("The first name must be at least 2 character long");
        RuleFor(x => x.FirstName)
            .MinimumLength(2).
            WithMessage("The first name must be at least 2 character long");
        
        RuleFor(x => x.LastName)
            .NotNull()
            .WithMessage("The last name must be at least 2 character long");
        RuleFor(x => x.LastName)
            .MinimumLength(2)
            .WithMessage("The last name must be at least 2 character long");

        RuleFor(x => x.Birthday)
            .InclusiveBetween(DateTime.Now.AddYears(-150).Date, DateTime.Now)
            .WithMessage("The birthday must not be longer ago than 150 years and can not be in the future");
            
        RuleFor(x => x.Age)
            .InclusiveBetween(0, 150)
            .WithMessage("The minimum age is 0 and the maximum age is 150 years");
    }
}  
```

### Startup

In the Startup.cs, I register my services, validators and configure other parts of the application like AutoMapper, the database context or Swagger. This part should be self-explanatory and I will talk about <a href="/document-your-microservice-with-swagger" target="_blank" rel="noopener noreferrer">Swagger</a> or <a href="/rabbitmq-in-an-asp-net-core-3-1-microservice" target="_blank" rel="noopener noreferrer">RabbitMQ</a> later.

<div id="attachment_2475" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Register-the-classes-and-configure-the-services.jpg"><img aria-describedby="caption-attachment-2475" loading="lazy" class="wp-image-2475" src="/assets/img/posts/2020/04/Register-the-classes-and-configure-the-services.jpg" alt="Register the classes and configure the services" width="700" height="439" /></a>
  
  <p id="caption-attachment-2475" class="wp-caption-text">
    Register the classes and configure the services
  </p>
</div>

## Data

The Data project contains everything needed to access the database. I use Entity Framework Core, an in-memory database and the repository pattern.

### Database Context

In the database context, I add a list of customers that I will use to update an existing customer. The database context is created for every request, therefore updated or created customers will be lost after the request. This behavior is fine for the sake of this demo.

```csharp
public CustomerContext(DbContextOptions<CustomerContext> options) : base(options)
{
    var customers = new[]
    {
        new Customer
        {
            Id = Guid.Parse("9f35b48d-cb87-4783-bfdb-21e36012930a"),
            FirstName = "Wolfgang",
            LastName = "Ofner",
            Birthday = new DateTime(1989, 11, 23),
            Age = 30
        },
        new Customer
        {
            Id = Guid.Parse("654b7573-9501-436a-ad36-94c5696ac28f"),
            FirstName = "Darth",
            LastName = "Vader",
            Birthday = new DateTime(1977, 05, 25),
            Age = 43
        },
        new Customer
        {
            Id = Guid.Parse("971316e1-4966-4426-b1ea-a36c9dde1066"),
            FirstName = "Son",
            LastName = "Goku",
            Birthday = new DateTime(1937, 04, 16),
            Age = 83
        }
    };

    Customer.AddRange(customers);
    SaveChanges();
}

public DbSet<Customer> Customer { get; set; } 
```

If you want to use a normal database, all you have to do is delete the adding of customers in the constructor and change the following line in the Startup class to take your connection string instead of using an in-memory database.

```csharp  
services.AddDbContext<CustomerContext>(options => options.UseInMemoryDatabase(Guid.NewGuid().ToString()));  
```

You can either hard-code your connection string in the Startup class or better, read it from the appsettings.json file.

```csharp  
services.AddDbContext<CustomerContext>(options => options.UseSqlServer(Configuration["Database:ConnectionString"]));  
```

### Repository

I have a generic repository for CRUD operations which can be used for every entity. This repository has methods like AddAsync and UpdateAsync.

```csharp  
public class Repository<TEntity> : IRepository<TEntity> where TEntity : class, new()
{
    protected readonly CustomerContext CustomerContext;

    public Repository(CustomerContext customerContext)
    {
        CustomerContext = customerContext;
    }

    public IEnumerable<TEntity> GetAll()
    {
        try
        {
            return CustomerContext.Set<TEntity>();
        }
        catch (Exception ex)
        {
            throw new Exception($"Couldn't retrieve entities: {ex.Message}");
        }
    }

    public async Task<TEntity> AddAsync(TEntity entity)
    {
        if (entity == null)
        {
            throw new ArgumentNullException($"{nameof(AddAsync)} entity must not be null");
        }

        try
        {
            await CustomerContext.AddAsync(entity);
            await CustomerContext.SaveChangesAsync();

            return entity;
        }
        catch (Exception ex)
        {
            throw new Exception($"{nameof(entity)} could not be saved: {ex.Message}");
        }
    }

    public async Task<TEntity> UpdateAsync(TEntity entity)
    {
        if (entity == null)
        {
            throw new ArgumentNullException($"{nameof(AddAsync)} entity must not be null");
        }

        try
        {
            CustomerContext.Update(entity);
            await CustomerContext.SaveChangesAsync();

            return entity;
        }
        catch (Exception ex)
        {
            throw new Exception($"{nameof(entity)} could not be updated {ex.Message}");
        }
    }
} 
```

Additionally to the generic repository, I have a CustomerRepository that implements a Customer specific method, GetCustomerByIdAsync.

```csharp  
public class CustomerRepository : Repository<Customer>, ICustomerRepository
{
    public CustomerRepository(CustomerContext customerContext) : base(customerContext)
    {
    }

    public async Task<Customer> GetCustomerByIdAsync(Guid id, CancellationToken cancellationToken)
    {
        return await CustomerContext.Customer.FirstOrDefaultAsync(x => x.Id == id, cancellationToken);
    }
} 
```

The OrderRepository has more Order specific methods. The CustomerRepository inherits from the generic repository and its interface inherits from the repository interface. Since the CustomerContext has the protected access modified in the generic repository, I can reuse it in my CustomerRepository.

## Domain

The Domain project contains all entities and no business logic. In my microservice, this is only the Customer or Order entity.

## Messaging.Send

The Messaging.Send project contains everything I need to send Customer objects to a RabbitMQ queue. I will talk about the specifics of the implementation in a later post.

```csharp  
public void SendCustomer(Customer customer)
{
    var factory = new ConnectionFactory() { HostName = _hostname, UserName = _username, Password = _password };
    
    using (var connection = factory.CreateConnection())
    using (var channel = connection.CreateModel())
    {
        channel.QueueDeclare(queue: _queueName, durable: false, exclusive: false, autoDelete: false, arguments: null);

        var json = JsonConvert.SerializeObject(customer);
        var body = Encoding.UTF8.GetBytes(json);

        channel.BasicPublish(exchange: "", routingKey: _queueName, basicProperties: null, body: body);
    }
}  
```

## Service

The Service project is split into Command and Query. This is how CQRS separates the concerns of reading and writing data. I will go into the details in a later post. For now, all you have to know is that commands write data and queries read data. A query consists of a query and a handler. The query indicates what action should be executed and the handler implements this action. The command works with the same principle.

```csharp  
public class CreateCustomerCommandHandler : IRequestHandler<CreateCustomerCommand, Customer>
{
    private readonly ICustomerRepository _customerRepository;

    public CreateCustomerCommandHandler(ICustomerRepository customerRepository)
    {
        _customerRepository = customerRepository;
    }

    public async Task<Customer> Handle(CreateCustomerCommand request, CancellationToken cancellationToken)
    {
        return await _customerRepository.AddAsync(request.Customer);
    }
}  
```

The handler often calls the repository to retrieve or change data.

## Tests

For my tests, I like to create a test project for each normal project wheres the name is the same except that I add .Test at the end. I use xUnit, FakeItEasy, and FluentAssertions. Currently, there are no tests for the RabbitMQ logic.

## Run the Microservice

In the previous section I only talked about the Customer service but the Order service has the same structure and should be easy to understand.

Now that the base functionality is set up, it is time to test both microservice. Before you can start them, you have to make two small changes to be actually able to start them. Currently, we have no queue and therefore the microservices will generate an exception. In the future, it would be nice if the microservices could work even without a queue.

## Edit the Customer Service

Open the CustomerCommandHandler in the Service project and comment out the following line _customerUpdateSender.SendCustomer(customer);

```csharp  
public async Task<Customer> Handle(UpdateCustomerCommand request, CancellationToken cancellationToken)
{
    var customer = await _customerRepository.UpdateAsync(request.Customer);

    // _customerUpdateSender.SendCustomer(customer);

    return customer;
}  
```

This line is responsible for publishing the Customer to the queue

## Edit the Order Service

In the Order API, you have to comment out services.AddHostedService<CustomerFullNameUpdateReceiver>(); in the Startup class of the API project.

```csharp  
services.AddTransient<IRequestHandler<GetPaidOrderQuery, List<Order>>, GetPaidOrderQueryHandler>();  
services.AddTransient<IRequestHandler<GetOrderByIdQuery, Order>, GetOrderByIdQueryHandler>();  
services.AddTransient<IRequestHandler<GetOrderByCustomerGuidQuery, List<Order>>, GetOrderByCustomerGuidQueryHandler>();  
services.AddTransient<IRequestHandler<CreateOrderCommand, Order>, CreateOrderCommandHandler>();  
services.AddTransient<IRequestHandler<PayOrderCommand, Order>, PayOrderCommandHandler>();  
services.AddTransient<IRequestHandler<UpdateOrderCommand>, UpdateOrderCommandHandler>();  
services.AddTransient<ICustomerNameUpdateService, CustomerNameUpdateService>();

// services.AddHostedService<CustomerFullNameUpdateReceiver>();  
```

This line would register a background service that listens to change in the queue and would pull these changes.

## Test the Microservice

After you made the changes to both APIs, you can start them. This should display the Swagger GUI which gives you information about all actions and models and also lets you send requests. The GUI should be self-explanatory but <a href="/document-your-microservice-with-swagger" target="_blank" rel="noopener noreferrer">I will talk more about it in my next post</a>.

<div id="attachment_1881" style="width: 556px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/The-Swagger-GUI-with-the-available-actions-and-models.jpg"><img aria-describedby="caption-attachment-1881" loading="lazy" class="size-full wp-image-1881" src="/assets/img/posts/2020/04/The-Swagger-GUI-with-the-available-actions-and-models.jpg" alt="The Swagger GUI with the available actions and models" width="546" height="627" /></a>
  
  <p id="caption-attachment-1881" class="wp-caption-text">
    The Swagger GUI with the available actions and models
  </p>
</div>

## Conclusion

Today, I talked about the structure and the features of my microservices. This is just the beginning but both applications are working and could be already deployed. It is important to keep in mind that each microservice has its own data storage and is kept as simple as possible.

<a href="/document-your-microservice-with-swagger" target="_blank" rel="noopener noreferrer">In my next post, I will talk about Swagger</a> and how you can use it to easily and quickly document your microservice while providing the opportunity to test requests.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.