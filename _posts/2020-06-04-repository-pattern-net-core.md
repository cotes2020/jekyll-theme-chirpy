---
title: Repository Pattern in .Net Core
date: 2020-06-04T23:08:45+02:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: [.net core, 'C#', entity framework core]
---
A couple of years ago, I wrote about the <a href="/repository-and-unit-of-work-pattern/" target="_blank" rel="noopener noreferrer">Repository and Unit of Work pattern</a>. Although this post is quite old and not even .net core, I get many questions about it. Since the writing of the post, .Net core matured and I learned a lot about software development. Therefore, I wouldn&#8217;t implement the code as I did back then. Today, I will write about implementing .the repository pattern in .Net core

## Why I am changing the Repository Pattern in .Net Core

Entity Framework Core already serves as unit of work. Therefore you don&#8217;t have to implement it yourself. This makes your code a lot simpler and easier to understand.

## The Repository Pattern in .Net Core

For the demo, I am creating a simple 3-tier application consisting of controller, services, and repositories. The repositories will be injected into the services using the built-in dependency injection. You can find the code for the demo on <a href="https://github.com/WolfgangOfner/.NetCoreRepositoryAndUnitOfWorkPattern" target="_blank" rel="noopener noreferrer">Github</a>.

In the data project, I have my models and repositories. I create a generic repository that takes a class and offers methods like get, add, or update.

### Implementing the Repositories

```csharp  
public class Repository<TEntity> : IRepository<TEntity> where TEntity : class, new()
{
    protected readonly RepositoryPatternDemoContext RepositoryPatternDemoContext;

    public Repository(RepositoryPatternDemoContext repositoryPatternDemoContext)
    {
        RepositoryPatternDemoContext = repositoryPatternDemoContext;
    }

    public IQueryable<TEntity> GetAll()
    {
        try
        {
            return RepositoryPatternDemoContext.Set<TEntity>();
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
            await RepositoryPatternDemoContext.AddAsync(entity);
            await RepositoryPatternDemoContext.SaveChangesAsync();

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
            RepositoryPatternDemoContext.Update(entity);
            await RepositoryPatternDemoContext.SaveChangesAsync();

            return entity;
        }
        catch (Exception ex)
        {
            throw new Exception($"{nameof(entity)} could not be updated: {ex.Message}");
        }
    }
} 
```

This repository can be used for most entities. In case one of your models needs more functionality, you can create a concrete repository that inherits from Repository. I created a ProductRepository which offers product-specific methods:

```csharp  
public class ProductRepository : Repository<Product>, IProductRepository
{
    public ProductRepository(RepositoryPatternDemoContext repositoryPatternDemoContext) : base(repositoryPatternDemoContext)
    {
    }

    public Task<Product> GetProductByIdAsync(int id)
    {
        return GetAll().FirstOrDefaultAsync(x => x.Id == id);
    }
}  
```

The ProductRepository also offers all generic methods because its interface IProductRepository inherits from IRepository:

```csharp  
public interface IProductRepository : IRepository<Product>  
{  
    Task<Product> GetProductByIdAsync(int id);  
}  
```

The last step is to register the generic and concrete repositories in the Startup class.

```csharp  
services.AddTransient(typeof(IRepository<>), typeof(Repository<>;));  
services.AddTransient<IProductRepository, ProductRepository>();  
services.AddTransient<ICustomerRepository, CustomerRepository>();  
```

The first line registers the generic attributes. This means if you want to use it in the future with a new model, you don&#8217;t have to register anything else. The second and third line register the concrete implementation of the ProductRepository and CustomerRepository.

### Implementing Services which use the Repositories

I implement two services, the CustomerService and the ProductService. Each service gets injected a repository. The ProductServices uses the IProductRepository and the CustomerService uses the ICustomerRepository;. Inside the services, you can implement whatever business logic your application needs. I implemented only simple calls to the repository but you could also have complex calculations and several repository calls in a single method.

```csharp  
public class CustomerService : ICustomerService  
{  
private readonly ICustomerRepository _customerRepository;

public CustomerService(ICustomerRepository customerRepository)  
{  
_customerRepository = customerRepository;  
}

public async Task<List<Customer>> GetAllCustomersAsync()  
{  
return await _customerRepository.GetAllCustomersAsync();  
}

public async Task<Customer> GetCustomerByIdAsync(int id)  
{  
return await _customerRepository.GetCustomerByIdAsync(id);  
}

public async Task<Customer> AddCustomerAsync(Customer newCustomer)  
{  
return await _customerRepository.AddAsync(newCustomer);  
}  
}  
```

```csharp  
public class CustomerService : ICustomerService
{
    private readonly ICustomerRepository _customerRepository;

    public CustomerService(ICustomerRepository customerRepository)
    {
        _customerRepository = customerRepository;
    }

    public async Task<List<Customer>> GetAllCustomersAsync()
    {
        return await _customerRepository.GetAllCustomersAsync();
    }

    public async Task<Customer> GetCustomerByIdAsync(int id)
    {
        return await _customerRepository.GetCustomerByIdAsync(id);
    }

    public async Task<Customer> AddCustomerAsync(Customer newCustomer)
    {
        return await _customerRepository.AddAsync(newCustomer);
    }
}  
```

## Implementing the Controller to test the Application

To test the application, I implemented a really simple controller. The controllers offer for each service method a parameter-less get method and return whatever the service returned. Each controller gets the respective service injected.

```csharp  
public class CustomerController : Controller
{
    private readonly ICustomerService _customerService;

    public CustomerController(ICustomerService customerService)
    {
        _customerService = customerService;
    }

    public async Task<ActionResult<Customer>> CreateCustomer()
    {
        var customer = new Customer
        {
            Age = 30,
            FirstName = "Wolfgang",
            LastName = "Ofner"
        };

        return await _customerService.AddCustomerAsync(customer);
    }

    public async Task<ActionResult<List<Customer>>> GetAllCustomers()
    {
        return await _customerService.GetAllCustomersAsync();
    }

    public async Task<ActionResult<Customer>> GetCustomerById()
    {
        return await _customerService.GetCustomerByIdAsync(1);
    }
}  
```

```csharp  
public class ProductController : Controller
{
    private readonly IProductService _productService;

    public ProductController(IProductService productService)
    {
        _productService = productService;
    }

    public async Task<ActionResult<Product>> GetProductById()
    {
        return await _productService.GetProductByIdAsync(1);
    }

    public async Task<ActionResult<Product>> CreateProduct()
    {
        var product = new Product
        {
            Name = "Name",
            Description = "Desc",
            Price = 99.99m
        };

        return await _productService.AddProductAsync(product);
    }
} 
```

When you call the create customer action, a customer object in JSON should be returned.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Test-the-creation-of-a-customer.jpg"><img aria-describedby="caption-attachment-2150" loading="lazy" class="size-full wp-image-2150" src="/assets/img/posts/2020/06/Test-the-creation-of-a-customer.jpg" alt="Test the creation of a customer Repository Pattern in .Net Core " /></a>
  
  <p>
    Test the creation of a customer
  </p>
</div>

## Use the database

If you want to use the a database, you have to add your connection string in the appsettings.json file. My connection string looks like this:

```json  
"ConnectionString": "Server=localhost;Database=RepositoryPatternDemo;Integrated Security=False;Persist Security Info=False;User ID=sa;Password=<;YourNewStrong@Passw0rd>;"  
```

By default, I am using an in-memory database. This means that you don&#8217;t have to configure anything to test the application

```csharp  
//services.AddDbContext<RepositoryPatternDemoContext>(options => options.UseSqlServer(Configuration["Database:ConnectionString"]));  
services.AddDbContext<RepositoryPatternDemoContext>(options => options.UseInMemoryDatabase(Guid.NewGuid().ToString()));  
```

I also added an SQL script to create the database, tables and test data. You can find the script <a href="https://github.com/WolfgangOfner/.NetCoreRepositoryAndUnitOfWorkPattern/blob/master/NetCoreRepositoryAndUnitOfWorkPattern.Data/DatabaseScript/database.sql" target="_blank" rel="noopener noreferrer">here</a>.

## Conclusion

In today&#8217;s post, I gave my updated opinion on the repository pattern and simplified the solution compared to my post a couple of years ago. This solution uses entity framework core as unit of work and implements a generic repository that can be used for most of the operations. I also showed how to implement a specific repository, in case the generic repository can&#8217;t full fill your requirements. Implement your own unit of work object only if you need to control over your objects.

You can find the code for the demo on <a href="https://github.com/WolfgangOfner/.NetCoreRepositoryAndUnitOfWorkPattern" target="_blank" rel="noopener noreferrer">Github</a>.