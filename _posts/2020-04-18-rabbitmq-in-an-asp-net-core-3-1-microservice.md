---
title: RabbitMQ in an ASP .NET Core 3.1 Microservice
date: 2020-04-18T11:38:51+02:00
author: Wolfgang Ofner
categories: [Docker, ASP.NET]
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, Swagger]
---
[In my last posts](https://www.programmingwithwolfgang.com/document-your-microservice-with-swagger), I created two microservices using ASP .NET Core 3.1. Today, I will implement RabbitMQ, so the microservices can exchange data while staying independent. RabbitMQ can also be used to publish data even without knowing the subscribers. This means that you can publish an update and whoever is interested can get the new information.

## What is RabbitMQ and why use it?

RabbitMQ describes itself as the most widely deployed open-source message broker. It is easy to implement and supports a wide variety of technologies like Docker, .NET or Go. It also offers plugins for monitoring, managing or authentication. I chose RabbitMQ because it is well known, quickly implemented and especially can be easily run using Docker.

## Why use a Queue to send data?

Now that you know what RabbitMQ is, the next question is: why should you use a queue instead of directly sending data from one microservice to the other one. There are a couple of reasons why using a queue instead of directly sending data is better:

  * Higher availability and better error handling
  * Better scalability
  * Share data with whoever wants/needs it
  * Better user experience due to asynchronous processing

### Higher availability and better error handling

Errors happen all the time. In the past, we designed our systems to avoid errors. Nowadays we design our systems to catch errors and handle them in a user-friendly way.

Let&#8217;s say we have an online shop and the order services send data to the process order service after the customer placed his order. If these services are connected directly and the process order service is offline, the customer will get an error message, for example, &#8220;An error occurred. Please try it again later&#8221;. This user probably won&#8217;t return and you lost the revenue of the order and a customer for the future.

If the order service places the order in a queue and the order processing service is offline, the customer will get a message that the order got placed and he or she might come back in the future. When the order processing service is back online, it will process all entries and the queue. You might know this behavior when booking a flight. After booking the flight it takes a couple of minutes until you get your confirmation per email.

### Better scalability

When you place messages in a queue, you can start new instances of your processor depending on the queue size. For example, you have one processor running, when there are ten items in the queue, you start another one and so on. Nowadays with cloud technologies and serverless features, you can easily scale up to thousands of instances of your processor.

### Share data with whoever wants/needs it

Most of the time, you don&#8217;t know who wants to process the information you have.

Let&#8217;s say our order service publishes the order to the queue. Then the order processing service, reporting services, and logistics services can process the data. Your service as a publisher doesn&#8217;t care who takes the information. This is especially useful when you have a new service in the future which wants your order data too. This service only has to read the data from the queue. If your publisher service sends the data directly to the other services, you would have to implement each call and change your service every time a new service wants the data.

### Better user experience due to asynchronous processing

Better user experience is the result of the three other advantages. The user is way less likely to see an error message and even when there is a lot of traffic like on Black Friday, your system can perform well due to the scalability and the asynchronous processing.

## Implement RabbitMQ with .NET Core

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.

RabbitMQ has a lot of features. I will only implement a simple version of it to publish data to a queue in the Customer service and to process the data in the Order service.

### Implementation of publishing data

I am a big fan of Separation of Concerns (SoC) and therefore I am creating a new project in the CustomerApi solution called CUstomerApi.Message.Send. Next, I install the RabbitMQ.Client NuGet package and create the class CustomerUpdateSender in the project. I want to publish my Customer object to the queue every time the customer is updated. Therefore, I create the SendCustomer method which takes a Customer object as a parameter.

#### Publish data to RabbitMQ

Publishing data to the queue is pretty simple. First, you have to create a connection to RabbitMQ using its hostname, a username and a password using the ConnectionFactory. With this connection, you can use QueueDeclare to create a new queue if it doesn&#8217;t exist yet. The QueueDeclare method takes a couple of parameters like a name and whether the queue is durable.

[code language=&#8221;CSharp&#8221;]  
public void SendCustomer(Customer customer)  
{  
var factory = new ConnectionFactory() { HostName = \_hostname, UserName = \_username, Password = _password };

using (var connection = factory.CreateConnection())  
using (var channel = connection.CreateModel())  
[/code]

After creating the connection to the queue, I convert my customer object to a JSON object using JsonConvert and then encode this JSON to UTF8.

[code language=&#8221;CSharp&#8221;]  
var json = JsonConvert.SerializeObject(customer);  
var body = Encoding.UTF8.GetBytes(json);  
[/code]

The last step is to publish the previously generated byte array using BasicPublish. BasicPublish has like QueueDeclare a couple of useful parameters but to keep it simple, I only provide the queue name and my byte array.

[code language=&#8221;CSharp&#8221;]  
channel.BasicPublish(exchange: "", routingKey: _queueName, basicProperties: null, body: body);  
[/code]

That&#8217;s all the logic you need to publish data to your queue. Before I can use it, I have to do two more things though. First, I have to register my CustomerUpdateSender class in the Startup class. I am also providing the settings for the queue like the name or user from the appsettings. Therefore, I have to read this section in the Startup class.

[code language=&#8221;CSharp&#8221;]  
public void ConfigureServices(IServiceCollection services)  
{  
services.AddOptions();

services.Configure<RabbitMqConfiguration>(Configuration.GetSection("RabbitMq"));  
services.AddTransient<ICustomerUpdateSender, CustomerUpdateSender>();  
[/code]

The last step is to call the SendCustomer method when a customer is updated. This call is in the Handle method of the UpdateCustomerCommandHandler. If you commented out the line before to test the application without RabbitMQ, you have to uncomment the call now.

[code language=&#8221;CSharp&#8221;]  
public class UpdateCustomerCommandHandler : IRequestHandler<UpdateCustomerCommand, Customer>  
{  
private readonly ICustomerRepository _customerRepository;  
private readonly ICustomerUpdateSender _customerUpdateSender;

public UpdateCustomerCommandHandler(ICustomerUpdateSender customerUpdateSender, ICustomerRepository customerRepository)  
{  
_customerUpdateSender = customerUpdateSender;  
_customerRepository = customerRepository;  
}

public async Task<Customer> Handle(UpdateCustomerCommand request, CancellationToken cancellationToken)  
{  
var customer = await _customerRepository.UpdateAsync(request.Customer);

_customerUpdateSender.SendCustomer(customer);

return customer;  
}  
}  
[/code]

### Implementation of reading data from RabbitMQ

Implementing the read functionality is a bit more complex because we have to constantly check the queue if there are new messages and if so, process them. I love .NET Core and it comes really handy here. .NET Core provides the abstract class BackgroundService which provides the method ExecuteAsync. This method can be overriden and is executed regularly in the background.

In the OrderApi solution, I create a new project called OrderApi.Messaging.Receive, install the RabbitMQ.Client NuGet and create a class called CustomerFullNameUpdateReceiver. This class inherits from the BackgroundService class and overrides the ExecuteAsync method.

In the constructor of the class, I initialize my queue the same way as in the CustomerApi using QueueDeclere. Additionally, I register events that I won&#8217;t implement now but might be useful in the future.

[code language=&#8221;CSharp&#8221;]  
private void InitializeRabbitMqListener()  
{  
var factory = new ConnectionFactory  
{  
HostName = _hostname,  
UserName = _username,  
Password = _password  
};

_connection = factory.CreateConnection();  
\_connection.ConnectionShutdown += RabbitMQ\_ConnectionShutdown;  
\_channel = \_connection.CreateModel();  
\_channel.QueueDeclare(queue: \_queueName, durable: false, exclusive: false, autoDelete: false, arguments: null);  
}  
[/code]

#### Reading data from the queue

In the ExecuteAsync method, I am subscribing to the receive event and whenever this event is fired, I am reading the message and encode its body which is my Customer object. Then I am using this Customer object to call another service that will do the update in the database.

[code language=&#8221;CSharp&#8221;]  
protected override Task ExecuteAsync(CancellationToken stoppingToken)  
{  
stoppingToken.ThrowIfCancellationRequested();

var consumer = new EventingBasicConsumer(_channel);  
consumer.Received += (ch, ea) =>  
{  
var content = Encoding.UTF8.GetString(ea.Body.ToArray());  
var updateCustomerFullNameModel = JsonConvert.DeserializeObject<UpdateCustomerFullNameModel>(content);

HandleMessage(updateCustomerFullNameModel);

_channel.BasicAck(ea.DeliveryTag, false);  
};  
consumer.Shutdown += OnConsumerShutdown;  
consumer.Registered += OnConsumerRegistered;  
consumer.Unregistered += OnConsumerUnregistered;  
consumer.ConsumerCancelled += OnConsumerConsumerCancelled;

\_channel.BasicConsume(\_queueName, false, consumer);

return Task.CompletedTask;  
}  
[/code]

That&#8217;s all you have to do to read data from the queue. The last thing I have to do is to register my CustomerFullNameUpdateReceiver class as a background service in the Startup class.

[code language=&#8221;CSharp&#8221;]  
services.AddHostedService<CustomerFullNameUpdateReceiver>();  
[/code]

## Run RabbitMQ in Docker

The publish and receive functionalities are implemented. The last step before testing them is to start an instance of RabbitMQ. The easiest way is to run it in a Docker container. If you don&#8217;t know what Docker is or haven&#8217;t installed it, download Docker Desktop for Windows from <a href="https://docs.docker.com/docker-for-windows/" target="_blank" rel="noopener noreferrer">here</a> or for Mac from <a href="https://docs.docker.com/docker-for-mac/" target="_blank" rel="noopener noreferrer">here</a>. After you installed Docker, copy the two following lines in Powershell or bash:

[code language=&#8221;powershell&#8221;]  
docker run -d &#8211;hostname my-rabbit &#8211;name some-rabbit -e RABBITMQ\_DEFAULT\_USER=user -e RABBITMQ\_DEFAULT\_PASS=password rabbitmq:3-management  
[/code]

[code language=&#8221;powershell&#8221;]docker run -it &#8211;rm &#8211;name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management[/code]

Don&#8217;t worry if you don&#8217;t understand them. Simplified these two lines download the RabbitMQ Docker image, start is as a container and configure the ports, the name, and the credentials.

<div id="attachment_1912" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Run-RabbitMQ-in-a-Docker-container.jpg"><img aria-describedby="caption-attachment-1912" loading="lazy" class="wp-image-1912" src="/assets/img/posts/2020/04/Run-RabbitMQ-in-a-Docker-container.jpg" alt="Run RabbitMQ in a Docker container" width="700" height="478" /></a>
  
  <p id="caption-attachment-1912" class="wp-caption-text">
    Run RabbitMQ in a Docker container
  </p>
</div>

After RabbitMQ is started, you can navigate to localhost:15672 and login with guest as user and guest as password.

<div id="attachment_1913" style="width: 586px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Login-into-the-RabbitMQ-management-portal.jpg"><img aria-describedby="caption-attachment-1913" loading="lazy" class="size-full wp-image-1913" src="/assets/img/posts/2020/04/Login-into-the-RabbitMQ-management-portal.jpg" alt="Login into the RabbitMQ management portal" width="576" height="243" /></a>
  
  <p id="caption-attachment-1913" class="wp-caption-text">
    Login into the RabbitMQ management portal
  </p>
</div>

Navigate to the Queues tab and you will see that there is no queue yet.

<div id="attachment_1914" style="width: 656px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/No-queues-are-created-yet.jpg"><img aria-describedby="caption-attachment-1914" loading="lazy" class="wp-image-1914 size-full" src="/assets/img/posts/2020/04/No-queues-are-created-yet.jpg" alt="A message was published and consumed" width="646" height="474" /></a>
  
  <p id="caption-attachment-1914" class="wp-caption-text">
    A message was published and consumed
  </p>
</div>

Now you can start the OrderApi and the CustomerApi project. The order how you start them doesn&#8217;t matter. After you started the CustomerApi, the CustomerQueue will be created and you can see it in the management portal.

<div id="attachment_1915" style="width: 883px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/The-CustomerQueue-was-created-from-the-CustomerApi-solution.jpg"><img aria-describedby="caption-attachment-1915" loading="lazy" class="wp-image-1915" src="/assets/img/posts/2020/04/The-CustomerQueue-was-created-from-the-CustomerApi-solution.jpg" alt="No queues are created yet" width="873" height="528" /></a>
  
  <p id="caption-attachment-1915" class="wp-caption-text">
    No queues are created yet
  </p>
</div>

Click on CustomerQueue and you will see that there is no message in the queue yet and that there is one consumer (the OrderApi).

<div id="attachment_1916" style="width: 616px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Overview-of-the-CustomerQueue-and-its-Consumers.jpg"><img aria-describedby="caption-attachment-1916" loading="lazy" class="wp-image-1916" src="/assets/img/posts/2020/04/Overview-of-the-CustomerQueue-and-its-Consumers.jpg" alt="Overview of the CustomerQueue and its Consumers" width="606" height="700" /></a>
  
  <p id="caption-attachment-1916" class="wp-caption-text">
    Overview of the CustomerQueue and its Consumers
  </p>
</div>

Go to the Put action of the CustomerApi and update a customer. If you use my in-memory database you can use the Guid &#8220;9f35b48d-cb87-4783-bfdb-21e36012930a&#8221;. The other values don&#8217;t matter for this test.

<div id="attachment_1917" style="width: 541px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Update-a-customer.jpg"><img aria-describedby="caption-attachment-1917" loading="lazy" class="size-full wp-image-1917" src="/assets/img/posts/2020/04/Update-a-customer.jpg" alt="Update a customer" width="531" height="688" /></a>
  
  <p id="caption-attachment-1917" class="wp-caption-text">
    Update a customer
  </p>
</div>

After you sent the update request, go back to the RabbitMQ management portal and you will see that a message was published to the queue and also a message was consumed.

<div id="attachment_1918" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/A-message-was-published-and-consumed.jpg"><img aria-describedby="caption-attachment-1918" loading="lazy" class="wp-image-1918" src="/assets/img/posts/2020/04/A-message-was-published-and-consumed.jpg" alt="A message was published and consumed" width="700" height="387" /></a>
  
  <p id="caption-attachment-1918" class="wp-caption-text">
    A message was published and consumed
  </p>
</div>

## Shortcomings of my Implementation

I wanted to keep the implementation and therefore it has a couple of shortcomings. The biggest problem is that the OrderApi won&#8217;t start if there is no instance of RabbitMQ running. Also, the CustomerApi will crash when you try to update a customer and there is no instance of RabbitMQ running. There is no exception handling right now. Also if there is an error while processing a message, the message will be deleted from the queue and therefore be lost.

Another problem is that after the message is read, it is removed from the queue. This means only one receiver is possible at the moment. There are also no unit tests for the implementation of the RabbitMQ client.

## Conclusion

This post explained why you should use queues to decouple your microservices and how to implement RabbitMQ using Docker and ASP .NET Core 3.1. Keep in mind that this is a quick implementation and needs some work to be production-ready.

<a href="/dockerize-an-asp-net-core-microservice-and-rabbitmq" target="_blank" rel="noopener noreferrer">In my next post</a>, I will dockerize the application which will make it way easier to run and distribute the whole application.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.