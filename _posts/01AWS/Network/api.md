

### Scale APIs in the cloud with Security, Reliability and Quality of Service.


having a microservices architecture made scaling to millions and even a billion users possible.

Container technology married with microservices is a natural fit.  A container using Docker can be instantiated and start operating in less than 100 milliseconds.
1. You need to make the service secure
   - follow best practices for security.
   - standard things like creating multiple subnets for different components to protect data and services from simple attacks.
   - scanning for virus’s and malware, etc.
   - check your code for valid safe versions of open source software so mistakes are not made using bad software.
   - detect intrusions and monitor logs to insure appropriate activities are going on.
   - need to assume your network is compromised.
   - need to make sure that certificates for containers are valid and that the security practices of your APIs enforce TLS and TLS HSTS and other protocol enhancements available like TLS 1.2.
   - perform penetration tests and other intrusion tests using standard tools.
   - need to make sure your databases are secure, possibly encrypted and file systems to the extent they need to be and that authentication and authorization are being done right.
   - do code scanning for potential security flaws.
   - If the security practices above are not automated you will find yourself cutting short security.
   - Automating all your security processes is non-trivial amount of work
   - try to do it by hand you will eventually make a mistake and become next weeks story.
   - Integrating security into your DevOps automation is described below and important to insure best practices and is essential to actually being secure.

2. You need a container management framework
   - components deploying by hand is no longer fun or safe.
   - Similarly monitoring them and numerous other management tasks are best done by a container management framework.
   - This is what Docker Swarm, Kubernetes, Mesos and other products do.
   - The container management framework is critical for some security functions as well.  It provides the ability to deploy containers safely.

3. You need to make it reliable
   - Active/Active and Active / Passive servers, service discovery and registration, heartbeats, transaction protocols, replication of databases and other means.
   - Most likely you will use a service registry and heartbeat with active/active servers for most of your reliability.
   - bring in a message queue or some other transactional system.   Databases typically have their own reliability mechanisms.   Your service needs to support these components and all the other pieces they need to implement fast recovery of failed components and hardware.
   - Today, a lot of this can be implemented by policy and with standard components like Zookeeper, Consul, etcd which allow you to monitor heartbeats set up automatic restart and restore configuration instantly.

1. You need scalability
   - You need components to help you scale your API service.
   - Having a microservice architecture doesn’t automatically make it scalable.
   - Kubernetes for instance has the ability to monitor and keep load constant across a bunch of containers in a cluster.   You can say, for instance, when the load on these containers goes above 50% on any server create a new instance automatically.   You can use this to scale single microservice to multiple instances and keep response time consistent across your service even as load builds rapidly.
   - Sometimes this type of scaling is not enough.  You may need a more policy based scaling component that can scale multiple components in different ways depending on different load indications.

2. You need DevOps Fullstack Automation with security
   - evelopment environment, test environment, production or other staging environments need to be completely in sync in order to be sure that when you deploy something it actually works.
   - In order to achieve the improvement in agility the cloud can give you, you need a finely automated infrastructure that you can replicate stacks of components reliably over and over.
   - deploy multiple production environments for different regions, different customers or for any number of reasons.
   - Different APIs may share a lot of similar underlying infrastructure.
   - If you have multiple APIs automating your stacks will make it easier to deploy new APIs and new services.
   - do a synthetic test of your APIs or services with 1000 times the load you expect like AWS and Netflix and other sophisticated services in the cloud do.
   - It is critical that the DevOps automation include automatic security configuration and checks otherwise mistakes will be made.  People are notoriously bad at doing routine tedious things over and over.   Expecting that you can do security as a separate function from DevOps is a mistake.

3. You need to upgrade the stack of all these things regularly
   - In the past many companies were scared to do upgrades because they put the production environment at risk.   Therefore, companies let the components become stale and let upgrades go for a year or more before trying a big upgrade.
   - As components are upgraded frequently these upgrades provide needed performance, security or bug fixes that your customers will need.  Since a component microservices architecture shares services across all components if one service needs the upgrade you will end up upgrading all the uses of that microservice component.
   - Therefore, you will be forced to upgrade more frequently and you will be forced to test all your stack and components with the new upgrade.
   - Since so many upgrades are coming through you will need a way to do these upgrades with minimal interruption.
   - Several other components can help you make upgrades and security patches without bringing down any users of your services.

1. You need to have test suites for the stacks not just the API
   - Since you will be upgrading your components many times a year you will want to build test suites to test your stack of components in your service to see if any upgrade or security patch breaks some part of the stack.
   - The new upgrade may seem great and when you test it against your application it works fine but other components in the stack that are not your responsibility may use that component and they may not work with the upgrade.
   - So, you need to build test suites before you deploy anything to production for your stacks so you can run the test suites against a deployed copy of your production environment to see if anything breaks.

2. You need to keep the automation up to date and the test suites for the stacks
   - Over time upgrades of components, new features, adding components will mean you will have to maintain your automation.
   - Each change to the automation is essentially like an upgrade of a component and you need to test the automation and the resulting deployments as you would for an upgrade or security patch.
   - detect flaws in your test suites you need to keep them up to date and modify them.
   - constantly be improving them to consider new test cases and potential issues.

3. You need to understand the costs of the components in your stack and to manage that cost
   - Finally, you have all of this infrastructure working and your APIs are secure, reliable, scalable, upgradable, automated.  You can make as many changes to your APIs as before and you are seeing a successful service that is growing.   You are ecstatic.
   - costs of your infrastructure are growing non-linearly
   - get a bill from your cloud provider that shows you thousands of lines of detail but you have no idea how to translate those servers and usage to the underlying services.
   - figure out what is taking the most money or if that is reasonable.

4. You need to implement a data gathering and instrument all your services so they produce information you can use to diagnose what is using what and how you could save money.
   - Maybe some services shouldn’t be scaled arbitrarily.
   - Possibly you should limit the scaling to a certain amount.
   - Maybe a configuration change or frequency of doing something could drastically improve the costs.
   - Possibly a component you selected is too expensive using way more resources than it should.
   - Maybe this is a bug or a different component which does substantially the same function is less expensive.

5. You need to instrument your services and make that instrumentation part of your automation and you need to build the analysis tools or use tools to mine the data to find where your problems are or how you might be able to save money.





---

### Architect APIs for Scale and Security


creating RESTful APIs
- using HTTP methods, such as GET, POST, DELETE to perform operations against the API.


Amazon API Gateway
- make it easy for developers to create APIs at any scale without managing any servers.
- API Gateway will handle all of the heavy lifting needed including traffic management, security, monitoring, and version/environment management.


GraphQL APIs are relatively new, with a primary design goal of allowing clients to define the structure of the data that they require.

AWS AppSync allows you to create flexible APIs that access and combine multiple data sources.



#### REST APIs

Architecting a REST API is structured around creating combinations of resources and methods.
- Resources are paths that are present in the request URL and methods are HTTP actions that you take against the resource.
- For example
- define a resource called “cart”: `https://myapi.somecompany.com/cart`
- The cart resource can respond to HTTP POSTs for adding items to a shopping cart
- or HTTP GETs for retrieving the items in your cart.


With API Gateway, you would implement the API like this:

![Arch-comparison-1-1024x565](https://i.imgur.com/nVf8A6F.jpg)

Behind the scenes, you can integrate with nearly any backend to provide the compute logic, data persistence, or business work flows.
- For example
- configure an AWS Lambda function to perform the addition of an item to a shopping cart (HTTP POST).
- use API Gateway to directly interact with AWS services like Amazon DynamoDB.
- using API Gateway to retrieve items in a cart from DynamoDB (HTTP GET).


RESTful APIs tend to use Path and Query parameters to inject dynamic values into APIs.
- For example,
- to retrieve a specific cart with an id of abcd123,
- design the API to accept a query or path parameter that specifies the cartID:
- `/cart?cartId=abcd123` or `/cart/abcd123`


when you need to add functionality to your API
- the typical approach would be to add additional resources.
- For example
- add a checkout function, you could add a resource called `/cart/checkout`



#### GraphQL APIs


Architecting GraphQL APIs
- is not structured around resources and HTTP verbs
- instead you define your data types and configure where the operations will retrieve data through a resolver.
- An operation is either a query or a mutation.
- Queries simply retrieve data while mutations are used when you want to modify data.
- If we use the same example from above, you could define a cart data type as follows:


```py
type Cart {
  cartId: ID!
  customerId: String
  name: String
  email: String
  items: [String]
}
```

Next, configure the fields in the Cart to map to specific data sources.
- AppSync is then responsible for executing resolvers to obtain appropriate information.
- Your client will send a HTTP POST to the AppSync endpoint with the exact shape of the data they require.
- AppSync is responsible for executing all configured resolvers to obtain the requested data and return a single response to the client.

![Arch-comparison-2-1024x608](https://i.imgur.com/hlGEJez.jpg)



With GraphQL, the client can change their query to specify the exact data that is needed.
- The above example shows two queries that ask for different sets of information.
- The first getCart query asks for all of the static customer (customerId, name, email) and a list of items in the cart.
- The second query just asks for the customer’s static information.
- Based on the incoming query, AppSync will execute the correct resolver(s) to obtain the data.
- The client submits the payload via a HTTP POST to the same endpoint in both cases.
- The payload of the POST body is the only thing that changes.

As we saw above, a REST based implementation would require the API to define multiple HTTP resources and methods or path/query parameters to accomplish this.

AppSync also provides other powerful features that are not possible with REST APIs such as real-time data synchronization and multiple methods of authentication at the field and operation level.


---


### example

![restAPI-1-1024x369](https://i.imgur.com/vfXnxIj.jpg)

1. <font color=red> synchronous, tightly coupled architecture </font>
   - <font color=blue> the request must wait for a response from the backend integration (RDS) </font>

   - This API accepts GET requests to retrieve a user’s cart
     - by using a Lambda function to perform SQL queries against a relational database managed in RDS.
     - If receive a large burst of traffic,
       - both API Gateway and Lambda will scale in response to the traffic.
       - but relational databases typically have limited memory/cpu capacity and will quickly exhaust the total number of connections.

   - solution
     1. <font color=red> defining API Keys and requiring your clients to deliver a key with incoming requests. </font>
        - to track each application or client who is consuming your API.
        - to create Usage Plans and throttle your clients according to the plan you define.
        - For example, you if you know your architecture is capable of of sustaining 200 requests per second, you should define a Usage plan that sets a rate of 200 RPS and optionally configure a quota to allow a certain number of requests by day, week, or month.


     2. API Gateway lets you define throttling settings for the whole stage or per method.
        - If GET operation is less resource intensive than a POST operation
        - you can override the stage settings and set different throttling settings for each resource.


![restAPI-2-1024x548](https://i.imgur.com/BKu0OoH.jpg)


2. <font color=red> an asynchronous, loosely coupled architecture </font>
   - A decoupled architecture
   - separates the data ingestion from the data processing and allows you to scale each system separately.

   - This architecture enables ingestion of orders directly into a <font color=blue> highly scalable and durable data store </font> such as Amazon Simple Queue Service (SQS).
   - the backend can process these orders at any speed that is suitable for your business requirements and system ability.
   - Most importantly, <font color=blue> the health of the backend processing system does not impact your ability to continue accepting orders </font>




---


### API design?

As a company that creates an API management platform, we’re built on open-source API technology. We’ve always had an API-first mentality. All of our UIs are built as a presentation layer, which is powered by an API. Users configure the Kong API gateway using the Admin API to control the Kong platform. All this aims to allow users easy CI/CD integration with our internal and third-party tools.

Real-time, simple, secure, and fast. As an infrastructure-as-a-service provider, our API must be streaming updates as they happen, rather than requiring the client to poll for updates. The key questions we think about are, “How can we make the API easy to use?” and “Can we make it fast globally?” The answers are core to the developer experience. We also have to make sure the API is secure.

For the past couple years, during the migration from our Ruby on Rails monolith to a service-oriented architecture (SOA), our data service APIs looked similar to create, read, update, and delete operations with ActiveRecord. However, as our SOA matures, we’re rethinking our API design from extensibility and capability perspectives, instead of focusing on how to minimize breakage during a large architectural migration.

In the Creative Cloud, creative assets are a foundational element, and often our API design will start with building out models that represent one or more kinds of creative assets. From there, we often take inspiration and learning from our in-product APIs to provide some consistency in functionality and developer experience. Ultimately, APIs should be easily understood, well-documented, consistent, and provide useful feedback when something works and when something fails. Where there are standards, we try to stick to them, and where we have to make our own API surface, we need to remember the human who’s going to be using the API.



---


### API documentation?

We take a “spec-first” approach to development. Wasting developer time on creating documentation, which often goes out of date the moment the API is published, is a real problem. To address this, we did a couple things.
First, we acquired Insomnia last year, which is an API and GraphQL testing tool. We’re extending it and open-sourcing those extensions to do spec-first development for APIs using the OpenAPI Specification (formerly Swagger). This allows users to do both definition and testing of APIs in one tool.
Second, the Kong Developer Portal consumes these OpenAPI Specifications and auto-generates live documentation users can leverage to test and explore their APIs. If they don’t have or don’t want to write OpenAPI Specifications, Kong Brain, which automates API documentation and configuration, can automatically generate an OpenAPI Specification from traffic on the Kong gateway. These OpenAPI Specifications can then be sent into a File API for consumption by the Kong Developer Portal—the whole API life cycle, really.

Supporting all platforms leads to bulky documentation. As you scale and support more APIs, it quickly becomes cumbersome for internal teams to keep up and for developers—the primary audience—to navigate. Despite this challenge, we strive for consistency across platforms and optimizing for developer expectations and experience. This problem is never-ending, so we’ve dedicated a team to enhancing the developer journey. We use drivers such as data outcomes to drive our priorities, and we’re looking for a higher trials-to-sign-up ratios and lower average time to trial.

Enabling block comments for our Apache Thrift Interface Description Language (IDL) adds context at the endpoint, request, and response field levels in addition to metadata about the services, such as communication channels for the service owner or the technical design document. Our Thrift IDL with documentation comments is automatically parsed and displayed in a web UI upon code deployment. This enables our internal documentation website to be up to date, as any changes are picked up whenever the service is deployed.

Where possible, we try to leverage Swagger documentation. We’re also thinking beyond the standard reference documentation and creating other enablement materials like code samples, tutorials, and other longform documentation that goes into detail on specific aspects of the API. We primarily rely on storing markdown files in GitHub and have converters that turn the markdown files into more consumable HTML pages. This makes the documentation more accessible (because it looks better), allows us to track issues in the repo itself, and empowers our developers to fix or enhance the documentation with their own pull requests.




---


### What kinds of resources does your organization dedicate to building APIs? What does that organizational structure look like?


Each team is in charge of documenting and building the API endpoints for their part of the product. For instance, the developer portal team has developed OpenAPI Specification files, which can be leveraged in the default portal configuration. All our teams love Insomnia and use it to test their APIs as they develop them—it’s a great debugging tool. Our sales engineers get the best feedback because they use it to do proofs of concept for customers all the time.

We’re an API company, which means everyone is dedicated to APIs. It’s not just our engineers—the customer success team, marketing team, and product team are all focused on delivering excellent API experiences. We use a process for API product design that involves learning from our customers’ needs. Once we understand the requirements and design principles, we follow an RFC process, which defines the technical components required to deploy a production-grade, global, real-time API. From an engineering team perspective, we have architects who design and guide implementation direction; DevOps, which improves our in-house developer and runtime experience; and engineers who implement and build the software to deliver API value.

A dedicated service framework engineering team (part of our platform infrastructure organization) creates reusable components and infrastructure to build reliable, scalable services and APIs. We use Apache Thrift to define our APIs for our remote procedure call framework and annotations. We also invested in a team to provide out-of-the-box options and standardization for our thousands of service APIs. The team has enabled many other features, such as rate limiting and traffic replay mechanisms, to be configurable with annotations instead of requiring each service owner to build or integrate their own solution.

Increasingly, we pair API engineering teams with technical product managers who can flesh out API requirements and provide input on the developer experience. We also have models where a technical writer is embedded with the teams to ensure documentation is relevant and stays up to date. For APIs that are going to be used by teams across Adobe and used to power major services, an architecture council defines an API specification and manages any changes to the specification. The service teams are then responsible for implementing the API in accordance with the specification so clients leveraging the services are developing against a consistent API surface.



---


### manage API changes?


We focus on backward compatibility and follow major/minor version bumps, like most open-source software. Backward-incompatible changes are only allowed in major releases once a year. We mark endpoints that are going away as “deprecated” for several releases before we officially communicate their retirement.

First and foremost, we version our APIs.
- Each API can be maintained as a new API, or the backend can be deployed to support the new and old formats. We take advantage of Swagger and OpenAPI to do that. We’re all for making sure customers on older versions of our APIs are still able to make use of PubNub. Especially for mission-critical services that are built on top of PubNub, it’s not right to force customers to upgrade to a new version of an API unless there’s a good reason for it. We try to keep it simple when it comes to versioning, using a version ID at the front of the path, such as `/v2/api-name`.

Engineers are empowered to modify their own service APIs. Airbnb once had service framework engineers as mandatory reviewers, but as the number of services and APIs multiplied and engineers became more comfortable with Thrift, we automated a series of tests that check for backward incompatibility. With that tooling in place, service owners decide whether to move forward with the API change. We’ve also placed standardized observability into the client services that call each of our API endpoints, which allows us to understand what to migrate and monitor for modifications to the API.


avoid making unnecessary changes by carefully architecting up front and using standards where they make sense—no point reinventing the wheel. But when change is necessary, we ask: Is it worth it? Is the fix super expensive? Do we need to add some backward compatibility options? The goal is to see what dependents we could break, and plan from there.

For our cloud APIs, potential changes are documented and brought to an architecture council.
- The council reviews the change, votes on it, and, if it’s approved, updates the underlying specification.
- Support documentation that illustrates how to convert from the old to the new API surface are always part of the release plan.
- The key here is time.
- Dependent teams need to know about big changes in advance so they can make plans to adjust.

Clients are notified throughout the process, and once the specification has been updated, service teams are tasked with scheduling the changes to their implementation of the API. The schedule is coordinated across teams.



---


### test and collect feedback on its API?

feedback from both our enterprise customers and our open-source community. The best thing about being open-source software is there are plenty of people willing to help and give you input. They’re also the most vocal when you make a mistake! So we pay a lot of attention to perception when we release and deprecate our APIs.

Globally distributed test servers monitor the speed and deliverability of real-time API data. We use our own tools, as well as third parties like Catchpoint and Pingdom. Outside of the technical monitoring tools, it’s important to make sure we keep track of how often these APIs are actually used to make more informed decisions going forward, including whether to keep the API alive and whether we need to rethink our go-to-market around it. And, of course, we keep in touch with our customers to make sure the API is serving their needs.

In terms of usability, we rely heavily on feedback from our pilot integrations with other engineers at Airbnb. Upon reviewing production incidents, we also evaluate our APIs and create action items for improving instrumentation or standardization. We’re working on better, automated solutions, such as an easy way to perform load testing.

We tend to have slightly different processes for our internal and external-facing APIs. As part of exposing APIs to external developers, we’ve built a process that allows us to validate and improve an API in multiple stages. After identifying a specific API or set of APIs to expose, an internal team is tasked with creating integrations on those APIs. The team also carves out time to provide early feedback on the API so they can suggest changes while they work on the integration. After that, we give a select group of partners access to the APIs so they can start creating their own integrations. They help identify any feature gaps or developer experience issues. Once the API team has tracked and, where applicable, made changes based on that feedback, we open it up to our wider third-party developer community and record any feedback they have. We actively seek that feedback in individual partner communications, surveys, forums, and at in-person events like MAX Adobe’s annual creativity conference.

---


### API design, future of APIs will hold


As we move from being solely an API company to being a cloud connectivity company, we’re seeing a shift from REST APIs to services that connect systems. The new world of APIs will be gRPC, TCP custom protocols, Kafka, GraphQL, and more.
- We think the connectivity promise of APIs will continue to be realized, but the methods will expand.
- It’s like the evolution of APIs from XML/SOAP to JSON and specifications.
- This is exciting and shouldn’t be scary.

Modern APIs leverage the best of modern transports such as HTTP2 and gRPC.
- We’ve deployed these two modern API transports to offer the best possible service, and we continue to keep an eye on upcoming technologies.
- Just because we provide a great technology platform, doesn’t mean we can’t keep making it better!

GraphQL is more expressive and flexible with schema stitching. Because of this, we’re exploring how it interacts with frontend clients and how it enables querying of data across multiple services. Another hope for the future is more out-of-the-box resilience and robustness features—such as timeouts and management of hard versus soft dependencies with fallback values—instead of requiring organizations to custom build these.

We’ve been investing in Hypermedia as the Engine of Application State (HATEOAS) REST APIs because of the way it decouples the client and server and allows us to improve and augment our services without breaking clients. We’ve been particularly interested in how to improve the developer experience and how to document Hypermedia APIs effectively. We’ve also started emphasizing events and webhooks as a complement to our APIs and providing flexibility to developers in terms of how they interact with our services. APIs exist to enable developers to connect disparate pieces in useful and often innovative ways. We need to keep the long view in mind—so many products and tools have gone by the wayside because an endpoint or API no longer functions. A commonly heard phrase at Adobe is “democratize feature development,” and we believe a well-built API surface and platform should enable anyone to do just that.
