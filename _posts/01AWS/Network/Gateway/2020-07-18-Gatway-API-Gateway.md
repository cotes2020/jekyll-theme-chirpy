---
title: AWS - VPC Gateway - API Gateway
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Network]
tags: [AWS, Network, VPC]
toc: true
image:
---

- [AWS API Gateway](#aws-api-gateway)
  - [benefits](#benefits)
  - [Security](#security)
    - [Application Firewall](#application-firewall)
    - [Resource Policy](#resource-policy)
    - [Authorization](#authorization)

---

# AWS API Gateway

---

## benefits

1. fully managed service

2. <font color=red> create a unified API frontend for multiple microservices </font>
   - to loosely couple systems
   - does not compute information from an API.

3. for developers to create, publish, maintain, monitor, scale, and secure APIs at any scale.
   - in AWS Management Console
   - throttle, meter, and monetize API usage by thrid-aprty developers
   - handles all the tasks involved in accepting and processing thousands of concurrent API calls
     - including traﬃc management, authorization and access control, monitoring, and API version management.
   - capabilities does the API Gateway possess:
     - Publish APIs
     - Create APIs
     - Scale APIs
       - APIs can be scaled when using the API Gateway.
     - Monitor APIs
     - Manage APIs

4. create an API that acts as a “front door” for applications to access data, business logic, or functionality from your back-end services
   - designed to handle workloads such as workloads running on EC2,
   - code running on AWS Lambda, or any web application.

5. security:
   - DDos protection and throttling for the backend
   - <font color=red> Throttle and monitor requests to protect your backend </font>
     - if you have a bad actor or someone who has bad code, you can turn off their access to your data
   - <font color=red> authenticate and authorize requests to a backend </font>
   - Reduced latency and DDoS, protection through Amazon CloudFront

6. Managed cache to store API responses

7. SDK generation for iOS, Android, and JavaScript


8. OpenAPI specification, or Swagger, support.
   - Swagger is a specification and complete framework implementation for describing, producing, consuming, and visualizing RESTful web services.

9. Request and response data transformation


10. API endpoint works closely with APIs; they supply the ending point for API communication.

11. Host and use multiple versions and stages of your APIs
12. Create and distribute API keys to developers
13. Leverage signature version 4 to authorize access to APIs



The Amazon API Gateway integrates with:
- AWS Lambda
  - Lambda can be used to compute data supplied from an API.
- AWS Marketplace
- Endpoints with Private VPCs
- Fargate
  - used to deploy containers but also for its computing power.

---

![Screen Shot 2020-07-10 at 17.03.24](https://i.imgur.com/5VSPDj4.png)

a serverless architecture using the API Gateway.
- includes a variety of Amazon SDKs.
- Route 53 performs the DNS resolution
- Amazon CloudFront serves cached data from the S3 bucket where the static images are stored.
- After the API Gateway receives a response to a request to backend,
  - it caches the response in its own cache.
    - When the same request comes through again
    - the API Gateway checks its cache for the request and returns it without check with the origin instances.
- The API Gateway can work in conjunction with AWS Lambda and any of the web services shown, like DynamoDB and EC2 instances.



---

## Security

Security with API Gateway falls into three major buckets

### Application Firewall

- enable <font color=red> AWS Web Application Firewall (WAF) for the entire API </font>
- WAF will inspect all incoming requests and block requests that fail your inspection rules.
- For example
- WAF can inspect requests for SQL Injection, Cross Site Scripting, or whitelisted IP addresses.

### Resource Policy
- apply a <font color=red> Resource Policy that protects your entire API </font>
- an IAM policy applied to the API
- use this to white/black list client IP ranges or allow AWS accounts and AWS principals to access your API.


### Authorization

- IAM:
  - This AuthZ option requires clients to sign requests with the AWS v4 signing process.
  - The associated IAM role or user must have permissions to perform the `execute-api:Invoke` action against the API.

- Cognito:
  - This AuthZ option requires clients to login into Cognito and then pass the returned ID or Access JWT token in the `Authentication` header.

- Lambda Auth:
  - This AuthZ option is the most flexible
  - lets you execute a Lambda function to perform any custom auth strategy needed.
  - A common use case for this is OpenID Connect.






.
