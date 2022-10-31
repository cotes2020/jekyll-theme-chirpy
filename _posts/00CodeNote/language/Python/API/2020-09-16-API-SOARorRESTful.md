---
title: API - SOAR or RESTful
author: Cotes Chung
date: 2020-09-15 11:11:11 -0400
categories: [Web]
tags: [Web, API]
toc: true
image:
---

[toc]

---

# API - SOAR or RESTful

---

## video line

helpful video for basic concepte before start:

- [What is REST API? | Web Service](https://www.youtube.com/watch?v=qVTAB8Z2VmA&ab_channel=Telusko)
- [REST API concepts and examples](https://www.youtube.com/watch?v=7YcW25PHnAA&ab_channel=WebConcepts)
- [REST API & RESTful Web Services Explained | Web Services Tutorial](https://www.youtube.com/watch?v=LooL6_chvN4&ab_channel=CleverTechie)


[apigee](https://apigee.com/410-console) transfer to [google](https://cloud.google.com/blog/products/apigee)

---

![Screen Shot 2020-09-16 at 11.41.59](https://i.imgur.com/DAmvZIg.png)

---


## Web Services API

In recent years, there is a tendency to move some business logic to the client side, in a architecture known as `Rich Internet Application (RIA)`.
- In RIA, the server provides the clients with data retrieval and storage services, becoming a `'web service' or 'remote API'`.


**API** (Application Programming Interface)
- web API generally refers to <font color=red> both sides of computer systems communicating over a network</font>:
  - the `API services` offered by a server
  - and the `API` offered by the client such as a web browser.
- a set of routines, protocols and tools for building software applications.
- It exposes a software component in terms of its operations, input and output, that are independent of their implementation.
- A Web Service is a web application that is available to your application as if it was an API.

There are several protocols that the RIA clients can communicate with a web service or remote API,
- such as `RPC (Remote Procedure Call)`,
- `SOAP (Simplified Object Access Protocol)`
- and `REST (Representational State Transfer)`


The **server-side portion of the web API** is a `programmatic interface to a defined request-response message system`.
- There are several design models for web services
- the two most dominant are `SOAP` and `REST`.


![soapvsrest](https://i.imgur.com/uqRbjf4.jpg)


---



## SOAP vs REST

**SOAP** provides the following advantages when compared to REST:
- Language, platform, and transport independent
  - <font color=red> (REST requires use of HTTP) </font>
- Works well in distributed enterprise environments
  - <font color=red> (REST assumes direct point-to-point communication) </font>
- Standardized
- Provides significant pre-build extensibility in the form of the `WS* standards`
- Built-in error handling
- Automation when used with certain language products


**REST** (REpresentational State Transfer)
- easier to use for the most part and is more flexible
- Uses easy to understand standards like swagger and OpenAPI Specification 3.0
- Smaller learning curve
- Efficient (SOAP uses XML for all messages, REST mostly uses smaller message formats like JSON)
- Fast (no extensive processing required)
- Closer to other Web technologies in design philosophy

As one REST API tutorial put it: SOAP is like an envelope while REST is just a postcard.

![soapvsrest](https://i.imgur.com/uqRbjf4.jpg)


---


### SOAP - Simple Object Access Protocol
- SOAP relies heavily on XML, and together with schemas, defines a very strongly typed messaging framework.
- Every operation the service provides is explicitly defined, along with the XML structure of the request and response for that operation.
- Each input parameter is similarly defined and bound to a type: for example an integer, a string, or some other complex object.
- All of this is codified in the WSDL – Web Service Description (or Definition, in later versions) Language. The WSDL is often explained as a contract between the provider and the consumer of the service. In programming terms the WSDL can be thought of as a method signature for the web service.


---

# API design

1. Naming
2. Parameter define
3. Response object define
4. error define
5. should had no side effects (doing everthing in one funtion)

big API
- paging
- fragment

## with HTTP request


```bash
    #  routing
POST www.website.tech/chat_message/getAdmin/v1(optional)

POST www.website.tech/chat_message/admin?groupID=123

request:
{
  "groupID": "123"
}

response
{
  "admin": [
    "id": "123"
    "name:" "grace"
  ]
}

```


















。
