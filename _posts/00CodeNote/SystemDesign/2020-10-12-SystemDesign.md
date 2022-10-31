---
title: System Design - Key Concepts
# author: Grace JyL
date: 2020-10-12 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote]
tags: []
math: true
# pin: true
toc: true
# image: /assets/img/note/tls-ssl-handshake.png
---

- [System Design - Key Concepts](#system-design---key-concepts)
  - [Handling the Question](#handling-the-question)
  - [Design](#design)
  - [Algorithms that Scale](#algorithms-that-scale)
- [Key Concepts](#key-concepts)
  - [Horizontal vs. Vertical Scaling](#horizontal-vs-vertical-scaling)
  - [Load Balancer](#load-balancer)
  - [Database Denormalization and NoSQL](#database-denormalization-and-nosql)
  - [Database Partitioning (Sharding)](#database-partitioning-sharding)
  - [Caching](#caching)
  - [Asynchronous Processing & Queues](#asynchronous-processing--queues)
  - [Networking Metrics](#networking-metrics)
  - [MapReduce](#mapreduce)
  - [Considerations](#considerations)

---


# System Design - Key Concepts

---


## Handling the Question


* **Communicate**: A key goal of system design questions is to evaluate your ability to communicate. Stay engaged with the interviewer. Ask them questions. Be open about the issues of your system.
* **Go broad first**: Don't dive straight into the algorithm part or get excessively focused on one part.
* **Use the whiteboard**: Using a whiteboard helps your interviewer follow your proposed design. Get up to the whiteboard in the very beginning and use it to draw a picture of what you're proposing.
* **Acknowledge interview concerns**: Your interviewer will likely jump in with concers. Don't brush them off; validate them. Acknowledge the issues your interviewer points out and make changes accordingly.
* **Be careful about assumptions**: An incorrect assumption can dramatically change the problem.
* **State your assumptions explicitly**: When you do make assumptions, state them. This allows your interviewer to correct you if you're mistaken, and shows that you at least know what assumptions you're making.
* **Estimate when necessary**: In many cases, you might not have the data you need. You can estimate this with other data you know.
* **Drive**: As the candidate, you should stay in the driver's seat. This doesn't mean you don't talk to your interviewer; in fact, you _must_ talk to your interviewer. However, you should be driving through the question. Ask questions. Be open about tradeoffs. Continue to go deeper. Continue to make improvements.

---


## Design

1.  Scope the Problem
2.  Make Reasonable Assumption
3.  Draw the Major Components
4.  Identify the Key Issues
5.  Redesign for the Key Issues

---


## Algorithms that Scale

In some cases, you're being asked to design a single feature or algorithm, but you have to do it in a scalable way.

1.  Ask Questiosn
2.  Make Believe
3.  Get Real
4.  Solve Problems


---

# Key Concepts

## Horizontal vs. Vertical Scaling

* Vertical scaling
  * increasing the resoures of a specific node.
  * For example
  * add additional memory to a server to improve its ability to handle load changes.

* Horizontal scaling
  * increasing the number of nodes.
  * For example
  * add additional servers, thus decreasing the load on any one server.

Vertiacal scaling is generally easer than horizontal scaling, but it's limited.

---

## Load Balancer

Typically, some frontend parts of a scalable website will be thrown behind a load balancer. This allows a system to distribute the load evenly so that one server doesn't crash and take down the whole system.
- To do so, of course, you have to build out a network of cloned servers that all have essentially the same code and access to the same data.

---

## Database Denormalization and NoSQL

Joins in a relational database such as SQL can get very slow as the system grows bigger. For this reason, you would generally avoid them.

- **Denormalization** is one part of this.
  - adding redundant information into a database to speed up reads.
  - For example, imagine a database describing projects and tasks (in addition to the project table).
- Or, you can go with a NoSQL database.
  - A NoSQL database does not support joins and might structure data in a different way.
  - It is designed to scale better..



---



## Database Partitioning (Sharding)

- Sharding means splitting the data across multiple machines
- while ensuring you have a way of figuring out which data is on which machine.

A few common ways of partitioning include:

* **Vertical Partitioning**:
  * This is basically `partitioning by feature`.

* **Key-Based (or Hash-Based) Partitioning**:
  * This uses some part of the data to partition it.
  * A very simple way to do this is to allocate N servers and put the data on `mode(key, n)`.
  * One issue with this is that the number of servers you have is effectively fixed.
  * Adding additional servers means reallocating all the data -- a very expensive task.

* **Directory-Based Partitioning**:
  * In this scheme, you maintain a lookup table for where the data can be found.
  * This makes it relatively easy to add additional servers, but it comes with two major drawbacks.
    * First the lookup table can be a single point of failure.
    * Second, constantly access this table impacts performance.


---



## Caching

- An in-memory cache can deliver very rapid results.
- It is a simple key-value pairing and typically sits between your application layer and your data store.

---


## Asynchronous Processing & Queues

- Slow operations should ideally be done asynchronously.
- Otherwise, a user might get stuck waiting and waiting for a process to complete.

---

## Networking Metrics

* **Bandwidth**: This is the maximum amount of data that can be transferred in a unit of time. It is typically expressed in bits per seconds.
* **Throughput**: Whereas bandwidth is the maximum data that can be transferred in a unit of time, throughput is the actual amoutn of data that is transferred.
* **Latency**: This is how long it takes data to go from one end to the other. That is, it is the delay between the sender sending information (even a very small chunk of data) and the receiver receiving it.

---

## MapReduce

A MapReduce program is typically used to process large amounts of data.

* Map takes in some data and emits a pair
* Reduce takes a key and a set of associated values and reduces them in some way, emitting a new key and value.

MapReduce allows us to do a lot of processing in parallel, which makes processing huge amounts of data more scalable.


---

## Considerations

* **Failures**: Essentially any part of a system can fail. You'll need to plan for many or all of these failures.
* **Availability and Reliability**:
  * Availability is a function of `the percentage of time` the system is operatoinal.
  * Redliability is a function of `the probability` that the system is operational for a certain unit of time.
* **Read-heavy vs. Write-heavy**:
  * Whether an application will do a lot of reads or a lot of writes implacts the design.
  * If it's write-heavy, you could consider queuing up the writes (but think about potential failure here!).
  * If it's read-heavy, you might want to cache.
* **Security**:
  * Security threats can, of course, be devastating for a system.
  * Think about the tyupes of issues a system might face and design around thos.
