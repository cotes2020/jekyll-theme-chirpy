---
title: AWS - CodeDevelop - ElasticBeanstalk
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

- [AWS ElasticBeanstalk     ￼￼](#aws-elasticbeanstalk-----)
  - [basics](#basics)
  - [benefits](#benefits)
  - [Blue/green deployment](#bluegreen-deployment)
  - [Elastic Beanstalk for docker](#elastic-beanstalk-for-docker)

---

# AWS ElasticBeanstalk     ￼￼


![ElasticBeanstalk](https://i.imgur.com/poyMxI7.png)



## basics

- AWS compute service option.
- PaaS

- deploys your code on:
  - Apache Tomcat for Java applications;
  - Apache HTTP Server for PHP and Python applications;
  - NGINX or Apache HTTP Server for Node.js applications;
  - Passenger or Puma for Ruby applications;
  - Microsoft Internet Information Services (IIS) for .NET applications, Java SE, Docker, and Go.

<img src="https://i.imgur.com/pr7Dh9z.png" width="400">

Setup:
1. Elastic Beanstalk > create a web app
   - single docker container or multiple docker container
   - upload application code zip
     - `dockerfile`
     - `application.py`
   - create application
   - get the url
     - <font color=red> web application server from the container running in Elastic Beanstalk </font>

2. upgrade
   - select <font color=blue> upload and deploy </font>
   - go to version page and select the previous version.


---

## benefits

- fully managed
  - The entire platform is already built, <font color=red> only need to upload code </font>

- fast and simple way
  1. to <font color=red> get web application up and runninng </font>
     - quick deployment, scaling, and management of web applications and services.
       - Choose instance type, database, set and adjust automaticscaling, update application, access the server log files, and enable HTTPS on the load balancer.
     - provides all the application services that you need for your application.
       - make deploying your application a quick and easy process.
       - Use the AWS Management Console, a Git repository, or an integrated development environment(IDE) such as Eclipse or Visual Studio to upload your application.
       - deploy code through the AWS Management Console, AWS CLI, Visual Studio, and Eclipse.
       - supports a broad range of platforms (Docker, Go, Java, .NET, Node.js, PHP, Python, and Ruby).
  2. <font color=red> supports the deployment of Docker containers </font>
     - Docker containers: are self-containered and include all the cofiguration information and software your web application required to run,
       - libraries, system tools, code&runtime

- <font color=red> automated deployment scaling service for web applications </font>
  - Elastic Beanstalk automatically handles the deployment details of
    - <font color=blue> capacity provisioning, Load balancing, auto scaling, and application health monitoring </font>
    - Application platform management
    - automated infrastructure management
    - Code deployment

- <font color=red> improve developer productivity </font>
  - focusing on writing code
  - instead of managing and configuring servers, databases, load balancers, firewalls, and networks.
  - AWS updates the underlying platform that runs your application with patches and updates.

- Elastic Beanstalk is <font color=red> difficult to outgrow </font>
  - With Elastic Beanstalk, the application can handle peaks in workload or traffic while minimizing your costs.
  - It automatically scales your application up or down based on your application's specific needs by using easily adjustable automatic scaling settings.
  - use CPU utilization metrics to trigger automatic scalingactions.

- free to select the AWS resource (like EC2 instance type) optimal for the application.
  - retain full control over the AWS resources that power the application
  - and can access the underlying resources at any time.
  - If decide to take over some (or all) of the elements of the infrastructure, do so seamlessly by using the management capabilities that are provided by Elastic Beanstalk.


- no additional charge for AWS Elastic Beanstalk.
  - pay for the AWS resources (for example, EC2 instances or S3 buckets) created to store and run your application.
  - only pay for what you use, as you use it.
  - no minimum fees or upfront commitments.



---

## Blue/green deployment

One of the challenges of automating deployment is the cutover,
- when take software from the final stage of testing to live production.


Blue/green deployment on AWS Elastic Beanstalk
- Blue/green
  - the live production environment is “blue”
  - the new deployment environment is “green”
  - <img src="https://i.imgur.com/lkVk2sc.png" width="300">

- <fonr color=red> test new hardware or applications without going fully into production </font>
- quickly deploy the application without downtime for web application.
  - deploy updates to the green deployment and attach it to your load balancer.
  - After the green deployment is complete and functional, begin to shut down or upgrade the blue deployment.
  - also can rapid roll back switching back to blue deployment if the green environment is not working properly.


Blue/green deployment on AWS CloudFormation
- AWS CloudFormation templates were used instead of Elastic Beanstalk.
  - It takes little more effort than the Elastic Beanstalk approach.
  - <img src="https://i.imgur.com/kVF1Tav.png" width="400">
- use AWS CloudFormation to implement the blue/green deployment.
  - Traffic was trickled from Stack 1 to Stack 2 until it was apparent that Stack 2 was functional.
  - After Stack 2 was functional, the connection to Stack 1 (former production environment) was taken away.
  - Stack 2 became the new production environment, and the old production environment was torn down.

- used if your code is using a supported runtime (Ruby, Python, etc.)
- when needing minimal to no admin overhead
- key architecture components in Elastic Beanstalk
  - an application
    - The base entity of Elastic Beanstalk is an application.
    - An Elastic Beanstalk's application can be thought of as a container.
  - environment
    - work environment or web server environment.
    - allows for quick environment deployment and management of an application
    - an application can contain zero to multiple environments
    - Each environment has a different URL
      - can use each URL for A/B testing to see which application version is better for users.
    - Environments live in an application container and it references a specific application version
    - application container > environment > a single application version
  - An application version
    - a distinct version of an app's code that's packaged into a source bundle.


---

## Elastic Beanstalk for docker

1. deploy docker container
   - <font color=red> single docker container </font>
     - run a <font color=blue> single docker container on an EC2 instance </font> provisioned by Elastic Beanstalk
   - <font color=red> multiple docker container </font>
     - use Elastic Beanstalk to <font color=blue> build an ECS cluster and deploy multiple docker container on each instance </font>

2. deploy your code
   - upload a zip file containing the code bundle and Elastic Beanstalk will do the rest.

3. upload your code
   - upgrade your application to a new version
   - one easy step in the concole to upload and deploy.

---




.
