---
title: AWS - CodeDevelop - CodeBuild - buildspec.yml Template
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

- [buildspec.yml Template](#buildspecyml-template)

---

# buildspec.yml Template


```yml

# example from AWS WhitePaper, no real credential inside, no real credential inside

version: 0.2

#env:
  #variables:
     # key: "value"
     # key: "value"
  #parameter-store:
     # key: "value"
     # key: "value"

phases:
  install:
    runtime-versions:
        docker: 18
    commands:
      - nohup /usr/local/bin/dockerd --host=unix:///var/run/docker.sock --host=tcp://127.0.0.1:2375 --storage-driver=overlay2&
      - timeout 15 sh -c "until docker info; do echo .; sleep 1; done"
  pre_build:
    commands:
    - echo Logging in to Amazon ECR....
    - aws --version
    # update the following line with your own region
    - $(aws ecr get-login --no-include-email --region eu-central-1)
  build:
    commands:
    - echo Build started on `date`
    - echo Building the Docker image...
    # update the following line with the name of your own ECR repository
    - docker build -t mydockerrepo .
    # update the following line with the URI of your own ECR repository (view the Push Commands in the console)
    - docker tag mydockerrepo:latest 757250003982.dkr.ecr.eu-central-1.amazonaws.com/mydockerrepo:latest
  post_build:
    commands:
    - echo Build completed on `date`
    - echo pushing to repo
    # update the following line with the URI of your own ECR repository
    - docker push 757250003982.dkr.ecr.eu-central-1.amazonaws.com/mydockerrepo:latest
#artifacts:
    # - location
    # - location
  #discard-paths: yes
  #base-directory: location
#cache:
  #paths:
    # - paths
```
