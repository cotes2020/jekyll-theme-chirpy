---
title: AWS - AWS Step Functions
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, Compute]
tags: [AWS]
math: true
image:
---

- [AWS Step Functions](#aws-step-functions)
  - [basic](#basic)

---

# AWS Step Functions

---

## basic

- `coordinate multiple AWS services into serverless workflows` to build and update apps quickly.

- `design and run workflows that stitch together services` such as AWS Lambda and Amazon ECS into feature-rich applications.
  - Workflows are made up of a series of steps, with the output of one step acting as input into the next.
  - Application development is simpler and more intuitive using Step Functions, because it translates workflow into a state machine diagram that is easy to understand, explain to others, and change.

- automatically triggers and tracks each step, and retries when there are errors, so application executes in order and as expected.
  - `monitor each step of execution` as it happens, which means can identify and fix problems quickly.

- `State machines` are used by Step Functions.
  - This product is serverless and can orchestrate long-running workflows involving other AWS services and human interaction.
