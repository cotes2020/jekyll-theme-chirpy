---
title: Software Development Lifecycle (SDLC)

# author: Grace JyL
date: 2020-09-28 11:11:11 -0400
description:
excerpt_separator:
categories: [14AppSec, SecureSoftwareDevelopment]
tags: [SecureSoftwareDevelopment, SDLC]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

# Software Development Lifecycle

- [Software Development Lifecycle](#software-development-lifecycle)
  - [Software Development Lifecycle (SDLC)](#software-development-lifecycle-sdlc)
  - [Framework](#framework)
  - [SDLC Models](#sdlc-models)
  - [ENV](#env)
    - [DEV](#dev)
    - [UAT](#uat)
    - [PROD](#prod)

---

## Software Development Lifecycle (SDLC)

- a comprehensive framework that describes the various stages involved in the development of software, from initial planning and requirements gathering to deployment and maintenance.

- The SDLC model follows a systematic and iterative approach to ensure that the software is developed efficiently, meets high-quality standards, and aligns with the needs of stakeholders.

---

## Framework

The Typical Phases of the Software Development Life Cycle (SDLC):

- **Plan**:
  - **Planning and Requirements Gathering**:
    - the initial phase where the objectives of the software project are defined, potential stakeholders are identified, and their requirements are gathered.
    - A detailed project plan is created, outlining the scope, timeline, and budget.

  - **Defining Requirements**:
    - the gathered requirements are analyzed, refined, and prioritized.
    - Functional and non-functional requirements are specified, providing a clear understanding of what the software needs to achieve.
  - When the development team collects requirements from stakeholders
    - Stakeholders can be:
    - Customers
    - Product managers
    - Management
    - Etc...
  - These requirements get turned into specification documents

- **Design**:
  - The software architecture and design are developed during this phase.
  - The system's structure is visualized, and detailed diagrams are created to represent the different components and how they will interact.
  - Software developers analyze the requirements and identify the best solutions to write software for those requirements
  - Including security in the planning phase helps the team come up with solutions that take into consideration security requirements
  - Implementing security requirements later in the SDLC is costly and frustrating

- **Implement**:
  - The actual coding of the software is performed in this phase.
  - The software is developed based on the detailed design specifications and requirements.
  - Code Review

- **Test**:
  - Thorough testing is conducted in this phase to identify and fix bugs, errors, and performance issues in the software.
  - Various testing methodologies, such as unit testing, integration testing, and system testing, are applied.

- **Deploy**:
  - Once the software is thoroughly tested and meets the required quality standards, it is deployed to the target environment.
  - This may involve installing the software on servers, configuring the system, and training end-users.
  - Production is the software and environment that costumers use
  - Prior to that, teams work in development, testing, and build environments

- **Maintain**:
  - Once software is in production and being used by customers, we need to continuously maintain it
  - This is the ongoing phase where:
    - the software is monitored, maintained, and updated to meet changing user needs and address any issues that arise after deployment. 
    - Bug fixes, enhancements, and performance improvements are continuously performed.
    - monitoring the software and the environment for overall performance issues and for security purposes

The SDLC model can be iterative, allowing for continuous feedback and improvement throughout the development process. Different organizations may adopt specific SDLC methodologies, such as Agile, Waterfall, Scrum, Kanban, or Extreme Programming, based on the nature of the project and their organizational preferences.

---

## SDLC Models

The SDLC gives us a framework, but there are multiple different models or implementations of that framework that organizations can choose to implement

Two common and well known examples include:

1. Waterfall
2. Agile

Regardless of which model an organization chooses, SDLC aims to integrate security as part of its process


---

## ENV

---

### DEV

**Dev (Development):**

- The 'Dev' environment is the initial stage where software development and testing begin.

- It is a controlled and isolated environment where developers can work freely, make code changes, experiment with new features, and run tests without affecting the main system.

- Characteristics of the Dev environment:

  - It is a copy of the production system but may not reflect the exact data or configurations.

  - It is accessible only to developers and a select few testers.

  - It is used for continuous integration (CI) and continuous deployment (CD) pipelines, where code is automatically built, tested, and deployed to this environment.


---

### UAT

- User Acceptance Testing

- a critical phase in the software development life cycle (SDLC) where the end-users or client representatives evaluate the software application or system to determine whether it meets their requirements and expectations.


- The 'UAT' environment is a staging area that bridges the gap between the development and production environments.

- It is designed to simulate the real-world production environment as closely as possible.

- Once the software is tested and deemed stable in the Dev environment, it is migrated to the UAT environment for final testing by end-users, business stakeholders, and QA testers.

- The purpose of UAT is to validate the software's functionality, usability, and performance in a real-world setting before it is deployed to production.

- Characteristics of the UAT environment:

  - It is a copy of the production environment with realistic data and configurations.

  - It is accessible to end-users, business stakeholders, and QA testers for testing purposes.

  - Any issues identified in UAT are addressed before the software is deployed to Prod.


Key Purpose of UAT:

- **Enhance User Experience**: UAT allows users to interact with the software in a realistic environment, providing feedback on usability, functionality, and user interface issues.
- **Confirm Compliance**: Users verify that the system aligns with their business processes, regulatory requirements, and data security standards.
- **Identify and Fix Errors**: UAT helps uncover bugs, bugs, performance problems, or compatibility issues that may have been overlooked during previous testing stages.
- **End-user Validation**: Ultimately, UAT ensures that the software is fits the purpose for which it was developed and will be adopted successfully by the end-users.


Types of User Acceptance Testing:

- **Alpha Testing**: Performed by internal stakeholders, such as developers, quality assurance testers, or a special testing team, to simulate real-world usage and identify critical issues.

- **Beta Testing**: Involves real end-users in an uncontrolled environment to gather feedback, identify usability problems, and measure performance.

- **Migration Testing**: Used when a system is migrated from an old environment to a new one, ensuring data integrity and smooth transition.

- **Regression Testing**: Verifies that existing features and functionalities are still working correctly after new changes or updates are made.

- **Integration Testing**: Ensures seamless interaction between different software modules or components.

- **Installation Testing**: Compares the actual installation process with the documented requirements, verifies data integrity, and checks for any errors.
UAT is a vital step in the software development process, bridging the gap between development and end-user deployment. It helps ensure that the delivered software meets the user's needs, enhances user satisfaction, and increases the likelihood of a successful software implementation.


---

### PROD

**Prod (Production):**

- The 'Prod' environment is the live operational environment where the final, tested, and approved software is running and serving end-users.

- It is the actual system that businesses rely on for their day-to-day operations.

- Deploying software to the Prod environment should be done with utmost caution and rigorous testing, as any issues can impact real users and business processes.

- Characteristics of the Prod environment:

  - It is the real-world live system with actual data and configurations.

  - It is accessible only to authorized end-users and business personnel.

  - Any changes or updates made to the Prod environment are closely monitored and controlled.
