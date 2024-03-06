---
title: SecConcept - Attack Surface Analysis
date: 2020-11-11 11:11:11 -0400
categories: [10SecConcept]
tags: [SecConcept]
toc: true
image:
---

[toc]

---


# Attack Surface Analysis

ways of doing Attack Surface Analysis and managing an application's Attack Surface

Attack Surface Analysis:
- map out what parts of a system need to be reviewed and tested for security vulnerabilities.
- to understand the risk areas in an application,
- to make developers and security specialists aware of what parts of the application are open to attack, to find ways of minimizing this
- to notice when and how the Attack Surface changes and what this means from a risk perspective.
- usually done by security architects and pen testers.
- But developers should understand and monitor the Attack Surface as they design and build and change a system.

Attack Surface Analysis helps you to:
- identify what functions and what parts of the system need to review/test for security vulnerabilities
- identify high risk areas of code that require defense-in-depth protection - what parts of the system that you need to defend
- identify when you have changed the attack surface and need to do some kind of threat assessment

---

![Scout-AS-transparent](https://i.imgur.com/duCoh41.png)


7 common attack vectors for web application vulnerabilities

1. V1 Security mechanisms (SM):
   - How the traffic between users and the application is secured
   - i.e is there authentication in place?
2. V2 Page creation method (PCM):
   - This depends on the code the website has been developed in
   - i.e some code languages are more exposed than others and as new versions are released this will fix these issues.
   - Developing a website with insecure code means there are allot more potential vulnerabilities from old and not up to date code which is easy for a hacker to exploit.
3. V3 Degree of distribution (DOD):
   - The more pages you have, the more risks there are, therefore all pages must be identified and vulnerabilities uncovered at all levels.
4. V4 Authentication (AUTH):
   - Authentication is the process of verifying the identity of an individual accessing your application.
   - Access to certain actions or pages can be restricted using user levels set up by the administrator and critical to keeping the bad guys out.
5. V5 Input vectors (IV):
   - The attack surface increases with the number of different input fields you have on a web application and can lead to XSS attack.
6. V6 Active contents (ACT):
   - As soon as an application runs scripts, we have active contents and depending on the way those scripts have been implemented, the attack surface could increase if a website has been developed using several active content technologies.
7. V7 Cookies (CS):
   - Cookies are essential for real time application security,
   - by monitoring session activity and ensuring anyone who sends requests to your website are allowed access and keep hackers away from unauthorized areas.

---

## Defining the Attack Surface of an Application

The Attack Surface describes all of the different points where an attacker could get into a system, and where they could get data out.

The Attack Surface of an application is the sum of
1. all <font color=red> paths for data/commands </font> into and out of the application
2. the <font color=red> code that protects these paths </font>, including
   - resource connection and authentication,
   - authorization,
   - activity logging,
   - data validation and encoding
3. all <font color=red> valuable data </font> used in the application, including
   - secrets and keys,
   - intellectual property,
   - critical business data,
   - personal data and PII
4. <font color=red> the code that protects these data </font>, including
   - encryption and checksums,
   - access auditing,
   - and data integrity
   - and operational security controls


overlay this model with the different types of users - roles, privilege levels - that can access the system (whether authorized or not).
- Complexity increases with the number of different types of users.
- But focus especially on the two extremes:
  - <font color=red> unauthenticated, anonymous users </font>
  - and <font color=red> highly privileged admin users </font> (e.g. database administrators, system administrators).


Group each type of attack point into buckets based on risk (external-facing or internal-facing), purpose, implementation, design and technology.
- then count the number of attack points of each type,
- then choose some cases for each type,
- and focus your review/assessment on those cases.

With this approach, you don't need to understand every endpoint in order to understand the Attack Surface and the potential risk profile of a system.
- Instead, you can count the different general type of endpoints and the number of points of each type.
- With this you can budget what it will take to assess risk at scale, and you can tell when the risk profile of an application has significantly changed.


---


## Identifying and Mapping the Attack Surface

1. building a baseline description of the Attack Surface
2. reviewing design and architecture documents from an attacker's perspective.
3. Read through the source code and identify different points of entry/exit:
   - User interface (UI) forms and fields
   - HTTP headers and cookies
   - APIs
   - Files
   - Databases
   - Other local storage
   - Email or other kinds of messages
   - Runtime arguments
   - ...Your points of entry/exit

4. The total number of different attack points can add up to thousands
   - To make this manageable, break the model into different types based on function, design and technology:
     - Login/authentication entry points
     - Admin interfaces
     - Inquiries and search functions
     - Data entry (CRUD) forms
     - Business workflows
     - Transactional interfaces/APIs
     - Operational command and monitoring interfaces/APIs
     - Interfaces with other applications/systems
     - ...Your types

5. identify the valuable data in the application,
   - (e.g. confidential, sensitive, regulated)
   - by interviewing developers and users of the system
   - by reviewing the source code.

6. build up a picture of the Attack Surface by scanning the application.
   - For web apps you can use a tool like
     - the OWASP ZAP or Arachni or Skipfish or w3af
     - or one of the many commercial dynamic testing and vulnerability scanning tools
     - or services to crawl your app and map the parts of the application that are accessible over the web.
   - Some web application firewalls (WAFs) may also be able to export a model of the application's entry points.

7. Validate and fill in your understanding of the Attack Surface
   - walking through some of the main use cases in the system:
     - signing up and creating a user profile,
     - logging in,
     - searching for an item, placing an order, changing an order, and so on.
   - Follow the flow of control and data through the system,
     - see how information is validated and where it is stored,
     - what resources are touched and what other systems are involved.
   - There is a recursive relationship between <font color=red> Attack Surface Analysis </font> and <font color=red> Application Threat Modeling </font>
     - changes to the Attack Surface should trigger threat modeling,
     - and threat modeling helps understand the Attack Surface of the application.

The Attack Surface model may be rough and incomplete to start, especially if you haven't done any security work on the application before.
- Fill in the holes as you dig deeper in a security analysis, or as you work more with the application and realize that your understanding of the Attack Surface has improved.

---

## Measuring and Assessing the Attack Surface

1. have a map of the Attack Surface, identify the high risk areas.
   - Focus on remote entry points – interfaces with outside systems and to the Internet – and especially where the system allows anonymous, public access.
   - Network-facing, internet-facing code
   - Web forms
   - Files from outside of the network
   - Backward compatible interfaces with other systems
     - old protocols, sometimes old code and libraries, hard to maintain and test multiple versions
   - Custom APIs
     - protocols etc – likely to have mistakes in design and implementation
   - Security code
     - anything to do with cryptography, authentication, authorization (access control) and session management
   - These are often where you are most exposed to attack.

2. Then understand your control
   - what compensating controls you have in place,
   - operational controls like network firewalls and application firewalls, and intrusion detection or prevention systems to help protect your application.

Measure:
1. <font color=red> Relative Attack Surface Quotient (RSQ) </font>
   - Michael Howard at Microsoft and other researchers have developed this method for measuring the Attack Surface of an application, and to track changes to the Attack Surface over time
   - this method calculate an overall attack surface score for the system,
   - and measure this score as changes are made to the system and to how it is deployed.


1. Researchers at Carnegie Mellon built on this work to develop a formal way to calculate an Attack Surface Metric for large systems like SAP.
   - They calculate the Attack Surface as the sum of
     - all entry and exit points, channels (the different ways that clients or external systems connect to the system, including TCP/UDP ports, RPC end points, named pipes...)
     - and untrusted data elements.
   - Then they apply a damage potential/effort ratio to these Attack Surface elements to identify high-risk areas.

increases the Attack Surface.
- deploying multiple versions of an application,
- leaving features in that are no longer used just in case they may be needed in the future,
- or leaving old backup copies and unused code increases the Attack Surface.

decrease the Attack Surface.
- Source code control and robust change management/configurations practices should be used
- to ensure the actual deployed Attack Surface matches the theoretical one as closely as possible.
- Backups of code and data - online, and on offline media - are an important but often ignored part of a system's Attack Surface.
- Protecting your data and IP by writing secure software and hardening the infrastructure will all be wasted if you hand everything over to bad guys by not protecting your backups.


---


## Managing the Attack Surface
Once you have a baseline understanding of the Attack Surface, you can use it to incrementally identify and manage risks going forward as you make changes to the application. Ask yourself:
- What has changed?
- What are you doing different? (technology, new approach, ….)
- What holes could you have opened?


1. The first web page that you create opens up the system's Attack Surface significantly and introduces all kinds of new risks.
   - If you add another field to that page, or another web page like it, while technically you have made the Attack Surface bigger, you haven't increased the risk profile of the application in a meaningful way.
   - Each of these incremental changes is more of the same, unless you follow a new design or use a new framework.

2. If you add another web page that follows the same design and using the same technology as existing web pages, it's easy to understand how much security testing and review it needs.
   - If you add a new web services API or file that can be uploaded from the Internet, each of these changes have a different risk profile again - see if if the change fits in an existing bucket, see if the existing controls and protections apply.
   - If you're adding something that doesn't fall into an existing bucket, this means that you have to go through a more thorough risk assessment to understand what kind of security holes you may open and what protections you need to put in place.

3. Changes to session management, authentication and password management directly affect the Attack Surface and need to be reviewed.
   - So do changes to authorization and access control logic, especially adding or changing role definitions, adding admin users or admin functions with high privileges.
   - Similarly for changes to the code that handles encryption and secrets.
   - Fundamental changes to how data validation is done.
   - And major architectural changes to layering and trust relationships, or fundamental changes in technical architecture – swapping out your web server or database platform, or changing the runtime operating system.

4. add new user types or roles or privilege levels, you do the same kind of analysis and risk assessment.
   - Overlay the type of access across the data and functions and look for problems and inconsistencies.
   - It's important to understand the access model for the application, whether it is positive (access is deny by default) or negative (access is allow by default).
   - In a positive access model, any mistakes in defining what data or functions are permitted to a new user type or role are easy to see.
   - In a negative access model, you have to be much more careful to ensure that a user does not get access to data/functions that they should not be permitted to.

5. threat or risk assessment can be done periodically, or as a part of design work in serial / phased / spiral / waterfall development projects, or continuously and incrementally in Agile / iterative development.




.
