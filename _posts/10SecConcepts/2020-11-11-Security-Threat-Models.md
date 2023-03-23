---
title: SecConcept - Security Threat Models
date: 2020-11-11 11:11:11 -0400
categories: [10SecConcept]
tags: [SecConcept]
toc: true
image:
---

[toc]

---

## Security Threat Models

---


five aspects to security threat modeling:

security assessment should broadly include two components:

- **Security review**: A collaborative process that includes `identifying security issues and their level of risk, preparing a plan to mitigate these risks`.
  1. Create a core assessment team.
  2. Review existing security policies.
  3. Create a database of IT assets.
     1. model the system either with `data flow diagrams (DFDs)` or `UML deployment diagrams`
  4. Identify threats.
     1. From these diagrams, identify `entry points to system` such as data sources, application programming interfaces (APIs), Web services and the user interface itself.
     2. Because an adversary gains access to system via entry points, they are your starting points for understanding potential threats.
     3. To help identify security threats
        1. ![databaseTesting](https://i.imgur.com/l4HGSBM.jpg)
        2. add "`privilege boundaries`" with dotted lines
        3. to explain the boundaries applicable to testing a relational database.
        4. A privilege boundary separates processes, entities, nodes and other elements that have different trust levels.
     4. Wherever aspects of system cross a privilege boundary, security problems can arise.
     5. For example, system's ordering module interacts with the payment processing module. Anybody can place an order, but only manager-level employees can credit a customer's account when he or she returns a product.
     6. At the boundary between the two modules, someone could use functionality within the order module to obtain an illicit credit.
  5. Understand the threat
     1. understand the potential threats at an entry point,
     2. identify any security-critical activities that occur and imagine what an adversary might do to attack or misuse system.
     3. Ask yourself questions such as "How could the adversary use an asset to modify control of the system, retrieve restricted information, manipulate information within the system, cause the system to fail or be unusable, or gain additional rights. In this way, determine the chances of the adversary accessing the asset without being audited, skipping any access control checks, or appearing to be another user.
     4. To understand the threat posed by the interface between the order and payment processing modules, you would identify and then work through potential security scenarios. For example, an adversary who makes a purchase using a stolen credit card and then tries to get either a cash refund or a refund to another card when he returns the purchase.
  6. Categorize the threats.
     1. consider the STRIDE (Spoofing, Tampering, Repudiation, Information disclosure, Denial of Service, and Elevation of privilege) approach.
     2. Classifying a threat is the first step toward effective mitigation.
     3. For example, if you know that there is a risk that someone could order products from your company but then repudiate receiving the shipment, you should ensure that you accurately identify the purchaser and then log all critical events during the delivery process.
  7. Identify mitigation strategies.
     1. create a diagram `threat tree`.
        1. root of the tree is the threat itself
        2. its children (or leaves) are the conditions that must be true for the adversary to realize that threat.
        3. Conditions may in turn have subconditions.
     2. For example, under the condition that an adversary makes an illicit payment. The fact that the person uses a stolen credit card or a stolen debit/check card is a subcondition.
     3. For each of the leaf conditions, you must identify potential mitigation strategies; in this case, to verify the credit card using the XYZ verification package and the debit card with the issuing financial institution itself.
     4. Every path through the threat tree that does not end in a mitigation strategy is a system vulnerability.

  8. Estimate the impact.
     1. For example, what impact would a credit card data breach have on your business?
     2. The impact could be in monetary terms, loss of clients, or loss of brand value or credibility.
     3. Categorize the impact as “high, “medium,” or “low” based on its severity and estimated cost.
  9.  Determine the likelihood.
     4. Categorize the likelihood that each potential risk would happen as “high,” “medium,” or “low.”
  10. Plan the controls.
     5. List the `existing control systems` in place and outline further actions that can help mitigate the identified risks.
     6. These controls can include a `change in policies or procedures, application procurement, training content and configurations, or implementation of new applications and/or hardware`.
  11. creating a risk matrix like the example below to assess your security posture.
     7. Prepare a report that summarizes your findings
     8. Take steps to implement the needed actions


- **Security testing**: The process of `finding vulnerabilities` in software applications or processes.
  - `Cyberattack simulation tests`. Authorized simulation attacks on your computer system help identify the weaknesses as well as the strengths of your existing system. For example, a phishing simulation tool can help identify risky employee behavior while training them to spot scam emails.
  - `Security scanning`. Use security software to run a complete scan of applications, networks, and devices at least once a month to identify threats and risks. Most security software provides real-time and automatic scanning features. If you don't have security software in place, implementing such a system should be a priority.
  - `Vulnerability scanning`. vulnerability assessment is a set of processes that help you identify vulnerabilities and rate them based on the severity of issue they can potentially cause. Some ways identify vulnerabilities include:
    - Check whether you are using outdated versions of software.
    - Use an Active Directory management tool to identify users with weak domain passwords. Eighty-one percent of security breaches leveraged stolen or weak passwords in 2017.
    - Use vulnerability management software to automatically scan your systems and detect weaknesses.
  - `Survey employees to identify weaknesses`.
    - human error is a major cause of cyber attacks.
    - Interviewing employees helps to identify risky behavior and correct bad practices.
    - Here are some sample questions ask when conducting your security assessment, as well as some potential responses to look out for and recommended actions take to mitigate risk:
  - `Ensure supplier compliance`.
    - ensure security compliance within your organization,
    - verify the credentials of your vendors and other business partners.
    - Electronics company Acer suffered a breach because of a security issue at one of its third-party payment processing companies. The data of over 34,000 customers was reported stolen.
    - As we saw above, the breach at Target was also engineered when hackers targeted a third-party vendor. Routinely check with your suppliers and business partners through surveys and questionnaires to ensure that they arecompliant with all industry regulations.




















。
