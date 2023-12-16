---
title: Palo Alto Networks - Cortex SOAR Security Orchestration
# author: Grace JyL
date: 2020-10-18 11:11:11 -0400
description:
excerpt_separator:
categories: [SOC, PaloAlto]
tags: [SOC]
math: true
# pin: true
toc: true
image: /assets/img/note/prisma.png
---

[toc]

---


# Cortex SOAR Security Orchestration

an environment with applications targeted at current security operations team needs

![Screen Shot 2020-12-08 at 23.25.31](https://i.imgur.com/BRTXerl.png)

![Screen Shot 2020-12-08 at 23.26.07](https://i.imgur.com/NFi9Uct.png)

![Screen Shot 2020-12-08 at 23.27.22](https://i.imgur.com/Y744Rjp.jpg)


![Screen Shot 2020-12-02 at 01.59.36](https://i.imgur.com/leY439c.png)


## Positioning Cortex XSOAR

1. **Orchestration**
   - Orchestration coordinates disparate technologies through workflows so they function together.
   - It uses both security-specific and non-security-specific technologies.
   - `Management, Orchestration, and Collaboration`: Cortex XSOAR pulls together these together to provide security operations functionality that is greater than the sum of its parts.
   - ![small](https://i.imgur.com/ugRB2jt.png)

2. **Automation**
   - Automation is execution by machine rather than humans.
   - But security operations cannot now be fully automated.
   - automate some tasks in a larger context of mixed human and machine execution.
   - enables automation of `many well-defined, routine tasks` for which all required information is available or automatically obtainable.

3. **Incident Management and Response**
   - Incident management and response is another key element of SOAR.
   - SOAR facilitates a comprehensive, end-to-end understanding of incidents by security teams, and this results in `faster, more accurate, and more effective response`.

4. **Dashboards and Reports**
   - Dashboards and reports are important for effective SOAR.
   - Data visualizations where incidents are `easily seen, correlated, triaged, documented, and measured` help achieve quicker and more comprehensive response.
   - help provide information critical for process improvement.
   - These reports can be edited, customized, or created new and can help meet compliance requirements.

5. Playbook and Orchestration Force Multiplier
   - playbooks help realize a concrete benefit of orchestration.
     - Analysts develop habits of using go-to technologies in their workflows.
     - These habits exclude other technologies that may be available to the SOC.
     - Cortex XSOAR can tie all these technologies together with a single command or task.
   - For example,
     - had a few different sandbox products in environment, we could use a single command to send a sample to all of them.
     - We’d be making better use of our security resources and products.
     - Cortex XSOAR can bring a lot of technology-specific requirements into a single abstraction and become a force multiplier.

---


## user cases

> Customer
> Esri applies advanced geospacial technology to help more than 350,000 customers.

> Challenge
> Esri's vast customer base created an alert volume of greater than 10,000 per week. This caused significant fatigue among their five SOC analysts.
> They were not addressing false positives and duplicate incidents. They also wanted to streamline their threat indicator management processes, which were distributed and complex. They saw risk and a drain on SOC resources.


> Response
> Esri deployed Cortex XSOAR Enterprise in addition to their existing SIEM and Network Monitoring solutions.
> For quicker triage and response to their high incident volume, they used custom playbooks that combined automated and manual tasks. These playbooks incorporated analyst knowledge to enable standardized responses to specific attacks.
> For false positives and deduplication, Esri used Cortex XSOAR's historical cross-correlation capabilities. By quickly highlighting common artifacts and indicators across incidents, Esri analysts were able to find duplicate attacks reduce redundant investigations.


> Customer
> Esri applies advanced geospacial technology to help more than 350,000 customers.

> Challenge
> With more than 2.8 million subscribers, it is critical for this customer to protect its digital and infrastructure assets from compromise. The customer offers a broad range of services, and security is a multi-team effort. It was a challenge to coordinate among security, development, and production teams, for both security operations and incident response.
> There was no defined SOC team, a hundred daily alerts, and lost time during incident handoffs. There were a variety of ingestion and detection sources. A SIEM aggregated logs and machine data into alerts, but some incidents also flowed in via mailboxes where employees forwarded suspected phishing emails. There was no unified console to view alerts and respond at scale.


> Response
> The customer solves these challenges by deploying Cortex XSOAR along with its existing SIEM, threat intelligence, email, and behavioral analysis solutions.
> Cortex XSOAR's orchestration enabled alerts to be ingested across sources, and the customer directed alerts from its SIEM and mailboxes into Cortex XSOAR for visibility, triage, and response.
> The customer deployed a custom playbook that coordinated across a range of products for automated malware enrichment and response.
> The playbook receives alerts from the SIEM and runs initial threat intelligence actions to obtain indicator reputation. It then retrieves endpoint details using integrations with relevant tools and runs behavioral analytics using a customer-owned custom tool. It uses Cortex XSOAR to control infected endpoints.


**Business Use Cases**
- `MTTRs` (reduce customers' mean time to response) and `SLAs` (achieve their service level agreements).
- Customers commonly have security SLAs with no specific tools to implement them. The SLA discussion can be productive when it moves to how Cortex XSOAR enforces SLAs at the task and playbook level.
- Orchestration, automation, and incident management are all good discussions around MTTR.


**Technical Use Cases**
- Technical use cases are guided by customer needs, but are often productive discussions about how Cortex SOAR facilitates the handline of phishing, false positive alerts, automated incident response, cloud security, and threat hunting.


**Customer’s Environment**
- Teams and Team Sizes
- Centralized SOC
- Managed Security Provider
- Ticketing
- Scripts
- SLA Enforcement
- Current Products
- Speed and Scale
- Forensics



**Engaging Customer Personas for Cortex XSOAR**
- `CISO`
  - If there is a Chief Information Security Officer, target the CISO.
  - The CISO will most likely be the key decision maker.
  - If there is not CISO, a direct report to the head of IT in charge of security might be the key decision maker.
- SOC Lead
  - If there is a SOC or security operations team, target the lead of this group.
  - The SOC lead will inform and influence the key decision maker.
  - If there is no formal SOC or security operations team, a security engineer often fills this role.
- Security Analyst
  - Security analysts will be the end user of Cortex SOAR.
  - In some cases, this will be the security engineer.
  - Security analysts will also give opinions and advice to the key decision maker, and may become champions within the customer.


**Engaging Speedboat Specialists**
- Return on Investment
  - if they think they are getting a good `Return on Investment (ROI)` on their existing tool investments.

- Customer Dissatisfaction
  - Are there any products not doing their jobs properly? If so, Cortex SOAR is likely to dramatically increase ROI on the customer's existing investments and be a compelling solution. SEs should engage specialists.

- Mean Time to Resolve
  - If MTTR is important to the customer, SEs should bring a Cortex XSOAR specialist to help. Cortex XSOAR is very successful at reducing MTTR and Cortex XSOAR speedboat specialists are prepared to show it.


---


## SOC and SOAR Challenges

1. Alert Handling
   - a precipitous rise in the number of alerts and the time and expertise required to handle them.
   - There are more attacks, attacks that do more things to be alerted about, and more and better tools that identify activity that generates alerts.


2. Talent Shortage
   - small pool of cybersecurity professionals.
   - skills gap in those professionals they can hire.


3. Process Issues
   - Current security operations processes lead to redundant activity but still don't thoroughly and consistently address their security objectives.
   - The processes are manual, and error prone, and lead to inefficient allocation of people and tools.


4. Point Products as Tools
   - Most tools are point products with narrow roles. lack integration.
   - These tools span across vendors, functions, and data standards.
   - extra effort to switch context, centralize data, and coordinate actions across different browser tabs and user interface consoles.
   - And groups of tools targeting specific functions need better interconnectivity with groups of tools targeting other functions.
   - They need data organization standards and data transfer standards, and the ability to conveniently control actions across security tools. This action control


5. SIEMs
   - rely heavily on SIEM tools for incident ingestion and enrichment.
   - also rely on them for investigation, and for tracking metrics and performance.
   - SIEM tools lack the functionality necessary to properly handle these tasks.


---


## Incident Response Challenges

incident resolution processes

- **Automated Data Enrichment**
  - enrich data through automated requests to threat intelligence feeds, SIEMs, and other data sources.
  - Ideally, alert-related data can also be automatically enriched with root-cause or behavioral analysis.


- **Mobile Application Support**
  - Customers want mobile access to certain key functions.
  - The strongest demands are to be able to view dashboards, manage incident and task lists, assign actions to analysts, set incident severity levels, and close incidents from mobile devices.


- **Autodocumentation**
  - manual documentation mechanisms, results in `fragmented and inconsistent documentation`.
  - Better documentation tools `automatically capture information that can be used for post-incident review` and used to improve processes moving forward.
  - Documentation should `include all tasks, comments, and actions` associated with incident response activity.
  - Automated documentation can be complete and consistent.


- **Single Evidence Repository**
  - Customers need something like an evidence board to enable effective and efficient response to the whole set of incidents security operations teams face.
  - This is a place to collect key information or evidence to help reconstruct attack chains and to help find common elements across incidents.
  - An evidence board allows analysts to pivot from one suspicious indicator to another while tracking and gathering critical evidence.
  - It's also helpful for tools to autopopulate information associated with items in the evidence board.


- **Live Run on Incident**
  - A live run of a playbook on an incident is a way to trace a playbook's progress through an incident.
  - This provides visibility into incident response processes and makes task-specific troubleshooting much easier.

![extraLarge](https://i.imgur.com/kYobMEh.png)


---

---

## Cortex as a Solution


![original](https://i.imgur.com/jStbU6i.png)


---

### Integrations

Integrations are the way Cortex XSOAR connects to the world.

1. **Integrations from Third-Party Sources**
   - Integrations are used by Cortex XSOAR to communicate with third-party security products, services, tools, and platforms. Integrations can work with anything that has a public API.
   - Cortex XSOAR includes more than 300 integrations for security products. These integrations are organized into customizable categories.
   - Palo Alto Networks and customers can create their own additional integrations, which typically takes a few weeks.
   - The Cortex XSOAR user community makes integrations available on GitHub.

2. **Incidents from Alerts**
   - Third-party security offerings provide alerts, incidents, tickets, events, or cases to Cortex XSOAR, and Cortex XSOAR ingests them as incidents.
   - Cortex XSOAR integrations allow alerts from SIEMs and other services and servers to be ingested as incidents. But Cortex XSOAR integrations can also be used to enrich existing incidents.
   - Integrations are programmed to automatically associate these outside triggers with appropriate incidents, and to place information from third-party sources into predefined or custom-defined fields.
   - Integrations also enable API requests to these sources to enrich the data associated with an alert or its incident. Integrations are configured and can be further fine-tuned to minimize ingestion of false positives.


---

### Three Modes of Deployment

Cortex XSOAR can be deployed as
- a `fully hosted service`

- a `managed security service provider (MSSP)`
  - Cortex XSOAR enables MSSPs to gain immediate ROI from their existing threat intelligence investments by optimizing their SOC or by monetizing SOAR services.
  - This in turn allows MSSPs to provide immediate ROI to their customers.

- a product integrated with `cloud-based services`
  - When Cortex XSOAR is delivered from the cloud, Cortex Data Lake provides it cloud-based, centralized log storage and aggregation for on premises, private cloud and public cloud firewalls, for Prisma Access, and for other cloud-delivered services such as Cortex XDR.

- or as a `customer’s on-premises system`.


----


### Incidents


Cortex XSOAR provides a comprehensive, end-to-end treatment of incidents by security teams
- resulting in better and more informed response.
- This combines with other product functionality such as autodocumentation and playbook development to enable `continuous improvement of incident response processes`.


- Ingestion
  - An incident is created when an alert from a source such as a `SIEM, email security tool, or firewall` is brought into Cortex XSOAR.
  - Incidents can also be created manually.

- Enrichment
  - Additional information associated with alerts helps with incident response.
  - This information assists investigation efforts, helps to reduce false positives, and facilitates duplication of alerts from multiple sources.

- Investigation
  - Analysts can investigate incidents automatically using `playbooks` or `manually`.
  - The **playbook**
    - automatically assigned to an incident when it is ingested based on the incident's type.
    - Cortex XSOAR provides customizable incident dashboards and reports to help investigation.
  - Investigation page in Cortex XSOAR.
    - This page is based on collected data
    - This page help make decisions about the case.
    - show at a glance things in the environment that indicate a false positive or things that warrant more scrutiny.
    - We can customize this page by incident type.
    - For example,
    - can show different pages for `Phishing incidents vs. Cortex XDR Prevent incidents vs. SIEM correlations`.
    - can add or remove fields, define custom fields, and remove, add, or reorder sections on the page.
    - We can also build views that show things beyond data in incident fields.
    - For example, we can show such things as a visualization of a database lookup.

- Response
  - Cortex XSOAR works with products that enforce and provide response actions such as Cortex XDR and next-generation firewalls.
  - Responses are orchestrated to be automatic or manually controlled.


---


## Playbooks

- self-documenting procedures that query, analyze, and perform other actions on incidents using the information associated with those incidents.
- organize and document security monitoring orchestration and response activities for Cortex XSOAR.
- Playbooks for common investigation scenarios, such as phishing, come out-of-the-box and customers can create their own playbooks.

- **Playbooks and Tasks**
  - Playbooks comprise of tasks that each perform a specific action.
  - Examples of tasks used in playbook
  - extracting an IP address from a suspected email, checking the reputation of an indicator, sending an email or phoning a manager, and performing a comparison or transformation on a field of an incident.

- **Creating Playbooks**
  - A playbook is created as a sequence of tasks and decision points, normally using a WYSIWYG editor.
  - Playbooks are like flowcharts. They have different branches that can do different things.
  - Decisions influence the flow of the book.
  - These decisions can be made automatically, manually, or with a mix of the two.
  - Cortex XSOAR includes a rich environment to support customers' development and testing of playbooks to meet their specific needs.


- **Subplaybooks**
  - Subplaybooks are themselves playbooks included in larger playbooks.
    - reused just as a subroutine in a library.
    - simply dragged and dropped into a new or edited playbook.
  -  technology or configuration change that requires adjustment to a playbook’s workflow,
    - only have to update the workflow in the playbook that’s used as a subplaybook.
     - don’t have to touch every individual playbook that makes use of this workflow.
  - Cortex XSOAR comes with a large number of playbooks and sub-playbooks out-of-the-box, so we don’t have to build them all on our own.
  - For custom use cases, playbooks can be easy to build.
  - Palo Alto Networks will help customers build playbooks and will also build custom playbooks.


---


## Indicators

Cortex XSOAR indicators
- objects with assigned reputations used to help orchestration and hunting.
- They come from threat intelligence sources such as `AutoFocus` or `URLscan.io`, and their reputations are typically automatically assigned based on information from those sources.
- They're stored in a repository that can be viewed in a dashboard according to analysts' preferences.

- **Indicator Type**
  - Indicators are classified by their type.
  - Types are `automatically` or `manually` assigned.
  - There are predefined indicator types such as `IP addresses, registry paths, hashes, and file enhancement scripts`, but customers can also define their own indicator types.


- **Indicator Fields**
  - indicators have other out-of-the-box or custom fields associated with them:
  - "Type" is an indicator field
  - Other useful fields of an indicator: its source, reputation, related incidents, and when it was first seen and last seen.


- **Value of Indicators**
  - `Indicators` can be used to create `incidents`,
  - and those incidents can use indicator fields, such as their type, to orchestrate and automate responses when the created incident is assigned a playbook.
  - This automation and orchestration speeds the work for analysts who monitor security incidents, manually hunt for indicators, and then take action.



---

## The War Room and Evidence Board

We use the War Room and Evidence Board to centralize incident response.

**War Room**
- The War Room is a chronological journal of all investigation actions, artifacts, and collaboration associated with an incident made automatically or manually by investigation stakeholders.
- The War Room allows a team to collaborate while tracking their playbooks, scripts, and commands in a controlled and audited environment.


Evidence Board
- The Evidence Board provides a place to collect the evidence or indicators associated with an incident and do further collaborative investigation.



Benefits
- With the War Room and Evidence Board, explore and execute incident response activities from one place.
- All activities and commands are recorded, so every action is self-documenting and auditable.
- This simplifies the incident review process and improves its accuracy.
- Many customers have found that when they do not review cases immediately after their completion, they depend on human memory and lose significant information about how the case was handled.


---

## Cortex XSOAR’s collaboration tools.

- **Facilitating Teamwork**
  - Real-time collaboration tools facilitate the human in end-to-end incident management.
    - Analysts can share autodocumenting security insights and actions.
  - These combine with interactive investigation features for more productivity and better correlation and analysis of indicators across incidents.


- **Related Incident Map**
  - Analysts can use `chat functions` and an `investigative toolkit` to build and share a customizable map of related incidents across time.


- **Machine Learning-based Suggestions**
  - There's a machine learning-based security bot that helps run commands and suggests analyst ownership and future courses of action.




















.
