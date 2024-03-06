---
title: Palo Alto Networks - CortexXDR 2.0 - Architecture, Analytics, and Causality Analysis
# author: Grace JyL
date: 2020-10-18 11:11:11 -0400
description:
excerpt_separator:
categories: [SOC, PaloAlto]
tags: [SOC, CortexXDR]
math: true
# pin: true
toc: true
image: /assets/img/note/prisma.png
---


# Cortex XDR 2.0 - Architecture, Analytics, and Causality Analysis

- [Cortex XDR 2.0 - Architecture, Analytics, and Causality Analysis](#cortex-xdr-20---architecture-analytics-and-causality-analysis)
  - [overall](#overall)
  - [need reformate](#need-reformate)
  - [need re format](#need-re-format)

---

## overall

Cortex is designed to
- reduce alert fatigue,
- address the problems associated with using disparate security products,
- support the effective use of security expertise,
- and reduce the complexity of SIEM use.

Integrating Technology
- Cortex collects data from different sources into one place
- processes the data from the entire infrastructure together rather than processing the data in silos.


Behavioral Analytics
- When Cortex finds something it needs to respond to, it responds back to the entire infrastructure and automates as much of the whole process as possible.
- With the rich data collected, Cortex applies ML-based data analytics to incorporate user behavior into its detection and response capabilities.

> SIEM:
> The variety of alert schemes and data forms provided by SIEMs makes it hard to automate searches and analysis
> The alerts provided by SIEMs often do not contain enough rich context data to allow for effective analytics.

> SOC: analyzing and responding to security events
> Cloud computing changes the security model SOCs must work with.
> SOCs must process more alerts from more security products.


![original](https://i.imgur.com/rAypCfw.png)

> How are XDR and EDR different?
> XDR uses more data sources.


![XDR and EDR](https://i.imgur.com/7Si3IFN.png)

---


## need reformate

```

### The Product Suite.

1. Cortex XDR and Cortex XSOAR
   - **Cortex XDR**
     - automates a big part of the investigative process.
     - it hunts for attacks that previously humans hunted for, so humans can now hunt for attacks that Cortex XDR can’t.
     - it automates alert reduction by consolidating many different alerts into single incidents.
     - On average, Cortex XDR takes 50 alerts and converts them into a single incident, as a result of investigation automation, investigations are eight times faster using Cortex XDR than they are without it.

   - Another part of Cortex is Cortex XSOAR.
     - takes meaningful data from Cortex XDR and elsewhere.
     - It automates the process of enriching that data, running playbooks against it,
     - and making modifications to anything in the infrastructure that needs to be modified to deal with an attack.
     - on average, 95% of their alerts are `handled automatically`. in stead of overwhelmed and dropping alerts, now able to handle all their alerts. The alerts customers handle with human involvement are on average handled 90% faster. Customer capacity to handle these alerts has increased tenfold.


Cortex XDR and Cortex XSOAR provide the key functionality for Cortex.

![small](https://i.imgur.com/lWqln89.png)


2. Cortex Data Lake
   - the central repository for data used by Cortex.
   - a scalable cloud-based service with global locations to support local data residency and privacy requirements.
   - automatically collects, integrates, and normalizes data across a customer’s security infrastructure.

![small-1](https://i.imgur.com/TqTHDBH.png)


3. Cortex’s Threat Intelligence Provider
   - AutoFocus provides threat intelligence to Cortex to enable deep visibility into attacks.
   - AutoFocus' intel is crowdsourced from the industry’s largest footprint of network, endpoint, and cloud intel sources.
   - It can be embedded in any tool customers use through a custom threat feed and APIs.
   - AutoFocus' sources include `WildFire®, PAN-DB, DNS Security, Cortex Prevent, Prisma Access, Unit 42 Research, and third-party sources`.
   - This data can be tagged, searched, and analyzed and fed through APIs, EDLs, Custom Feeds, Cortex, and Prisma Cloud.

![small-2](https://i.imgur.com/MmjHbdf.png)



```
Cortex XDR Environment

People
- roles such as security analysts, security architects, security engineers, and administrators.

ASK:
Which roles do their people play and which skills do they have?
Who has overall responsibility for cybersecurity?
What is the level of skill available to those who would work with Cortex XDR?
How does the security team work with the network team?
What is the big picture about how these people organize their cybersecurity work?
Do they have red teams and blue teams?
Do they regularly engage penetration testing or do other security assessment?
Does the customer have tightly defined formal cybersecurity roles?


alerts
- Most customers have too many alerts, and this is a huge issue to manage.
- These alert issues are exactly the issues that Cortex XDR helps them address.
ASK:
Does the customer have orders-of-magnitude too many alerts?
How do they approach the challenge of too many alerts?
How do they determine which alerts to investigate?
When the customer receives related alerts, how are they linked together?
Is it easy for the customer to make full use of information about linked alerts?


SIEMs
- Cortex XDR is not a SIEM and does not replace SIEM functionality.
- SIEMs collect logs from various sources. They often normalize these logs and prioritize alerts.
- Cortex XDR also collects data from various sources. It has native integration with these sources to both collect data and enforce protection and SIEMs don’t do this.
- It collects rich data from network, endpoint, and cloud sources.
- The data SIEMs collect is often not sufficiently rich. Cortex XDR uses this rich data to help security operations teams investigate threats and to detect and help hunt even sophisticated attacks. SIEMs are beginning to offer detection capability but generally do not do detection and hunting. Cortex XDR also responds and provides enforcement back to those points, and traditional SIEMs do not do this.
- Cortex XDR is better suited to work with a SIEM, rather than replacing one that has already been deployed.
ASK
Which SIEMs does the customer use?
Where do the SIEMs get their data?
Is the data coming from their SIEMs consistent?
Are they able to do analysis on SIEM data?
How do they do analysis on their SIEM data?



```


### TTP and MITRE ATT&CK

Techniques, Tactics, and Procedures (TTP)
- an industry standard term for approaching `advanced persistent attacks (APTs)`.
- This method is used to break down the attack and analyze how they are meant to work.
- Cortex XDR Prevent is very good at recognizing attack techniques.
- Cortex XDR Pro’s strength lies in what it does with the rich data it collects.
- Cortex XDR Pro uses data about the events and their context, helps hunting by showing the probable paths of the attack.
- Its `causal analysis` traces back to the attack vector and can help identify an attack source by combining rich data from sensors and threat intelligence sources.
- This approach also identifies and presents reconnaissance efforts to analysts.
- Cortex XDR uses all of this data and the big picture of an attack and identifies possible targets of an attack.
- Cortex XDR can identify and quarantine or control compromised endpoints.


**MITRE ATT&CK**
- MITRE ATT&CK is one of the most respected evaluation frameworks for TTP in the industry.
- It provides coverage of capabilities and techniques that attackers use in real-world attacks.
  - For example,
  - the MITRE ATT&CK Matrix for Enterprise looks at an organization’s ability to protect against everything from `initial access through privilege escalation, C2, exfiltration, and final impact`.
- use MITRE ATT&CK to evaluate their cybersecurity effectiveness.
- According to the MITRE ATT&CK evaluation framework, Cortex XDR provides the best endpoint visibility and the highest coverage across different attack techniques among the ten tested endpoint detection and response vendors.
- Cortex XDR’s highly automated AI approach also provides more coverage, consistency, and no delays compared with vendors who rely on human processes.



---

## Cortex XDR Functionality

**Single View of Alerts**
- Cortex XDR assists SOC analysts
- allowing them to view all the alerts from all Palo Alto Networks products in one place, telling the full story of what has actually happened, in seconds.


**Analysis, Threat Hunting, and Response**
- providing the abilities to prevent, detect what cannot be prevented, and automate as much prevention, detection, and response as possible.
- consumes data from the Cortex Data Lake to correlate and stitch together logs provided by different sensors.
- This data is used to derive how events are ordered and create a causality chain. Agents on endpoints also provide their own protection.
- Analysis: correlates data from endpoint, network, user, applications, and cloud.
- `Threat Hunting`:
  - helps humans hunt for threats when it cannot automatically hunt them.
  - `Displays Process Activity`: Identifies and displays to analysts when there is a launch or installation of a process.
  - `Sensing Event Sequences`: Accelerates hunting when it automatically determines a sequence of events with thread level visibility
  - `Root Causes and Ramifications`: Automatically presents ramifications of the root cause associated with the attack
- Response: enables rapid and seamless response to threats across an infrastructure.


**Simple and Advanced Attacks**
- Cortex XDR Prevent protects against simple attacks by comparing signature hashes.
- It also protects against more sophisticated attacks by checking for exploit and malware techniques.
- Agents on endpoints protect against even zero-day attacks.
- Cortex XDR Pro applies ML and analytics to identify even advanced persistent threats.


![small](https://i.imgur.com/Db2pauP.png)

---

### Data

**The Right Data**
- Good analysis through machine learning is all about having the right data.
- Cortex XDR leverages all the data that firewalls and WildFire collects and extends that with third-party sources.
- It has built a broad set of curated data, sorting known good data from known bad data.
- This data trains the local analysis engine.
- The more good data that’s used for this, the better the analysis will be.


the importance of selecting the right data:
- Machine Learning
  - Palo Alto Networks evaluated hundreds of thousands of file attributes to determine which are the most accurate in predicting whether a never-before-seen file is good or bad.
  - Palo Alto Networks selected the right attributes among those to use for machine learning analytics.
  - This data is time dependent, and modules used for training machine learning analytics are updated daily or weekly as needed.

- Detection of Suspicious Activity
  - Palo Alto `Networks firewall session log data` is the richest context data available in the industry for detection analytics.
  - Combined with endpoint baseline data, it enables Cortex XDR analytics to provide far better coverage than is typically enabled by SIEM provided data.


![extraLarge](https://i.imgur.com/5m7wOQ3.png)


### Types and Sources of Data


Cortex XDR uses and correlates five types of data, collected from endpoint, network, and cloud sources.
1. Network Data
   - such as `source and destination IP addresses and ports comes from `PAN-OS™ traffic logs collected from the datacenter or cloud deployments, Cortex XDR Prevent agents, or third-party firewall logs.
2. User Data:
   - such as `usernames, login ID, and user organizational unit comes from `PAN-OS software, the Cortex XDR Prevent agent, and the Directory Sync Service.
3. Process Data:
   - such as the `name of the executable, who created it, and WildFire's analysis of the process` comes from the Cortex XDR Prevent agent, the Directory Sync Service, and WildFire.
4. Host Data:
   - such as `hostnames, MAC addresses, system organization units, and operating systems and versions` comes from PAN-OS enhanced application logs, Cortex XDR Prevent agents, and the Directory Sync Service.
5. Application Data:
   - such as an `applications protocol, domain name, and application context` comes from PAN-OS App-Id, enhanced application logs, and the Cortex XDR Prevent agent.

6. Integration with Other Vendors
   - Cortex XDR accepts data from third parties, including competitors, and uses that data to maximum effectiveness.
   - For example, Cortex XDR collects alerts and logs from Check Point.
   - It uses those for analytics, investigation, and response.

---

### Cortex XDR and Time

Cortex XDR saves time for analysts in several ways.

1. Alert Root Cause Identification
   - Cortex XDR save significant `threat hunting and exploration time` when it automatically identifies the root cause of an alert and presents the ramifications of that event to analysts.

2. Eliminates Many False Positive Alerts
   - Analysts `save the time for false positives` when Cortex XDR uses good, rich data and machine learning analytics to eliminate them.

3. Quickly Contains Threats
   - With its `integrated response across network, endpoint, and cloud enforcement points`, Cortex XDR saves time by quickly containing threats.

4. Allows Access and Control from User Interface
   - Cortex XDR allows users to access and control endpoints directly from its user interface.
   - For example, it can look at processes or tasks and kill them, or control USB device access.


---

## Architecture and Components

![Screen Shot 2020-10-23 at 23.51.17](https://i.imgur.com/nc8RKWC.png)

![Screen Shot 2020-10-23 at 23.52.51](https://i.imgur.com/VFJYgBA.png)

![Screen Shot 2020-10-23 at 23.55.35](https://i.imgur.com/uF0c22B.png)

![Screen Shot 2020-10-23 at 23.56.33](https://i.imgur.com/QZ7Ztgk.png)

![Screen Shot 2020-10-23 at 23.57.20](https://i.imgur.com/H5osWZ8.png)

![Screen Shot 2020-10-23 at 23.58.34](https://i.imgur.com/qxvF37l.png)

![Screen Shot 2020-10-23 at 23.59.44](https://i.imgur.com/nmzz16J.png)


The Cortex XDR application provides complete visibility into all data in the Cortex Data Lake.
- It provides a single interface from which you can investigate and triage alerts, take remediation actions, and define policies to detect the malicious activity in the future.

1. Cortex Analytics Engine
   - a cloud-based network security service
   - utilizes data from the Cortex Data Lake to automatically detect and report on post-intrusion threats.
   - The analytics engine does this by identifying good (normal) behavior on your network, so that it can notice bad (anomalous) behavior.

1. Palo Alto Networks next-generation on premises or virtual firewalls
   - provide rich network data and enforce network security policies in in the data centers of campuses, branch offices, and the cloud.

1. Cortex XDR Prevent
   - Cortex XDR Prevent agents protect endpoints from known and unknown malware and malicious behavior and techniques.
   - Cortex XDR Prevent performs its own analysis locally on the endpoint but also consumes WildFire threat intelligence.
   - The agent reports all endpoint activity to the Cortex Data Lake for analysis by Cortex XDR applications.

1. Cortex Data Lake
   - The Cortex Data Lake is a cloud-based logging infrastructure that centralizes the collection and storage of logs from log data sources.


### Suggested SE Approaches
SEs have provided suggestions on how to approach customer opportunities.

Proof of Concept
- Proof of Concepts (POCs) with Cortex XDR are often not as effective as POCs with other products, it needs a 30-day baseline of behavior, which is often hard to arrange.
- And it is sometimes hard to arrange for an advanced persistent threat or other threat that Cortex XDR shines with.
- Generally, use a POC when the customer wants the product's functionality, but needs to ensure it doesn't impact the customer's operation.
- However, a guided evaluation can be very effective. During one sales opportunity and guided evaluation, a customer engaged a penetration tester, and the SE identified an anomalous behavior of ipconfig /all. This led to the discovery of the tester, and convinced the customer to purchase the product.
- Before discussing POCs or guided evaluations with customers, synchronize with sales on rules, context, scope, strategy, and expected outcomes. Stay aware of product lock down schedules.


Existing Palo Alto Networks Firewall Use
- Many customers still use our firewalls primarily on their perimeter, and do not segment their networks.
- Educate these customers on the value of segmentation.
- It is often a good strategy to sell only Cortex XDR endpoint prevention to customers who don’t segment.
- Cortex XDR analytics using endpoint data only is also a possibility, but this typically provides only about half of the Cortex XDR analytics functionality. It's a better product when combined with network logs from a fully segmented environment.


SOCs
- Many customers do not currently have SOCs, but still run into the same challenges that SOCs face.
- Customers do not need large SOCs to benefit from purchasing Cortex XDR.


Strategically Preventing Unknown Threats
- Customers often need to be explicitly taught the value of preventing unknown threats. They all understand the risk of applying patches but many may not fully appreciate the risk of not applying patches. Cortex XDR reduces that risk.
- Cortex XDR Prevent is very strong as a best of breed EDR. But credibly explaining and demoing the XDR approach of combining rich data from multiple sources to enable more accuracy and a better picture of what is happening with an attack is often what convinces customers to go with Cortex XDR.



![Screen Shot 2020-10-24 at 00.05.22](https://i.imgur.com/3U1WKCt.png)

![Screen Shot 2020-10-24 at 00.05.56](https://i.imgur.com/YNl3NIr.png)

![Screen Shot 2020-10-24 at 00.06.30](https://i.imgur.com/cV4B9fk.png)

---


## Analytics Overview

![Screen Shot 2020-10-24 at 00.09.20](https://i.imgur.com/n7U738P.png)

![Screen Shot 2020-10-24 at 00.10.25](https://i.imgur.com/uLl6Y9S.png)

![Screen Shot 2020-10-24 at 00.11.30](https://i.imgur.com/jKj8kiH.png)

![Screen Shot 2020-10-24 at 00.12.22](https://i.imgur.com/QUhQxeS.png)

---

## Causality Analysis Overview

![Screen Shot 2020-10-24 at 00.22.41](https://i.imgur.com/viosBcj.png)

![Screen Shot 2020-10-24 at 00.23.25](https://i.imgur.com/oTVEpxP.png)

![Screen Shot 2020-10-24 at 00.24.03](https://i.imgur.com/oGIfs93.png)

![Screen Shot 2020-10-24 at 00.24.44](https://i.imgur.com/jCP1xzs.png)

![Screen Shot 2020-10-24 at 00.25.26](https://i.imgur.com/neJcTXt.png)

![Screen Shot 2020-10-24 at 00.26.21](https://i.imgur.com/8jd9nE8.png)

![Screen Shot 2020-10-24 at 00.27.22](https://i.imgur.com/FX0apxi.png)

![Screen Shot 2020-10-24 at 00.28.25](https://i.imgur.com/CEN4BaF.png)

![Screen Shot 2020-10-24 at 00.29.07](https://i.imgur.com/VoB0hyS.png)


![Screen Shot 2020-10-24 at 00.30.06](https://i.imgur.com/iWFBTRy.png)

![Screen Shot 2020-10-24 at 00.30.52](https://i.imgur.com/iWfIzSm.png)

![Screen Shot 2020-10-24 at 00.31.09](https://i.imgur.com/2fhlO6Q.png)

![Screen Shot 2020-10-24 at 00.32.05](https://i.imgur.com/WRy97t1.png)

![Screen Shot 2020-10-24 at 00.32.28](https://i.imgur.com/syau5FC.png)


![Screen Shot 2020-10-24 at 00.33.21](https://i.imgur.com/EI6zCJj.png)

![Screen Shot 2020-10-24 at 00.33.45](https://i.imgur.com/iMvYdjd.png)

![Screen Shot 2020-10-24 at 00.34.40](https://i.imgur.com/8XGlgzo.png)

![Screen Shot 2020-10-24 at 01.21.52](https://i.imgur.com/tOyV03i.png)

![Screen Shot 2020-10-24 at 01.30.15](https://i.imgur.com/tT1KTOn.png)


---

## Incident Management

![Screen Shot 2020-10-24 at 01.32.25](https://i.imgur.com/IKJ9cbg.png)

![Screen Shot 2020-10-24 at 01.33.01](https://i.imgur.com/njF3yR6.png)

![Screen Shot 2020-10-24 at 01.33.11](https://i.imgur.com/myoVAK5.png)

![Screen Shot 2020-10-24 at 01.33.49](https://i.imgur.com/EjPJLLL.png)

![Screen Shot 2020-10-24 at 01.33.58](https://i.imgur.com/zdk1POj.png)

![Screen Shot 2020-10-24 at 01.35.46](https://i.imgur.com/AHN9CIN.png)

![Screen Shot 2020-10-24 at 01.36.45](https://i.imgur.com/8xxxOpa.png)

![Screen Shot 2020-10-24 at 01.38.22](https://i.imgur.com/cFQinIW.png)

![Screen Shot 2020-10-24 at 01.38.52](https://i.imgur.com/kLxAGs1.png)

![Screen Shot 2020-10-24 at 01.43.36](https://i.imgur.com/4MV0c8h.png)

![Screen Shot 2020-10-24 at 01.45.50](https://i.imgur.com/iGFZw4Q.png)


![Screen Shot 2020-10-24 at 01.46.42](https://i.imgur.com/Ej4sr8v.png)

![Screen Shot 2020-10-24 at 01.49.54](https://i.imgur.com/2yUTqOV.png)

![Screen Shot 2020-10-24 at 01.50.52](https://i.imgur.com/o5J38RG.png)

![Screen Shot 2020-10-24 at 01.52.01](https://i.imgur.com/2dBVo4d.png)

![Screen Shot 2020-10-24 at 01.53.22](https://i.imgur.com/4COY4th.png)

![Screen Shot 2020-10-24 at 01.53.33](https://i.imgur.com/zKyIKKt.png)

![Screen Shot 2020-10-24 at 01.54.51](https://i.imgur.com/YKZdWOs.png)

![Screen Shot 2020-10-24 at 01.55.19](https://i.imgur.com/4Ci0PvR.png)

![Screen Shot 2020-10-24 at 01.56.04](https://i.imgur.com/tLETxYN.png)

![Screen Shot 2020-10-24 at 01.57.17](https://i.imgur.com/OIQaMrN.png)

![Screen Shot 2020-10-24 at 01.58.08](https://i.imgur.com/vRkskFJ.png)

![Screen Shot 2020-10-24 at 01.59.08](https://i.imgur.com/wi2GHnC.png)

![Screen Shot 2020-10-24 at 01.59.54](https://i.imgur.com/THW5eSa.png)



---

## Analysis Actions

![Screen Shot 2020-10-24 at 03.15.45](https://i.imgur.com/b72hum0.png)

![Screen Shot 2020-10-24 at 03.15.56](https://i.imgur.com/HnkRfBk.png)

![Screen Shot 2020-10-24 at 03.16.07](https://i.imgur.com/zSAiGc0.png)

![Screen Shot 2020-10-24 at 03.16.54](https://i.imgur.com/4ZiXH3v.png)

![Screen Shot 2020-10-24 at 03.17.12](https://i.imgur.com/S5bxjf2.png)

![Screen Shot 2020-10-24 at 03.17.32](https://i.imgur.com/0mfijFJ.png)

![Screen Shot 2020-10-24 at 03.17.48](https://i.imgur.com/NrMzyXn.png)

![Screen Shot 2020-10-24 at 03.18.14](https://i.imgur.com/odBRFun.png)

![Screen Shot 2020-10-24 at 03.18.39](https://i.imgur.com/BJ9ezDL.png)

![Screen Shot 2020-10-24 at 03.20.05](https://i.imgur.com/DIALD41.png)

![Screen Shot 2020-10-24 at 03.20.25](https://i.imgur.com/AcSOZBq.png)

![Screen Shot 2020-10-24 at 03.21.10](https://i.imgur.com/1nt19VE.png)

![Screen Shot 2020-10-24 at 03.21.48](https://i.imgur.com/QolubaU.png)

![Screen Shot 2020-10-24 at 03.21.59](https://i.imgur.com/fCzg3x4.png)

![Screen Shot 2020-10-24 at 03.22.13](https://i.imgur.com/2XofdSU.png)

![Screen Shot 2020-10-24 at 03.23.04](https://i.imgur.com/81Vr0mY.png)

![Screen Shot 2020-10-24 at 03.24.04](https://i.imgur.com/T42Au2A.png)

![Screen Shot 2020-10-24 at 03.24.59](https://i.imgur.com/6iBLZLF.png)

![Screen Shot 2020-10-24 at 03.25.13](https://i.imgur.com/XSBub6L.png)

![Screen Shot 2020-10-24 at 03.26.37](https://i.imgur.com/NRN4yOk.png)

![Screen Shot 2020-10-24 at 03.26.53](https://i.imgur.com/ITwMnqx.png)

![Screen Shot 2020-10-24 at 03.28.47](https://i.imgur.com/2DGDXPL.png)

![Screen Shot 2020-10-24 at 03.29.11](https://i.imgur.com/KSCuiZN.png)

![Screen Shot 2020-10-24 at 03.32.02](https://i.imgur.com/cMHOyL6.png)

![Screen Shot 2020-10-24 at 03.35.24](https://i.imgur.com/Lg5I8Wp.png)

![Screen Shot 2020-10-24 at 03.35.52](https://i.imgur.com/b6GoNHA.png)



---

## Analysis Actions

![Screen Shot 2020-10-24 at 02.02.58](https://i.imgur.com/pM5r5tw.png)

![Screen Shot 2020-10-24 at 02.04.08](https://i.imgur.com/VXprE9U.png)

![Screen Shot 2020-10-24 at 02.03.48](https://i.imgur.com/hnEhhO7.png)

![Screen Shot 2020-10-24 at 02.04.57](https://i.imgur.com/gqldi3e.png)

![Screen Shot 2020-10-24 at 02.05.26](https://i.imgur.com/hM64LSh.png)

![Screen Shot 2020-10-24 at 02.06.04](https://i.imgur.com/uycqz3O.png)

![Screen Shot 2020-10-24 at 02.06.18](https://i.imgur.com/MzJbx8X.png)

![Screen Shot 2020-10-24 at 02.06.32](https://i.imgur.com/G9MytFI.png)

![Screen Shot 2020-10-24 at 02.07.12](https://i.imgur.com/77vdXYH.png)

![Screen Shot 2020-10-24 at 02.07.42](https://i.imgur.com/dLgz6aE.png)

![Screen Shot 2020-10-24 at 02.08.43](https://i.imgur.com/UZ7GtP0.png)

![Screen Shot 2020-10-24 at 02.09.23](https://i.imgur.com/WnM7kvc.png)

![Screen Shot 2020-10-24 at 02.09.53](https://i.imgur.com/aByBYTJ.png)

![Screen Shot 2020-10-24 at 02.10.15](https://i.imgur.com/CZJ7cpU.png)

![Screen Shot 2020-10-24 at 02.10.39](https://i.imgur.com/bx6Am6f.png)

![Screen Shot 2020-10-24 at 02.10.58](https://i.imgur.com/9TVn72s.png)

![Screen Shot 2020-10-24 at 02.11.25](https://i.imgur.com/do6mMjc.png)

![Screen Shot 2020-10-24 at 02.11.58](https://i.imgur.com/JYioIXK.png)

![Screen Shot 2020-10-24 at 02.12.50](https://i.imgur.com/x0QLuYz.png)

![Screen Shot 2020-10-24 at 02.12.59](https://i.imgur.com/8EDWdzf.png)

![Screen Shot 2020-10-24 at 02.13.52](https://i.imgur.com/ebyJLp3.png)

![Screen Shot 2020-10-24 at 02.15.08](https://i.loli.net/2020/10/24/EtL7RjNrJbkHimY.png)

![Screen Shot 2020-10-24 at 02.15.27](https://i.loli.net/2020/10/24/qGRxA3DMVgTk15z.png)

![Screen Shot 2020-10-24 at 02.16.40](https://i.loli.net/2020/10/24/CZ4sLSou9KnbwfQ.png)

![Screen Shot 2020-10-24 at 02.16.58](https://i.loli.net/2020/10/24/epiKwVorP1bFUGC.png)

![Screen Shot 2020-10-24 at 02.17.19](https://i.loli.net/2020/10/24/MaflIp8QZGCsLi6.png)

![Screen Shot 2020-10-24 at 02.17.57](https://i.loli.net/2020/10/24/1Lj2JUqmYe6ncEf.png)

![Screen Shot 2020-10-24 at 02.17.33](https://i.loli.net/2020/10/24/rZnt5QRN4jJObi8.png)

![Screen Shot 2020-10-24 at 02.18.20](https://i.loli.net/2020/10/24/1ehTgq39dXObpEa.png)

![Screen Shot 2020-10-24 at 02.18.51](https://i.loli.net/2020/10/24/ANnwPUzIDQ6CR8L.png)

![Screen Shot 2020-10-24 at 03.00.54](https://i.loli.net/2020/10/24/mbjXIqr3WpL28k4.png)

![Screen Shot 2020-10-24 at 03.01.22](https://i.loli.net/2020/10/24/wvOYlWaMSgLXnoZ.png)

![Screen Shot 2020-10-24 at 03.03.05](https://i.loli.net/2020/10/24/RQYF8aAb6rdmzhM.png)

![Screen Shot 2020-10-24 at 03.04.07](https://i.loli.net/2020/10/24/dzebSVasTKCWwx1.png)

![Screen Shot 2020-10-24 at 03.04.59](https://i.loli.net/2020/10/24/7LAjbT9qwDcZWSx.png)

![Screen Shot 2020-10-24 at 03.05.18](https://i.loli.net/2020/10/24/CzAystS4jxnweUJ.png)

![Screen Shot 2020-10-24 at 03.05.30](https://i.loli.net/2020/10/24/VJ9KD2rQENk4GCF.png)

![Screen Shot 2020-10-24 at 03.10.37](https://i.loli.net/2020/10/24/rdV4IaLBiTl7UMe.png)

![Screen Shot 2020-10-24 at 03.10.58](https://i.loli.net/2020/10/24/837KRoHatxnCZuj.png)

![Screen Shot 2020-10-24 at 03.11.38](https://i.loli.net/2020/10/24/KDcUrvwzGT3FoQn.png)

![Screen Shot 2020-10-24 at 03.12.35](https://i.imgur.com/K3E6mCu.png)

![Screen Shot 2020-10-24 at 03.13.57](https://i.imgur.com/xHanaFo.png)

---

## Managing Rules
![Screen Shot 2020-10-24 at 03.37.44](https://i.imgur.com/JVk1BFI.png)

![Screen Shot 2020-10-24 at 03.38.38](https://i.imgur.com/wkE9lzN.png)

![Screen Shot 2020-10-24 at 03.41.11](https://i.imgur.com/NUDFyjW.png)

![Screen Shot 2020-10-24 at 03.41.56](https://i.imgur.com/A4gM5Dt.png)

![Screen Shot 2020-10-24 at 03.42.12](https://i.imgur.com/R8ZEnHO.png)

![Screen Shot 2020-10-24 at 03.42.38](https://i.imgur.com/K89mJBY.png)

![Screen Shot 2020-10-24 at 03.43.27](https://i.imgur.com/Uc48sd0.png)

![Screen Shot 2020-10-24 at 03.44.07](https://i.imgur.com/uHHmwwr.png)


![Screen Shot 2020-10-24 at 03.44.43](https://i.imgur.com/fatpUSv.png)

![Screen Shot 2020-10-24 at 03.46.25](https://i.imgur.com/ynC0jm5.png)

![Screen Shot 2020-10-24 at 03.46.38](https://i.imgur.com/pgHZFNL.png)

![Screen Shot 2020-10-24 at 03.46.59](https://i.imgur.com/A5LpVRL.png)

![Screen Shot 2020-10-24 at 03.47.17](https://i.imgur.com/dod8Z6v.png)

![Screen Shot 2020-10-24 at 03.47.54](https://i.imgur.com/yNvh7Io.png)

![Screen Shot 2020-10-24 at 03.48.27](https://i.imgur.com/5calQxC.png)


![Screen Shot 2020-10-24 at 03.48.47](https://i.imgur.com/yBl9I5w.png)

![Screen Shot 2020-10-24 at 03.49.10](https://i.imgur.com/H89FGRX.png)

![Screen Shot 2020-10-24 at 03.49.19](https://i.imgur.com/bpe5NOu.png)


![Screen Shot 2020-10-24 at 03.51.38](https://i.imgur.com/p8sduII.png)

![Screen Shot 2020-10-24 at 03.52.16](https://i.imgur.com/VgutcZR.png)

![Screen Shot 2020-10-24 at 03.52.59](https://i.imgur.com/YxpQzMg.png)

![Screen Shot 2020-10-24 at 03.55.35](https://i.loli.net/2020/10/24/jYmJt7RH2f3v5EC.png)

---

## Search and Investigation

![Screen Shot 2020-10-24 at 03.57.14](https://i.loli.net/2020/10/24/tnreibA92mQl6Ro.png)

![Screen Shot 2020-10-24 at 03.57.39](https://i.loli.net/2020/10/24/9CL5slOW2KHeYTg.png)

![Screen Shot 2020-10-24 at 03.57.50](https://i.loli.net/2020/10/24/ipGvabYoFtH9Jjn.png)

![Screen Shot 2020-10-24 at 03.58.07](https://i.loli.net/2020/10/24/Gnwf1A5Tihu7M8v.png)

![Screen Shot 2020-10-24 at 03.58.49](https://i.loli.net/2020/10/24/7gSDJrIHAfY6bE8.png)

![Screen Shot 2020-10-24 at 03.59.41](https://i.loli.net/2020/10/24/wiWREAQvINxKFgo.png)

![Screen Shot 2020-10-24 at 03.59.54](https://i.loli.net/2020/10/24/tbZn9Ci6mBjlOwa.png)

![Screen Shot 2020-10-24 at 04.00.10](https://i.loli.net/2020/10/24/H6D4lRCjTmifw3h.png)

![Screen Shot 2020-10-24 at 04.00.35](https://i.loli.net/2020/10/24/hxT5QCZRpDBaI2V.png)


---

## check


From where on the management console can rerun a query?
- Query Center

For the All `Actions option` on the Query Builder, which two entities can be specified?
- Host
- Process

Which entity is created based on the result of running a query?
- a table

Which two `entities` can be used to create queries from the Query Builder?
- Network
- Process

Which two options can be considered as a “lead” in the threat-hunting context?
- endpoints that have been reported as acting abnormally
- well-defined threat information from online articles


Which option can be considered as a use case of the rule exceptions?
- to prevent false positives


Which option describes the `severity attribute` of a Cortex XDR rule?
- severity of alert to create


Which two types are shown in the type list on the page for adding IOC rules?
- Destination IP
- File Name


can create a BIOC rule based on which two entities?
- Network
- Registry


Which action option is correct about global BIOC rules?
- can clone


Which two options are correct about the Live Terminal action?
- based on only WebSocket
- can save session log at the end of the session


What does the Pending status mean for an action performed in multiple endpoints?
- Not all endpoints have started to run the action yet.

For which entity can apply the Exclude action?
- alerts


Which action can be performed against network attacks?
- EDL


Which action can be used for both response and investigation?
- Live Terminal


In the Timeline View, what does a gray icon imply?
- informational alert


After select a `node in the CI chain` on the `Causality View page`, which two tabs can click? ​ (Choose two)
- NETWORK
- PROCESS


Which two alert severity levels are visible on the `Causality View page`?
- High
- Medium


After right-click a `node in the CI chain` on the `Causality View page`, which two actions can perform?
- Open in VirusTotal
- Terminate


Which option describes why the context menu of an alert shows the “Analyze” action enabled but the “Investigate in timeline” action dimmed? Palo Alto Networks - Cortex XDR 2.0 Architecture, Analytics, and Causality Analysis
- an `unstitched alert` generated by an XDR agent


Which two entities can appear under the section `Key Assets` on the incident details page?
- hosts
- users

On the `incident details page`, informational alerts are displayed in which section?
- Insights

Which generator can appear in the ALERT SRC field of alerts in the management console?
- PAN NGFW


Which alert attribute is blank for unstitched alerts?
- CGO signer


Which two `incident attributes` are editable from the Cortex XDR management console</kbd>?
- severity ​
- status ​



Which entity can be identified as every immediate child process (and thread) of a spawner?
- causality group owner


Which entity can be responsible for initiating an attack?
- causality group owner


Which two options occur during the Cortex XDR log stitching process?
- causation
- correlation


Which profile contains the setting to <kbd>enable or disable the collection of enhanced endpoint data</kbd>?
- agent settings


How often in minutes is the enhanced endpoint data uploaded?
- 5


What two types of data does "enhanced endpoint data" contain?
- file access `logs`
- network access `logs`


Which analysis technique is most effectively applied to `block fileless threats`?
- behavioral


Which term is used to describe the `process of learning the normal behavior of an entity`?
- profiling


Which option is the unit of the `Analytics engine` that organizes its analytics activities per attack type?
- detector



Which tactic does Cortex XDR block by detecting changes in connectivity patterns such as increased rates of connections, failed connections, and port scans?
- discovery


Which attack tactic is described as techniques to receive data from a network such as valuable enterprise data?
- exfiltration


Which analysis technique can block fileless threats?
- behavioral


Which option describes the attacks or threats that have already evaded network defenses but haven't yet done their full damage?
- post-intrusion threats


Which two services are provided by the Broker Service?
- proxy service
- syslog collector


Which component is required in `agentless` Cortex XDR deployments?
- PathFinder


Which Cortex XDR offering `supports the enhanced endpoint data collection` feature?
- <kbd>Cortex XDR Pro per Endpoint</kbd>


Which two features are supported by <kbd>Cortex XDR Prevent</kbd>?
- device control
- endpoint management


Which two engines does Cortex XDR Pro <kbd>per endpoint</kbd> have?
- Analytics
- Causality Analysis



What does the Palo Alto Networks-invented term `XDR` stand for?
- Cross-Platform Detection & Response

---

# Cortex XDR: Managed Threat Hunting (EDU-194)


---


## Cortex XDR: Managed Threat Hunting

Managed Threat Hunting Overview

The Cortex XDR Managed Threat Hunting service is designed to find new, unnoticed threats before they pose risk to network.

Proactive Threat Hunting: Problems and Solutions
- Proactive threat hunting is difficult to conduct due to `lack of time, lack of resources, and missed attacks`.
- Cortex XDR addresses these issues through its <kbd>Managed Threat Hunting</kbd> service.


Problems
- Most teams don’t have the time to proactively hunt for threats.
- Teams rarely have advanced `threat hunters` dedicated to finding attacks.
  - Finding stealthy attacks requires vast amounts of data and threat intelligence that must be available at cloud scale.
  - Threat hunting is a data science problem that requires a unique combination of analytics expertise with a deep understanding of how adversaries work.
  - Without manual threat hunting, organizations may not find the stealthiest attacks.


Solutions > managed theat hunting
- can augment team with around-the-clock threat hunting,
- and cut dwell times and reduce risk without hiring more staff.
- enables to find all attacks with dedicated hunting across endpoint, network, cloud, and third-party data collected by Cortex XDR.
- can get detailed reports with context, intelligence, and guidance.

![Screen Shot 2020-10-28 at 00.55.29](https://i.imgur.com/5NALAxp.png)


<kbd>Cortex XDR Research</kbd>
- Researchers analyze emerging threats.
- They also develop and test new detections.

Cortex XDR and Managed Threat Hunting
- Cortex XDR and Managed Threat Hunting create a powerful combination of services can leverage.
- **Cortex XDR** offers visibility, analytics, data exploration, and threat intel.
- **Managed Threat Hunting** offers monitoring, knowledge, and early access to relevant information.

<kbd>Cortex XDR Platform</kbd>
- New detections are released to Cortex XDR customers on a regular basis.
- Cortex XDR provides threat prevention
- gives full visibility across endpoint, network, and cloud.
- Analytics and threat detection across all data sources generate leads for hunting.
- a powerful data exploration and integrated threat intel tool.


<kbd>Managed Threat Hunting</kbd>
- build on Cortex XDR
  - Analytics on integrated endpoint, network, and cloud data drive threat hunting.
- enriched with context.
  - High-fidelity threat intel provides information about threats and generates impact reports.
- This threat hunting capability is backed by `Unit 42`.
  - Dedicated threat hunters continuously monitor environment for attacks.
- leverage the **Cortex XDR platform**, **tools** such as Cortex XDR, and **ongoing research** to uncover hidden threats.
- find new threats by `leveraging the Cortex XDR platform for research and testing`.
  - The Cortex XDR platform gives `Unit 42 threat hunters` access to `emerging research and detections`.
  - The Cortex XDR platform gives the threat hunters full access to `Cortex XDR analytics and data exploration engine`.
- monitor environment to discover new threats and complex attacks by advanced adversaries.
- get deep knowledge of Cortex XDR data sources and Palo Alto Networks threat intelligence.
- rely on the managed threat hunting team to monitor network
- get email notifications if new threats are found.


Unit 42
- The Cortex XDR Managed Threat Hunting is backed by Unit 42.
- Unit 42 comprises a global threat intelligence team, threat hunting specialists, malware analysis authorities, and researchers.
- A Global threat intelligence team
  - years of defense, cyber warfare, intelligence, and industry experience.
- Threat hunting specialists
  - deep knowledge of attacker tools, techniques, and procedures.
- Malware analysis authorities
  - adept at forensics analysis and reverse engineering malware.
- Researchers
  - Researchers partner with the security community and law enforcement.


Reducing Risk with Managed Threat Hunting Reports
- Cortex XDR enables to get detailed reports based on thorough investigations of threats in organization.
- Cortex XDR provides both <kbd>threat reports</kbd> and <kbd>impact reports</kbd>.

**Threat Reports**
- The Cortex XDR Managed Threat Hunting service provides threat reports that include `technical details` of the attack, `guidance`, and `information` about additional resources.
  - Technical Details
    - the scope of the attack, the probable source, and the tools and techniques used in the attack.
  - Guidance
    - guidance on remedial steps take.
  - Additional Resources
    - links to additional resources use for investigation.

Impact Reports
- summary information about a threat and its impact for organization.
- The report also provides information on ways to get assistance.
  - Threat
    - provide information about emerging attack campaigns, malware, and vulnerabilities.
  - Summary
    - summary on a reported attack.
    - The summary gives details about the nature and the scope of the reported attack.
  - Impact
    - information on whether organization was affected by the attack.
  - Assistance
    - getting assistance from Unit 42 analysts.

---

# XDR Prevent


---

## Exploit Prevention Approaches

Cortex XDR Prevent uses multiple methods to prevent exploits.

---

### The traditional methods
- The **traditional** methods
  - preventing known exploits by
  - antivirus, blacklisting, whitelisting, and aggressively applying patches supplied by software vendors.
  - problematic in preventing even known exploits
  - almost useless in preventing unknown exploits.
    - Signature matching and aggressive patching help prevent known attacks.
    - They cannot prevent unknown attacks.
    - But in practice, signatures and patches are not always current and do not provide adequate prevention against even known attacks.

---

## Multi-Method Approach for Exploits



![Screen Shot 2020-12-02 at 02.45.03](https://i.imgur.com/CZtj61N.png)


### Exploit Protection for Protected Processes

In a typical attack scenario, an attacker attempts to gain control of a system by
- first `corrupting or bypassing
memory allocation or handlers`.
    - Using memory-corruption techniques, such as `buffer overflows` and `heap
    corruption`, a hacker trigger a bug in software or exploit a vulnerability in a process.

- The attacker must then `manipulate a program to run code` provided or specified by the attacker while evading detection.

- If the attacker `gains access to the operating system`, the attacker can then `upload malware`, such as Trojan horses
(programs that contain malicious executable files), or use the system to their advantage.

The Cortex XDR agent prevents such exploit attempts by employing roadblocks / traps at each stage of an
exploitation attempt.


![Screen Shot 2020-12-10 at 12.26.45](https://i.imgur.com/jRoEM8E.png)


When a user opens a non-executable file, such as a PDF or Word, and the process that opened the file is protected, the Cortex XDR agent seamlessly injects code into the software.
- This occurs at the earliest possible stage before any files belonging to the process are loaded into memory.
- The Cortex XDR agent then activates one or more protection modules inside the protected process.
- Each protection module targets a specific exploitation technique and is designed to prevent attacks on program vulnerabilities based on memory corruption or logic flaws.

In addition to automatically protecting processes from such attacks, the Cortex XDR agent reports any security events to Cortex XDR and performs additional actions as defined in the endpoint security policy.
Common actions that the Cortex XDR agent performs include collecting forensic data and notifying the user about the event.


> execution
> -> whitelist/blacklist
> -> publisher tursted?
> not trusted -> WildFire
> unknown -> conduct static analysisInc


- **Multi-Method** Approach for Exploits
  - `Cortex XDR Prevent` uses a multi-method prevention approach for exploits.
  - Rather than looking only at `signatures` and relying on `software and OS patches`,
  - it identifies exploit techniques and prevents them from succeeding.
  - All these methods work together to stop known and unknown attacks.
  - This has security advantages, but also has operational advantages.

```
Memory Corruption Prevention
Exploits sometimes manipulate the operating system's memory management.
This manipulation can be used to allow a weaponized data file to redirect applications to execute an attacker's intended commands.


Logic Flaw Prevention
exploits manipulate an application's process management.
allows the exploit to do things like alter system privilege management mechanisms.
can manipulate the local file system or modify the location where dynamic link libraries (DLLs) are loaded and replace a legitimate DLL with the exploit’s malicious DLL with sufficient privilege.
This is called “DLL hijacking.”
The Logic Flaw Prevention method recognizes these exploitation techniques and stops them before they succeed.


Code Execution Prevention
Exploits to execute an attacker’s malicious commands, embedded in an exploit data file.
Code Execution Prevention recognizes exploitation techniques that allow an attacker’s malicious codes to execute and blocks them before they succeed.

```

---


### Malware Protection
The Cortex XDR agent provides malware protection in a series of four evaluation phases:


![Screen Shot 2020-12-10 at 12.37.56](https://i.imgur.com/ITS6Szb.png)

1. **Phase 1: Evaluation of Child Process Protection Policy**
   - When a user attempts to run an executable, the operating system attempts to run the executable as a process.
   - If the process tries to launch any child processes, the Cortex XDR agent first evaluates the <kbd>child process protection policy</kbd>.
   - If the parent process is a known targeted process that attempts to launch a restricted child process, the Cortex XDR agent blocks the child processes from running and reports the security event to Cortex XDR.
   - For example,
   - if a user tries to open a Microsoft Word document (using the winword.exe process) and that document has a macro that tries to run a `blocked child process (such as WScript)`, the Cortex XDR agent blocks the child process and reports the event to Cortex XDR.
   - If the parent process does not try to launch any child processes or tries to launch a child process that is not restricted, the Cortex XDR agent next moves to Phase 2: Evaluation of the Restriction Policy.

2. **Phase 2: Evaluation of the Restriction Policy**
   - When a user or machine attempts to open an executable file, the Cortex XDR agent first evaluates the child process protection policy as described in Phase 1: Evaluation of Child Process Protection Policy.
   - The Cortex XDR agent next verifies that the executable file does not violate any <kbd>restriction rules</kbd>.
   - For example,
   - have a restriction rule that blocks executable files launched from network locations.
   - If a restriction rule applies to an executable file, the Cortex XDR agent blocks the file from executing and reports the security event to Cortex XDR and, depending on the configuration of each restriction rule, the Cortex XDR agent can also notify the user about the prevention event.
   - If no restriction rules apply to an executable file, the Cortex XDR agent next moves to Phase 3: Evaluation of Hash Verdicts.


3. **Phase 3: Hash Verdict Determination**
   - The Cortex XDR agent `calculates a unique hash using the SHA-256` algorithm for every file that attempts to run on the endpoint.
   - Depending on the features that you enable, the Cortex XDR agent performs additional analysis to determine whether an unknown file is malicious or benign.
   - The Cortex XDR agent can also submit unknown files to Cortex XDR for in-depth analysis by WildFire.
   - ![Screen Shot 2020-12-02 at 02.52.24](https://i.imgur.com/ZXWNCOS.png)
   - To determine a verdict for a file, the Cortex XDR agent evaluates the file in the following order:
     1. **Hash exception**
        - A hash exception enables you to override the verdict for a specific file without affecting the settings in your Malware Security profile.
        - The hash exception policy is evaluated first and takes precedence over all other methods to determine the hash verdict.
        - For example, you may want to configure a hash exception for any of the following situations:
          - • You want to block a file that has a benign verdict.
          - • You want to allow a file that has a malware verdict to run.
            - recommend that you only override the verdict for malware after you use available threat intelligence resources—such as WildFire and AutoFocus—to determine that the file is not malicious.
          - • You want to specify a verdict for a file that has not yet received an official WildFire verdict.
        - After you `configure a hash exception`, Cortex XDR distributes it at the next heartbeat communication with any endpoints that have previously opened the file.
        - `When a file launches on the endpoint, the Cortex XDR agent first evaluates any relevant hash exception for the file`.
          - The hash exception specifies whether to treat the file as malware.
          - If the file is assigned a benign verdict, the Cortex XDR agent permits it to open.
          - If a hash exception is not configured for the file, the Cortex XDR agent next evaluates the verdict to determine the likelihood of malware.
        - multi-step evaluation process in the following order to determine the verdict: Highly trusted signers, WildFire verdict, and then Local analysis.
     2. **Highly trusted signers (Windows and Mac)**
        - The Cortex XDR agent distinguishes highly trusted signers such as Microsoft from other known signers.
        - To keep parity with the signers defined in WildFire, Palo Alto Networks regularly reviews the list of highly trusted and known signers and delivers any changes with content updates.
        - `The list of highly trusted signers also includes signers in allow list from Cortex XDR`.
        - When an unknown file attempts to run, the Cortex XDR agent applies the following evaluation criteria:
          - Files signed by highly trusted signers are permitted to run
          - files signed by prevented signers are blocked, regardless of the WildFire verdict.
          - when a file is not signed by a highly trusted signer or by a signer included in the block list, the Cortex XDR agent next evaluates the WildFire verdict.
          - For Windows endpoints, evaluation of other known signers takes place if WildFire evaluation returns an unknown verdict for the file.
     3. **WildFire verdict**
        - If a file is not signed by a highly trusted signer on Windows and Mac endpoints, the Cortex XDR agent performs a hash verdict lookup to determine if a verdict already exists in its local cache.
        - If the executable file has a malware verdict, the Cortex XDR agent reports the security event to the Cortex XDR and, depending on the configured behavior for malicious files, the Cortex XDR agent then does one of the following:
          - • Blocks the malicious executable file
          - • Blocks and quarantines the malicious executable file
          - • Notifies the user about the file but still allows the file to execute
          - • Logs the issue without notifying the user and allows the file to execute.
        - If the verdict is benign, the Cortex XDR agent moves on to the next stage of evaluation (Phase 4: Evaluation of Malware Protection Policy).
        - If the hash does not exist in the local cache or has an unknown verdict, the Cortex XDR agent next evaluates whether the file is signed by a known signer.
     4. **Local analysis**
        - When an unknown executable, DLL, or macro attempts to run on a Windows or Mac endpoint, the Cortex XDR agent uses local analysis to determine if it is likely to be malware.
          - On Windows endpoints, if the file is signed by a known signer, the Cortex XDR agent permits the file to run and does not perform additional analysis.
          - For files on Mac endpoints and files that are not signed by a known signer on Windows endpoints, the Cortex XDR agent `performs local analysis to determine whether the file is malware`.
        - Local analysis uses a statistical model that was developed with machine learning on WildFire threat intelligence.
        - The model enables the Cortex XDR agent to `examine hundreds of characteristics for a file and issue a local verdict (benign or malicious) while the endpoint is offline or Cortex XDR is unreachable`.
        - The Cortex XDR agent can rely on the local analysis verdict until it receives an official WildFire verdict or hash exception.
        - Local analysis is enabled by default in a Malware Security profile.
        - Because local analysis always returns a verdict for an unknown file,
          - if enable the Cortex XDR agent to Block files with unknown verdict, the agent only blocks unknown files if a local analysis error occurs or local analysis is disabled.
        - To change the default settings (not recommended), see Add a New Malware Security Profile.

4. **Phase 4: Evaluation of Malware Security Policy**
   - If the prior evaluation phases do not identify a file as malware, the Cortex XDR agent observes the behavior of the file and applies additional malware protection rules.
   - If a file exhibits malicious behavior, such as encryption-based activity common with ransomware, the Cortex XDR agent blocks the file and reports the security event to the Cortex XDR.
   - If no malicious behavior is detected, the Cortex XDR agent permits the file (process) to continue running but continues to monitor the behavior for the lifetime of the process.


---

## Exploits and Patches

Cortex XDR prevents known as well as unknown exploits and malware, even on unpatched systems.
- This technique-oriented approach reduces risk between patching cycles.

Cortex XDR Prevent `Behavioral Rules` BIOCs.
- These can be predefined or custom rules.
- These rules match criteria including `process, file, registry, or network` information to identify attack tactics, techniques, and procedures.
- Analysts can also save queries they've used for threat hunting as rules to detect future attacks and help automate threat hunting.
  - **Anti-Ransomware Rules**:
    - `target encryption-based activity` associated with ransomware.
    - These rules can analyze and halt ransomware activity before any data loss occurs.
  - **Behavioral Threat Rules**:
    - prevent sophisticated attacks that leverage built-in OS executables and common administration utilities.
    - They do this by continuously monitoring endpoint activity for chains of behavior that appear malicious.
  - **Data Execution Prevention (DEP) Rules**
    - prevent areas of memory that are designated to contain only data from running executable code.


---

## Machine Learning Analytics


Behavioral Analytics
- Cortex XDR Pro applies behavioral analytics to network and endpoint data to find hard-to-detect attack activity.
- This activity includes behavior such as
  - low and slow reconnaissance
  - or one machine attempting to control another machine.
- This works because even after malware is installed, an attacker must often perform thousands of actions.
  - Each single action, such as a user connecting to an unknown site, might look innocent.
  - But by profiling a baseline of behavior, organizations can detect behavioral changes that attackers cannot conceal.


Cortex XDR applies machine learning to its behavioral analytics with `models of behavior` and `profiles of objects`.
- Cortex XDR uses three behavior models in its analysis:
  - command and control communication,
  - lateral movement,
  - and exfiltration.
  - Cortex XDR looks for chains of events to fit these models to automatically detect them.
- Profiles
  - Time profiles: `compare past user and device activity to current` user and device activity.
  - Peer profiles: `compare peer users and devices` to the activities of the user or device being analyzed.
  - Entity profiles: compare the device type or user type to other device types or user types `exhibiting the same behavior`.


Examples of Machine Learning Analytics
- Some of the attacks that can typically only be detected using machine-learning informed behavioral analytics are the stealthiest and most dangerous types of threats.
- They are the ones already acting inside a network that can lead to costly data breaches.


Attacks That `Behavioral Analytics` Can Detect:
- `Targeted Manual Attacks by external attackers`.
  - These often lead to costly breaches.
  - The average cost per incident of these breaches is $3.6 million.
  - But large scale breaches can cost hundreds of millions of dollars.

- `Malicious Insiders`
  - exploit trusted credentials and cause major damage.
  - take months to discover, because the users are trusted.
  - But Cortex XDR Pro can detect this anomalous behavior.

- `Reckless Users`
  - Risky activity by well-meaning but reckless users can also lead to data breaches.
  - Human error, such as a user posting valuable data on the internet, is directly responsible for about 15% of all breaches.
  - But risky behavior can also invite external attacks.
  - It's hard to detect threats and the increased risk of attack when users upload large files to unsanctioned sites or share credentials.
  - Cortex XDR Pro detects this behavior.

- `Compromised Endpoints`
  - Compromised endpoints represent exploits that have succeeded, and often can be the source of attacks that go undetected.
  - These also can be detected by applying machine learning-informed behavioral analytics that Cortex XDR provides.

- `Other Cases`
  - Cortex XDR is particularly good at detecting other use cases such as anomaly detection or lateral movement without legitimate credentials.



Response Capability in Cortex XDR:
- `Native Integration with Endpoints and Firewalls`
  - Cortex XDR coordinates enforcement with Cortex XDR endpoint agents and with network and cloud-based firewalls.

- `Live Terminal Facilitates Analysts Response`
  - Live Terminal provides the ability to investigate and shut down attacks directly on endpoints.

- `Create BIOC Rules`
  - apply knowledge gained from investigations to detect and prevent similar future attacks by incorporating that knowledge in behavioral rules.

- `Leverage WildFire`
  - New protections will be automatically distributed to all WildFire users by coordinating with WildFire.


---

## Managed Threat Hunting Techniques

The Cortex XDR platform enables to conduct both manual and semi-automated threat hunting.

<kbd>Manual Hunting</kbd>
- gathers information from a variety of data sources.
- uses the data for initial investigation followed by deep investigation, and then writes a report.

> - Data Sources
> collects data from network, endpoint, cloud, and third-party sources provided by customers.
> - Idea
> Based on findings from cases or a published exploit or attack, designs a query to search for the attack.
> - Hypothesis
> validates the hypothesis, checks results, and refines the hypothesis until discovered threats or is confident that no threat exists.
> - Investigation
> conducts deeper investigation of the findings and evidence.
> - Report
> sends a report with findings to the customer.


![Screen Shot 2020-10-28 at 00.48.20](https://i.imgur.com/BaVVQ9I.png)


<kbd>Semi-Automated Hunting</kbd>
- Like manual hunting, semi-automated hunting involves collecting data from a variety of sources.
- The data is analyzed, enriched, and prioritized.
- Then the hunter uses the data for initial investigation followed by deep investigation, and writes a report.

> - Data Sources
> Data is collected from network, endpoint, cloud, and third-party sources provided by customers.
> - Signals
> Smart signals analyze all collected data to discover threats. Signals are based on one or all customers.
> - Enrichment and Prioritization
> Cortex XDR enriches the data that is analyzed by the signals and prioritizes the incident.
> - Initial Investigation
> As a first step, the hunter validates the signal before investigating it in the Cortex XDR management console.
> - Deep Investigation
> The hunter performs a manual investigation to confirm the threat and to understand the full scope of the attack.
> Report
> sends a report with findings to the customer.

![Screen Shot 2020-10-28 at 00.52.08](https://i.imgur.com/XBEoMAF.png)

---

## Threat Hunting Tools

Cortex XDR enables managed threat hunting by leveraging several threat hunting tools such as `AutoFocus` and `WildFire`.

- **Unit 42**:
  - team of expert threat hunters.

- **Cortex XDR**:
  - A special version of Cortex XDR enables to keep an eye on all managed threat hunting customers, to pose questions, and to perform investigations.

- **AutoFocus**:
  - provides a high-fidelity threat intelligence feed powered by WildFire findings.

- **WildFire**:
  - a cloud-delivered malware analysis service
  - uses data and threat intelligence from the industry's largest community.
  - applies advanced analysis to automatically identify unknown threats and stop attackers in their tracks.
  - ![Screen Shot 2020-12-02 at 02.49.44](https://i.imgur.com/iEgKx36.png)
  - ![Screen Shot 2020-12-02 at 02.51.32](https://i.imgur.com/vukaied.png)

- **Cortex XSOAR**
  - applies playbooks to aggregate and normalize threat intel, enrich incidents, reduce false positives, deduplicate activities, and produce experimental signals.

- external resources:
  - The threat hunting team uses several external resources, such as `VT, Cuckoo, URL Analyzer, and GCP`.


---

Demo: Cortex XDR Managed Threat Hunting

![Screen Shot 2020-10-28 at 00.58.22](https://i.imgur.com/LyxAq9I.png)

![Screen Shot 2020-10-28 at 00.59.24](https://i.imgur.com/MbYDxP5.png)

![Screen Shot 2020-10-28 at 00.59.42](https://i.imgur.com/z1Wg1Ue.png)

![Screen Shot 2020-10-28 at 00.59.59](https://i.imgur.com/NQW3QxQ.png)

![Screen Shot 2020-10-28 at 01.00.26](https://i.imgur.com/1buNwua.png)

![Screen Shot 2020-10-28 at 01.00.57](https://i.imgur.com/mw9Zn6w.png)

![Screen Shot 2020-10-28 at 01.01.11](https://i.imgur.com/DaqWx5N.png)

![Screen Shot 2020-10-28 at 01.01.47](https://i.imgur.com/IXdfeQM.png)

![Screen Shot 2020-10-28 at 01.02.53](https://i.imgur.com/A7lguEb.png)

![Screen Shot 2020-10-28 at 01.03.09](https://i.imgur.com/UYlCIYz.png)

### check

Which three types of data exist in a threat report? (Choose three.)
- the scope of the attack
- probable source
- attack tools and techniques


Which two hunting techniques does the Managed Threat Hunting team use? (Choose two.)  5245097
- semi-automated & manual


What three benefits does the Cortex XDR Managed Threat Hunting service bring? (Choose three.)  5245097
- It augments teams with 24/7 threat hunting.
- It unmasks threats anywhere in an organization.
- It enables teams to quickly respond with recommended next steps.


Which two report types should a customer expect to receive from the Managed Threat Hunting team? (Choose two.)
- threat report
- impact report


In which two places are Managed Threat Hunting incidents reported? (Choose two.)  5245097
- Cortex XDR UI & email


The Managed Threat Hunting team analyzes all alerts that exist in Cortex XDR.  5245097
- False


Which two items describe the Managed Threat Hunting service? (Choose two.)  5245097
- It is supported by Unit 42.
- It is enriched with context.



---


## Cortex Data Lake


![Screen Shot 2020-10-28 at 01.12.47](https://i.imgur.com/sKWW9BN.png)


![Screen Shot 2020-10-28 at 01.13.48](https://i.imgur.com/TaIu0kP.png)

![Screen Shot 2020-10-28 at 01.16.12](https://i.imgur.com/7RGURkL.png)

![Screen Shot 2020-10-28 at 01.17.12](https://i.imgur.com/HGhFYfv.png)

![Screen Shot 2020-10-28 at 01.17.34](https://i.imgur.com/3KONLyc.png)

![Screen Shot 2020-10-28 at 01.17.51](https://i.imgur.com/9iWS2pG.png)

![Screen Shot 2020-10-28 at 01.18.09](https://i.imgur.com/34pCbp3.png)

![Screen Shot 2020-10-28 at 01.18.41](https://i.imgur.com/yP1tuvE.png)

![Screen Shot 2020-10-28 at 01.19.13](https://i.imgur.com/54yjXdP.png)

![Screen Shot 2020-10-28 at 01.20.32](https://i.imgur.com/6BY7k4M.png)

![Screen Shot 2020-10-28 at 01.20.05](https://i.imgur.com/Z4dAq0I.png)

![Screen Shot 2020-10-28 at 01.28.46](https://i.imgur.com/cjbOrRM.png)

![Screen Shot 2020-10-28 at 01.29.43](https://i.imgur.com/smRw4Mu.png)

![Screen Shot 2020-10-28 at 01.32.11](https://i.imgur.com/yfknfG8.png)

![Screen Shot 2020-10-28 at 11.32.06](https://i.imgur.com/inWJdU4.png)

![Screen Shot 2020-10-28 at 11.33.53](https://i.imgur.com/uApRL0D.png)

![Screen Shot 2020-10-28 at 11.34.27](https://i.imgur.com/aTwatYs.png)

![Screen Shot 2020-10-28 at 11.34.58](https://i.imgur.com/vD5vSIM.png)

![Screen Shot 2020-10-28 at 11.35.17](https://i.imgur.com/YnKj44k.png)

![Screen Shot 2020-10-28 at 11.35.31](https://i.imgur.com/Nc7vYGW.png)

![Screen Shot 2020-10-28 at 11.35.41](MySQL Circuit Breaker Open)

![Screen Shot 2020-10-28 at 11.36.32](https://i.imgur.com/F7eiARF.png)

![Screen Shot 2020-10-28 at 11.36.57](https://i.imgur.com/Eqc3iaP.png)


![Screen Shot 2020-10-28 at 11.38.30](https://i.imgur.com/CfT6kMO.png)

![Screen Shot 2020-10-28 at 11.39.08](https://i.imgur.com/TGmTANi.png)

![Screen Shot 2020-10-28 at 11.39.00](https://i.imgur.com/HcKfqw4.png)

![Screen Shot 2020-10-28 at 11.40.24](https://i.imgur.com/SrKxH6r.png)

![Screen Shot 2020-10-28 at 11.40.33](https://i.imgur.com/sYVei5D.png)

![Screen Shot 2020-10-28 at 11.41.05](https://i.imgur.com/wLKoE3A.png)

![Screen Shot 2020-10-28 at 11.41.12](https://i.imgur.com/E32Uc3M.png)

![Screen Shot 2020-10-28 at 11.41.40](https://i.imgur.com/6yWNrkj.png)

![Screen Shot 2020-10-28 at 11.41.58](https://i.imgur.com/W0jqweE.png)

![Screen Shot 2020-10-28 at 11.42.48](https://i.imgur.com/rpiJNrz.png)

![Screen Shot 2020-10-28 at 11.43.26](https://i.imgur.com/sVCW5v3.png)

![Screen Shot 2020-10-28 at 11.43.46](https://i.imgur.com/Q8WlWwi.png)

![Screen Shot 2020-10-28 at 11.45.25](https://i.imgur.com/9IOAVmq.png)

![Screen Shot 2020-10-28 at 11.45.39](https://i.imgur.com/LGPA93x.png)

### troubleshoot

![Screen Shot 2020-10-28 at 11.53.44](https://i.imgur.com/R95CKHY.png)

![Screen Shot 2020-10-28 at 11.55.09](https://i.imgur.com/QE1TnQz.png)

![Screen Shot 2020-10-28 at 11.56.12](https://i.imgur.com/oDEE8FV.png)

![Screen Shot 2020-10-28 at 11.57.02](https://i.imgur.com/Ua3Wu9a.png)

![Screen Shot 2020-10-28 at 11.57.13](https://i.imgur.com/HpiOzwA.png)

![Screen Shot 2020-10-28 at 11.58.14](https://i.imgur.com/j7ssAHM.png)

![Screen Shot 2020-10-28 at 11.58.33](https://i.imgur.com/M6tthwL.png)

![Screen Shot 2020-10-28 at 11.58.45](https://i.imgur.com/0RCKvZk.png)


![Screen Shot 2020-10-28 at 11.59.57](https://i.imgur.com/Uipklvm.png)

![Screen Shot 2020-10-28 at 12.04.23](https://i.imgur.com/AXLvHYA.png)

![Screen Shot 2020-10-28 at 12.04.43](https://i.imgur.com/s5uy60G.png)


![Screen Shot 2020-10-28 at 12.05.25](https://i.imgur.com/nBUi4b3.png)


## check

What is the maximum logs-per-second rate per TB of storage that Cortex Data Lake supports?
- 1,000

By what quantity can you increase your storage in Cortex Data Lake?
- 1GB


Every log is stored in how many different databases?
- 2


For fault tolerance, each log is replicated how many times in each database?
- 3


If you have purchased Traps, where do you go to `activate Cortex Data Lake?`
- Cortex hub


If you need firewalls to forward logs to the Logging Service, where do you go to activate Cortex Data Lake?
- Customer Support Portal


Older logs in Cortex Data Lake get purged during which two conditions? (Choose two.)
- maximum days of log retention
- log storage quota exceeded


True or false? Firewalls need to connect securely to Cortex Data Lake.
- True


True or false? For Cortex Data Lake to work, the time in the firewall and Panorama must be synchronized with the NTP server.
- True

True or false? It is the best practice to use the paloalto-logging-service App-ID.
- True


True or false? Logs stored in Cortex Data Lake can be forwarded to an external syslog receiver.
- True

True or false? Magnifier is an application running on the Palo Alto Networks Application Framework.
- True

True or false? Cortex Data Lake enables the Palo Alto Network Application Framework.
- True


True or false? Cortex Data Lake offers multi-tenancy.
- True


True or false? You activate Cortex Data Lake license by using the auth code.
- True


Where is the Log Forwarding App available?
- Cortex hub

What is the purpose of the one-time password (OTP)?
- establish a certificate for secure connection between Cortex Data Lake and Panorama


Which entity does Panorama contact to get a one-time password (OTP)?
- License Server

Which three TCP ports need to be opened for connectivity between Cortex Data Lake and Panorama? (Choose three.)
- 443
- 444
- 3978


True or false? If you subscribe to Cortex Data Lake, then you do not need to own any Log Collectors or be concerned about deploying them.
- True

True or false? Cortex Data Lake provides built-in log redundancy.
- True
```

---

## need re format

```

Which attack prevention technique does Cortex XDR use?
memory corruption protection

In which two ways does Cortex XDR Prevent complement Palo Alto Networks perimeter protection? (Choose two.)
Endpoints sometimes are operated by their users outside the corporate network perimeter.
Cortex XDR can prevent malevolent process execution spawned by traffic the NGFW allows through.


Which sensor captures forensic information about a security event that occurs on an endpoint?
Cortex XDR agent


Which option best describes the functionality of Cortex XDR Prevent for endpoints?
prevention


Which statement is true regarding Cortex XDR Prevent Execution Restrictions?
They define where and how users can run executable files.


What is an advantage of Cortex XDR cloud-based analysis?
It puts attack steps in context for security analysts, even when each step in itself may look innocent.



Which statement describes the malware protection flow in Cortex XDR Prevent?
A trusted signed file is exempt from local static analysis.




Which Cortex XSOAR functionality always is part of accessing external sources for alert enrichment?
Integrations


What are two sources of alert enrichment for Cortex XSOAR? (Choose two.)
SIEMs
AutoFocus



What are two sources of log data for Cortex XDR? (Choose two.)
next-generation firewalls
agents on endpoints


Which advantage is provided by unknown attack prevention?
It provides protection before OS patches are applied.


Which statement is true about advanced cyberthreats?
A zero-day vulnerability is a product security flaw of which the product's vendor has no prior awareness.



Which two analysis methods does WildFire use to detect malware? (Choose two.)
Static
Dynamic


How does Cortex XDR use machine learning?
It learns about normal user and process behavior in an infrastructure so it can recognize anomalous behavior.



When is an existing Cortex XDR customer a suboptimal prospect for Cortex XSOAR?
when they have no interest in automation



What should a customer do to obtain a Cortex XSOAR dashboard that caters to its needs and processes?
quickly design and build the dashboard they need within minutes


What should a customer do that wants to keep a set of specific information for every event of a certain type?
add custom fields to incidents representing events of that type



Which action is required before a new integration can ingest a typed alert and automatically run a playbook for the resulting incident?
An instance of the integration must be created.



What should a customer do that wants to keep a set of specific information for every event of a certain type?
add that information in the Evidence Board when investigating the incident









```










.
