---
title: SecurityDefend - Control Types
date: 2018-05-18 11:11:11 -0400
categories: [18DefendSec]
tags: [SecurityDefend]
toc: true
image:
---

[toc]

---


> book:
> security+ 7ch
> CISSP chap13

---


# Control Types

CompTIA lists the following control types in the objectives:
- TAP principle:
- <font color=red> Technical controls </font> use technology.
- <font color=red> Administrative controls </font> use administrative or management methods.
- <font color=red> Physical controls </font>, controls you can physically touch.
  - Increase response time.
- <font color=red> Preventive controls </font> attempt to prevent an incident from occurring.
- <font color=red> Detective controls </font> attempt to detect incidents after they have occurred.
- <font color=red> Corrective controls </font> attempt to reverse the impact of an incident.
- <font color=red> Deterrent controls </font> attempt to discourage individuals from causing an incident.
- <font color=red> Compensating controls </font> are alternative controls used when a primary control is not feasible.

---


## Types of access control

- <font color=red> Identify and authenticate </font> users or other subjects attempting to access resources.
- <font color=red> Determine </font> whether the access is authorized.
- <font color=red> Grant or restrict access </font> based on the subject’s identity.
- <font color=red> Monitor and record </font> access attempts.


Access control | goal | example
---|---|---
<font color=red> Preventive Access Control </font> | thwart or stop occurring | - data classification, penetration testing, access control methods, encryption, antivirus software, firewalls, IPS, OS hardening <br> - fences, locks, biometrics, mantraps, lighting, alarm, auditing, smartcards, callback procedures, security cameras, closed circuit television (CCTV), Security guards, <br> - separation of duties, job rotation, security policies, security awareness training
---|---|---
<font color=red> Detective Access Control </font> | discover or detect only after it has occurred | Examples: <br> - audit trails, honeypots or honeynets, security audits, IDS, system logs <br> - security guards, motion detectors, recording and reviewing of events in CCTV, job rotation policies, mandatory vacation policies, violation reports, supervision and reviews of users, incident investigations,
---|---|---
<font color=red> Corrective Access Control </font> | modifies the environment to return systems to normal after an unwanted or unauthorized activity has occurred. correct any problems that occurred by security incident. | Corrective controls can be simple: <br> terminating malicious activity, <br> rebooting a system, <br> antivirus solutions remove or quarantine a virus, <br> backup / restore plans, restored lost data,<br> active IDS, modify the environment to stop an attack in progress. <br> IPS<br> Security guard perform a post-incident review.<br> Alternate site
---|---|---
<font color=red> Deterrent 制止 Access Control </font> | discourage security policy violations. | (deterrent controls: often depend on individuals deciding not to take an unwanted action. <br> preventive control: In contrast, just blocks the action.) <br> Examples: <br> policies, <br> security awareness training, locks, fences, security badges, guards, mantraps security cameras. <br> Warning signs<br> Lighting <br> Login banners
---|---|---
<font color=red> Recovery Access Control </font> | repair or restore resources, functions, and capabilities after a security policy violation. | Recovery controls are an extension of corrective controls but have more advanced or complex abilities. <br> Examples: backups and restores, fault-tolerant drive systems, system imaging, server clustering, antivirus software, database or virtual machine shadowing.
---|---|---
<font color=red> Directive Access Control </font> | direct, confine, or control the actions of subjects to force or encourage compliance with security policies. | Examples: security policy requirements or criteria, posted notifications, escape route exit signs, monitoring, supervision, and procedures.
---|---|---
<font color=red> Compensation 补偿 Access Control </font> | provides an alternative when it isn’t possible to use a primary control, or when necessary to increase the effectiveness of a primary control. | The organization could issue hardware tokens to employees as a compensating control, provide stronger authentication than just a username and password. <br> Example, a security policy might dictate the use of smartcards by all employees but it takes a long time for new employees to get a smartcard. <br> Backup generator


---

## Technical Controls

- Take the form of sorftware or hardware devices.
- use technology to reduce vulnerabilities.

An administrator installs and configures a technical control, and the technical control then provides the protection automatically.

Examples:

- <font color=red> Encryption </font>:
  - strong technical control used to protect the confidentiality of data.
  - This includes data transferred over a network and data stored on devices, such as servers, desktop computers, and mobile devices.

- <font color=red> Antivirus software </font>:
  - antivirus software provides protection against malware infection.

- <font color=red> Intrusion detection/prevention systems (IDS IPS) </font>:
  -  monitor a network or host for intrusions
  -  and provide ongoing protection against various threats.

- <font color=red> Firewalls </font>:
  - Network firewalls restrict network traffic going in and out of a network.

- <font color=red> Least privilege </font>:
  - The principle of least privilege specifies that individuals or processes are granted only the privileges they need to perform their assigned tasks or functions, but no more.
  - Privileges are a combination of <font color=blue> rights and permissions </font>

- <font color=red> Proxy, permissions, auditing and similar technologies </font>:

- <font color=red> authentication methods </font>
  - such as passwords, smartcards, and biometrics

constrained interfaces, access control lists, protocols, routers, and clipping levels.


---

## Administrative / operational / management controls.

- Take the form of use methods mandated by <font color=red> organizational policies or guidelines </font>
- Many of these assessments provide an ongoing review of an organization’s risk management capabilities.
- controls focus on personnel and business practices.


Many administrative controls are also known as operational or management controls.
- help ensure that day-to-day operations of an organization comply with the organization’s overall security plan.
- People (not technology) implement these controls.


Some common administrative controls are:
policies, procedures, hiring practices, background checks, classifying and labeling data, reports and reviews, personnel controls, and testing.

- <font color=red> Risk assessments </font>.
  - help quantify and qualify risks in an organization so they can focus on the serious risks.
  - example:
    - a <font color=blue> quantitative risk assessment </font> uses cost and asset values to quantify risks based on <font color=blue> monetary values. </font>
    - A <font color=blue> qualitative risk assessment </font> uses judgments to categorize risks based on <font color=blue> probability and impact. </font>

- <font color=red> Vulnerability assessments </font>.
  - discover current vulnerabilities or weaknesses.
  - When necessary, an organization implements additional controls to reduce the risk from these vulnerabilities.

- <font color=red> Penetration tests </font>.
  - a stepfurther than a vulnerability assessment by attempting to exploit vulnerabilities.
  - example,
  - a vulnerability assessment might discover a server isn’t kept up to date with current patches, making it vulnerable to some attacks.
  - A penetration test would attempt to compromise the server by exploiting one or more of the unpatched vulnerabilities.

- <font color=red> Security Awareness and training </font>.
  - The importance of training to reduce risks cannot be overstated.
  - Training helps users maintain password security, follow a clean desk policy, understand threats such as phishing and malware, and much more.

- <font color=red> Configuration and change management </font>.
  - Configuration management often uses baselines to ensure that systems start in a secure, hardened state.
  - Change management helps ensure that changes don’t result in unintended configuration errors.

- <font color=red> Contingency planning </font>.
  - help an organization plan and prepare for potential system outages.
  - The goal is to reduce the overall impact on the organization if an outage occurs.

- <font color=red> Media protection </font>.
  - Media includes physical media such as USB flash drives, external and internal drives, and backup tapes.

- <font color=red> Physical and environmental protection </font>.
  - This includes physical controls, such as cameras and door locks,
  - and environmental controls, such as heating and ventilation systems.
  - example,
  - an access list identifies individuals allowed into a secured area.
  - Guards then verify individuals are on the access list before allowing them in.

---

## Pysical Controls


---

## Preventive Controls
Ideally, an organization won’t have any security incidents and that is the primary goal of preventive controls—to prevent security incidents.
- eliminating vulnerabilities.


Some examples include:

- <font color=red> Strong authentication </font>
  - For example
  - strong authentication of cloud user’s enables only authorized personnel to access the data.

- code that disables inactive ports
  - to ensure that there are no available entry points for hackers

- <font color=red> Hardening </font>.
  - Hardening is the practice of making a system or application more secure than its default configuration.
  - This uses a <font color=blue> defense-in-depth </font> strategy with <font color=blue> layered security </font>
  - This includes
  - disabling unnecessary ports and services,
  - implementing secure protocols,
  - using strong passwords along with a robust password policy,
  - and disabling default and unnecessary accounts.

- <font color=red> Security awareness and training </font>.
  - Ensuring that users are aware of security vulnerabilities and threats helps prevent incidents.
  - When users understand how social engineers operate, they are less likely to be tricked.
  - example,
  - uneducated users might be tricked into giving a social engineer their passwords,
  - but educated users will see through the tactics and keep their passwords secure.

- <font color=red> Security guards </font>.
  - Guards prevent and deter many attacks.
  - example, guards can prevent unauthorized access into secure areas of a building by first verifying user identities. Although a social engineer might attempt to fool a receptionist into letting him into a secure area, the presence of a guard will deter many social engineers from even trying these tactics.

- <font color=red> Change management </font>.
  - Change management ensures that changes don’t result in unintended outages.
  - instead of administrators making changes on the fly
  - submit the change to a change management process.
  - it’s both an operational control and a preventive control, which attempts to prevent incidents.

- <font color=red> Account disablement policy </font>.
  - An account disablement policy ensures that useraccounts are disabled when an employee leaves.
  - This prevents anyone, including ex-employees, from continuing to use these accounts.

---

## Detective Controls

- detect when vulnerabilities have been exploited, resulting in a security incident.

- <font color=red> discover the event after it’s occurred </font>
- the detective control will signal the preventive or corrective controls to address an issue

Some examples of detective controls are:

- <font color=red> IDS and Network security monitoring tools SIEM</font>.
  - IDS is generally implemented in line with an intrusion prevention system (IPS).
  - continuously monitors computer systems for policy violation and malicious activity.
  - As soon as it detects either of them, it can alert the system administrator.
  - Advanced IDS may allow an organization to record information about security incidents and help security teams in retrieving information such as IP address and MAC address of the attack source.

- <font color=red> Anti-virus/anti-malware tool </font>
  - An anti-virus tool is generally installed on every system in an organizational network.
  - regular scanning along with real-time alerts and updates.
  - While traditional tools heavily rely on virus signatures, ideal anti-virus tools use behavior detection to discover viruses, worms, ransomware, trojan horses, and other malicious files regularly.
  - A dedicated policy may support the organization-wide implementation of an anti-virus tool.


- <font color=red> Log monitoring to detect incidents </font>.
  - Several different logs record details of activity on systems and networks.
  - Some automated methods of log monitoring automatically detect potential incidents and report them right after they’ve occurred.
  - example,
    - firewall logs record details of all traffic that the firewall blocked.

- <font color=red> Log monitoring to detect trends, Trend analysis </font>.
  - example,
  - an intrusion detection system (IDS) attempts to detect attacks and raise alerts or alarms.
  - By analyzing past alerts, you can identify trends,
    - such as an increase of attacks on a specific system.

- <font color=red> Security audit </font>.
  - Security audits can examine the security posture of an organization.
  - example,
  - a password audit can determine if the password policy is ensuring the use of strong passwords.
  - a periodic review of user rights can detect if users have more permissions than they should.

- <font color=red> Video surveillance </font>.
  - A closed-circuit television (CCTV) system can record activity and detect what occurred.
  - It’s worth noting that video surveillance can also be used as a deterrent control.

- <font color=red> Motion detection </font>.
  - Many alarm systems can detect motion from potential intruders and raise alarms.

---

### Detection vs Prevention Controls
the differences between detection and prevention controls.
- detective control <font color=red> can’t predict </font> when an incident will occur and it can’t prevent it.
- prevention control <font color=red> prevent the incident from occurring at all </font>


---

## Corrective Controls

- attempt to reverse the impact of an incident or problem after it has occurred.

Some examples of corrective controls are:
- <font color=red> intrusion prevention system IPS </font>.
  - attempts to detect attacks and then modify the environment to block the attack from continuing.

- <font color=red> Backups and system recovery </font>.
  - Backups ensure that personnel can recover data if it is lost or corrupted.
  - system recovery procedures ensure administrators can recover a system after a failure.

---

## Deterrent Controls

- attempt to 劝阻
  - discourage a threat.
  - discourage potential attackers from attacking,
  - discourage employees from violating a security policy.

- often describe many deterrent controls as preventive controls.
  - Example: imagine an organization hires a security guard to control access to a restricted area of a building. This guard will deter most people from trying to sneak in simply by discouraging them from even trying. This deterrence prevents security incidents related to unauthorized access.

The following list identifies some physical security controls used to deter threats:
- <font color=red> Cable locks </font>.
  - Securing laptops to furniture with a cable lock deters thieves from stealing the laptops. Thieves can’t easily steal a laptop secured this way. If they try to remove the lock, they will destroy it. Admittedly, a thief could cut the cable with a large cable cutter.
  - However, someone walking around with a four-foot cable cutter looks suspicious.

- <font color=red> Hardware locks </font>.
  - Other locks such as locked doors securing a wiring closet or a server room also deter attacks.
  - Many server bay cabinets also include locking cabinet doors.


---

## Compensating Controls
- are <font color=red> alternative controls used instead of a primary control </font>

- example,
- an organization might require employees to use smart cards when authenticating on a system.
- However, it might take time for new employees to receive their smart card.
- To allow new employees to access the network and still maintain a high level of security, the organization might choose to implement a Time-based One-Time Password (TOTP) as a compensating control.
- The compensating control still provides a strong authentication solution.


---

# Types of Security in Cloud Computing

- <font color=red> Network Segmentation </font>
  - in multi-tenant SaaS environments, determine, assess, and isolate customer data

- <font color=red> Access Management </font>
  - Using robust access management and user-level privileges
  - Access to cloud environments, applications, etc. should be issued by role, and audited frequently.

- <font color=red> Password Control </font>
  - never allow shared passwords.
  - Passwords should be combined with authentication tools to ensure the greatest level of security.

- <font color=red> Encryption </font>
  - to protect your data at rest and transit.

- <font color=red> Vulnerability Scans and Management </font>
  - revolves around regular security audits and patching of any vulnerabilities.

- <font color=red> Disaster Recovery </font>
  - have a plan and platforms in place for data backup, retention, and recovery.

- <font color=red> Security Monitoring, Logging, and Alerting </font>
  - Continuous monitoring across all environments and applications is a necessity for cloud computing security.


- <font color=red> Procedural controls </font>
  - such as security awareness education, security framework compliance training, and incident response plans and procedures


- <font color=red> Compliance controls </font>
  - such as privacy laws and cybersecurity frameworks and standards designed to minimize security risks.
  - These typically require an information security risk assessment, and impose information security requirements, with penalties for non-compliance.
  - The National Institute of Standards and Technology (NIST) Special Publication 800-53, Security and Privacy Controls for Federal Information Systems and Organizations
  - The International Organization for Standardization (ISO) standard ISO 27001, Information Security Management
  - The Payment Card Industry Data Security Standard (PCI DSS)
  - The Health Insurance Portability and Accountability Act (HIPAA)






.
