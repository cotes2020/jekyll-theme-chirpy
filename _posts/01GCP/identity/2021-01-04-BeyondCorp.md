---
title: GCP - IdenAccessManage - Zero-trust BeyondCorp
date: 2021-01-04 11:11:11 -0400
categories: [01GCP, Identity]
tags: [GCP]
toc: true
image:
---

- [Zero-trust BeyondCorp](#zero-trust-beyondcorp)
  - [basic](#basic)
  - [BeyondCorp](#beyondcorp)
    - [Guiding Principles of BeyondCorp](#guiding-principles-of-beyondcorp)
    - [Benefits to users](#benefits-to-users)
    - [use cases](#use-cases)
    - [Common signals](#common-signals)
    - [BeyondCorp Enterprise access protection overview](#beyondcorp-enterprise-access-protection-overview)
    - [The Reference Architecture](#the-reference-architecture)
  - [How BeyondCorp Enterprise works](#how-beyondcorp-enterprise-works)
  - [quickstart](#quickstart)
  - [step](#step)
    - [Configure Chrome, create DLP rules, and set up alerts](#configure-chrome-create-dlp-rules-and-set-up-alerts)
    - [View the audit log and security reports, and perform investigations](#view-the-audit-log-and-security-reports-and-perform-investigations)
    - [BeyondCorp Threat and Data Protection URLs](#beyondcorp-threat-and-data-protection-urls)
    - [Before you make the apps and resources context-aware, need to:](#before-you-make-the-apps-and-resources-context-aware-need-to)
    - [Secure the apps and resources with IAP](#secure-the-apps-and-resources-with-iap)
      - [Virtual machine resources](#virtual-machine-resources)
    - [Creating an access level with Access Context Manager](#creating-an-access-level-with-access-context-manager)
    - [Applying access levels](#applying-access-levels)
    - [Enabling device trust and security with Endpoint Verification](#enabling-device-trust-and-security-with-endpoint-verification)

---

# Zero-trust BeyondCorp


- ref:
  - [BeyondCorp playlist](https://www.youtube.com/watch?v=e1Y7NDLSHfI&list=PLIivdWyY5sqLvoPf2pMI2uIz1FLSfphCh&index=2&ab_channel=GoogleCloudPlatform)

---

## basic

![Screen Shot 2021-02-15 at 12.18.08](https://i.imgur.com/bDMGWHr.png)

Perimeter security
- good idea a while back
- hard shell on the outside, soft, gooey center inside.
- revolution needed
  - co-worker, contractor access

![Screen Shot 2021-02-15 at 12.18.34](https://i.imgur.com/MD4UZLX.png)

change security posture
- phishing-resistant keys
- Endpoint device regularly health monitor
- context-aware access
  - control access based on identity and device instead on location or network
  - Rule Engine

---


## BeyondCorp

![Screen Shot 2021-02-15 at 12.18.41](https://i.imgur.com/1AwfQVd.png)

![BeyondCorp_Enterprise.max-2800x2800](https://i.imgur.com/6fvomCy.png)

![unnamed](https://i.imgur.com/gIqkZz7.png)

- A zero trust solution
  - enables secure access with integrated threat and data protection.
  - enables an organization's workforce to access web applications securely from anywhere, without the need for VPN and without fear of malware, phishing, and data loss.
  - manage access for apps on Google Cloud, other clouds, and on-premises,
  - define and enforce access policies based on user identity, device health, and other contextual factors
  - and make apps more accessible and responsive through Google's global network.

- <font color=red> Scalable, reliable foundation </font>
  - <font color=blue> Rely on Google Cloud’s global infrastructure </font>
  - Built on the backbone of Google’s planet-scale network and infrastructure
    - Benefit from the scale, reliability, and security of Google's network
    - with 144 edge locations in over 200 countries and territories.
  - provide a seamless and secure experience with integrated DDoS protection, low-latency connections, and elastic scaling.

- <font color=red> Identity and context-aware access control </font>
  - Provide secure access to critical apps and services
  - Easily configure policies based on user identity, device health, and other contextual factors to enforce granular access controls to applications, VMs, and Google APIs.
  - Implement strong authentication and authorization policies to ensure users have access to the resources they need.
  - Richer access controls protect access to secure systems (applications, virtual machines, APIs, and so on) by using the context of an end-user's request to ensure each request is authenticated, authorized, and as safe as possible.


- Safeguard the information with <font color=red> integrated threat and data protection </font>
  - <font color=blue> Continuous end-to-end protection </font>
  - A layered approach to security across users, access, data, and applications
  - Prevent data loss, malware infections, fraud and thwart threats with real-time alerts and detailed reports,
    - such as
    - exfiltration risks, copy and paste, extending DLP protections into the browser,
    - malware and phishing
    - Strong phishing-resistant authentication to ensure that users are who they say they are.
  - all built into the Chrome Browser with no agents required.
  - real-time alerts
    - Continuous authorization for every interaction between a user and a BeyondCorp-protected resource.
    - End-to-end security from user to app and app to app (including microsegmentation) inspired by the BeyondProd architecture.
    - Automated public trust SSL certificate lifecycle management for internet-facing BeyondCorp endpoints powered by Google Trust Services.

- Simplify the experience for admins and end-user with an <font color=red> agentless approach </font>
  - Easy adoption with our agentless approach
  - non-disruptive overlay to the existing architecture,
  - no need to install additional agents, agentless support delivered through the Chrome Browser,
  - seamless, familiar, easy-to-use experience.

- <font color=red> Open and extensible ecosystem </font>
  - Integrates posture information and signals from leading security vendors, for extra protection.
  - Support the environment: cloud, on-premises, or hybrid
  - Access SaaS apps, web apps, and cloud resources whether they are hosted on Google Cloud, on other clouds, or on-premises.
  - Built on an expanding ecosystem of technology partners in BeyondCorp Alliance which democratizes zero trust and allows customers to leverage existing investments.
  - Open at the endpoint to incorporate signals from partners
    - such as Crowdstrike and Tanium
    - customers can utilize this information when building access policies.
  - Extensible at the app to integrate into best-in-class services from partners
    - such as Citrix and VMware.

---


### Guiding Principles of BeyondCorp

1. <font color=red> Perimeterless Design </font>
   - Connecting from a particular network must not determine which services you can access.
   - Access to services must not be determined by the network from which you connect


2. <font color=red> Context-Aware </font>
   - Access to services is granted based on what we know about you and your device.
   - Access to services is granted based on contextual factors from the user and their device

3. <font color=red> Dynamic Access Controls </font>
   - All access to services must be authenticated, authorized and encrypted.



---

### Benefits to users

For administrators:
- Strengthen security posture to account for dynamic changes in a user's context.
- Shrink the access perimeter to only those resources that an end user should be accessing.
- Enforce device security postures for employees, contractors, partners, and customers for access
  - no matter who manages the devices.
- Extend security standards with per-user session management and multifactor authentication.

For end users:
- Allow all end users to be productive everywhere without compromising security.
- Allow the right level of access to work applications based on their context.
- Unlock access to personally-owned devices based on granular access policies.
- Access internal applications without being throttled by segmented networks.

---

### use cases

When to use
- when you want to establish fine-grained access control based on a wide range of attributes and conditions including what device is being used and from what IP address.
- Making your corporate resources context-aware improves your security posture.
- manage access for apps on Google Cloud, other clouds, and on-premises,
- define and enforce access policies based on user, device, and other contextual factors,
- and make apps more accessible and responsive through Google's global network.
- can also apply BeyondCorp Enterprise to Google Workspace apps.


Common use cases
- As end users work outside of the office more often and from many different types of devices, enterprises have common security models they are looking to extend to all users, devices, and applications

- Allow non-employees
  - to access a web application deployed on Google Cloud or other cloud services platforms without requiring VPN.
  - to access data from their personal or mobile devices as long as they meet a minimum security posture.

- Ensure employees are prevented from copying and pasting sensitive data
  - into email
  - or saving data into personal storage such as Google Drive.
  - Provide DLP protections for corporate data.

- Only allow enterprise-managed devices to access certain key systems.

- Gate access based on a user's location.

- Protect applications in hybrid deployments that use a mix of Google Cloud, other cloud services platforms, or on-premises resources.



---

### Common signals

BeyondCorp Enterprise offers common signals enterprises can take into account when making a policy decision, including:

- User or group information
- Location (IP or geographic region)
- Device
  - Enterprise-managed devices
  - Personally-owned devices
  - Mobile devices
- Third-party device signals from partners in the BeyondCorp Alliance:
  - Check Point
  - CrowdStrike
  - Lookout
  - Tanium
  - VMware
- Risk scores


---


### BeyondCorp Enterprise access protection overview

BeyondCorp Enterprise
- Based on the BeyondCorp security model
- an approach that utilizes a variety of Google Cloud offerings
- to enforce granular access control <font color=red> based on a user's identity and context of the request </font>

Example
- depending on the policy configuration, your sensitive app/resource can:
  - Grant access to all employees <font color=red> if they're using a trusted corporate device from your corporate network </font>
  - Grant access to employees in the Remote Access group <font color=red> if they're using a trusted corporate device with a secure password and up-to-date patch level, from any network </font>
  - Grant administrators access to the Google Cloud Console (via UI or API) <font color=red> only if they are coming from a corporate network </font>
  - Grant developers SSH access to virtual machines.


---


### The Reference Architecture
- Google's architecture is made up of a number of coordinated components
- can be used as reference for any organization looking to move towards their own like-minded system.

![Screen Shot 2021-02-15 at 12.55.02](https://i.imgur.com/iB9LnLU.png)

![Screen Shot 2021-02-15 at 13.30.53](https://i.imgur.com/JSXt1lk.png)

- Device Inventory Service
  - A system that continuously collects, processes, and publishes changes about the state of known devices.

- Trust Inferer
  - A system that continuously analyzes and annotates device state to determine the maximum trust tier for accessing resources.

- Resources
  - The applications, services, and infrastructure that are subject to access control by the system.

- Access Control Engine
  - A centralized policy enforcement service that provides authorization decisions in real time.

- Access Policy
  - A programmatic representation of the resources, trust tiers, and other predicates that must be satisfied for successful auth.

- Gateways
  - SSH servers, web proxies, and 802.1x-enabled wireless networks that perform authorization actions.

---

## How BeyondCorp Enterprise works

Implementing BeyondCorp Enterprise enacts a zero trust model.
- No one can access your resources unless they meet all the rules and conditions.
- Instead of securing your resources at the network-level, access controls are put on individual devices and users.

![endpoint-verification-flow](https://i.imgur.com/Q2KarE3.png)

![high-level_architecture_1.max-2800x2800](https://i.imgur.com/DKHpOMs.jpg)


BeyondCorp Enterprise works by leveraging four Google Cloud offerings:
- <font color=red> Endpoint Verification </font>
  - A Google Chrome extension
    - <font color=blue> collects user device details </font>
      - including encryption status, OS, and user details.
    - gathers and reports device information, constantly syncing with Google Workspace.
    - The end result is an inventory of all the corporate and personal devices accessing your corporate resources.
  - Once enabled through the Google Workspace Admin Console
    - deploy the Endpoint Verification Chrome extension to corporate devices.
    - Employees can also install it on their managed, personal devices.
  - The attributes collected can be used by Access Context Manager to control access to Google Cloud and Google Workspace resources.

![Screen Shot 2021-02-15 at 15.18.56](https://i.imgur.com/uJdAtls.png)

- <font color=red> Access Context Manager </font>
  - A rules engine
  - enables fine-grained access control.
  - Through Access Context Manager, access levels are created
    - Access levels applied on your resources with IAM Conditions enforce fine-grained access control
    - restrict access based on the following attributes:
      - IP subnetworks
      - Regions
      - Members
      - Device policy
  - When create a device-based access level
    - Access Context Manager references the inventory of devices created by Endpoint Verification.
    - Example
    - an access level can restrict access to only employees who are using encrypted devices.
    - Coupled with IAM conditions, you could make this access level more granular by also restricting access to the time between 9am and 5pm.
    - ![Screen Shot 2021-02-15 at 15.14.54](https://i.imgur.com/1YwJlJe.png)
  - can also tag individual devices and mark company-owned devices.
    - Manual device tagging: enforced by creating a device access level that requires device approval.
    - Company-owned devices: enforced by creating a device access level that requires company-owned devices.

![0_zRpXauT263IHpz3Z](https://i.imgur.com/2kuM5jt.png)

- <font color=red> Identity-Aware Proxy (IAP) </font>
  - A service
    - the base of BeyondCorp Enterprise
    - IAP ties everything together
    - Once you've secured your apps and resources behind IAP, your organization can gradually extend BeyondCorp Enterprise as richer rules are needed.
  - establish a <font color=blue> central authorization layer </font>
    - for your Google Cloud resources accessed by HTTPS and SSH/TCP traffic.
  - establish a <font color=blue> resource-level access control model </font>
    - grant members access to your HTTPS apps and resources.
    - instead of relying on network-level firewalls.
  - enables employees to access corporate apps and resources from untrusted networks without the use of a VPN.
    - Once secured, resources are accessible to any employee, from any device, on any network, that meets the access rules and conditions.
  - Extended BeyondCorp Enterprise resources can limit access based on properties
    - such as user device attributes, time of day, and request path.
  - apply IAM conditions on Google Cloud resources.



- <font color=red> Identity and Access Management (IAM) </font>
  - The identity management and authorization service for Google Cloud.
  - define and enforce conditional, attribute-based access control for Google Cloud resources.
  - With IAM Conditions
    - grant permissions to members only if configured conditions are met.
    - IAM Conditions can limit access with a variety of attributes, including access levels.
    - Conditions are specified in the IAP role bindings of a resource's IAM policy.
    - When a condition exists, the role is only granted if the condition expression evaluates to true.
    - Each condition expression is defined as a set of logic statements allowing you to specify one or many attributes to check.


---


## quickstart


1. choose the software
   - lumen
   - had a specific use for small team and fewoutside dependency
2. got leadership but-in
   - introduce the plan to IT, security and marketing team
   - why and timeline
   - benefit:
     - increase access from worker outside of office
     - address risk of unauthorized access or application compromise
     - multi-region availability
     - reduced latency
     -
3. get Lumen run on GCP
   - run on VMs compute engine instead on on-premise VMs
4. active Identity-Aware Proxy
   - as part of the HTTPS load balancer in front of the cluster
   - create simple rules in IAP
     - to allow the marketing team to view the web app
     - IT team to manage access
5. start shifting traffic
   - from on-prem instance of Lumen to GCP instance
   - watching for errors

![Screen Shot 2021-02-15 at 15.44.46](https://i.imgur.com/h9TG57D.png)



---

## step

> applying BeyondCorp Enterprise to the Google Cloud and on-premises resources.
> implement enhanced user protections in Chrome
> BeyondCorp Threat and Data Protection features are available only for customers who have purchased BeyondCorp Enterprise.


### Configure Chrome, create DLP rules, and set up alerts

1. Set up Chrome Management
   - Setup either Cloud Management for Chrome Browser or Chrome Device Management

2. Set up Chrome browser policies
   - To enable additional protections against data loss and malware in Chrome
   - enable Chrome Enterprise connectors
     - so content gathered in Chrome is uploaded to Google Cloud for analysis.
     - must be enabled for DLP rules to integrate with Chrome.

3. Set up data protection rules
   - create DLP rules.
   - These rules are specific to Chrome and warn of or block the sharing of sensitive data.
   - The rules trigger alerts and messages in the Chrome Browser, letting users know that file uploads or downloads are blocked, or warning that sensitive data might be shared.

4. Set up activity alert rules
   - Set up alert center rules so analysts are notified of certain security events.




### View the audit log and security reports, and perform investigations

- use the Rules audit log and security dashboard security reports to monitor security events.
- use the investigation tool to learn more about alert notifications.

View the Rules audit log
- Use the Rules audit log to track user attempts to share sensitive data.
- The Rules audit log tracks Device ID and Device Type audit data types for BeyondCorp-related events.

View security dashboard reports
- View reports in the security dashboard. Security reports related to BeyondCorp are:
  - Chrome threat summary
  - Chrome data protection summary
  - Chrome high risk users
  - Chrome high risk domains

Use the investigation tool to examine security issues
- further investigate the source of the alert in the security investigation tool
- identify, triage, and take action on security and privacy issues in your domain.


### BeyondCorp Threat and Data Protection URLs

- These URLs are used by Chrome to check for updates when running BeyondCorp Threat and Data Protection.
- Chrome must access the following URLs when BeyondCorp Threat and Data Protection is implemented.



### Before you make the apps and resources context-aware, need to:

1. [create a few Cloud Identity accounts](https://support.google.com/cloudidentity/answer/7332836?hl=en)

2. Determine a resource you want to protect.
   1. Configure one of the following if you don't have a resource.
   - A web app running behind an HTTPS load balancer on Google Cloud.
     - This includes web apps like
     - App Engine apps, apps running on-premises, and apps running in another cloud.
   - A virtual machine on Google Cloud.

3. Determine members that you want to grant and limit access to.


### Secure the apps and resources with IAP

IAP establishes a <font color=red> central identity awareness layer </font> for apps and resources accessed by HTTPS and TCP.
- can control access on each individual app and resource instead of using network-level firewalls.

Secure the Google Cloud app and all its resources by selecting one of the following guides:
- App Engine standard and flexible environment
- Compute Engine
- Google Kubernetes Engine
- extend IAP to non-Google Cloud environments like on-premises as well as other clouds.


#### Virtual machine resources

control access to administrative services like SSH and RDP on the backends by
- setting tunnel resource permissions
- and creating tunnels that route TCP traffic through IAP to virtual machine instances.


### Creating an access level with Access Context Manager

after secured the apps and resources with IAP
- set richer access policies with access levels
- Access Context Manager creates access levels.
- Access levels can limit access based on the following attributes:
  - IP subnetworks
  - Regions
  - Access level dependency
  - Members
  - Device policy (Endpoint Verification must be set up.)



### Applying access levels

An access level doesn't take effect until you apply it on a IAP-secured resources' IAM policy.
- This step is done by adding an `IAM Condition` on the IAP role used to grant access to the resource.
- Once you've applied the access level, the resources are now secured with BeyondCorp Enterprise.



### Enabling device trust and security with Endpoint Verification

To further strengthen the security of the BeyondCorp Enterprise secured resources
- apply device-based trust and security access control attributes with access levels.
- Endpoint Verification enables this control.
  - Endpoint Verification is a Chrome extension for Windows, Mac, and Chrome OS devices.
  - Access Context Manager references the device attributes gathered by Endpoint Verification to enforce fine grained access control with access levels

Next steps
- Set up Cloud Audit Logs


---














.
