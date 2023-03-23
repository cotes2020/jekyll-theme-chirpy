---
title: Palo Alto Networks - Cortex Data Lake
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

[toc]

---

# Cortex Data Lake

---

```bash

# data lake > log forwarding > query empty

# Data Lake Terabyte?


```

---

## overview

Palo Alto Networks® Cortex Data Lake
- provides cloud-based, centralized log storage and aggregation
- secure, resilient, and fault-tolerant
- ensures logging data is up-to-date and available when need it.
- provides a scalable logging infrastructure that alleviates the need for to plan and deploy Log Collectors to meet log retention needs.
- If already have on premise **Log Collectors**, the new Cortex Data Lake can easily complement existing setup. can augment existing log collection infrastructure with the **cloud-based Cortex Data Lake** to expand operational capacity as business grows, or to meet the capacity needs for new locations.
- With this service, Palo Alto Networks takes care of the ongoing maintenance and monitoring of the logging infrastructure so that can focus on business.

![Screen Shot 2020-10-28 at 01.12.47](https://i.imgur.com/sKWW9BN.png)

![compare](https://i.imgur.com/TaIu0kP.png)

![Screen Shot 2020-10-28 at 01.16.12](https://i.imgur.com/7RGURkL.png)

![Screen Shot 2020-10-28 at 01.17.12](https://i.imgur.com/HGhFYfv.png)

## Required Ports and FQDNs Required for Cortex Data Lake

If using a <kbd>Palo Alto Networks firewall</kbd> to secure traffic between Panorama, the firewalls, and Cortex Data Lake,
- use the <kbd>App-ID</kbd> `paloalto-logging-service` in a Security policy rule to
  - **allow Panorama and the firewalls to connect to Cortex Data Lake**
  - and **forward logs on TCP 444 and 3978** (default ports for the application)
- If firewall has an Applications and Threats content version earlier than 8290, must also allow the panorama app-id in a security policy rule.

If using <kbd>another vendor’s firewall</kbd>
- use the following table to identify the `fully qualified domain names (FQDNs) and ports` that must allow traffic to ensure that `Panorama` and the `firewalls` can successfully connect to `Cortex Data Lake`.

[link](https://docs.paloaltonetworks.com/cortex/cortex-data-lake/cortex-data-lake-getting-started/get-started-with-cortex-data-lake/ports-and-fqdns.html)


![Screen Shot 2020-10-28 at 01.17.34](https://i.imgur.com/3KONLyc.png)

![Screen Shot 2020-10-28 at 01.17.51](https://i.imgur.com/9iWS2pG.png)

host Cortex Data Lake in the following regions:
- Americas (US)
- Europe (Netherlands)
- UK
- Singapore
- Canada
- Japan

![location](https://i.imgur.com/34pCbp3.png)

![privacy](https://i.imgur.com/yP1tuvE.png)

![compliance](https://i.imgur.com/54yjXdP.png)

---

## Cortex Data Lake Log Sources

Here are the products and services that can send logs to Cortex Data Lake:

sources | log type | Note
---|---|---
**Palo Alto Networks Firewalls** [link](https://www.paloaltonetworks.com/products/product-selection) | firewalls log | onboard individual firewalls directly to Cortex Data Lake. Use the `Explore app` to view all log records that the firewalls forward to Cortex Data Lake.
**Panorama-Managed Firewalls** [link](https://www.paloguard.com/Panorama.asp)| firewalls log | onboard firewalls to Cortex Data Lake at scale, instead of onboarding each individual firewall. All Cortex Data Lake logs are visible directly in Panorama.
**Prisma Access** | the logs, ACC, and reports from Panorama for remote network and mobile user traffic | Prisma Access deploys and manages the security infrastructure globally to secure remote networks and mobile users. Prisma Access logs directly to Cortex Data Lake. <br> - To enable logging for Prisma Access, must purchase a Cortex Data Lake license. <br> - Log traffic does not use the licensed bandwidth purchased for Prisma Access.
**Cortex XDR** | Cortex XDR alerts are automatically written to Cortex Data Lake as log records | other apps can read and respond to alerts. <br> - These log records are not visible in `Explore`; <br> - can use `Log Forwarding app` to forward XDR alerts to the email or Syslog destination and configure email alert notifications within XDR.


![Screen Shot 2020-10-28 at 01.20.32](https://i.imgur.com/6BY7k4M.png)

![Screen Shot 2020-10-28 at 01.20.05](https://i.imgur.com/Z4dAq0I.png)

![Screen Shot 2020-10-28 at 01.28.46](https://i.imgur.com/cjbOrRM.png)

![Screen Shot 2020-10-28 at 01.29.43](https://i.imgur.com/smRw4Mu.png)

![Screen Shot 2020-10-28 at 01.32.11](https://i.imgur.com/yfknfG8.png)

---

## Cortex Data Lake Log Types

[link](https://docs.paloaltonetworks.com/cortex/cortex-data-lake/cortex-data-lake-getting-started/get-started-with-cortex-data-lake/overview/log-types.html)

In the Cortex Data Lake app, can set how much of overall log storage would like to allocate to the following log types:

Log Type | Description
---|---
. | <kbd>Cortex XDR Logs</kbd>
`alert` | Information for all alerts raised in Cortex XDR.
`xdr` | (Cortex XDR Pro per Endpoint only) All EDR data collected on the endpoint.
. | <kbd>Common Logs</kbd> |
`config` / Configuration logs | entries for changes to the firewall configuration.
`system` / System logs | entries for each system event on the firewall.
. | <kbd>Firewall Logs</kbd> |
`auth` / Authentication logs | information about authentication events that occur when end users try to access network resources for which access is controlled by Authentication Policy rules.
`eal` Enhanced application logs | data that increases visibility into network activity for Palo Alto Networks apps and services, like Cortex XDR.
`extpcap` Extended packet capture | packet captures in a proprietary Palo Alto Networks format. The firewall only collects these if enable extended capture in Vulnerability Protection or Anti-Spyware profiles.
`file_data` Data filering logs | entries for the security rules that help prevent sensitive information such as credit card numbers from leaving the area that the firewall protects.
`globalprotect` | GlobalProtect system logs <br> LSVPN/satellite events <br> GlobalProtect portal and gateway logs <br> Clientless VPN logs
`hipmatch` HIP Match logs | the security status of the end devices accessing network.
`iptag` IP-Tag logs | how and when a source IP address is registered or unregistered on the firewall and what tag the firewall applied to the address.
`sctp` Stream Control Transmission Protol logs | events and associations based on logs generated by the firewall while it performs stateful inspection, protocol validation, and filtering of SCTP traffic.
`threat` Threat logs | entries generated when traffic matches one of the Security Profiles attached to a security rule on the firewall.
`traffic` Traffic logs | entries for the start and end of each session.
`tunnel` Tunnel Inspection logs | entries of non-encrypted tunnel sessions.
`url` URL Filtering logs | entries for traffic that matches the URL Filtering profile attached to a security policy rule.
`userid` User-ID logs | information about IP address-to-username mappings and
Authentication Timestamps, such as the sources of the mapping information
and the times when users authenticated.

![Screen Shot 2020-10-28 at 11.32.06](https://i.imgur.com/inWJdU4.png)

## Cortex Data Lake Apps

Cortex Data Lake includes the following apps to manage and view logs.
1. <kbd>Cortex Data Lake</kbd>:
   1. After activating Cortex Data Lake, the Cortex Data Lake app is listed on the hub.
   2. If have multiple instances of Cortex Data Lake, choose which instance of the app want to open.
   3. Use the Cortex Data Lake app to configure log storage quota and to check the status of a Cortex Data Lake instance.
2. <kbd>Explore</kbd>:
   1. Use Explore to search, filter, and export log data.
   2. This app offers critical visibility into enterprise's network activities by allowing to easily examine network log data.
3. <kbd>Log Forwarding app</kbd>:
   1. Use this app to archive the logs send to Cortex Data Lake for long-term storage, SOC, or internal audit directly from the Cortex Data Lake.
   2. forward Cortex Data Lake logs to an external destination such as a Syslog server or an email server. (Or, continue to forward logs directly from the firewalls to Syslog receiver).

---

## Cortex Data Lake License activation

Cortex Data Lake collects log data from `next-generation firewalls, Prisma Access, and Cortex XDR`.
- When purchase Cortex Data Lake, all firewalls registered to support account receive a Cortex Data Lake license.
- also receive an auth code to activate Cortex Data Lake instance.

activation
- Use the hub to activate Cortex Data Lake.
- Use the hub to activate Cortex XDR.
  - To activate the Cortex XDR app, you must be assigned a required role and locate activation email containing a link to begin activation in the hub.
  - Activating Cortex XDR automatically includes activation of Cortex Data Lake.
    1. Begin activation.
    2. Provide details about the Cortex XDR app to activate
    3. Review the end user license agreement and Agree & Activate.
    4. Manage Apps > current status of apps.
    5. log in to Cortex XDR app to confirm access for the Cortex XDR app interface.
    6. [Allocate Log Storage for Cortex XDR](https://docs.paloaltonetworks.com/cortex/cortex-xdr/cortex-xdr-pro-admin/get-started-with-cortex-xdr-pro/allocate-log-storage-for-cortex-xdr.html)
       1. [Allocate Storage Based on Log Type](https://docs.paloaltonetworks.com/cortex/cortex-data-lake/cortex-data-lake-getting-started/get-started-with-cortex-data-lake/set-log-storage-quota)
       2. storage allocation for Cortex Data Lake and adjust the quota as needed.
          1. Select Cortex Data Lake instance.
          2. Select Configuration > logging storage settings.
          3. The Cortex Data Lake depicts storage allocation graphically. As you adjust storage allocation, the graphic updates to display the changes to storage policy.
          4. The **Cortex Data Lake storage policy** specifies the distribution of total storage allocated to each app or service and the minimum retention warning (not supported with Cortex XDR).
          5. Allocate quota for <kbd>Cortex XDR</kbd>:
             - If purchased quota for **firewall logs**, allocate quota to the Firewall log type.
             - To use the same Cortex Data Lake instance for **both firewall logs and Cortex XDR logs**, ust first associate Panorama with the Cortex Data Lake instance before you allocate quota for firewall logs.
             - Review storage allocation for Cortex XDR according to the formula:
             - 1TB for every 200 Cortex XDR Pro endpoints for 30 days
             - By default,
             - `80% available storage is assigned to logs and data`,
             - and `20% is assigned to alerts`.
             - It is recommended to review the status of Cortex Data Lake instance after about two weeks of data collection and make adjustments as needed but to use the default allocations as a starting point.
             - Apply changes.
             - Monitor data retention.
             - From Cortex XDR > Cortex XDR License > Endpoint XDR Data Retention:
             - Current number of days your data has been stored in `Cortex XDR Data Lake`.
             - The count begins the as soon as you activate Cortex XDR.
             - Number of retention days permitted according to the quota you allocated.
       3. You must be an assigned an `Instance Administrator or higher role` to for Cortex Data Lake to manage logging storage.


![workflow](https://i.imgur.com/uApRL0D.png)

![activate](https://i.imgur.com/aTwatYs.png)

![Screen Shot 2020-10-28 at 11.34.58](https://i.imgur.com/vD5vSIM.png)

![Screen Shot 2020-10-28 at 11.35.17](https://i.imgur.com/YnKj44k.png)

![Screen Shot 2020-10-28 at 11.35.31](https://i.imgur.com/Nc7vYGW.png)


![Screen Shot 2020-10-28 at 11.36.32](https://i.imgur.com/F7eiARF.png)

![Screen Shot 2020-10-28 at 11.36.57](https://i.imgur.com/Eqc3iaP.png)

## Start Sending Logs to Cortex Data Lake

Before you start sending logs to Cortex™ Data Lake, you must:
- Configure the XDR Agent
  - Enable the Cortex XDR agent to **Monitor and Collect Enhanced Endpoint Data** [link](https://docs.paloaltonetworks.com/cortex/cortex-xdr/cortex-xdr-pro-admin/endpoint-security/customizable-agent-settings/add-agent-settings-profile.html)
- **Activate your Cortex Data Lake instance**
  - Connect Firewalls to Cortex Data Lake
- Sending log data to Cortex Data Lake from
  - [Cortex Data Lake License](https://docs.paloaltonetworks.com/cortex/cortex-data-lake/cortex-data-lake-getting-started/get-started-with-cortex-data-lake/license-activation.html#id183GAI00VDE)
  - **Connect the firewall to Cortex Data Lake**.
  - other sources
  - Panorama-managed firewalls	Forward Logs to Cortex Data Lake (Panorama-Managed)
  - Prisma™ Access	Configure the Service Infrastructure
  - Cortex XDR Prevent
  - Cortex XDR Pro per Endpoint
    - Begin activation.
    - Provide details about the Cortex XDR app activating.
    - Review the end user license agreement and Agree & Activate.
    - Manage Apps to view the current status of your apps.
    - When your app is available, log in to your Cortex XDR app to confirm that you can successfully access the Cortex XDR app interface.
    - Allocate Log Storage for Cortex XDR.
    - Assign roles to additional administrators, if needed.
    - Complete your configuration.

  - Cortex XDR Pro per TB

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
