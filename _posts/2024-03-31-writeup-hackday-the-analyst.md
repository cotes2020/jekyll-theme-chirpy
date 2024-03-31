---
layout: post
title: 'Write Up HackDay 2024: The Analyst'
categories:
- CTF
- Forensic
tags:
- forensic
- kerberos
- hackday
- windows
- ad
- ctf
lang: en
date: 2024-03-31 16:00 +0100
description: This is the official write-up of the forensic challenge "The Analyst"
  which took place during the 2024 HackDay finals
image: "/assets/img/HackDay2024/9_hackday.png"
---
## Context
As a challenge creator of the HackDay CTF since 2022, I created a Forensic challenge for the 2024 HackDay finals.
Each year a new theme is chosen, this year the theme of espionage has been chosen.
This is the official write-up of this challenge.

## Statement
Our chief has recovered a Syndicate file given to him by our secret agent Hawkeye. It contains communications from a Syndicate member who is planning a mass attack.

Our chief is asking you to **decrypt** this information in order to trace his future actions:

* `TGT renewal date` in dd/mm/yyyy format
* `User's machine name` in uppercase (example: SERVER99)
* `Last subkey value` (example: 99zae54e450ef88ee55511900666115441cd)
* `User SID` (example: S-1-5-21-123456789-987456321-10000000000-1000)

## Files

The following two files are given to participants in order to solve the challenge:
* PCAP capture: [analysis_admin.pcapng](/files/HackDay2024/analysis_admin.pcapng)
* Credentials file: [creds.txt](/files/HackDay2024/creds.txt)

## Solving

The first step is to analyze what we have (files and instructions) in order to understand what will be our next move.

So we see that we are given a PCAP file containing what seems like a **Kerberos** authentication: 
![pcap](/assets/img/HackDay2024/1_pcap.png)

We know that Kerberos packets contain encrypted parts, so it's a good thing we've got credentials!

Indeed the `creds.txt` contains both `krbtgt` and `odin` `AES-256` keys:
![creds](/assets/img/HackDay2024/0_creds.png)

Now it's clear that we need to decrypt these packets, but how?

### Decrypting

By opening **WireShark** preferences and going to "KRB5" (Kerberos) protocol we can see that by giving a `Keytab` file, it is possible to decrypt Kerberos packets.
![preferences](/assets/img/HackDay2024/2_preferences.png)

With a little digging on the Internet, we can find a tool made by [dirkjanm](https://github.com/dirkjanm) that will help us create the Keytab file:

> [https://github.com/dirkjanm/forest-trust-tools/blob/master/keytab.py](https://github.com/dirkjanm/forest-trust-tools/blob/master/keytab.py)

In order to create the Keytab, we need to edit the script to insert both `AES-256` keys:
![keytab.py](/assets/img/HackDay2024/3_keytabpy.png)

Now we can execute the script and TADA we just created our Keytab!

Finally, we can go back to WireShark preferences and load our KeyTab file.
Also, don't forget to check both boxes:
![decrypting](/assets/img/HackDay2024/4_decrypting.png)

At this point, we successfully decrypted some encrypted parts of the Kerberos communications such as parts in the Ticket Granting Ticket (TGT).

### Analysing Kerberos packets

Now let's search for the requested information!

#### TGT renewal date:

It can be found in the **encrypted part** of a **TGT** inside an `AP-REQ` request (Kerberos funny easter egg), which is in the **pre-authentication data** of a `TGS-REQ` request:
![renew](/assets/img/HackDay2024/5_renew.png)

#### User's machine name in uppercase:

The user's machine name can be found in multiple locations, the simple way is to retrieve it from the req-body of an `AS-REQ` request (which is not encrypted):
![machinename](/assets/img/HackDay2024/6_machinename.png)

#### Last subkey value:

A subkey in the Kerberos authentication process can be found inside an `AP-REP` request.
In this case, the last subkey is located inside the last LDAP response which uses the **GSS-API** to carry out the negotiation using an `AP-REP` request inside a Kerberos blob.

Inside this request, we can retrieve the **subkey** value inside the encrypted part:
![subkey](/assets/img/HackDay2024/7_subkey.png)

#### User SID:

Finally, the SID of the user "odin" can be found in the "**Authorization-data**" field which is inside the encrypted part of a TGT that can be found inside an `AP-REP` request.

![SID](/assets/img/HackDay2024/8_SID.png)

> The **Authorization-data** field here contains the **PAC** (Privilege Attribute Certificate) of the user "odin".
{: .prompt-info }

## References
- https://medium.com/tenable-techblog/decrypt-encrypted-stub-data-in-wireshark-deb132c076e7
- https://github.com/dirkjanm/forest-trust-tools/blob/master/keytab.py
