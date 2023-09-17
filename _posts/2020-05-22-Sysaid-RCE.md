---
title: SysAid On-Premise 20.1.11 - Unauthenticated RCE
date: 2020-05-22 13:33:37 +/-TTTT
categories: [cve-2020-10569]
tags: [ghostcat,cve-2020-10569]     # TAG names should always be lowercase
---

## Introduction

In 2020, I identified a vulnerability in the "SysAid Help Desk Software". This security flaw was significant enough to be assigned the reference CVE-2020-10569. In this article, I'll detail my discovery process, explain the nature of the vulnerability, and discuss its implications for users of the software.

## Sys-Aid Help Desk software

Sys-aid help desk is a software to help corporations with automating and tracking tickets, in addition to reporting and asset management. The product has both SaaS and on-prem solutions, we aim to target the on-prem solutions in this case. 

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/fddd01bf-cc94-4929-92dc-71c67f706767)


## Discovery

During a red team engagement, I encountered a specific login page.

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/981b1606-fe23-4c5d-96c2-608eba6ec58b)

At first glance, I was unfamiliar with the product behind this login page. However, a brief online research provided me with a clearer understanding of its functionalities and significance. In red teaming, maintaining a low profile is crucial to avoid detection. Notably, the `.jsp` file extensions indicated that the application was Java-based. Additionally, the presence of the spring icon hinted at the potential use of the `springboot` framework for its development.

## Vulnerability discovery

While there are numerous attack vectors to consider when attempting to compromise an asset, it's essential to opt for methods that minimize the risk of detection. Rather than directly targeting the application, I shifted my focus to the underlying hosting server, which often presents overlooked vulnerabilities.

One such vulnerability that crossed my mind was the GhostCat attack. To validate this hypothesis, I utilized `netcat` to check port `8009`. To my astonishment, the port was indeed open, indicating a potential avenue for further exploration.


## What is GhostCat attack?

> GhostCatcat permits an attacker to access arbitrary files throughout the web application. This includes directories such as `WEB-INF` and `META-INF`, as well as any other location accessible via ServletContext.getResourceAsStream(). Additionally, it enables the attacker to process any file within the web application as if it were JSP.

By default, remote code execution is not achievable. However, if an application on a vulnerable Tomcat version has a file upload flaw, GhostCatcat can be combined with this flaw to facilitate remote code execution. For this to be successful, the attacker needs to save the uploaded files to the document root and have direct access to the AJP port from outside the target network.


## Installing On-prem

![Desktop View](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/fddd01bf-cc94-4929-92dc-71c67f706767){: width="972" height="589" .w-50 .right}
Luckily, SysAid provide installation of the product with the trial version, therefore, I decided to install it locally and search for files. 

Surfing through the installation directory, we find a very interesting file `UploadIcon.jsp`! The file is used by the administrator to change the icon, Lets try to access the path directly without authentication. 

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/69f4b18c-a051-4228-820d-9beb3052b25b)
 

Aaaand it works! 

## Recap
So now, we have the following: 

- AJP connector with port `8009` accessible. This helps to read and interpret files from the webroot directory. 
- Unauthenticated file upload on `UploadIcon.jsp` 

## Exploitation

Do you remember how we are able to leverage Local file inclusion (LFI) vulnerabilities in PHP applications to RCE, even with restricted file upload? 

If you can upload anyfile (no matter what extension it is), you still can interpret it with the Local file inclusion vulnerability and get it executed. For example: upload `test.ico` file with PHP code inside and include it through the vulnerability. 

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/79f5e13e-fa06-4f5b-8144-f8df5ad025e0)
 

That will be the same case in this application, but we need to know what is the full path of the uploaded file. 

1. Luckily the application shows the full path of the uploaded file in a Javascript function named      `loadEvt`, we can see that the file has the following path: 

<img width="1643" alt="image" src="https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/28623b2a-4cb4-4380-9b93-c79cccd76544">


`C:\Program files\SysAidServer\tomcat\webapps\..\..\..\root\WEB-INF\temp\sysaidtemp_[random].dat`

2. Now, the interpretation of the file should be easy, as we will use the AJP connector to interpret the path of `WEB-INF\temp\sysaidtemp_[random].data` instead of `web.xml` file. 

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/e89b5716-4cf0-4020-b324-cc9949085368)


3. Enjoy the RCE 

![image](https://github.com/ahmedsherif/ahmedsherif.github.io/assets/4347574/91692570-678d-453f-aa65-445d1b4aac6c)
 

### Timeline disclosure

- 5th of April 2020 - Reported both unauthenticated file upload and GhostCatcat attack. 
- 7th of April 2020 - Recieved confirmation from SysAid and fix will be released the next version
- 5th of May 2020 - Sysaid fixed both vulnerabilites and assigned CVE-2020-10569 for it. 
