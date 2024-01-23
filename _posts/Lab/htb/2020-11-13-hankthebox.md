---
title: Lab - HTB - Setup
date: 2020-11-13 11:11:11 -0400
description: Learning Path
categories: [Lab, HackTheBox]
# img: /assets/img/sample/rabbit.png
tags: [Lab, HackTheBox]
---

- [Lab - HTB - Setup](#lab---htb---setup)
	- [starting point](#starting-point)
	- [invite](#invite)

---

# Lab - HTB - Setup


## starting point

Connections to the lab environment are made with OpenVPN, which comes pre-installed on Parrot and Kali.
- There are multiple different lab networks on Hack The Box, and you will require a connection pack for each.

Once you have downloaded your `.ovpn` connection pack on your virtual machine, you can connect to the lab network with the following terminal command: `sudo openvpn example.ovpn`
- `example.ovpn `should be replaced with the `full path or filename for your connection pack`.


```java
ports=$(nmap -p- --min-rate=1000 -T4 10.10.10.27 | grep ^[0-9] | cut -d '/' -f 1 | tr '\n' ',' | sed s/,$//)
nmap -sC -sV -p$ports 10.10.10.27
// Ports 445 and 1433 are open, which are associated with file sharing (SMB) and SQL Server.

smbclient -N -L \\\\10.10.10.27\\
// checking to see if anonymous access has been permitted, as file shares often store configuration files containing passwords or other sensitive information. We can use smbclient to list available shares.

smbclient -N \\\\10.10.10.27\\backups
// a share called backups. Let's attempt to access it and see what's inside.

// There is a dtsConfig file, which is a config file used with SSIS.

<DTSConfiguration>
 	<DTSConfigurationHeading>
 		<DTSConfigurationFileInfo GeneratedBy="..." GeneratedFromPackageName="..." GeneratedFromPackageID="..." GeneratedDate="20.1.2019 10:01:34"/>
 	</DTSConfigurationHeading>
 	<Configuration ConfiguredType="Property" Path="\Package.Connections[Destination].Properties[ConnectionString]" ValueType="String">
 		<ConfiguredValue>Data Source=.;Password=M3g4c0rp123;User ID=ARCHETYPE\sql_svc;Initial Catalog=Catalog;Provider=SQLNCLI10.1;Persist Security Info=True;Auto Translate=False;</ConfiguredValue>
 	</Configuration>
 </DTSConfiguration>

// We see that it contains a SQL connection string, containing credentials for the local Windows user ARCHETYPE\sql_svc

```


## invite

1. open the Chrome Developers Tools.
2. Go through the elements tab and you will find a script with source (src) as: /js/inviteapi.min.js
3. go to https://www.hackthebox.eu/js/inviteapi.min.js .
4. makeInviteCode looks interesting. go back to https://www.hackthebox.eu/invite and try to find its contents.
5. Goto console tab in Chrome Developer Tools, and type `makeInviteCode()`. You will get a 200 Success status and data as shown below.
6. the text is encrypted and the encoding type is ROT13.
7. decode that message
8. need to make a POST request to “https://www.hackthebox.eu/api/invite/generate”.
9. `curl -XPOST https://www.hackthebox.eu/api/invite/generate`
10. success message as:`{“success”:1,”data”:{“code”: “somerandomcharacters12345”, “format”: “encoded”}, “0”:200}`
11. decoding it

XRCDC-VBPZS-ROGKG-OUDBT-WTHIW


##











.
