---
title: Meow's Testing Tools - Windows Server 2012 Management Tools
# author: Grace JyL
date: 2019-09-17 11:11:11 -0400
description:
excerpt_separator:
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]


---

# Command-Line Reference  Nltest


---

## net

```
C:\Users\shufler>net help view

NET VIEW
[\\computername [/CACHE] | [/ALL] | /DOMAIN[:domainname]]

NET VIEW displays a list of resources being shared on a computer. When used
without options, it displays a list of computers in the current domain or
network.

\\ComputerName        Specifies the computer that contains the shared resources that you want to view.

/CACHE                Displays the offline client caching settings for the resources on the specified computer.

/ALL                  Displays all the shares including the $ shares

/domain[:DomainName]  Specifies the domain for which you want to view the available computers.
                      If you omit DomainName, /domain displays all of the domains in the network.

```


```c

net view
- returns the computers in the current domain or network.
- only show computers that have file and printer sharing enabled.

net view /all
- shows all shares available, including administrative shares like C$ and admin$

net view /domain:contoso.com
- list all of the sharing computers in the contoso.com domain

net view /all /domain
- shows all shares (regular and administrative) in the domain

net localgroup "administrator”
net group /dom "domain admins"
- see who are the domain admins:
```

---

## Nltest.exe

use nltest to:
- Get a list of domain controllers
- Force a remote shutdown
- Query the status of trust
- Test trust relationships and the state of domain controller replication in a Windows domain
- Force a user-account database to synchronize on Windows NT version 4.0 or earlier domain controllers


```bash
# Example 1: Verify domain controllers in a domain
nltest /dclist:fourthcoffee
Get list of DCs in domain 'ntdev' from '\\fourthcoffee-dc-01'.
    fourthcoffee-dc-01.forthcoffee.com       [DS] Site: Rome
    fourthcoffee-dc-03.forthcoffee.com       [DS] Site: LasVegas
    fourthcoffee-dc-04.forthcoffee.com       [DS] Site: LA
    fourthcoffee-dc-09.forthcoffee.com       [DS] Site: NYC
    fourthcoffee-dc-12.forthcoffee.com       [DS] Site: Paris
    fourthcoffee-dc-24.forthcoffee.com       [DS] Site: Chattaroy
    fourthcoffee-dc-32.forthcoffee.com       [DS] Site: Haifa
    fourthcoffee-dc-99.forthcoffee.com       [DS] Site: Redmond
    fourthcoffee-dc-63.forthcoffee.com [PDC] [DS] Site: London
The command completed successfully


# Example 2: Advanced information about users
nltest /user:"TestAdmin"
User: User1
Rid: 0x3eb
Version: 0x10002
LastLogon: 2ee61c9a 01c0e947 = 5/30/2001 13:29:10
PasswordLastSet: 9dad5428 01c0e577 = 5/25/2001 17:05:47
AccountExpires: ffffffff 7fffffff = 9/13/30828 19:48:05
PrimaryGroupId: 0x201
UserAccountControl: 0x210
CountryCode: 0x0
CodePage: 0x0
BadPasswordCount: 0x0
LogonCount: 0x33
AdminCount: 0x1
SecurityDescriptor: 80140001 0000009c 000000ac 00000014 00000044 00300002 000000
AccountName: User1
Groups: 00000201 00000007
LmOwfPassword: fb890c9c 5c7e7e09 ee58593b d959c681
NtOwfPassword: d82759cc 81a342ac df600c37 4e58a478
NtPasswordHistory: 00011001
LmPasswordHistory: 00010011
The command completed successfully



# Example 3: Verify trust relationship with a specific server
nltest.exe /server:fourthcoffee-dc-01 /sc_query:fourthcoffee
Flags: 30 HAS_IP  HAS_TIMESERV
Trusted DC Name \\fourthcoffee-dc-01.forthcoffee.com
Trusted DC Connection Status Status = 0 0x0 NERR_Success
The command completed successfully


# Example 4: Determine the PDC emulator for a domain
nltest /dcname:fourthcoffee
PDC for Domain fourthcoffee is \\fourthcoffee-dc-01
The command completed successfully


# Example 5: Show trust relationships for a domain

nltest /domain_trusts /all_trusts

nltest /domain_trusts
List of domain trusts:
    0: forthcoffee forthcoffee.com (NT 5) (Forest Tree Root) (Primary Domain)
The command completed successfully


```









---

ref
- [microsoft](https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/hh875576(v=ws.11))
.
