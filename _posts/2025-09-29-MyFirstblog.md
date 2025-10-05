---
title: My First Blog
date: 2025-09-29 11:30:00 +0800
categories: [CTF Writeup, HTB]
tags: [htb, crypto]
description:  
toc: true
---
# Summary 
Puppy is a Windows Active Directory pentest simulation. It starts with a set of creds in the HR group, which a common target of phishing attacks. That user has GenericWrite over the Developers group, so I’ll add my user and get access to SMB shares where I’ll find a KeePassXC database. I’ll crack the secret with John, and get auth as the next user. That uses is a member of Senior Devs, which has GenericAll over another user. I’ll reset that user’s password and get a WinRM session. This user has access to a site backup, where I’ll find a password to spray and get WinRM as the next user. Finally, I’ll abuse that user’s DPAPI access to get a saved credential for an administrator.

	

> Example line for prompt.
{: .prompt-warning }
