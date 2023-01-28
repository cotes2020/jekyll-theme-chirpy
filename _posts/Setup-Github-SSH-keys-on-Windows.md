---
layout: default
title: Setup Github SSH keys on Windows
date: 2022-10-04 00:00:00 00:00
comments: true
categories:
 - productivity
tags: 
 - ssh
 - keys
 - github
 - windows11
---

Recently, I need to setup ssh keys on my Github account and ran into an issue setting this up.  I thought I would document my setup procedure. For the most part, the process is pretty stright forward, but it's a little different than the linux environment. All my steps are using a PowerShell command window.
1. SSH keys are stored in the $ENV:USERPROFILE\.ssh folder.
2. Generating the SSH keys uses the ssh-keygen command. There are several key types to choose from.  In this example I'm using eliptical curve keys.
