---
title: Linux - Change Username and UID
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

# Change Username and UID

1. list user

```c
cat /etc/passwd
grep -w '^username' /etc/passwd
grep -w '^jerry' /etc/passwd
```

2. check user

```c
id tom

// login info
grep '^tom:' /etc/passwd

// group info
grep 'tom' /etc/group
groups tom

// Find home directory permissions
ls -ld /home/tom/

// see all Linux process owned by user and group named tom using the ps command:
ps aux | grep tom

```

3. rename

- id tom
- usermod -l jerry tom
- kill pid_number
- groupmod -n jerry tom
- usermod -d /home/jerry -m jerry
- id jerry


4. change user tom UID from 5001 to 10000
Type the usermod command as follows:

- id tom
- usermod -u 10000 tom
- id tom







.
