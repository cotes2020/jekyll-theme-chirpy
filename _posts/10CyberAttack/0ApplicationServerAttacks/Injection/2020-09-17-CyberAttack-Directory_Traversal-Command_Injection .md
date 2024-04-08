---
title: Meow's CyberAttack - Application/Server Attacks - Injection - Directory Traversal / Command Injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - Directory Traversal / Command Injection](#meows-cyberattack---applicationserver-attacks---injection---directory-traversal--command-injection)
  - [Directory Traversal / Command Injection](#directory-traversal--command-injection)
    - [directory traversal attack:](#directory-traversal-attack)
    - [command injection attack](#command-injection-attack)
  - [prevention](#prevention)

book:
- S+ 7th ch9
- CEH Prep ch13

---

# Meow's CyberAttack - Application/Server Attacks - Injection - Directory Traversal / Command Injection

---

## Directory Traversal / Command Injection

### directory traversal attack:

- <font color=LightSlateBlue> gain access to restricted directories (like root directory) through HTTP </font>>.

- is a specific type of command injection attack

- attempts to <font color=LightSlateBlue> access a file by including the full directory path, or traversing the directory structure </font>>.

  - gain access to the root directory of a system (limited by administrative users), essentially gain access to everything on the system.

  - the root directory of a website is far from the true root directory of the server;

  - an absolute path to the site’s root directory is something in IIS (Internet Information Services), like `C:\inetpub\wwwroot`.

  - If an attacker can get out of this directory and get to `C:\windows`, the possibility for inflicting harm is increased exponentially.

### command injection attack

- simplest ways to perform directory traversal.

- injects system commands into computer program variables such that they are executed on the web server.

Example:

- exploiting <font color=OrangeRed> weak IIS implementation </font> by calling up a web page along with parameter <font color=OrangeRed> cmd.exe?/c+dir+c:\ </font>, call the command shell and execute a directory listing of the root drive <font color=OrangeRed> (C:\) </font>.

- With Unicode support, entries such as <font color=OrangeRed> %c%1c </font> and <font color=OrangeRed> %c0%af </font> can be translated into `/` and `\`, respectively.


The ability to perform command injection is rare these days.

- Most vulnerability scanners will check for weaknesses with `directory traversal/command injection` and inform you of their presence.

- To secure, run such a scanner and keep the web server software patched.


Example:

- Unix systems, the passwd file includes user logon information, and it is stored in the `/etc` directory with a full directory path of `/etc/passwd`

  - Attackers use commands: `../../etc/passwd` or `/etc/passwd` to read the file.

- ![Pasted Graphic](/assets/img/Pasted%20Graphic_c414rj7mt.png)

  - a command was entered

  - the attacker was attempting to gain access to the password file within the `/etc` directory.

  - If the attacker tried to inject code, they would not use commands, but rather PHP, ASP, or another language.

- they could use a remove directory command (such as `rm -rf`) to delete a directory, including all files and subdirectories.

## prevention
prevent these types of attacks
- Input validation.
