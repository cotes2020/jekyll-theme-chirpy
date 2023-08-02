---
title: Nmap
author: dyl4n
date: 2023-04-21 10:29:00 +0700
summary: "Nmap: The Ultimate Network Scanner for Network Exploration and Security Auditing"
categories: [HackingTools]
tags: [hacking, tools]
render_with_liquid: false
comments: true
image:
  path: /thumbnails/nmap-network-scanner.png
---

## Overview

As the internet continues to grow, it becomes increasingly important to manage and secure the networks that connect us all. One tool that has become a staple in the arsenal of network administrators and security professionals is Nmap, the Network Mapper.

Nmap is a free and open-source tool that allows you to explore and audit networks by scanning hosts and identifying open ports, running services, and operating systems. It was first released in 1997 by Gordon Lyon (also known as Fyodor), and has since become one of the most popular network scanning tools available.

Nmap is incredibly versatile, and can be used for a wide range of tasks, such as:

- Network discovery
- Port scanning
- Operating system, Service, Firewall detection
- Determining device type, DNS name, Mac address
- Vulnerability scanning
- Network mapping



## Using Nmap

### Port States
- `open` indicates that an application is listening for connections on the port. The primary goal of port scanning is to find these.
- `closed` indicates that the probes were received but but there is no application listening on the port.
- `filtered` indicates that the probes were not received and the state could not be established.
- `unfiltered` indicates that the probes were received but a state could not be established. In other words, a port is accessible, but Nmap is unable to determine whether it is open or closed.
- `open/filtered` indicates that the port was filtered or open but Nmap couldn’t establish the state.
- `closed/filtered` indicates that Nmap is unable to determine whether a port is closed or filtered. It is only used for the IP ID idle scan.

### Nmap Switches
Some common Nmap Switches:

|       Nmap switches       | Description                                                                                                                   |
|:------------------------: |------------------------------------------------------------------------------------------------------------------------------ |
|           `-sS`           | Syn Scan (Stealth scans                                                                                                       |
|           `-sT`           | TCP Connect Scan (three-ways handshake)                                                                                       |
|           `-sU`           | UDP Scan                                                                                                                      |
|     `-p` port_number      | scan specific port                                                                                                            |
|           `-p-`           | scan all ports                                                                                                                |
|           `-sV`           | Detect the version of the services running on the target                                                                      |
|           `-O`            | OS detection                                                                                                                  |
|       `-v`, `-vv`         | verbosity level                                                                                                               |
|           `-oA`           | save the nmap results in three major formats (`-oN`: normal format, `-oG`: grepable format, `-oX`: XML format                 |
|           `-A`            | enable "aggressive" mode (activates service detection, operating system detection, a traceroute and common script scanning.)  |
|    `-sN`, `-sF`, `-sX`    | TCP NULL, FIN, Xmas port scans                                                                                                |
|           `-iL`           | input files                                                                                                                   |
|           `-sL`           | List scan                                                                                                                     |
| `--script=<script-name>`  | run specific script                                                                                                           |

Read more: [https://www.tutorialspoint.com/nmap-cheat-sheet](https://www.tutorialspoint.com/nmap-cheat-sheet){:target="_blank"}

### NSE Scripts

The Nmap Scripting Engine (NSE) is an incredibly powerful addition to Nmap, extending its functionality quite considerably. NSE Scripts are written in the Lua programming language, and can be used to do a variety of things: from scanning for vulnerabilities, to automating exploits for them.
#### NSE categories
There are many categories available. Some useful categories include:
+ `safe` Won't affect the target
+ `intrusive` Not safe: likely to affect the target
+ `vuln` Scan for vulnerabilities
+ `exploit` Attempt to exploit a vulnerability
+ `auth` Attempt to bypass authentication for running services (e.g. Log into an FTP server anonymously)
+ `brute` Attempt to bruteforce credentials for running services
+ `discovery` Attempt to query running services for further information about the network (e.g. query an SNMP server).

#### Script Types and Phases

Prerule scripts
: These scripts run before any of Nmap's scan phases, so Nmap has not collected any information about its targets yet.
: They can be useful for tasks which don't depend on specific scan targets, such as performing network broadcast requests to query DHCP and DNS SD servers

Host scripts
: Scripts in this phase run during Nmap's normal scanning process after Nmap has performed host discovery, port scanning, version detection, and OS detection against the target host. 

Service scripts
: These scripts run against specific services listening on a target host.
Postrule scripts
: These scripts run after Nmap has scanned all of its targets.

#### Working with NSE

Using `-sC` to run a script scan using the default set of scripts.

To run a specific script, we would use `--script=<script-name>` , e.g. `--script=http-fileupload-exploiter`


Multiple scripts can be run simultaneously in this fashion by separating them by a comma. For example: 

`--script=smb-enum-users,smb-enum-shares`.

Some scripts require arguments (for example, credentials, if they're exploiting an authenticated vulnerability). These can be given with the `--script-args` Nmap switch. An example of this would be with the `http-put` script (used to upload files using the PUT method). This takes two arguments: the URL to upload the file to, and the file's location on disk.  For example:

`nmap -p 80 --script http-put --script-args http-put.url='/dav/shell.php',http-put.file='./shell.php'`

Note that the arguments are separated by commas, and connected to the corresponding script with periods (i.e.  `<script-name>.<argument>`).



