---
title: Create SOCKS Tunnel to route Web Traffic Securely Without a VPN
# author: Grace JyL
date: 2020-10-02 11:11:11 -0400
description:
excerpt_separator:
categories: [30System, Kali]
tags: [Linux, SOCKS]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

[toc]

# Create SOCKS Tunnel to route Web Traffic Securely Without a VPN

---

## Prerequisites
- Step 1 (macOS/Linux) — Setting Up the Tunnel
- Step 1 (Windows) — Setting Up the Tunnel
- Step 2 — Configuring Firefox to Use the Tunnel
- Step 3 — Reverting the Proxy in Firefox
- Step 4 (macOS/Linux) — Creating Shortcuts for Repeated Use
- Step 5 (Optional) — Troubleshooting: Getting Through Firewalls

---

## SOCKS 5 proxy tunnel.
- A SOCKS proxy is an SSH encrypted tunnel
- configured applications forward their traffic down, and then, on the server-end, the proxy forwards the traffic to the general Internet.
- Unlike a VPN, a SOCKS proxy has to be configured on an `app-by-app` basis on the client machine,
- client can set up apps without specialty software as the app is capable of using a SOCKS proxy.
- server-side, all need to configure is SSH.

---

## Step 1 (macOS/Linux) — Setting Up the Tunnel
- create an SSH key
- make sure the public side is added to the `‘authorized_keys’ file on your SSH Droplet`.
- create an SSH tunnel with SOCKS proxy enabled.

```bash
1. Set up the tunnel:
ssh -i ~/.ssh/id_rsa -D 1337 -f -C -q -N username@your_domain

# Explanation of arguments
# - -i: The path to the SSH key to be used to connect to the host
# - -D: Tells SSH that we want a SOCKS tunnel on the specified port number (1025 ~ 65536)
# - -f: Forks the process to the background
# - -C: Compresses the data before sending it
# - -q: Uses quiet mode
# - -N: Tells SSH that no command will be sent once the tunnel is up

2. Verify that the tunnel is running with this command:
ps aux | grep ssh
# Output
username    14345   0.0  0.0  2462228    452   ??  Ss    6:43AM   0:00.00 ssh -i ~/.ssh/id_rsa -D 1337 -f -C -q -N username@your_domain
# - quit terminal application and the tunnel will stay up.
# the -f argument, put the SSH session into the background:

# - to terminate the tunnel
# grab the PID 14345.
# use the command kill 14345
```

---

## Step 2 — Configuring Firefox to Use the Tunnel
- for a SOCKS 5 tunnel to work, have to use a local application that can implement the tunnel;
  - Firefox has this capability:
- This step is the same for Windows, macOS, and Linux.
- Make sure you have the port number that you used in your SSH command;
  - in our examples we’ve used 1337.

Open Firefox.
- Network Settings > Settings
- `Manual proxy configuration`.
- `SOCKS Hostlocal`: `host` or `127.0.0.1`
- `port`: `1337`
- check the box `'Proxy DNS when using SOCKS v5’`
- Click the OK button to save and close your configuration


---

## Closing the Tunnel (macOS/Linux)

```bash
ps aux |grep ssh
# sammy    14345   0.0  0.0  2462228    452   ??  Ss    6:43AM   0:00.00 ssh -i ~/.ssh/id_rsa -D 1337 -f -C -q -N sammy@your_domain
kill 14345
```

---

## (macOS/Linux) — Creating Shortcuts for Repeated Use


### Clickable BASH Script
- The script will set up the tunnel and then launch Firefox

- On macOS
  - the Firefox binary that we can launch from the command line is inside Firefox.app. Assuming the app is in the Applications folder,
  - the binary will be found at `/Applications/Firefox.app/Contents/MacOS/firefox`.
- On Linux systems
  - if you installed Firefox via a repo or if it’s pre-installed, then its location should be `/usr/bin/firefox`.

```bash
nano ~/socks.sh

# Add the following lines:

#!/bin/bash -e
ssh -i ~/.ssh/id_rsa -D 1337 -f -C -q -N sammy@`your_domain`
/Applications/Firefox.app/Contents/MacOS/firefox &

# - Replace 1337 with your desired port number (it should match what you put in Firefox)
# - Replace sammy@your_domain with your SSH user @ your hostname or IP
# - Replace /Applications/Firefox.app/Contents/MacOS/firefox with the path to Firefox’s binary for your system
# - Save your script. For nano, type CONTROL + o, and then to quit, type CONTROL + x.
```

- Make the script executable, so that when you double click on it, it will execute. From the command line, use the chmod command to add execute permissions:
- chmod +x /path/to/socks.sh
- On macOS, you may have to perform an additional step to tell macOS that a .sh file should be executed like a program and not be opened in an editor. To do this, right click on your socks.sh file and select 'Get Info’.
- Locate the section 'Open with:’ and if the disclosure triangle isn’t pointing down, click on it so you can see the dropdown menu. Xcode might be set as the default app.Get Info
- Change it to Terminal.app. If Terminal.app isn’t listed, choose 'Other’, and then navigate to Applications > Utilities > Terminal.app (you may need to set the pull down menu 'Enable’ from 'Recommended Applications’ to 'All Applications’).
- To open your SOCKS proxy now, double click on the socks.sh file. The script will open a terminal window, start the SSH connection, and launch Firefox. Feel free to close the terminal window at this point. As long as you kept the proxy settings in Firefox, you can start browsing over your secure connection:
- This script will help you quickly stand up the proxy, but you’ll still have to perform the manual steps listed above to find the ssh process and kill it when you’re done.

---

### Command Line Alias
- Different Linux distributions and macOS releases save aliases in different places. The best bet is to look for one of the following files and search for alias to see where other aliases are currently being saved.
Possibilities include:
- ~/.bashrc
- ~/.zshrc
- ~/.bash_aliases
- ~/.bash_profile
- ~/.profile

- Once you’ve located the correct file, add the alias below to the end of the file.

```bash
alias firesox='ssh -i ~/.ssh/id_rsa -D 1337 -f -C -q -N sammy@your_domain && /Applications/Firefox.app/Contents/MacOS/firefox &'

# - Replace 1337 with your desired port number (it should match what you put in Firefox)
# - Replace sammy@your_domain with your SSH user @ hostname or IP
# - Replace /Applications/Firefox.app/Contents/MacOS/firefox with the path to Firefox’s binary
# - Your aliases are only loaded when you start a new shell, so close your terminal session and start a new one. Now when you type:

source ~/.bash_profile

firesox
- This alias sets up your tunnel, then launches Firefox for you and returns you to the command prompt. Make sure Firefox is still set to use the proxy. You can now browse securely.
```

---

refer:
- [1](https://www.digitalocean.com/community/tutorials/how-to-route-web-traffic-securely-without-a-vpn-using-a-socks-tunnel)
