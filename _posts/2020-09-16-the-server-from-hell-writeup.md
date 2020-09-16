---
layout: post
title: 'The Server From Hell - Writeup'
categories:
- Security
tags:
- ctf
- writeup
- tryhackme
---


## Introduction

---

The whole point of this challenge is having little to no information other than the fact that this box is 'from hell'. We are told one thing though: `Start at port 1337 and enumerate your way`. So let's just try scanning the machine for good measure and see what we're dealing with.

## Enumeration

---

Running a quick nmap scan (`nmap -sS -T5 -n -v -Pn <ip>`) on the target reveals the following:

![NMap Scan](/assets/img/media/server_from_hell_1.png)

The scan continued till the end and told me 30,000 ports were open. This tells us that there is some sort of firewall or program listening on all ports to make port scanning much more difficult. Let's instead head to port 1337 like the challenge description said.

![Connecting to port 1337](/assets/img/media/server_from_hell_2.png)

We get our first hint, telling us to go look for the trollface in the first 100 ports. Let's simply use netcat to automate this `nc 10.10.161.110 1-100 -v`.

![Printing banners from the first 100 ports](/assets/img/media/server_from_hell_3.png)

We get a message telling us to head to port 12345 next, so we also connect there using netcat.

![NFS Hint](/assets/img/media/server_from_hell_4.png)

This tells us there is an NFS share accesible on this host on the default port. You can explore it with nmap or the `showmount -e` command.

![Listing NFS shares](/assets/img/media/server_from_hell_5.png)

We see that `/home/nfs` is exported, so let's mount it locally and see what files we can access.

![Mounting the NFS share](/assets/img/media/server_from_hell_6.png)

We notice there is a backup.zip and when we try to unzip it, it asks for a password. Let's use `zip2john` to convert it to a format `john` will accept and run a hash cracking attack.

![Zip2John Warning](/assets/img/media/server_from_hell_7.png)

We can safely ignore the warning message and run `john` with the hash we just generated.

![JTR Zip Pass](/assets/img/media/server_from_hell_8.png)

Using the password to unzip the file we find a `.ssh` folder with the following files:

![Unzipped File](/assets/img/media/server_from_hell_9.png)

We got our first flag! Additionally, we got a hint to how to get into the system and an ssh private key! Let's explore:

![Exploring the ZIP](/assets/img/media/server_from_hell_10.png)

Great! Let's just check the public key to figure out for which user is this private key:

![SSH Public Key](/assets/img/media/server_from_hell_20.png)

We see that this key belongs to a user called `hades`. Let's use this key to login as him with ssh:

![Cannot SSH](/assets/img/media/server_from_hell_11.png)

It seems that SSH is not running on port 22 and we must find it. Remembering the contents of `hint.txt`, it seems to be telling us that the real SSH service is somewhere between the range of ports `2500-4500`. The hint provided with the challenge also mentions that only the real SSH service will respond properly while the others will just quit.

After enumerating for a long time using `nc 10.10.161.110 2500-4500`, I found out that the port is 3333. I'm sure that a python script using the `paramiko` library can be built or even something in bash but netcat was much more comvinient for me.

Let us try to log in as the user `hades` with the proper port now:

![SSH Login](/assets/img/media/server_from_hell_12.png)

And we're in! Now moving on to capturing `user.txt`.

## Escaping IRB and Capturing user.txt

---

Instead of a shell prompt, we get this odd little prompt at our terminal after logging in via SSH:

![IRB](/assets/img/media/server_from_hell_13.png)

After googling, we discover that `irb` is the interactive prompt for the programming language ruby. To escape and spawn a shell, we simply run `exec("/bin/bash")`.

![Capturing user.txt](/assets/img/media/server_from_hell_14.png)

Awesome! Now to enumerate the machine and escalate to root.

## Escalating to root

---

The hint for the challenge only mentions the command `getcap`, so let's list all the capabilities of any binary on the system.

![getcap](/assets/img/media/server_from_hell_15.png)

It seems like the `tar` binary has `cap_dac_read_search` which is the capability to read any file in the system as root! Let's abuse this to grab a copy of `/etc/passwd` and  `/etc/shadow` to crack the root password.

![Exploiting Tar](/assets/img/media/server_from_hell_16.png)

Extracting this archive and preparing for running `john` to crack the passwords on our own computer reveals that `root`'s password is actually easy to guess.

![Coping the archive](/assets/img/media/server_from_hell_17.png)

Running this file through `john` with the defailt rockyou.txt wordlist and `sha512crypt` is the format.

![Root's password](/assets/img/media/server_from_hell_18.png)

And we have cracked the password for the user `root`! Let us log in to root and get the `root.txt` flag:

![Final flag](/assets/img/media/server_from_hell_19.png)

And that is it!

## Conclusion

---

After abusing an open NFS share to download and crack a zip file containing an ssh key, we found the port the SSH service was running on and exploited an extra permission being given to the `tar` command to download and crack the hashes of the `root` user on the system.

I'm super proud of myself for making this first box, and hope you enjoyed hacking into it too!
