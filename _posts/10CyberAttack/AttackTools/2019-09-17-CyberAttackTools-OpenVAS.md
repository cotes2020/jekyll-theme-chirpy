---
title: Meow's Testing Tools - OpenVAS
# author: Grace JyL
date: 2019-09-17 11:11:11 -0400
description:
excerpt_separator:
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools, OpenVAS]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]

---


# OpenVAS on Kali Linux


---

## Setup Kali for installing OpenVAS

1. make sure Kali is up-to-date and install the latest OpenVAS.
2. run the `openvas-setup` command to setup OpenVAS, download the latest rules, create an admin user, and start up the various services.

```bash
root@kali:~# sudo apt-get update && apt-get dist-upgrade -y
root@kali:~# sudo reboot
root@kali:~# sudo apt-get install openvas -y

sudo gvm-setup

root@kali:~# sudo openvas-setup
[>] Checking redis.conf
[*] Editing redis.conf
[>] Checking openvassd.conf
[*] Adding to openvassd.conf
[>] Restarting redis-server
[>] Checking OpenVAS certificate infrastructure
[*] Creating OpenVAS certificate infrastructure
[>] Updating OpenVAS feeds
[*] Opening Web UI (https://127.0.0.1:9392) in: 5... 4... 3... 2... 1...
[>] Checking for admin user
[*] Creating admin user
User created with password 'e432aa97-2fd3-4c1b-8c16-1166cbd19d70'.
[+] Done


# !!!!
# When it’s done, it will show the admin login username and admin login password
sudo gvm-start
[*] Please note the password for the admin user
[*] User created with password 'd201fc56-6201-4f73-a646-c4787b1435fd'.


# Update feed for OpenVAS (Only required if there is new updates), when initializing, this step was done once already.
sudo gvm-feed-update


# Launch OpenVAS
sudo openvas-start
```

### Setup Errors

1. Occasionally, the ‘openvas-setup’ script will display errors at the end of the NVT download similar to the following.

```bash
(openvassd:2272): lib kb_redis-CRITICAL **: get_redis_ctx: redis connection error: No such file or directory
(openvassd:2272): lib kb_redis-CRITICAL **: redis_new: cannot access redis at '/var/run/redis/redis.sock'
(openvassd:2272): lib kb_redis-CRITICAL **: get_redis_ctx: redis connection error: No such file or directory
openvassd: no process found

# detects the issue and even provides the command to run to (hopefully) resolve the issue.
# see what component is causing issues.
openvas-check-setup
```


2. ` bash: …..: command not found error`

The Fix

```bash
# OpenVAS is changing the name, the new command gvm will replace all openvas commands.
# Since Kali Rolling updated repository, use gvm instead of openvas commands
sudo apt install gvm -y
sudo gvm-setup
sudo gvm-feed-update
sudo gvm-start


```

---

## Checking for OpenVAS ports
Once `openvas-setup` completes its process, the OpenVAS manager, scanner, and GSAD services should be listening:

```bash
root@kali:~# root@kali:~# netstat -antp
Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 127.0.0.1:80            0.0.0.0:*               LISTEN      4782/gsad
tcp        0      0 127.0.0.1:9392          0.0.0.0:*               LISTEN      4774/gsad
tcp        0      0 127.0.0.1:9390          0.0.0.0:*               LISTEN      4776/openvasmd
# 9392 is for WebGUI/OpenVAS Web Interface.
# TCP ports 9390 and 9392 listening on your loopback interface.

root@kali:~# ss -ant
State Recv-Q Send-Q Local Address:Port Peer Address:Port
LISTEN 0 128 127.0.0.1:9390 *:*
LISTEN 0 128 127.0.0.1:9392 *:*
```

---

## Checking OpenVAS services
run `openvas-check-setup` before launching OpenVAS just in case something went missing.

```bash

root@kali:~# openvas-check-setup
openvas-check-setup 2.3.7
  Test completeness and readiness of OpenVAS-9
  ...
 ERROR: Your OpenVAS-9 installation is not yet complete!

Please follow the instructions marked with FIX above and run this
script again.

If you think this result is wrong, please report your observation
and help us to improve this check routine:
http://lists.wald.intevation.org/mailman/listinfo/openvas-discuss
Please attach the log-file (/tmp/openvas-check-setup.log) to help us analyze the problem.
# The fix is given as well, run greenbone-scapdata-sync and it will sync OpenVAS SCAP database files.


root@kali:~# greenbone-scapdata-sync
OpenVAS community feed server - http://www.openvas.org/
This service is hosted by Greenbone Networks - http://www.greenbone.net/
sent 10,379 bytes  received 884,066,503 bytes  2,847,268.54 bytes/sec
total size is 926,410,667  speedup is 1.05
part 0 Done
part 1 Done
part 0 Done
part 1 Done
/usr/sbin/openvasmd
```

---

### Setup OpenVAS User account and changing password

```bash
# set manual password and create a new user from CLI.
root@kali:~# openvasmd --create-user=luo
User created with password '19c29356-c59e-481a-8c3d-80225f80302b'.
root@kali:~# openvasmd --create-user=blackmoreops
User created with password 'b4f70c8b-1c45-442d-a41b-b87b24f473b6'.

# new passwd
root@kali:~# openvasmd --user=blackmoreops --new-password=operations1
root@kali:~# openvasmd --user=admin --new-password=administrator1
root@kali:~# openvasmd --user=usrnamw --new-password=operations1

root@kali:~# openvasmd --get-users
admin
usrname
blackmoreops
```

---


## Start and end OpenVAS services

```bash
root@kali:~# openvas-start
[*] Please wait for the OpenVAS services to start.
[*] Web UI (Greenbone Security Assistant): https://127.0.0.1:9392

● greenbone-security-assistant.service - Greenbone Security Assistant
   Loaded: loaded (/lib/systemd/system/greenbone-security-assistant.service; disabled; vendor preset: disabled)
[*] Opening Web UI (https://127.0.0.1:9392) in: 5... 4... 3... 2... 1...

root@kali:~# openvas-stop
Stopping OpenVas Services
```


---

## Connect to the OpenVAS Web Interface Greenbone Security Assistant

1. https://127.0.0.1:9392
2. accept the self signed SSL certificate `confirm security exception`
3. Type in Admin username and password or one of the new users you’ve setup and bang

---

## Configure OpenVAS

---

### Configuring Credentials
Vulnerability scanners provide the most complete results when you are able to provide the scanning engine with credentials to use on scanned systems. OpenVAS will use these credentials to log in to the scanned system and perform detailed enumeration of installed software, patches, etc.
- add credentials via the “Credentials” entry under the “Configuration” menu.

---

### Target Configuration
OpenVAS, like most vulnerability scanners, can scan for remote systems
- but it’s a vulnerability scanner, not a port scanner.
- Rather than relying on a vulnerability scanner for identifying hosts,  much easier by using a dedicated network scanner like `Nmap` or `Masscan` and import the list of targets in `OpenVAS`.

```bash
# 1. identifying hosts
root@kali:~# nmap -sn -oA nmap-subnet-86 192.168.86.0/24
root@kali:~# grep Up nmap-subnet-86.gnmap | cut -d " " -f 2 > live-hosts.txt

# 2. import list of hosts
# import them under the “Targets” section of the “Configuration” menu.
```

![openvas6](https://i.imgur.com/nOU2p7d.png)

![openvas7](https://i.imgur.com/PQAxAKc.png)

---

### Scan Configuration
Prior to launching a vulnerability scan, you should fine-tune the Scan Config that will be used
- “Configuration” > "Scan Configs"
- can clone any of the default Scan Configs and edit its options, disabling any services or checks that don’t require.
- use Nmap to conduct some prior analysis of your target(s) to save hours of vulnerability scanning time.

---

### Task Configuration
credentials, targets, and scan configurations -> run a vulnerability scan.
- In OpenVAS, vulnerability scans are conducted as “Tasks”.
- When set up a new task, can further optimize the scan by either increasing or decreasing the concurrent activities that take place.
- With system with 3GB of RAM, we adjusted our task settings as shown below.

![openvas9](https://i.imgur.com/r1P5qaW.png)

With our more finely-tuned scan settings and target selection, the results of our scan are much more useful.

![openvas10](https://i.imgur.com/ppQvu3G.png)

---

## Automating OpenVAS

`openvas-automate.sh` by mgeeky
- a semi-interactive Bash script
- prompts you for a scan type and takes care of the rest.
- The scan configs are hard-coded in the script so if you want to use your customized configs, added under the “targets” section.

```bash
root@kali:~# apt -y install pcregrep
root@kali:~# ./openvas-automate.sh 192.168.86.61:: OpenVAS automation script.
mgeeky, 0.1[>] Please select scan type:
1. Discovery
2. Full and fast
3. Full and fast ultimate
4. Full and very deep
5. Full and very deep ultimate
6. Host Discovery
7. System Discovery
8. Exit

--------------------------------
Please select an option: 5

[+] Tasked: 'Full and very deep ultimate' scan against '192.168.86.61'
[>] Reusing target...
[+] Targets id: 6ccbb036-4afa-46d8-b0c0-acbd262532e5
[>] Creating a task...
[+] Task created successfully, id: '8e77181c-07ac-4d2c-ad30-9ae7a281d0f8'
[>] Starting the task...
[+] Task started. Report id: 6bf0ec08-9c60-4eb5-a0ad-33577a646c9b
[.] Awaiting for it to finish. This will take a long while...

8e77181c-07ac-4d2c-ad30-9ae7a281d0f8 Running 1% 192.168.86.61
```

[script](https://github.com/c610/tmp/blob/master/code16.py) by code16
- interacting with OpenVAS.
- need to make some slight edits to customize the scan type.

```bash
root@kali:~# ./code16.py 192.168.86.27
------------------------------------------------------------------------------
code16
------------------------------------------------------------------------------
small wrapper for OpenVAS 6[+] Found target ID: 19f3bf20-441c-49b9-823d-11ef3b3d18c2
[+] Preparing options for the scan...
[+] Task ID = 28c527f8-b01c-4217-b878-0b536c6e6416
[+] Running scan for 192.168.86.27
[+] Scan started... To get current status, see below:zZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzz
...
zZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzzZzz

[+] Scan looks to be done. Good.
[+] Target scanned. Finished taskID : 28c527f8-b01c-4217-b878-0b536c6e6416
[+] Cool! We can generate some reports now ... :)
[+] Looking for report ID...
[+] Found report ID : 5ddcb4ed-4f96-4cee-b7f3-b7dad6e16cc6
[+] For taskID : 28c527f8-b01c-4217-b878-0b536c6e6416

[+] Preparing report in PDF for 192.168.86.27

[+] Report should be done in : Report_for_192.168.86.27.pdf
[+] Thanks. Cheers!
```

---

# scan

```bash
# 1 Creating a target in OpenVAS
# 2 Configuring a scanning task in OpenVAS
# 3 Running the OpenVAS vulnerability scan

# Setting up a Host Discovery task
```


---

# OpenVAS with Docker


---

## Usage

Simply run:

```
# latest (9)
docker run -d -p 443:443 --name openvas mikesplain/openvas
docker run -d -p 443:443 -p 9390:9390 -p 9391:9391 -e OV_ADMIN_USERNAME=admin -e OV_PASSWORD=admin --name openvas mikesplain/openvas:8
docker run -d -p 443:443 -p 9390:9390 -p 9391:9391 -e OV_ADMIN_USERNAME=admin -e OV_PASSWORD=admin --name openvas mikesplain/openvas:9

# 9
docker run -d -p 443:443 --name openvas mikesplain/openvas:9
```

This will grab the container from the docker registry and start it up.
Once you see a `It seems like your OpenVAS-9 installation is OK.` process in the logs, the web ui is good to go.  Goto `https://<machinename>`

```
Username: admin
Password: admin
```

To check the status of the process, run:

```
docker top openvas
```

In the output, look for the process scanning cert data.  It contains a percentage.

To run bash inside the container run:

```
docker exec -it openvas bash
```

---

#### Specify DNS Hostname
By default, the system only allows connections for the hostname "openvas".  To allow access using a custom DNS name, you must use this command:

```
docker run -d -p 443:443 -e PUBLIC_HOSTNAME=myopenvas.example.org --name openvas mikesplain/openvas
```

---

#### OpenVAS Manager
To use OpenVAS Manager, add port `9390` to you docker run command:
```
docker run -d -p 443:443 -p 9390:9390 --name openvas mikesplain/openvas
```

---

#### Volume Support
mount your data directory to `/var/lib/openvas/mgr/`:
- local directory must exist prior to running.

```
mkdir data
docker run -d -p 443:443 -v $(pwd)/data:/var/lib/openvas/mgr/ --name openvas mikesplain/openvas
```

---

#### Set Admin Password

- specifying a password at runtime using the env variable `OV_PASSWORD`:

```
docker run -d -p 443:443 -e OV_PASSWORD=securepassword41 --name openvas mikesplain/openvas
```

--

#### Update NVTs

update your container by execing into the container and running a few commands:

```bash
docker exec -it openvas bash
## inside container
greenbone-nvt-sync
openvasmd --rebuild --progress
greenbone-certdata-sync
greenbone-scapdata-sync
openvasmd --update --verbose --progress

/etc/init.d/openvas-manager restart
/etc/init.d/openvas-scanner restart
```

---

#### Docker compose (experimental)

For simplicity a `docker-compose.yml` file is provided, as well as configuration for Nginx as a reverse proxy, with the following features:

* Nginx as a reverse proxy
* Redirect from port 80 (http) to port 433 (https)
* Automatic SSL certificates from [Let's Encrypt](https://letsencrypt.org/)
* A cron that updates daily the NVTs

To run:

* Change `"example.com"` in the following files:
  * [docker-compose.yml] - `(docker-compose.yml)`
  * [conf/nginx.conf] - `(conf/nginx.conf)`
  * [conf/nginx_ssl.conf] - `(conf/nginx_ssl.conf)`
* Change the "OV_PASSWORD" environmental variable in [docker-compose.yml] - (docker-compose.yml)
* Install the latest [docker-compose] - (https://docs.docker.com/compose/install/)
* run `docker-compose up -d`

---

#### LDAP Support (experimental)
Openvas do not support full ldap integration but only per-user authentication.
- A workaround is in place here by syncing ldap admin user(defined by `LDAP_ADMIN_FILTER `) with openvas admin users Every time the app start up.
- To use this, just need to specify the required ldap env variables:

```
docker run -d -p 443:443 -p 9390:9390 --name openvas -e LDAP_HOST=your.ldap.host -e LDAP_BIND_DN=uid=binduid,dc=company,dc=com -e LDAP_BASE_DN=cn=accounts,dc=company,dc=com -e LDAP_AUTH_DN=uid=%s,cn=users,cn=accounts,dc=company,dc=com -e LDAP_ADMIN_FILTER=memberOf=cn=admins,cn=groups,cn=accounts,dc=company,dc=com -e LDAP_PASSWORD=password -e OV_PASSWORD=admin mikesplain/openvas
```

---

#### Email Support
To configure the postfix server, provide the following env variables at runtime: `OV_SMTP_HOSTNAME`, `OV_SMTP_PORT`, `OV_SMTP_USERNAME`, `OV_SMTP_KEY`

```
docker run -d -p 443:443 -e OV_SMTP_HOSTNAME=smtp.example.com -e OV_SMTP_PORT=587 -e OV_SMTP_USERNAME=username@example.com -e OV_SMTP_KEY=g0bBl3de3Go0k --name openvas mikesplain/openvas
```

---


ref
- [openvas-docker](https://github.com/mikesplain/openvas-docker)
- [1](https://www.blackmoreops.com/2018/10/03/configure-tune-run-and-automate-openvas-on-kali-linux/)





.
