---
title: Lab - TRAFFIC ANALYSIS EXERCISES
date: 2020-11-08 11:11:11 -0400
description: Learning Path
categories: [Lab, codegame]
# img: /assets/img/sample/rabbit.png
tags: [Lab, codegame]
---

[toc]

---

# TRAFFIC ANALYSIS EXERCISES


[TRAFFIC ANALYSIS EXERCISES](http://malware-traffic-analysis.net/training-exercises.html)
password: infected



2020-03-14 -- Traffic analysis exercise - Mondogreek
2020-02-21 -- Traffic analysis exercise - All aboard the hot mess express!
2020-01-30 -- Traffic analysis exercise - Sol-Lightnet
2019-12-25 -- Traffic analysis exercise - It happened on Christmas day
2019-12-03 -- Traffic analysis exercise - Icemaiden
2019-11-12 -- Traffic analysis exercise - Okay-boomer
2019-10-05 -- Traffic analysis exercise - Tinsolutions
2019-08-20 -- Traffic analysis exercise - Spraline
2019-07-19 -- Traffic analysis exercise - So hot right now
2019-06-22 -- Traffic analysis exercise - Phenomenoc
2019-05-02 -- Traffic analysis exercise - BeguileSoft
2019-04-15 -- Traffic analysis exercise - StringrayAhoy
2019-03-19 -- Traffic analysis exercise - LittleTigers
2019-02-23 -- Traffic analysis exercise - Stormtheory
2019-01-28 -- Traffic analysis exercise - Timbershade
2018-12-26 -- Two pcaps I provided for UA-CTF in November 2018
2018-12-18 -- Traffic analysis exercise - Eggnog soup
2018-11-13 -- Traffic analysis exercise - Turkey and defence
2018-11-01 -- Two pcaps I provided for UISGCON CTF in 2018
2018-10-31 -- Traffic analysis exercise - Happy Halloween!
2018-09-27 -- Traffic analysis exercise - Blank clipboard
2018-08-12 -- Traffic analysis exercise - Sputnik House
2018-07-15 -- Traffic analysis exercise - Oh knoes! Torrentz on our network!
2018-06-30 -- Traffic analysis exercise - Sorting through the alerts
2018-05-11 -- Traffic analysis exercise - Night Dew
2018-04-11 -- Traffic analysis exercise - Dynaccoustic
2018-03-10 -- Traffic analysis exercise - Max Headroom
2018-02-13 -- Traffic analysis exercise - Office work
2018-01-16 -- Traffic analysis exercise - "Mars Smart"
2017-12-23 -- Traffic analysis exercise - Carlforce!
2017-12-15 -- Traffic analysis exercise - Two pcaps, two emails, two mysteries!
2017-11-21 -- Traffic analysis exercise - Juggling act: Find out what happened in 6 pcaps.
2017-10-21 -- Traffic analysis exercise - Doc Brown and Marty McFly: Back to the Present.
2017-09-19 -- Traffic analysis exercise - Mission possible.
2017-08-29 -- Traffic analysis pop quiz.
2017-08-24 -- Traffic analysis exercise - Mix and match.


2017-07-22 -- Traffic analysis exercise - Where dreams are made.


2017-06-28 -- Traffic analysis exercise - Infection at the Japan field office.


2017-05-18 -- Traffic analysis exercise - Fancy that.

2017-04-21 -- Traffic analysis exercise - Double Trouble.

2017-03-25 -- Traffic analysis exercise - Coworker suffers March madness.
2017-02-11 -- Traffic analysis exercise - A very special one.
2017-01-28 -- Traffic analysis exercise - Thanks, Brian.
2016-12-17 -- Traffic analysis exercise - Your holiday present.
2016-11-19 -- Traffic analysis exercise - A luminous future.
2016-10-15 -- Traffic analysis exercise - Crybaby businessman.
2016-09-20 -- Traffic analysis exercise - Halloween Super Costume Store!
2016-08-20 -- Traffic analysis exercise - Plain brown wrapper.
2016-07-07 -- Traffic analysis exercise - Email Roulette.
2016-06-03 -- Traffic analysis exercise - Granny Heightower at Bob's Donut Shack.
2016-05-13 -- Traffic analysis exercise - No decent memes for security analysts.
2016-04-16 -- Traffic analysis exercise - Playing detective.
2016-03-30 -- Traffic analysis exercise - March madness.
2016-02-28 -- Traffic analysis exercise - Ideal versus reality.
2016-02-06 -- Traffic analysis exercise - Network alerts at Cupid's Arrow Online.
2016-01-07 -- Traffic analysis exercise - Alerts on 3 different hosts.
2015-11-24 -- Traffic analysis exercise - Goofus and Gallant.
2015-11-06 -- Traffic analysis exercise - Email Roulette.
2015-10-28 -- Traffic analysis exercise - Midge Figgins infected her computer.
2015-10-13 -- Traffic analysis exercise - Halloween-themed host names.
2015-09-23 -- Traffic analysis exercise - Finding the root cause.
2015-09-11 -- Traffic analysis exercise - A Bridge Too Far Enterprises.
2015-08-31 -- Traffic analysis exercise - What's the EK? - What's the payload?
2015-08-07 -- Traffic analysis exercise - Someone was fooled by a malicious email.
2015-07-24 -- Traffic analysis exercise - Where'd the CryptoWall come from?
2015-07-11 -- Traffic analysis exercise - An incident at Pyndrine Industries.
2015-06-30 -- Traffic analysis exercise - Identifying the EK and infection chain.
2015-05-29 -- Traffic analysis exercise - No answers, only hints for the incident report.
2015-05-08 -- Traffic analysis exercise - You have the pcap.  Now tell us what's going on.
2015-03-31 -- Traffic analysis exercise - Identify the activity.
2015-03-24 -- Traffic analysis exercise - Answer questions about this EK activity.
2015-03-09 -- Traffic analysis exercise - Answer questions about this EK activity.
2015-03-03 -- Traffic analysis exercise - See alerts for Angler EK.  Now do a summary.
2015-02-24 -- Traffic analysis exercise - Helping out an inexperienced analyst.
2015-02-15 -- Traffic analysis exercise - Documenting a Nuclear EK infection.
2015-02-08 -- Traffic analysis exercise - Mike's computer is "acting weird."
2015-01-18 -- Traffic analysis exercise - Answering questions about EK traffic.
2015-01-09 -- Traffic analysis exercise - Windows host visits a website, gets EK traffic.
2014-12-15 -- Traffic analysis exercise - 1 pcap, 3 Windows hosts, and 1 EK.
2014-12-08 -- Traffic analysis exercise - Questions about EK traffic.
2014-12-04 -- Traffic analysis exercise - Questions about EK traffic.
2014-11-23 -- Traffic analysis exercise - Questions about EK traffic.

---

## 2014-11-16 -- Traffic analysis exercise - EK traffic.

QUESTIONS

LEVEL 1 QUESTIONS:

```c
1) What is the IP address of the Windows VM that gets infected

    filter DHCP communication: "bootp" or "udp.port==67".
    filter: "http.request"
    The infected Windows VM IP address is 172.16.165.165

2) What is the host name of the Windows VM that gets infected

    frame -> DHCP Data -> Option: Host Name: K34EN6W3N-PC

3) What is the MAC address of the infected VM

    frame -> DHCP Data -> Client MAC address: f0:19:af:02:9b:f1 (f0:19:af:02:9b:f1)


4) What is the IP address of the compromised web site

    filter the GET requests "http.request.method == GET"
    frame -> HTTP Data -> Host: www.ciniholland.nl\r\n
    -> get 5 sites: Bing, leaving www.ciniholland.nl, adultbiz*in, 24corp-shop*com, stand.trustandprobaterealty*com.
    -> compromised website: www.ciniholland.nl, 82.150.140.30.

    The user visited “ciniholland” and through the referrer of each GET requests, we see that it leads to a very suspicious website which initiates downloads on the machine.

5) What is the domain name of the compromised web site
    ciniholland.nl/

6) What is the IP address and domain name that delivered the exploit kit and malware
    its IP address is 37.200.69.143


7) What is the domain name that delivered the exploit kit and malware

    File->Export Object->HTTP
    -> Represents dll files, swf files, jar files, respectively
    -> so, stand.trustandprobaterealty.com 37.200.69.143
```

LEVEL 2 QUESTIONS:


```c
1) What is the redirect URL that points to the exploit kit (EK) landing page

    the exploit kit domain: stand.trustandprobaterealty.com
    filter: "tcp.stream eq 18"
    1079	22.647035	188.225.73.100	172.16.165.165	HTTP	1086		HTTP/1.1 200 OK  (text/html)
    open "Line-based text data: text/html (35 lines)"
    <html> \r\n
    [truncated]<body bgcolor=#ffffff><div align='center'><iframe src='http://stand.trustandprobaterealty.com/?PHPSSESID=njrMNruDMhvJFIPGKuXDSKVbM07PThnJko2ahe6JVg|ZDJiZjZiZjI5Yzc5OTg3MzE1MzJkMmExN2M4NmJiOTM' border=0 width=125 height=10 scro
    \r\n

    or

    File->Export Object->HTTP
    And find it in the extracted file:
    1074	22.627396	188.225.73.100	172.16.165.165	HTTP	1086	HTTP/1.1 200 OK  (text/html)
    After saving as a txt file, view the content and find
    So pointing to the EKs orientation (redirect, there is no translation into the redirect, because only the content of the page contains the pointing URL, not redirected) URL is http: 24corp-shop*com/


2) Beside the landing page (which contains the CVE-2013-2551 IE exploit), what other exploit(s) sent by the EK

    a Flash exploit and a Java exploit
    Media type: application/x-shockwave-flash (8227 bytes)
    Media type: application/java-archive (10606 bytes)


4) How many times was the payload delivered
    File->Export Object->HTTP->application/x-shockwave-flash
    3 times


5) Submit the pcap to VirusTotal and find out what snort alerts triggered.  What are the EK names are shown in the Suricata alerts
    Submit the file at www.virustotal.com/en to view the Suricata alerts information.

    Snort Alerts:
    - Sensitive Data was Transmitted Across the Network
      - (spp_sdf) SDF Combination Alert [1]
    - Potentially Bad Traffic
      - (http_inspect) LONG HEADER [19]
    - Unknown Traffic
      - (http_inspect) UNKNOWN METHOD [31]

    Suricata Alerts:
    "Potentially Bad Traffic"
    ET POLICY Reserved Internal IP Traffic [2002752]
    ET WEB_CLIENT Possible String.FromCharCode Javascript Obfuscation Attempt [2011347]
    ET POLICY Vulnerable Java Version 1.6.x Detected [2011582]
    ET CURRENT_EVENTS Exploit Kit Delivering JAR Archive to Client [2014526]
    ET CURRENT_EVENTS SUSPICIOUS JAR Download by Java UA with non JAR EXT matches various EKs [2016540]
    ET INFO Java File Sent With X-Powered By HTTP Header - Common In Exploit Kits [2017637]
    ET INFO JAR Size Under 30K Size - Potentially Hostile [2017639]
    ET INFO JAR Sent Claiming To Be Text Content - Likely Exploit Kit [2018234]
    "Attempted Information Leak"
    ET POLICY Java Url Lib User Agent [2002946]
    Potential Corporate Privacy Violation
    ET POLICY Suspicious Microsoft Windows NT 6.1 User-Agent Detected [2010228]
    ET POLICY Outdated Windows Flash Version IE [2014726]
    "Not Suspicious Traffic"
    ET POLICY Java JAR file download [2011854]
    "A Network Trojan was Detected"
    ET INFO JAVA - Java Archive Download By Vulnerable Client [2014473]
    ET INFO suspicious - gzipped file via JAVA - could be pack200-ed JAR [2017910]
    * ET CURRENT_EVENTS GoonEK encrypted binary (3) [2018297]
    * ET CURRENT_EVENTS Goon/Infinity URI Struct EK Landing May 05 2014 [2018441]
    * ET CURRENT_EVENTS RIG EK Landing URI Struct [2019072]
    "Detection of a Non-Standard Protocol or Event"
    GPL POLICY TRAFFIC Non-Standard IP protocol [2101620]
```

LEVEL 3 QUESTIONS:

```c
1) Checking my website, what have I (and others) been calling this exploit kit
    Rig EK
    Rig is similar to Infinity EK (originally identified as Goon in the spring of 2014). Some good info on Rig EK can be found at: http://www.kahusecurity.com/2014/rig-exploit-pack/


2) What file or page from the compromised website has the malicious script with the URL for the redirect

    1st malicious domain
    161	6.073686	172.16.165.165	82.150.140.30	HTTP	621	www.ciniholland.nl	GET / HTTP/1.1
    follow -> TCP stream
    Content-Encoding: gzip
    The index page for www.ciniholland.nl had the URL for http://24corp-shop .com/


3) Extract the exploit file(s).  What is(are) the md5 file hash(es)

    Flash exploit: 7b3baa7d6bb3720f369219789e38d6ab
    Java exploit: 1e34fdebbf655cebea78b45e43520ddf

    use a *nix command line tool
    or
    submit the files to Virus Total.



4) VirusTotal doesnt show all the VRT rules under the "Snort alerts" section for the pcap analysis. run your own version of Snort with the VRT ruleset as a registered user (or a subscriber), what VRT rules fire

    [1:30936:3] EXPLOIT-KIT Goon/Infinity/Rig exploit kit outbound uri structure
    [1:30934:2] EXPLOIT-KIT Goon/Infinity/Rig exploit kit encrypted binary download
    [1:25562:4] FILE-JAVA Oracle Java obfuscated jar file download attempt
    [1:27816:5] EXPLOIT-KIT Multiple exploit kit jar file download attempt

    Your results will very, depending on how you have your Snort installation configured. If you haven't tried to set up Snort on your own, check out some of the Snort Setup Guides at: https://www.snort.org/documents




```









































。
