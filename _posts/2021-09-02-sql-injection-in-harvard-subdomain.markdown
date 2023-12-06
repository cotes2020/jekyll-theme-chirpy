---
layout:	post
title:	"SQL injection in harvard subdomain"
date:	2021-09-02
medium_url: https://noob3xploiter.medium.com/sql-injection-in-harvard-subdomain-be67a5dbf664
categories: [Hacking, Bug Bounty]
tags: [Hacking, Web, Bug Bounty, Sql Injection]
---


  Hi. In this writeup, i will show you a sqli that i found in harvard and also, a xss as a bonus

While looking through the subdomains of harvard, i found this one interesting subdomain <https://schedule.med.harvard.edu/> . I fuzzed the directory using ffuf and found this one interesting endpoint availability.php

![](/img/1*uhtGFv7D2LZYqOrETw6Dhg.png)

Visiting that endpoint only gave me this.

![](/img/1*yfc7Z-jVLCZSNJF_bsIdMQ.png)

So i fuzzed the parameters using arjun and found an interesting parameter called users. I tried it again with the users parameter and saw this

![](/img/1*CVu5U82ADM6-nymvyohO7Q.png)

This is the same error message as before. So i guessed i only have to provide a year parameter. I did that and it worked.

![](/img/1*Oq_nETL4NodydxxB8vSFYg.png)

Again, its the same as before, i provided a month parameter and it worked.

![](/img/1*EGKLSMY7ilhXkd8riuE1jA.png)

It worked again, but now, its asking for a day parameter, i gave it and it showed me this

![](/img/1*1jwC2jCFZfsAj8e_Y0PcXA.png)

We can see that our input in users parameter is reflected so i tried to get an xss. And it worked

![](/img/1*cu_8Ly98IPnN13p6p1R3Yw.png)

So we have an xss. I quickly reported it and tried testing the other parameters. I tried adding ‘ in the day parameter and it gave me an sql error.

![](/img/1*dewqjX2vKKpGZP739Q40YQ.png)

So, i have an sqli injection here. Since i suck at sql injection, i just let sqlmap do the job for me and sqlmap worked.

![](/img/1*RgMX7V-NBonq8ErRETzobA.png)

I dumped the tables. I didnt go any further anymore and reported it to them.

The sqli got accepted but the xss does not. Apparently, harvard dont accept xss which sucks since i reported alot of xss to them

This is now fixed so i decided to publish it. Visiting the subdomain will show this

![](/img/1*pRsN0zHj4_J8aqv9PHXPVg.png)

And visiting the endpoint <https://schedule.med.harvard.edu/availability.php> will throw a 404 error.

Thats the end of the writeup, thanks for reading.

  