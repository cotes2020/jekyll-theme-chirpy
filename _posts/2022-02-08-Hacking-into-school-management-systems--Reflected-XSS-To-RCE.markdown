---
layout: post
title:	"Hacking into school management systems. Reflected XSS To RCE"
date:	2022-02-08
medium_url: https://noob3xploiter.medium.com/hacking-into-school-management-systems-reflected-xss-to-rce-74880c423024
categories: [Hacking, Code Review]
tags: [Hacking, Code Review, Web]
---

  As a hacker, we are asked a million times before if we can hack into their school system and change their grades.

![](/img/1*Ota1cQ3LccFNHGRf3MRL0Q.png)

So i decided to take it a little further and actually try to research on school management systems. I picked one of the most famous free school management system, Gibbon. Gibbon is a free and open source school management system that is used all over the world and Here’s what i found.

### CSRF

Gibbon, lack csrf protections on almost all of the functions. One functionality that is really interesting that lacks the csrf check is in password change.

![](/img/1*DqDref6EYdz7KtETaBlN6A.png)You might think this is vulnerable to csrf, but its not. Trying csrf on this doesn’t work and only logs us out.

![](/img/1*R1uevZ8QPum7mxQrWUGQmQ.png)After a little bit of researching, i found out that the csrf poc is not working because of HttpOnly and SameSite. So even though it has no csrf token protecting the functions, it is still unexploitable by simple csrf methods. And i found out that it can be bypassed if we have an xss <https://security.stackexchange.com/questions/121971/will-same-site-cookies-be-sufficient-protection-against-csrf-and-xss>.

### XSS

I started looking for reflected xss and found some, i will only be showing one instance of xss that i found. In StaffDashboard.php, the get parameter gibbonTTID is reflected into the page without proper filtering.

![](/img/1*ePMNXPVDcsNWbe9q4RnIJg.png)Leading to reflected xss.

![](/img/1*HdYUTfNidzSKpYwNRDuZDg.png)Now that we have a reflected xss, we can use this with the csrf from above.

### Account Takeover

With both xss and csrf available, i can now proceed with the account takeover by changing the password of user id 1. The user id 1 is normally the super admin of the site so taking over his account will give us full access to the system.

I came up with this payload

![](/img/1*0CainzLzPHdGU4jXS7CO7g.png)Then, i url encoded the payload and add it to my xss. After triggering the xss payload

![](/img/1*sDgZ9yW9o-BVu5-y3YgMqg.png)We can see that our payload works and it made a post request to change the username and password of user id 1 to admin:Password123!. Now we can login as the admin and have authenticated access .

### Authenticated RCE

As an authenticated user, we have access to the functionality to make lesson planners and in there, we can upload files.

![](/img/1*JgmQSwsqDcwq3f1EPS__iA.png)The file upload is properly implemented, it only allows certain file extensions and one of them is txt files. Keep this functionality in mind as we will use it later.

One interesting file that i found is export.php. This endpoint is only accessible by authenticated users.

![](/img/1*BEqpXPh9HHqQYDI4lmOr8Q.png)Here, you can see that it accept a parameter q, and use it in the include statement below. It also checks for .. so its not vulnerable to lfi. I found out, that if the file we included has php code in it, it will be executed even if the file we we include is not php. Combine with the file upload functionality from above, we can achieve an rce.

So i made a txt file called poc.txt and inside it is a simple <?php system(whoami); ?> And i uploaded it with the mentioned functionality above.

![](/img/1*i1Qf-R7fanXgdhxUYa9cVQ.png)The filename and the directory is shown in the responses. Then, i pass its path to the the q parameter in export.php

`http://localhost/gibon/export.php?q=%5Cuploads%5C2022%5C01/scratch_eyujxvxibrdess2s.txt/uploads/2022/01/poc_ckxsuyxkrm4pehkd.txt` .

I visited the link and….

![](/img/1*jujwOotg3ZrHdlaJ02XeBw.png)The command is executed. We now have an rce.

I reported these bugs to the vendor and they fixed them immediately the day after. Thank you to the gibbon team for being responsive.

Thanks for reading.

Join the bounty Hunter Discord Server: <https://discord.gg/bugbounty>

  