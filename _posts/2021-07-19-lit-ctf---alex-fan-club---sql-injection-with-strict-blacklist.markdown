---
layout:	post
title:	"Lit CTF — Alex Fan Club : Sql injection with strict blacklist"
date:	2021-07-19
medium_url: https://noob3xploiter.medium.com/lit-ctf-alex-fan-club-sql-injection-with-strict-blacklist-7abbcd402751
categories: [Hacking, CTF]
tags: [Hacking, CTF, Sql Injection, Web]
---

  This is my writeup on the Alex Fan Club challenge. I will show how i solved it all the way from the beginning. Lets start

![](/img/1*zAE3J11UZWYRZ2cPM4fYOQ.png)In the challenge, you can see that we are given a txt file and the vulnerable website. So i downloaded the zip file and visited the website.

![](/img/1*DUMIfpTR7hjbmQeulYdSmA.png)The website has a search field. A common vulnerability on search functionality is sqli so i tested it out by giving ‘

![](/img/1*51Zy_JahRJ51qlaS3IzXdg.png)And it gives us 500 meaning its vulnerable. Now i tried the normal ORDER BY sqli with a comment but it gave me this

![](/img/1*miU6TBdo4_P0RHzGE5SVbA.png)You can see that there is a blacklist on the characters “*-/ |%” . I dont know any bypass for it so i checked the zip file first. The zip file contain the source code of the website

![](/img/1*B_hMqezfcsxQeUWQeaPDdw.png)We can see that it directly get our input and insert it in the sql query making it vulnerable to sqli. Also, we can see that it checks for the existence of the characters *-/ |% . So what we now need to do is exploit a sqli without *-/ |%

One of my main problem is how can i inject comments since — is blocked and also /**/ and sqlite dont allow # as comment, i dont know what to do. So i asked in our discord server and someone has a really smart idea

![](/img/1*CHjIhRyY5FKZAoGZAlHY2A.png)Instead of comments, we will just complete the query so it is not invalid. So what i did is open up a sqlite playground in <https://sqliteonline.com/> and tried to replicate the query of the challenge

![](/img/1*AQa1rS1aDTAgaZRYF6FCyQ.png)<https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/SQL%20Injection/SQLite%20Injection.md>. I followed this cheat sheet and tried the extracting sqlite version with a union select

![](/img/1*c-OjgN79ddHG2BQc4_MxGw.png)This is the query that i came up with. I tried it in the server. Before that, we should know that in the challenge, there is only one column so we have to reduce our columns in the query

![](/img/1*pNR7KmwZ-DJsXExZzK3Dqw.png)And space it blacklisted. It doesnt work. After playing with it for a while, i found out that we can use tabs and new line instead of space. It took me awhile to find that out. Tabs is easier so i used it

![](/img/1*qV_HKkLWbO47oFyF7pxyrg.png)So i crafter a query again and tested it in the server and it give me the sqlite version meaning we have a working sqli now

![](/img/1*9rA55Pvpb91N0l_yb8_PhQ.png)Now, to dump the tables, we will just follow the cheatsheet. So i used the payload asdasd’ UNION SELECT tbl\_name FROM sqlite\_master WHERE type=’table’ and tbl\_name NOT like ‘sqlite\_ I replaced all spaces with tabs. And it works

![](/img/1*GDII2kr3kiwxQQtFv8Hu8w.png)Now we know that there is a table called flag\_is\_in\_here. Next up is extracting columns. Again, we followed the cheatsheet for that. The query that i come up is asd’ UNION SELECT sql FROM sqlite\_master WHERE type!=’meta’ AND sql NOT NULL AND name =’flag\_is\_in\_here’ AND ‘1’!=’ You can see in the end, i added '1'!=' It is because there is still a leftover %' from our query when we injected to it and this will get rid of it. So i tried it up and it worked

![](/img/1*dkBMjlI3T5m7I8VCROQACg.png)So now we know that there is a column called flag\_column. Now to extract the value of it, i used the query asdasd’ UNION SELECT flag\_column from flag\_is\_in\_here WHERE flag\_column like’. I tried it out and it worked

![](/img/1*nOg6ER5yrxsmlSpyKi3lhg.png)We got the flag.

Our team, noobzUnited, ranked 36 in the ctf. It is a hard ctf but fun. Thanks for reading

Join the discord server: <https://discord.gg/bugbounty>

  