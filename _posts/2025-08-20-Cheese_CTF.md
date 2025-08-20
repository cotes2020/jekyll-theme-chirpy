---
title: "TryHackMe | Cheese CTF"
date: 2025-08-20 03:00:00 +0100
categories: [TryHackMe]
tags: [thm, ctf, writeup]
---

# Reconnaissance

in the Reconnaissance phase i started by using nmap to enumerate the open ports, after a while nmap returns a very long list.
we see that a lot of ports of interest are open like http and ssh

<img width="634" height="499" alt="image" src="https://github.com/user-attachments/assets/d8ba15e1-ca3c-4d96-aae1-2ead76b33de0" />


<img width="280" height="67" alt="image" src="https://github.com/user-attachments/assets/8ae02006-f1e3-470b-b66e-f93587529ca4" />


we can access the web page by visiting `http://10.10.15.229:80/` 


# SQLI AUTH Bypass

i tried enumerating folders but it didn't lead anywhere.

<img width="828" height="492" alt="image" src="https://github.com/user-attachments/assets/290cc50e-a195-4dfb-97ff-beaa21d2f496" />

browsing the site we can find a login page on `http://10.10.15.229:80/login.php`

<img width="1275" height="792" alt="image" src="https://github.com/user-attachments/assets/f483f5bb-3e3b-41c5-a4fb-a0c05cfbbf7a" />

i tried testing basic login credentials like admin:admin and admin:Password123 but nothing worked, so i intercepted a login post reequest using burp suite and used it to test the login page using sqlmap for sql injections which tells me that the username parameter appears to be injectable

<img width="1253" height="328" alt="image" src="https://github.com/user-attachments/assets/536b334e-5752-4e70-a6d4-8c537e554d59" />

<img width="1183" height="70" alt="image" src="https://github.com/user-attachments/assets/3b783620-ee3b-48e0-931d-b244cb578ec3" />

i inetially tried to dump the database which was successful and revealed a single user but the password hash associated whit the user was uncrackable using online sites like crackstation and x.

<img width="599" height="248" alt="image" src="https://github.com/user-attachments/assets/428f4de6-5b17-4b8e-9c21-016d4212aa16" />

so knowing the username parameter is injectable i tried some payloads and eventually this worked `' || 1=1;-- -` and we were able to bypass the authetication process

<img width="1266" height="792" alt="image" src="https://github.com/user-attachments/assets/b37493e5-6140-450c-b102-6d46e8c87480" />

# LFI and RCE

visiting the messages section of the site we see a clickable link that sends you to a page with the message : If you know, you know :D 

upon further inspection we realise the url is using file parameter to give us the message along with a filter `http://10.10.15.229/secret-script.php?file=php://filter/resource=supersecretmessageforadmin`,
a simple test to try to read a world readable file turns out to be successful

<img width="1271" height="796" alt="image" src="https://github.com/user-attachments/assets/50fd13ed-d954-485d-8b53-0b3050845a6e" />

we have a LFI !

we can use the convert.base64-encode filter to read the secret-script.php file like so :

<img width="1103" height="163" alt="image" src="https://github.com/user-attachments/assets/afd2d90c-63f4-41b4-940f-c15fd5bdc992" />

to get a reverse shell we need to use this tool https://github.com/synacktiv/php_filter_chain_generator 


we set our netcat listener and then we create and send our payload 

<img width="1248" height="133" alt="image" src="https://github.com/user-attachments/assets/6f1363ef-1834-4007-ba88-f3b462b287b1" />

and we have a shell !

<img width="529" height="157" alt="image" src="https://github.com/user-attachments/assets/ad51943e-f1b4-4ecc-93eb-d33a5139a2aa" />

# user flag
when we access the user comte's home directory we can access his .ssh folder and we find an authorized_keys file,

<img width="473" height="120" alt="image" src="https://github.com/user-attachments/assets/d32f14a0-0f44-4641-a386-73ef3156b571" />

we can generate an ssh key and add it to the file to get a shell as comte scince the file is writable 

<img width="771" height="443" alt="image" src="https://github.com/user-attachments/assets/07df89d7-7b89-43f4-9b85-2798ebdcd780" />

<img width="1249" height="82" alt="image" src="https://github.com/user-attachments/assets/946e7f9b-5b7d-4713-935b-d808ffeaf4b6" />

now connect via ssh 

<img width="741" height="700" alt="image" src="https://github.com/user-attachments/assets/1ff1c680-b0f9-43bf-a665-3daeae9acb2c" />

# root flag

when we check the sudo priveleges for comte we see 

<img width="531" height="121" alt="image" src="https://github.com/user-attachments/assets/f603fc92-83cb-4a1a-bb7d-64fb8e311f65" />

we see that the user reload the configuration files for systemd as well start a script called exploit.timer which will start the service exploit.service, lets see the script 

<img width="703" height="284" alt="image" src="https://github.com/user-attachments/assets/422eef2e-ff90-421b-b84b-2b93cfc8e506" />

let's try and run the script

<img width="678" height="71" alt="image" src="https://github.com/user-attachments/assets/f875b906-27ee-4421-8160-5654cd7b953f" />

it wont run! thats because the OnBootSec= parameter is empty which specifies to systemd when to to start the timer after reload
lets add the missing value and execute the script

<img width="686" height="294" alt="image" src="https://github.com/user-attachments/assets/df0d082a-dac4-4296-ab05-5fa82c6a33e1" />

now checking for binaries with the suid set reveals a binary called xdd

<img width="1176" height="497" alt="image" src="https://github.com/user-attachments/assets/ec6e4a20-71ae-4ecd-9a62-d43b6125ae1b" />

we can check gtfobins and realize we can use it to read files so we use it to read  the flag 

<img width="578" height="171" alt="image" src="https://github.com/user-attachments/assets/1cfacc40-e166-4279-8584-5e2c55c11fa0" />









