---
title: "TryHackMe | Cheese CTF"
date: 2025-08-20 03:00:00 +0100
categories: [TryHackMe]
tags: [thm, ctf, writeup]
image:
  path: https://github.com/user-attachments/assets/6c2f9476-ee67-40e4-8f86-cdc842490bd0
---

# Reconnaissance

In the reconnaissance phase, I started by using Nmap to enumerate the open ports. After a while, Nmap returned a very long list.  
We can see that a lot of ports of interest are open, such as HTTP and SSH.

<img width="634" height="499" alt="image" src="https://github.com/user-attachments/assets/d8ba15e1-ca3c-4d96-aae1-2ead76b33de0" />

<img width="280" height="67" alt="Capture d'écran 2025-08-20 022212" src="https://github.com/user-attachments/assets/b6eff54c-e75b-4942-b4ae-d97ac37b184c" />

We can access the webpage by visiting `http://10.10.15.229:80/`.  

<img width="1257" height="750" alt="image" src="https://github.com/user-attachments/assets/25634e9a-9166-4e67-ba63-f4e9db4b7d61" />

I tried enumerating folders, but it didn't lead anywhere.

<img width="828" height="492" alt="image" src="https://github.com/user-attachments/assets/290cc50e-a195-4dfb-97ff-beaa21d2f496" />

While browsing the site, we find a login page at `http://10.10.15.229:80/login.php`.

<img width="1275" height="792" alt="image" src="https://github.com/user-attachments/assets/f483f5bb-3e3b-41c5-a4fb-a0c05cfbbf7a" />

# SQLi Authentication Bypass

I tried testing basic login credentials like `admin:admin` and `admin:Password123`, but nothing worked.  
So, I intercepted a login POST request using Burp Suite and tested it with SQLMap for SQL injection. SQLMap reported that the `username` parameter appeared to be injectable.

<img width="1253" height="328" alt="image" src="https://github.com/user-attachments/assets/536b334e-5752-4e70-a6d4-8c537e554d59" />

<img width="1183" height="70" alt="image" src="https://github.com/user-attachments/assets/3b783620-ee3b-48e0-931d-b244cb578ec3" />

I initially tried to dump the database, which was successful and revealed a single user. However, the password hash associated with the user was uncrackable using online sites like CrackStation and others.

<img width="599" height="248" alt="Capture d'écran 2025-08-20 025521" src="https://github.com/user-attachments/assets/e975e7ae-0e21-4f8b-bbf1-2b8bdb940c62" />

Knowing the `username` parameter was injectable, I tried some payloads, and eventually this one worked: `' || 1=1;-- -`


This allowed us to bypass the authentication process.

<img width="1266" height="792" alt="image" src="https://github.com/user-attachments/assets/b37493e5-6140-450c-b102-6d46e8c87480" />

# LFI and RCE

Visiting the `messages` section of the site, we see a clickable link that takes us to a page with the message:  
> If you know, you know :D  

Upon further inspection, we realize the URL is using a `file` parameter along with a filter:  

`http://10.10.15.229/secret-script.php?file=php://filter/resource=supersecretmessageforadmin`


A simple test to try reading a world-readable file turns out to be successful.

<img width="1271" height="796" alt="image" src="https://github.com/user-attachments/assets/50fd13ed-d954-485d-8b53-0b3050845a6e" />

We have an **LFI**!  

We can use the `convert.base64-encode` filter to read the `secret-script.php` file like so:

<img width="1103" height="163" alt="image" src="https://github.com/user-attachments/assets/afd2d90c-63f4-41b4-940f-c15fd5bdc992" />

To get a reverse shell, we need to use this tool:  
[PHP filter chain generator](https://github.com/synacktiv/php_filter_chain_generator)  

We set our `Netcat` listener and then create and send our payload.

<img width="1248" height="133" alt="image" src="https://github.com/user-attachments/assets/6f1363ef-1834-4007-ba88-f3b462b287b1" />

And we have a shell!

<img width="529" height="157" alt="image" src="https://github.com/user-attachments/assets/ad51943e-f1b4-4ecc-93eb-d33a5139a2aa" />

# User Flag

When we access the user `comte`'s home directory, we can access the `.ssh` folder and find an `authorized_keys` file.

<img width="473" height="120" alt="image" src="https://github.com/user-attachments/assets/d32f14a0-0f44-4641-a386-73ef3156b571" />

Since the file is writable, we can generate an SSH key and add it to the file to get a shell as `comte`.

<img width="771" height="443" alt="image" src="https://github.com/user-attachments/assets/07df89d7-7b89-43f4-9b85-2798ebdcd780" />

<img width="1249" height="82" alt="image" src="https://github.com/user-attachments/assets/946e7f9b-5b7d-4713-935b-d808ffeaf4b6" />

Now we connect via SSH:

<img width="741" height="700" alt="image" src="https://github.com/user-attachments/assets/1ff1c680-b0f9-43bf-a665-3daeae9acb2c" />

# Root Flag

When we check the sudo privileges for `comte`, we see:

<img width="531" height="121" alt="image" src="https://github.com/user-attachments/assets/f603fc92-83cb-4a1a-bb7d-64fb8e311f65" />

The user can reload the configuration files for systemd and start a script called `exploit.timer`, which in turn starts the service `exploit.service`. Let's see the script:

<img width="703" height="284" alt="image" src="https://github.com/user-attachments/assets/422eef2e-ff90-421b-b84b-2b93cfc8e506" />

Let's try running the script:

<img width="678" height="71" alt="image" src="https://github.com/user-attachments/assets/f875b906-27ee-4421-8160-5654cd7b953f" />

It won't run! That's because the `OnBootSec=` parameter is empty, which specifies to systemd when to start the timer after reload.  
Let's add the missing value and execute the script.

<img width="686" height="294" alt="image" src="https://github.com/user-attachments/assets/df0d082a-dac4-4296-ab05-5fa82c6a33e1" />

Now, checking for binaries with the SUID bit set reveals a binary called `xdd`.

<img width="1176" height="497" alt="image" src="https://github.com/user-attachments/assets/ec6e4a20-71ae-4ecd-9a62-d43b6125ae1b" />

Looking it up on [GTFOBins](https://gtfobins.github.io/), we see that it can be abused to read files.  
We use it to read the `root` flag:

<img width="579" height="171" alt="image" src="https://github.com/user-attachments/assets/8beec053-aeda-4ec7-819d-5786dd0d6399" />
