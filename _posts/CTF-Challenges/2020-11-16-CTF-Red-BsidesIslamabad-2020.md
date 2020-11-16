---
title: CTF-Red @ BsidesIslamabad-2020 - Pakistan's first ever Bsides
author:
date: 2020-11-16 15:00:00 +0500
comments: true
categories: [Categories, CTF-Challenges]
tags: [ctf, bsidespakistan, cyber security] # TAG names should always be lowercase
---

**Assalam-o-Alaikum folks!**
I hope you guys are safe and sound in this pandemic.

BsidesIslamabad-2020 was the first ever Bsides cyber security conference of Pakistan, conducted on _November 7th-8th 2020_.
For those who don't know what _Bsides_ is:

> BSides is a community-driven framework for building events for and by information security community members.

BsidesIslamabad had a lot of amazing speakers, workshops on hardware hacking and reverse-engineering and two CTFs:

- <span style="color:red"> CTF-Red (offensive) </span>
- <span style="color:blue"> CTF-Blue (defensive) </span>

In this blog post I am going to talk about some of the challenges from <span style="color:red">CTF-Red</span> of BsidesIslamabad 2020. The challenges were divided into different categories like web,binary,cloud etc.
The allowed time period was 6 hours and during that I was able to solve only 5 challenges of web categories, though after the ctf finished, I was able to solve a few more.

Enough for the boring talk, let's jump into the challenges. Here I am going to talk about following challenges:

1. Easy Peasy
2. InjectedOrg
3. Manipulator
4. ShaktimanDaDev
5. BrokenAuth
6. Remani
7. X-Pass
8. HackTheAdmin

# Easy Peasy

![Easy Peasy](/assets/img/posts/bsidesisb2020/easy-peasy.png)

The home page contains just a background image and nothing else. First two things I usually do in such scenario is to look into the source code for any javascript file, and 2nd thing is to run directory brutefoce to find any hidden content. Though there were no JS files, I did get some hidden directories.

Command:

```shell
dirsearch -u https://target-url/ -e [EXTENSIONS] -w [WORDLIST]
```

Output:

![Easy Peasy dirsearch](/assets/img/posts/bsidesisb2020/easy-peasy-dir-search.png)

There was nothing usefull inside _1.txt_ & _2.txt_. The file `index.php` had following content:
`hey to access notes just change the id parameter (for example 1,2,3)... you are not allowed to access secret file`

Well we usaully do something that we are specifically told not to do, right? So I tried to access the secret file:

```shell
https://target-url/index.php?id=secret.txt
```

Output of this file was something like this:
`....pssssttt.... flag is one directory up....its named flag.txt`

So now I replaced the `secret.txt` with `flag.txt`

![Easy Peasy Not Easy](/assets/img/posts/bsidesisb2020/easy-peasy-flag.txt.png)

Not that easy right? Well looking back at the output of `secret.txt` it says one directory up!

```
https://target-url/index.php?id=../flag.txt
```

**Voila!** challenge Solved!

## Vulnerability?

Well the vulnerability we just exploited is known as `Local File Inclusion`.

> LFI
>
> An attacker can use Local File Inclusion (LFI) to trick the web application into exposing or running files on the web server. Typically, LFI occurs when an application uses the path to a file as input. If the application treats this input as trusted, a local file may be used in the include statement.

---

# InjectedOrg

![InjectedOrg](/assets/img/posts/bsidesisb2020/injectedOrg.png)

Main page contains details of some users where I can edit their data, add new user or delete the existing users. Here I did tried a few things like IDOR, trying to find if there is any hidden user (may be an admin?), tried XSS (I don't know why I tried XSS), though I did got an alert box but did not know what to do with that ... sed life!

Looking back at the name of this challenge, I had an epiphany! Sql**Injection** (InjectedOrg => Injection, see! clever right?). Well I selected a user and the Url had an id paramter, simply added an apostrophe (**`'`**) and in response got a sql error!

Then the further steps were easy:

```shell
python sqlmap.py -u "http://target-url/read.php?id=1" --tables
```

![InjectedOrg DB Tables](/assets/img/posts/bsidesisb2020/injectedOrg-tables.png)

```shell
python sqlmap.py -u "http://target-url/read.php?id=1" -D bruhpls -T flag --dump
```

Output contained the flag, challenge solved!

## Vulnerability?

The vulnerability we just exploited is known as `SqlInjection`

> SQLi
>
> SQL Injection (SQLi) is a type of an injection attack that makes it possible to execute malicious SQL statements. These statements control a database server behind a web application. Attackers can use SQL Injection vulnerabilities to bypass application security measures. They can go around authentication and authorization of a web page or web application and retrieve the content of the entire SQL database. They can also use SQL Injection to add, modify, and delete records in the database.

---

# Manipulator

![Manipulator](/assets/img/posts/bsidesisb2020/manipulator.png)

We have a link shared by our friend who is existing user of this web application. Upon clicking the link we are welcomed with a registration form to register our account.
Here is the request that is sent to register the user:

![Manipulator Request](/assets/img/posts/bsidesisb2020/manipulator-register-request.png)

And here is the response to this request:

![Manipulator Rsponse](/assets/img/posts/bsidesisb2020/manipulator-register-response.png)

Looking at the request and response of this process, one thing that got my attention was `user_id` cookie in response. So what if we intercept the response and change the cookie value?

If we look at the link that we got from the already registered user, we can see a `referrerId` parameter in that link. So we have the id of our friend. Replace the id of newly registered user in response, with our friend's id.

![Manipulator Rsponse Changed](/assets/img/posts/bsidesisb2020/manipulator-register-response-changed.png)

And we successfully got access tot our friend's account:

![Manipulator Flag Found](/assets/img/posts/bsidesisb2020/manipulator-flag-found.png)

## Vulnerability?

I think the main issue here is that the authectication and authorization is solely handled by a single cookie value which is just the user-id. User-ids can be found by differnet ways so once a user-id is disclosed, that user is done!

---

# ShaktimanDaDev

![ShaktimanDaDev](/assets/img/posts/bsidesisb2020/ShaktimanDaDev.png)

There were two pages accessible of this web app. Home page containing the basic introduction of who shaktiman is and about page containing details about shaktiman.
While reading the about page, following paragraph caught my attention:

![ShaktimanDaDev about](/assets/img/posts/bsidesisb2020/ShaktimanDaDev-about.png)

So we have to read the source code of this web application right? Well there was some _`protection`_ to access the source code. Like we can't use usual ways like `Ctrl+u`, `Ctrl+j`, `Ctrl+Shift+i` etc.

So how can we access the source code? You know how, right? . . . . . No? Well google is your friend (is it?). A simple google search tells you to open developer tools from browser drop-down menu rather than `Ctrl+Shift+i` shortcut.

**There was a `security.js` file which tells us how to disable the protection, simply use `ctrl+c, ctrl+u`**

In the source code found another js file `home-css-js-js-change-nhp.js`. This was the file that Shaktiman referred to in his about page.

![ShaktimanDaDev JS File](/assets/img/posts/bsidesisb2020/ShaktimanDaDev-Js-content.png)

The content of file is not properly readable, so used an online [Js beautifier](https://beautifier.io/):

![ShaktimanDaDev JS File](/assets/img/posts/bsidesisb2020/ShaktimanDaDev-Js-content-deobfuscated.png)

From this file we have a url (which was the Shaktiman's secret portal), his username and password. Accessed that url and entered the found username & password:

![ShaktimanDaDev logedIn](/assets/img/posts/bsidesisb2020/ShaktimanDaDev-logedin.png)

Here we got an encoded string. At this stage I simply used `magic` tool of [CyberChef](https://gchq.github.io/CyberChef/), and got the flag:

![ShaktimanDaDev Flag found](/assets/img/posts/bsidesisb2020/ShaktimanDaDev-flag-found.png)

## Vulnerability?

Well the issue here is what we call `Security through Obscurity`. Simply disabling some shortcut keys to access source code, saving sensitive data (username,password and that url) by using so called `encryption` on some Js files is a secure practise.

---

# BrokenAuth

![Broken Auth](/assets/img/posts/bsidesisb2020/brokenAuth.png)

So we are moderator of this application and we have to delete a user to solve this challenge, but we are only allowed to view & edit a user's information, well atleast this is what the description and main page depicts:

![Broken Auth](/assets/img/posts/bsidesisb2020/brokenAuth-home.png)

After poking a little with the application I started to read the source code and found a JS file `BSides_Javascript.js`.
Upon reading the JS code, found the following function:

![Broken Auth JS](/assets/img/posts/bsidesisb2020/brokenAuth-js-content.png)

This function tells us that to delete a user we have to send a request to `deleting_a_user.php` with username as parameter value.

```
https://target-url/deleting_a_user.php?username=[USERNAME-HERE]
```

Username of a user can be found by viewing that user's details. Sending the above request solved the challenge.

![Broken Auth Flag Found](/assets/img/posts/bsidesisb2020/brokenAuth-flag-found.png)

## Vulnerability?

`Lack of proper access control`. There should be porper access cotrol implmented on endpoints that are not meant for a user (in this case the moderator).

---

# Remani

![Remani](/assets/img/posts/bsidesisb2020/remani.png)

Here we are welcomed with a form where we have to enter our debit card details to process the payment. But after I entered the details and pressed the Subscription button I recieved an error.

![Remani home](/assets/img/posts/bsidesisb2020/remani-home.png)

After exmanining the request within burp suite, came to know that the response is `403 Forbidden`.

![Remani Post](/assets/img/posts/bsidesisb2020/remani-post-request.png)

Unable to figure out the cause of this response, I started to look at the source code and found JS file `stripe.js`.
After reading the JS code in detail, following code caught my attention as it was responsible for making that payment request:

![Remani JS](/assets/img/posts/bsidesisb2020/remani-stripe-js.png)

If we closely read this code, we can see a **POST** request is made to **payment.php** endpoint with the token in the body. With this request two methods are chained which are:

- done()
- fail()

Well not a JS guru here (and you don't need be to understand the code), we can see that whenever we send this **POST** request, the `fail()` method is being called.
So what causes this `fail()` method to get called instead of `done()`? Well the first thing that came to my mind was that maybe the reponse of this request?

So I sent the **POST** request again but this time intercepted the response, changed the `403 Forbidden` to `200 Ok` and VOILA!

![Remani Post Changed](/assets/img/posts/bsidesisb2020/remani-post-request-changed.png)

![Remani flag-found](/assets/img/posts/bsidesisb2020/remani-flag-found.png)

## Vulnerability?

Well I think this is an example of vulnerable coding, where just changing the response caused the app to process the request differently.

---

# X_Pass

![X_Pass](/assets/img/posts/bsidesisb2020/x_pass.png)

Main page has an input box where we have to enter the right password to login. As suggested in the description of the challenge that there is a client side JS algorithm, I accessed the source code and started reading the code.
There found this interesting code:

![X_Pass JS](/assets/img/posts/bsidesisb2020/x_pass-js.png)

Here is a function `Validation()` which after doing multiple different operations, assigns a value to `alg_1` variable which is also logged in console.

Then there is an conditional statement where value of `alg_1` is comapred to `x` (this x is our supplied input)and if they are equal, then opens a new `url`.

So what I did here was that simply enter any value, hit enter, open the console and there was a value there (`alg_1`).

![X_Pass](/assets/img/posts/bsidesisb2020/x_pass-console.png)

Copy this value and paste it in the input box, hit enter and the challenge is solved!

![X_Pass](/assets/img/posts/bsidesisb2020/x_pass-flag-found.png)

## Vulnerability?

If we see, we were able to enter the correct password because that password was being logged in console. This is usually done by developers to debug the code but they may forget to remove these statments before making the application live.

---

# HackTheAdmin

![Hack The Admin](/assets/img/posts/bsidesisb2020/hacktheadmin.png)

Main page had simple login form to enter username and password.
![Hack The Admin Home](/assets/img/posts/bsidesisb2020/hacktheadmin-home.png)

After spending a few minutes and looking through the source code, did not find anything useful.Then I ran directory bruteforce through which I was able to access a few files:

![Hack The Admin Dir Brut](/assets/img/posts/bsidesisb2020/hacktheadmin-dir-search.png)

These files had the following content in them:

- **admin.txt:** `To use the admin panel, login. From there you can use the the panel to make requests to execute commands on our server using our secure api (found at yourdomain/api)!`

- **login.txt:** `The default user is admin. To login go to yoursite/index.html. If you are having problems, make sure your login.php file is correctly configured.`

So maybe there is something special in `login.php`? Well turns out there is nothing in reponse if we directly access the `login.php` file.

`info.php` had the following content:

![Hack The Admin info](/assets/img/posts/bsidesisb2020/hacktheadmin-info.png)

Accessing the **Login Help** we ended up on the following endpoint : `/helpdisplay.php?help=login.txt` containing the response of `login.txt` file.

Changed the `login.txt` with `login.php` and this time there was some content in reponse but not much meaningful. Upon this I accessed the same Url again `/helpdisplay.php?help=login.php` and analyzed the response in burp suite.

In the response we can see following php code:

![Hack The Admin disclose](/assets/img/posts/bsidesisb2020/hacktheadmin-php-disclose.png)

So following this code, we can see that we have another `.php` file but we can't access it directly.
Reading the code suggest that username is `admin` and there is cookie `user` whose value is base-64 encoded username `admin`.

So I sent a `GET` request to `/Cantbeaccessedwithoutlogin.php` with the cookie: `user= YWRtaW4=`

Here `YWRtaW4=` is base-64 encoding of the word `admin`

![Hack The Admin flag found](/assets/img/posts/bsidesisb2020/hacktheadmin-flag-found.png)

In response we receive a flag and the challenge is solved!

## Vulnerability?

`Source code disclosure` & `use of weak encoding`. Using base64 to encode some sensitive data and depending on that data for authentication or authorization puprose can lead to severe issues.

---

# Finishing note

Well thats all folks. I hope you guys enjoyed this post and learnt something new today. I would like to thank the team behind [BsidesIslamabad](https://www.bsidesislamabad.com/) for such an amazing event.

If anyone have any questions or you just wanna talk about cyber security in general, feel free to shoot a dm on twitter or linkedIn, as I am always open to talk, discuss and learn new things.

### Thank you!
