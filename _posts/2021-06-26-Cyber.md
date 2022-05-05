---
title: Cybertalents Wrong Token Web Challenge Writeup
author: Muhammad Adel
date: 2021-06-26 18:10:00 +0200
categories: [Cybertalents Writeups]
tags: [cybertalents, ctf, web]
---

## **Description**

Request to the flag is forbidden due to wrong CSRF token ... can you fix it and reveal the flag

**Difficulty**: hard

**Challenge Link:** [https://cybertalents.com/challenges/web/wrong-token](https://cybertalents.com/challenges/web/wrong-token)

## **Solution**

### **Exploration**

Opening the website we will find the following web page:

![](https://gblobscdn.gitbook.com/assets%2F-Mc-dhcC8XUrwR1pTDRF%2F-McZLF75KG134MmzWZ21%2F-McZNtqlxvbQDVBMqERV%2F1.png?alt=media&token=7568cdc4-bf54-4891-a2b5-c0016013c8c6)

Reading the source code we will find that we need to make a JSON request containing some data:

![](https://gblobscdn.gitbook.com/assets%2F-Mc-dhcC8XUrwR1pTDRF%2F-McZLF75KG134MmzWZ21%2F-McZOcK5PmlGfS8LwaGR%2F2.png?alt=media&token=87ec2fce-ceff-4517-b5d7-8a06334c3b01)


the request should look like the following:


```json
{"action":  "view_flag",  "_token":  "asdjhDJhfkjdI"}
```

if you send this request you will receive this error:

> Failed Comparison ( incoming CSRF token != Session CSRF token )

### **Exploitation**

It seems that there is some sort of verification on this CSRF token, so we need to think of a way to bypass it.

First, I tried to remove the whole parameter but it didn't work. I tried to remove the value only but still nothing.

Finally, I changed the data type from string to a boolean value equals **True** and it gives me the flag.

![](https://gblobscdn.gitbook.com/assets%2F-Mc-dhcC8XUrwR1pTDRF%2F-McZLF75KG134MmzWZ21%2F-McZQ9kBdsbw7mhEsNHW%2F4.jpg?alt=media&token=2bd1eeaf-08b8-454b-9fdd-6a69b18b9bad)

