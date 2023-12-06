---
layout:	post
title:	"How to get started Hacking Wordpress Plugins"
date:	2021-09-28
medium_url: https://noob3xploiter.medium.com/how-to-get-started-hacking-wordpress-plugins-to-earn-your-first-cve-b31ea5e834c0
categories: [Hacking, Code Review]
tags: [Hacking, Wordpress, Code Review, Web]
---

  Hi. In this writeup, i will teach you everything that i learnt and a methodology on how to get started hacking wordpress plugins. Keep in mind, i wont be teaching different vulnerability types, i will just be teaching how to look for vulnerabilities. I learnt it myself and its fun. Its a good way to get a cve and get started on code review too. Lets get started

### Following User Inputs

A good way to start hacking a wordpress plugin is by tracing user inputs. There are multiple way to get user inputs in PHP

`$_POST` is used for getting post parameters

`$_GET` is used for getting get parameters

`$_REQUEST` is used for getting either post or get parameters

`$_SERVER` is used to obtain the value of various http request headers

`$_COOKIE` is used for getting the value of cookies and

`$_FILES` is for files

These are the following ways the developers can obtain user inputs. So when hacking wordpress plugin, or just hacking any php code in general, its good to start looking at these, and follow the user inputs from there. We can do it by grepping.

#### **EXAMPLE**

As an example, we will be hacking and reproducing a local file inclusion in Mail Masta 1.0. This is the original report <https://www.exploit-db.com/exploits/50226>

So we start up by grepping for $\_GET to find out all the php code that accepts user input. While doing it, one php file seems interesting. inc/campaign/count\_of\_send.php

![](/img/1*KpphIU3_5r19Dz2fSX73qw.png)So i opened it up using my code editor

![](/img/1*YWnQxs6M38Mf7CG0tL2eqw.png)The code is fairly simple, in the line 4, it get the value of the get parameter ‘pl’, then pass it to the function include. Include may lead to lfi so we have an lfi here. Now, lets try to reproduce it.

![](/img/1*lPVS5AGM8DcqCEYsVzrTBg.png)You can see that it works. We have an lfi. Thats how you find vulnerabilities by following user inputs.

### HOOKS

In wordpress, we have something called hooks. Hooks are ways for developers to hook functions to pre-defined spots in wordpress. It can be done with `add_action()` function. For further reading, i recommend this <https://developer.wordpress.org/plugins/hooks/> and [https://developer.wordpress.org/reference/functions/add\_action/](https://developer.wordpress.org/reference/functions/add_action/)

Now while testing, there are certain hooks that i lookout for.

`wp_ajax_$action_name` used to hook to the ajax (admin-ajax.php)

`admin_post_$action_name` used to hook to the admin-post.php

`wp_ajax_nopriv_$action_name`, `admin_post_nopriv_$action_name` same as the above but doesnt require authentication

`admin_init` used to hook on every admin page load

`wp_loaded` used to hook when the plugin is installed

`profile_update` & `personal_options_update` called when a user edit his/her account.

Plugins loaded, `template_redirect`, init. These hooks are called on every page load.

There are alot more hooks there but these are what i usually find. Always refer to the documentation when you want to learn the use of a hook.

Now just like above, we have to find if there are functions hooked to these hooks. This can be done with grepping

#### **EXAMPLE**

For this example, we will be reproducing the sqli in Double Opt-In for Download 2.0.9. The original report can be found here. <https://www.exploit-db.com/exploits/39896>

So we start out by finding any actions hooked to `wp_ajax` using grep

![](/img/1*gBaiw-7H9I9YiqZzdEu7Hg.png)We found `wp_ajax_populate_download_edit_form` in `public/class-doifd.php`. Lets open the file and analyze the code.

![](/img/1*HQbAmOGz_JGU3gT7znVqTw.png)Here, we can see that it is hooked to the function populate\_download\_edit\_form. Lets analyze the function

![](/img/1*BFF04AAXh6aVj-0UGrfLgg.png)Here, we can see that it gets the value of the post parameter id and store it to the variable $value, then the $value variable is used in an sql query without filtering or preparing the statement making it vulnerable to sqli. Now lets reproduce it. The ajax hook is `wp_ajax_populate_download_edit_form` so the ajax action parameter is `populate_download_edit_form` and we will also provide the id parameter

![](/img/1*iH6ScKX9OjQUfF9PGzvUog.png)We will be using the **and 1=1** payload for a boolean based poc. Using the **and 1=1** throws a normal response

![](/img/1*e1JIUD3iBCXa6MBTW2EiKg.png)But using and 1=2 respond with a null meaning our sqli actually exist

![](/img/1*kOe91egnx6gNEsUo53_gyw.png)So thats how you check for hooks when hacking wordpress plugins. Remember to also check other hooks other than wp\_ajax when testing.

This is the end of the writeup. To learn the different vulnerability types, you have to learn it yourself. <https://www.wordfence.com/wp-content/uploads/2021/07/Common-WordPress-Vulnerabilities-and-Prevention-Through-Secure-Coding-Best-Practices.pdf> here’s a good article that will teach you some of the bug types in php and in wordpress. I have 3 possible upcoming cve soon with wordpress hacking alone and it taught me php code review thats why i really like wordpress hacking

Thanks for reading

Join the Bounty Hunter Discord server: <https://discord.gg/bugbounty>

  