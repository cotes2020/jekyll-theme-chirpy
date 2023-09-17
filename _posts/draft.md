---
title: ExpressionEngine CMS - From Code Review to RCE
date: 2023-03-20 13:33:37 +/-TTTT
categories: [codereview,cve-2023-22953]
tags: [codereview, cve-2023-22953]     # TAG names should always be lowercase
---

## Introduction

In this article, we will dive into the details of a recent vulnerability we discovered in the popular content management system, ExpressionEngine. Specifically, we will focus on a PHP object injection vulnerability, which was identified through a manual source code review. Through this article, I aim to provide a comprehensive understanding of the PHP object injection, including the methods and techniques used to discover it, and the steps taken to find a custom gadget chain, in order to achieve an RCE.

## ExpressionEngine CMS

As described by ExpressionEngine vendor `ExpressionEngine is a flexible, feature-rich, free open-source content management platform that empowers hundreds of thousands of individuals and organizations around the world to easily manage their web site.`

<img width="700" alt="image" src="https://user-images.githubusercontent.com/4347574/226133044-20b16b1a-fb1c-4ca5-8a1b-49a141d1a30a.png">


## Manual code review

It is acknowledged that the methodology and approach for conducting manual source code reviews can vary among individuals, and there is currently no standardized method in place. Additionally, the effectiveness of the review may be dependent on the reviewer's familiarity with the programming language, frameworks, and technology used in the application under review.

## sources and sinks

The most common approach while reviewing a code is to look at the paths by starting from the source or the sinks.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/4347574/226132644-40eb2e6e-fe1a-4978-96d7-867732078851.png">


> Example of **sources** in PHP:
> - `$_GET[]`
> - `$_POST[]`
> - `$_COOKIE[]`

> Example of interesting **sinks** in PHP:
> - `eval()`
> - `call_user_func()`
> - `require()`
> - `readfile()`
> - `unserialize()`

## Looking at the source code 

After cloning the project and looking around for the interesting **sinks**, there was a snippet that caught my eyes to look deeper into which is located at the following path:  `system/ee/ExpressionEngine/Service/File/ViewType.php`{: .filepath}

![image](https://user-images.githubusercontent.com/4347574/226138101-3e7593cc-c55a-4cdf-94d5-1a662a4a828f.png)
tracing back the source which is used to be parsed in the `unserialize()` function, it was found to be coming from the cookie input and the cookie name has to be `exp_viewtype`

<img width="700" alt="image" src="https://user-images.githubusercontent.com/4347574/226133669-2f5d0c2b-7c8f-42cf-8731-e751e916c602.png">

## Runtime debugging 

Now it is the time to poke around and test our inputs in the runtime, my prefered way to do so is through remote debugging by dockerizing the applications, setting-up debugging modules and connecting to debugger on VSCode. 

### VSCode configuration

The settings for VSCode and docker to setup remote debugging is quite easy, you can add configuration in VSCode with the following: 

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Listen for XDebug on Docker",
            "type": "php",
            "request": "launch",
            "port": 9003,
            "pathMappings": {
                "/var/www/html/": "${workspaceFolder}/"
            }
        }
    ]
}
```

<img width="700" alt="image" src="https://user-images.githubusercontent.com/4347574/226133777-cf88cdcc-baee-4b37-9d0e-c317b27c87fe.png">

### Docker configuration 

There are many docker templates which you can use to dockerize php application with database, you can use: https://github.com/amalendukundu/myonlineedu-php-mysql-docker/tree/master. A little bit of modification to add xdebug while running the app: 

```dockerfile
FROM php:7.2-apache

RUN apt-get update && apt-get install -y
RUN pecl install xdebug-3.1.2
RUN docker-php-ext-install mysqli pdo_mysql
RUN export XDEBUG_SESSION=1
**ADD xdebug.ini /usr/local/etc/php/conf.d/xdebug.ini**

RUN mkdir /app \
 && mkdir /app/ExpressionEngine \
 && mkdir /app/ExpressionEngine/www

COPY www/ /app/ExpressionEngine/www/

RUN cp -r /app/ExpressionEngine/www/* /var/www/html/.
```

Adding xdebug.ini as specified above with the following configuration: 

```

zend_extension=xdebug
[xdebug]
xdebug.mode=develop,debug
#xdebug.discover_client_host=1
xdebug.client_port = 9003
xdebug.start_with_request=yes
xdebug.client_host=host.docker.internal
xdebug.log='/var/logs/xdebug/xdebug.log'
xdebug.connect_timeout_ms=2000

```

### Running the application and testing our theories

Now I deployed the application with remote debugging on, it should be connecting back to our host machine on VSCode. We can simply test it by setting a breakpoint at the sink we would like to hit which should be in the following line 

```php 

$viewtype_prefs = unserialize(ee()->input->cookie('viewtype'));

```

We can see that our breakpoint was hit by sending the following request: 

```

GET /admin.php?/cp/addons/settings/filepicker HTTP/1.1
Host: localhost:30001
User-Agent: [Redacted]
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Purpose: prefetch
Accept-Encoding: gzip, deflate
Accept-Language: en-GB,en-US;q=0.9,en;q=0.8
Cookie: exp_last_visit=1671913623; exp_last_activity=1671968630; **exp_viewtype=a%3A1%3A%7Bs%3A3%3A%22all%22%3Bs%3A4%3A%22list%22%3B%7D**; exp_tracker=%7B%220%22%3A%22index%22%2C%22token%22%3A%223f92644cb590bcfc68dab723c0211297faea6108e6c788b23da23ff45100808e41675301e75d960c6ac60df9021f5acc%22%7D; exp_csrf_token=fe3239026bae4959dfe79f2701e87d181d9d4a07; exp_sessionid=2684f59ade507f3ee6f45f6eef4eeff6edd686d8
Connection: close
```

<img width="700" alt="image" src="https://user-images.githubusercontent.com/4347574/226135806-63f93804-de65-4995-895f-c7d161ed74ce.png">

### From PHP Object injection to authenticated RCE: 

