---
title: First Article
date: 2021-01-12 00:00
categories: [TEST]
tags: [test]     # TAG names should always be lowercase
---

# Introduction

I'm used to Mkdocs, let's see what we can do here with Jekyll. I will use this page to store what I'm used to use when I write an article.

## Header layer

The TOC is really useful on the right side.

Let's see how to import an image.

![Desktop View](/assets/img/2021-01-12/01.png)

Let's now use a little bit of HTML to write in various color.

<span style="color:red">**Let's write something in red**</span>

<span style="color:green">Or in green</span>

# Let's add some code

```powershell

Write-host "This is a Write-host"
$var = Get-process
Write-output "Processes are: $($var)"

```

# Alerts

{% include note.html content="This is my note." %}

{% include tip.html content="This is my tip." %}

{% include warning.html content="This is my warning." %}

{% include important.html content="This is my important info." %}
