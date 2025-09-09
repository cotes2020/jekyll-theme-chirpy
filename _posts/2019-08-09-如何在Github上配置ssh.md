---
title: 如何在Github上配置ssh
description: >-
  Get started with Chirpy basics in this comprehensive overview.
  You will learn how to install, configure, and use your first Chirpy-based website, as well as deploy it to a web server.
author: author_hai
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started, 中文]
pin: true
media_subpath: '/posts/20180809'
lang: zh-TW
---

## 如何在Github上配置ssh

When creating your site repository, you have two options depending on your needs:

### Option 1. 在本地電腦上生成SSH金鑰 (Recommended)

```console
ssh-keygen -t rsa -b 4096 -C "你的郵箱"
```

一路回車，會生成兩個文件（默認位置在 ~/.ssh/）：

id_rsa（私鑰，請勿泄露）

id_rsa.pub（公鑰，需要添加到 GitHub）
