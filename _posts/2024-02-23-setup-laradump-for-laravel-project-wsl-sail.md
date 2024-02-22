---
published: true
date: 2023-06-03
title: Setup Laradump for Laravel project (WSL + Sail)
---
1\. Follow the docs of Laradumps: [https://laradumps.dev/get-started/installation.html](https://laradumps.dev/get-started/installation.html)

2\. In WSL, get the desktop IP:

    cat /etc/resolv.conf
    

3\. In `.env`:

    DS_APP_HOST=172.18.240.1
    DS_APP_PORT=9191
    

4\. Start debugging 😀