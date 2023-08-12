---
title: Baby Logger Middleware Writeup - Cookie Arena
date: "2023-08-01 02:45:51"
categories:
  - CTF Writeup
  - Cookie Arena
tags:
  - middleware
  - SQL Injection
published: false
---

Link Challenge: [https://battle.cookiearena.org/challenges/web/baby-logger-middleware](https://battle.cookiearena.org/challenges/web/baby-logger-middleware){:target="\_blank"}.

![access log page](/posts/BabyLoggerMiddleware/access-log.PNG)

Có thể thấy đây là 1 trang log lại toàn bộ các request gửi đến. Dựa vào response ta có thể tiến hành khai thác HTTP header injection và SQL injection.

![error query](/posts/BabyLoggerMiddleware/error-sql.png)
_Có một số trường như User Agent, Referer, Cookie, URL đều có thể dễ dàng chỉnh sửa với Burp._

Phân tích syntax error 1 chút, ta hoàn toàn có thể inject 1 đoạn query hoàn chỉnh tùy ý.

![sql injection](/posts/BabyLoggerMiddleware/sql-injection-1.png)

Theo mô tả challenge, ta cần insert "**HACKER_WAS_HERE**" vào column "ip_address" là lấy được flag. Tuy nhiên, nhìn vào query có thể thấy "ip_address" nằm trước "user_agent", vậy làm sao để insert được đây? Tạo thêm 1 record khác nữa là được, ez game :v

![sql injection](/posts/BabyLoggerMiddleware/sql-injection-2.png)
_Có thể insert nhiều record cùng một lúc, mỗi record ngăn cách bởi 1 dấu phẩy_
