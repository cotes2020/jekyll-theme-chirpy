---
title: "[WRITEUP] Neonify - COOKIE ARENA"
categories:
- CTF Writeup
- Cookie Arena
tags:
- SSTI
- Ruby
date: '2023-11-15 16:00:00'
---

![Neonify](/posts/Neonify-CookieArena/Untitled.png)

Cơ bản, trang này có chức năng là nhận input từ người dùng và convert sang cái dòng chữ màu mè kia. 

![Neonify](/posts/Neonify-CookieArena/Untitled%201.png)

Khả năng param neon được xử lý và render ra màn hình.

Tuy nhiên, hầu hết các kí tự đặc biệt đều đã được filter và dừng xử lý.

![Neonify](/posts/Neonify-CookieArena/Untitled%202.png)

Vì admin không cung cấp source code nên cũng hơi bí bách. Ngồi vò đầu bứt tóc một lúc, vô tình trigger ra một exception và nhảy đến trang debug khi `neon=%ab`. (Nhắc lại đây chỉ là vô tình =)) )

![Neonify](/posts/Neonify-CookieArena/Untitled%203.png)

Lúc này đã biết đoạn code dòng 14 kia chỉ cho phép input nhập vào chỉ chứa chữ cái, chữ số và dấu khoảng trắng. 

![Neonify](/posts/Neonify-CookieArena/Untitled%204.png)

Vậy liệu chúng ta có thể bypass từ đây, hay đi tìm 1 hướng khác??

Sau 30p ngồi nghiên cứu cái docs của ruby, tui cũng lục ra được cái này:

Ở đây nè: [https://ruby-doc.org/core-2.4.1/Regexp.html#:~:text=^ - Matches beginning,end of line](https://ruby-doc.org/core-2.4.1/Regexp.html#:~:text=%5E%20%2D%20Matches%20beginning,end%20of%20line){:target="\_blank"}

![Neonify](/posts/Neonify-CookieArena/Untitled%205.png)

Tức là cái `^` nó match từ đầu dòng, và `$` match đến cuối dòng, hay nói cách khác, đoạn ReGex kia nó chỉ được áp dụng trong 1 dòng. Từ đó, ta có thể inject payload tùy ý trong dòng mới mà không phải qua cái filter kia.

![Neonify](/posts/Neonify-CookieArena/Untitled%206.png)

Yahh, bypass được ùi nhé, công việc còn lại là craft 1 payload hoàn chỉnh để rce.

Mệt quá, lười search google nên tui lôi thần chú ra đọc cho nhanh [úm ba la xì bùa](https://book.hacktricks.xyz/pentesting-web/ssti-server-side-template-injection#erb-ruby){:target="\_blank"}

![Neonify](/posts/Neonify-CookieArena/Untitled%207.png)

output trả về true tức là command đã thực thi thành công. Tiến hành đọc flag

![Neonify](/posts/Neonify-CookieArena/Untitled%208.png)

Done!
