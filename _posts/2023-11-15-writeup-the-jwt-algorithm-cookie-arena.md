---
title: "[WRITEUP] The JWT Algorithm - COOKIE ARENA"
categories:
- CTF Writeup
- Cookie Arena
tags:
- jwt
date: '2023-11-15 15:20:00'
published: true
---


![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled.png)

1 trang đăng nhập, có lẽ ta cần 1 credential để login lấy token.

Truy cập **`/robots.txt`** phát hiện 1 đường dẫn **/secret** và chỉ được phép truy cập với Googlebot

![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled%201.png)

<aside>
💡 Search google “user-agent string for googlebot”

</aside>

![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled%202.png)

Vậy đã có thông tin đăng nhập, getgo!

![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled%203.png)

Chúng ta cần có tài khoản của admin mới xem được flag, và bài này sử dụng JWT Token để xác thực.

![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled%204.png)

Thử case đơn giản nhất là jwt không sử dụng thuật toán, tức là sửa value của “alg” về “none”. Trong thực tế, JWT sử dụng tham số “alg” với giá trị “none” trong trường hợp nội dung của JWT đã được bảo mật bằng phương pháp khác ngoài chữ ký hoặc mã hóa (chẳng hạn như chữ ký trên cấu trúc dữ liệu chứa JWT). Tuy nhiên, một số web không áp dụng bước xác thực thêm nên gây ra lỗi này.

Đọc thêm: [rfc7519](https://datatracker.ietf.org/doc/html/rfc7519#:~:text=6.%20%20Unsecured%20JWTs%0A%0A%20%20%20To,as%20its%20JWS%20Payload){:target="\_blank"}

![The JWT Algorithm - CookieArena](/posts/the-jwt-algorithm-CookieArena/Untitled%205.png)

Cho bạn nào chưa biết thì extension mình đang sử dụng để sửa jwt là `JWT Editor` trong BurpSuite.

Sửa request và gửi lại sẽ lấy được flag.