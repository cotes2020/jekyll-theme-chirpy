---
title: Pwned 1 Writeup - VulnHub
author: dyl4n
categories:
  - CTF Writeup
  - VulnHub
tags:
  - hacking
  - pwned
date: "2023-07-25 19:40:01"
---

Download [PWNED: 1](https://www.vulnhub.com/entry/pwned-1,507/)

## Scanning

Scan ip với `arp-scan` thu được:

![scan ip](/posts/pwned-1/scan-ip.png)

Tiến hành scan tất cả ports

```shell
nmap 10.0.2.8 -p- -sC -A
```

![scan ports](/posts/pwned-1/scan-ports.png)

Có thể thấy, 3 ports dịch vụ đang mở là 21, 22 và 80.

## Enumeration

Đi tới dịch vụ web, không tìm thấy thông tin gì từ đây

![web](/posts/pwned-1/web.PNG)

Tiến hành enumerate directories với `gobuster`.

```shell
gobuster dir -u http://10.0.2.8/ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
```

![enumerate directories](/posts/pwned-1/enum-dir.PNG)

Nhìn cái tên thôi cũng thấy `/hidden_text` có vẻ thú vị hơn mấy cái kia. Nhưng ở bất kì trường hợp nào, manh mối cũng đều có thể xuất hiện nên chúng ta không nên bỏ xót.

![secret dictionary](/posts/pwned-1/secret_dic_file.PNG)

Check `/hidden_text` thấy có chứa 1 file `secret.dic`. Nội dung file này là 1 list các sub-directory. Khả năng 1 hoặc 1 vài trong số này có thể truy cập được. Lưu dic này và bruteforce để tìm ra đường dẫn hợp lệ.

![pwned.vuln](/posts/pwned-1/pwned-vuln.PNG)

Trong số này thì duy nhất `pwned.vuln` có status 200.

## Exploitation

![pwned.vuln](/posts/pwned-1/check-pwned-vuln.PNG)

Response trả về trong source có xuất hiện đoạn comment code php do dev sơ suất để lại. Hơn nữa, đoạn comment này đã leak tài khoản của 1 dịch vụ nào đó, check username và password cộng với việc port 21 mở ta có thể suy ra đây là user của ftp service.

![login ftp](/posts/pwned-1/login-ftp.PNG)

Login thành công. Get 2 file trong folder share về và đọc chúng

![clues](/posts/pwned-1/clues.PNG)

1 file có chứa private key để login SSH nhưng chứa 1 số kí tự gây nhiễu, file còn lại tưởng vô nghĩa nhưng thực chất có chứa thông tin user. Có thể thu thập được username của SSH service từ đây đó là `cmc`. Sửa lại file private key và dùng dữ kiện tìm được để login SSH.

![login ssh](/posts/pwned-1/login-ssh.PNG)

![user flag](/posts/pwned-1/user-flag.png)

Pwned cmc và capture cmc's flag thành công!

## Privilege Escalation

Kiểm tra quyền của user `cmc`

![cmc privilege](/posts/pwned-1/cmc-privilege.png)

Có thể thấy, _cmc_ có quyền thực thi file script `/home/messenger.sh` với tư cách là user `hoangmongto`. Vậy điều cần làm tiếp theo là chiếm quyền của `hoangmongto` và còn 1 user flag nằm ở home directory của user này.

![messenger.sh](/posts/pwned-1/messenger-sh.PNG)

### hoangmongto

Phân tích một chút về file script này, có 1 biến `msg` được nhập từ input, sau đó được chạy như 1 câu lệnh. Ngoài ra, các lỗi sẽ được chuyển hướng sang _/dev/null_ => Từ đó, ta có thể chèn 1 shell nhờ `/bin/bash`.

```shell
sudo -u hoangmongto /home/messenger.sh
```

![user flag-2](/posts/pwned-1/user-flag-2.png)

Spawn TTY shell và lấy được user flag thứ 2!

### root

![id hoangmongto](/posts/pwned-1/id.png)

Như chúng ta có thể thấy, `hoangmongto` thuộc group `docker`. May mắn thay, ta có thể leo lên root nhờ đó, đọc thêm [tại đây](https://flast101.github.io/docker-privesc/).

```shell
docker run -v /:/mnt --rm -it alpine chroot /mnt sh
```

Payload tại [GTFOBins](https://gtfobins.github.io/gtfobins/docker/)

![root flag](/posts/pwned-1/root-flag.PNG)
