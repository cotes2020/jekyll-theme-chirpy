---
layout: post
title: "[FIA] Bài Test member vào CLB FIA"
summary: "Test skill"
author: technical
date: '2023-01-12 9:00:00'
category: News
thumbnail: assets/img/logo/logo_FIA.png
keywords: Hacking, CTF
permalink: /blog/Test-skill/
usemathjax: true
---


<table align=center style="margin-left: auto; margin-right: auto;">
<h2 align=center>MỤC LỤC CHALLANGE</h2>
<tr>
<td markdown="1">

|STT | Link |
| :--------- | :-- |
| [Flag1] Sanity check | [Link](#flag1-sanity-check) | 
| [Flag2] Hidden | [Link](#flag2-hidden) | 
| [Flag3] Or does anybody | [Link](#flag3-or-does-anybody) | 
| [Flag4] Save Bory's times | [Link](#flag4-save-borys-times) | 
| [Flag5] Owner permission | [Link](#flag5-owner-permission)|

</td>
<td markdown="1">

|STT | Link |
| :--------- | :-- | 
| [Flag6] Log files | [Link](#flag6-log-files) | 
| [Flag7] Hide and seek | [Link](#flag7-hide-and-seek) | 
| [Flag8] Malmand | [Link](#flag8-malmand) | 
| [Flag9] EXEC_ME | [Link](#flag9-exec_me) | 
| [Flag10] Privilege Escalation| [Link](#flag10-privilege-escalation) |

</td>
</tr>
</table>








------------------------
## [Flag1] Sanity check
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211604649-2e0c1815-e50a-4495-963b-fa5b8d8f16ce.png">
</p>
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211604198-cfc52f9b-1462-4086-84f1-560d1ee75015.png">
</p>

Đề bài cho chúng ta `username` và `password` để đăng nhập vào trong server của bài thi. Đối với bài thi này, mình sẽ sử dụng `window terminal` để làm bài thay cho `kali` Cách cài `window terminal` mọi người có thể xem [tại đây](https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh)

Sau khi cài xong, sử dụng cú pháp của ssh: `ssh username@ipAddress` **(nhớ bật openVPN trước nha)** để vào trong server 

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211606742-4e9652f9-ad64-4302-843b-b961b16ab570.png">
</p>

Tới đây thì nhập password `fia` vào là được.

Sau khi vào được rồi thì dùng lệnh `ls` để xem trong đây có những file gì

```
fia@fia:~$ ls
Flag1.txt  Flag3  Flag4  Flag5  Flag9
```
Thấy Flag1 rồi đó =)) rồi giờ dùng lệnh `cat` để xem trong file có gì thôi 

```
fia@fia:~$ cat Flag1.txt
FIA{Welcome_to_the entrance_test}
```
#### Flag: FIA{Welcome_to_the entrance_test}
-----------------------------
## [Flag2] Hidden
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211607932-c78cb324-1383-4435-a969-d153f1dfd253.png">
</p>

Với bài này thì mình chỉ cần dùng lệnh `ls -la` để hiện ra tất cả các file đã bị ẩn thôi là được.

```
fia@fia:~$ ls -la
total 76
drwxr-xr-x 9 fia  fia   4096 Dec 29 03:04 .
drwxr-xr-x 3 root root  4096 Dec 27 12:32 ..
-rw------- 1 fia  fia    302 Jan 10 15:51 .bash_history
-rw-r--r-- 1 fia  fia    220 Feb 25  2020 .bash_logout
-rw-r--r-- 1 fia  fia   3771 Feb 25  2020 .bashrc
drwx------ 2 fia  fia   4096 Dec 26 04:46 .cache
-rw-rw-r-- 1 fia  fia     33 Dec 26 05:04 Flag1.txt
drwxrwxr-x 2 fia  fia   4096 Dec 26 08:01 Flag3
drwxrwxr-x 3 fia  fia   4096 Dec 26 08:26 Flag4
---------- 1 fia  fia  15952 Dec 26 08:50 Flag5
drwxrwxr-x 2 fia  fia   4096 Dec 27 09:35 Flag9
drwxrwxr-x 2 fia  fia   4096 Dec 26 05:12 .hidden
drwxrwxr-x 3 fia  fia   4096 Dec 26 05:11 .local
-rw-r--r-- 1 fia  fia    807 Feb 25  2020 .profile
-rw-rw-r-- 1 fia  fia     66 Dec 26 14:12 .selected_editor
drwx------ 2 fia  fia   4096 Dec 26 04:46 .ssh
-rw-r--r-- 1 fia  fia      0 Dec 26 05:01 .sudo_as_admin_successful
```

Do file có dấu chấm đằng trước tên file nên không thể sử dụng lệnh `ls` bình thường mà xem được. Có một file được gọi là `.hidden`. Thử dùng lệnh `cd` vào bên trong thư mục và dùng lệnh `ls -la` để xem có file nào không 

```
fia@fia:~$ cd .hidden/
fia@fia:~/.hidden$ ls -la
total 12
drwxrwxr-x 2 fia fia 4096 Dec 26 05:12 .
drwxr-xr-x 9 fia fia 4096 Dec 29 03:04 ..
-rw-rw-r-- 1 fia fia   34 Dec 26 05:12 .Flag2.txt
```
Ở đây có một file `.Flag2.txt`. Ta chỉ cần cat nó ra là được thôi. 
```
fia@fia:~/.hidden$ cat .Flag2.txt
FIA{y0u_f0und_a_v3ry_h1dd3n_Fl4g}
```
#### Flag: FIA{y0u_f0und_a_v3ry_h1dd3n_Fl4g}
----------------------
## [Flag3] Or does anybody
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211609858-7b6b1957-7df9-410f-80d1-15ba59be247b.png">
</p>

Vào thư mục Flag3, chúng ta có 2 file trong đó, một là `Flag3.txt` và `NOTME.txt`. Theo bản năng `cat` thằng `Flag3.txt` xem sao
```
fia@fia:~/Flag3$ cat Flag3.txt
Or is it just me ?
aHR0cHM6Ly93d3cueW91dHViZS5jb20vd2F0Y2g/dj1jdXhOdU1EZXQwTQ==
```
Nhìn nó là một đoạn string được mã hóa `base64`. Đi giải mã đoạn string đó ra
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211610927-47b06274-b671-4d87-bb0e-48e06ff6243a.png">
</p>

Đó là link [Youtube](https://www.youtube.com/watch?v=cuxNuMDet0M)

***Tác giả tâm sự là rất buồn khi không ai nghe bài hát đó cả :(***

Quay trở lại và xem thử file NOTME.txt với câu lệnh `cat`
```
fia@fia:~/Flag3$ cat NOTME.txt
BZh91AY&SY���
SA1�4���d`G��l��y& 24���c�@�� CI�2N� ���" -5�4��`�ItV+n�ǔ@�&;eE2�C[1y9Ws��a��2�C�TQ�Z�ƣ��2�C��5�l�J0l��S���x+fc"�`7rE8P����
```
Đât là những gì chúng ta thấy được ở bên trong file đó. Thấy hơi khả nghi nên mình đã dùng lệnh `file` để xem file nó là gì 
```
fia@fia:~/Flag3$ file NOTME.txt
NOTME.txt: bzip2 compressed data, block size = 900k
```
Á à hóa ra nó là file bzip. Rename nó lại đúng với file extension `.bz2` với câu lệnh `move` với cú pháp `mv tên_file_hiện_tại tên_file_mới`
```
fia@fia:~/Flag3$ mv NOTME.txt NOTEME.bz2
```
Sau đó ta tiếp tục decompress file với lệnh `bzip2 -d tên_file.bz2`
```
fia@fia:~/Flag3$ bzip2 -d NOTEME.bz2
```
Decompress xong lại lòi ra thêm một file `NOTME`. Sử dụng lệnh `file` lần nữa xem lần này là cái gì. 

```
fia@fia:~/Flag3$ file NOTEME
NOTEME: POSIX tar archive (GNU)
```
Đây là file `POSIX tar archive` hay còn được gọi là `tar.gz`. Tiếp tục dùng `mv` để rename lại tên file
```
fia@fia:~/Flag3$ mv NOTEME NOTEME.tar.gz
```
Sử dụng command `tar xvf tên_file` để có thể giải nén 
```
fia@fia:~/Flag3$ tar xvf NOTEME.tar.gz
Flag3.txt
```
Rồi ra flag rồi đó =)) Giờ mình đi `cat` thôi 
```
fia@fia:~/Flag3$ cat Flag3.txt
�w3�1.���F�0
```
Lại bịp nên tiếp tục thực hiện quy trình `file --> mv ` 
```
fia@fia:~/Flag3$ file Flag3.txt
Flag3.txt: gzip compressed data, was "Flag3.txt", last modified: Mon Dec 26 06:00:37 2022, from Unix, original size modulo 2^32 28

fia@fia:~/Flag3$ mv Flag3.txt Flag3.gz

fia@fia:~/Flag3$ gzip -d Flag3.gz
```
Sau đó dùng `cat` để ra flag thôi
```
fia@fia:~/Flag3$ cat Flag3
FIA{4lways_ch3ck_SuS_F1l3s}
```
#### Flag: FIA{4lways_ch3ck_SuS_F1l3s}

--------------------------------
## [Flag4] Save Bory's times
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211616496-ee6b4e7c-fee5-4df9-a8e6-caef34e689f3.png">
</p>
Đến với chall này, chúng ta có hai file `README` và `Source` thử dùng `cat` đọc file `README`

```
fia@fia:~/Flag4$ cat README

Bory very hate literature. He is planning to delete all the source file that he has use to review for the final
exam. But accidentally, he has dropped some of his Flag for the linux challenge some where in this source. He
need your help to find it back !!!
```
Ở đây có vẻ như là challange sử dụng lệnh `grep`. Thử vào trong file `source` xem có gì. 
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211617232-727fe119-4112-435b-a876-a52cd21a0c20.png">
</p>
Một đống file, để có thể tìm ra flag chúng ta dùng câu lệnh `grep -ri FIA{`
   <ul>
    <li>-r dùng để đọc tất cả các file</li>
    <li>-i đùng dể chấp nhận hoa thường đều được</li>
   </ul>
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211617932-85349e5a-5eb0-4cb7-8ee0-72294e958b78.png">
</p>

#### FIA{Gr3p_1s_th3_b3st_c0mmand_t0_f1nd_m1ssing_th1ngs}
---
## [Flag5] Owner permission
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211618709-089cf3b4-e23d-4fe6-ba09-50937701f3ad.png">
</p>
Đối với chal này, chúng ta thử sử câu lệnh `ls -la` tại `home` xem sao

```
fia@fia:~$ ls -la
total 76
drwxr-xr-x 9 fia  fia   4096 Dec 29 03:04 .
drwxr-xr-x 3 root root  4096 Dec 27 12:32 ..
-rw------- 1 fia  fia    302 Jan 10 15:51 .bash_history
-rw-r--r-- 1 fia  fia    220 Feb 25  2020 .bash_logout
-rw-r--r-- 1 fia  fia   3771 Feb 25  2020 .bashrc
drwx------ 2 fia  fia   4096 Dec 26 04:46 .cache
-rw-rw-r-- 1 fia  fia     33 Dec 26 05:04 Flag1.txt
drwxrwxr-x 2 fia  fia   4096 Jan 10 17:03 Flag3
drwxrwxr-x 3 fia  fia   4096 Dec 26 08:26 Flag4
---------- 1 fia  fia  15952 Dec 26 08:50 Flag5
drwxrwxr-x 2 fia  fia   4096 Dec 27 09:35 Flag9
drwxrwxr-x 2 fia  fia   4096 Dec 26 05:12 .hidden
drwxrwxr-x 3 fia  fia   4096 Dec 26 05:11 .local
-rw-r--r-- 1 fia  fia    807 Feb 25  2020 .profile
-rw-rw-r-- 1 fia  fia     66 Dec 26 14:12 .selected_editor
drwx------ 2 fia  fia   4096 Dec 26 04:46 .ssh
-rw-r--r-- 1 fia  fia      0 Dec 26 05:01 .sudo_as_admin_successful
```
Để ý Flag5 không có một quyền truy cập gì cả, user hay group đều không có, đến đây chúng ta sẽ bắt đầu sử dụng lệnh `chmod` để cấp quyền cho thư mục với câu lệnh căn bản: `chmod (quyền truy cập) tên_file`

Tại đây mình cấp cho quyền truy cập 777 cho lẹ :(
```
fia@fia:~$ chmod 777 Flag5
```
Thử dùng lệnh `file` để xem thử đó là gì thì phát hiện ra đó là file `elf`

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211619959-cdc68c4f-34fa-4d3f-be3d-942821d4cee5.png">
</p>
Thử chạy file thì chạy không được 

```
fia@fia:~$ ./Flag5

./Flag5: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ./Flag5)
```

Sử dụng câu lệnh `cat` chắc chắn không được, mình dùng `nano tên_file` một công cụ của linux để có thể viết và xem các string bên trong file. Mở lên và search chữ `"FIA{"`

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211622776-e9e707ec-32a0-40b9-b820-450ccf7dbc31.png">
</p>

Và chúng ta đã có flag

#### FIA{f1l3_p3rm1SSion_1s_v3RY_1mP0rt4nt}
---
## [Flag6] Log files
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211623065-ad09d513-f44d-4404-95e3-636fcfc1397a.png">
</p>

Bài này hỏi về log file, thì chúng ta sẽ đến thư mục `log` dựa trên những gì mình biết, `log` file sẽ được lưu ở trong `/var/log/`. Các bạn có thể tìm hiểu [tại đây](https://blogd.net/linux/cac-file-log-quan-trong-tren-linux/)


*  Dùng lệnh `cd /var/log` để tới thư mục `log`.

*  Dùng lệnh `ls -a` để xem hết tất cả các thư mục 

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211624300-a7a26662-7ce4-40fc-8735-ef22f8659a67.png">
</p>

Tại đây có một file tên `flagsix`. Dùng `cat` để xem file và ta có flag

#### FIA{l0gs_file_on_th3_SYST3M_1s_h3r3}

----
## [Flag7] Hide and seek
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211624868-3bc40adf-7386-4953-af77-b30d2a27e77f.png">
</p>

Tại đây nếu dùng `ls -la` để hiển thị tất cả các thư mục mà bị ẩn, chúng ta sẽ không thấy thằng file `flag7` ở đâu. Vì thế, ở chal này, chúng ta phải bắt buộc dùng lệnh `find` để có thể tìm ra flag. Cấu trúc câu lệnh `find [nơi bắt đầu] -name [tên file]`

Tìm ở thư mục `/` là tốt nhất, tại vì nó là thư mục gốc, nơi bắt đầu của mọi thứ

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211625643-5453affb-2242-42b7-ba22-9c66e24d1133.png">
</p>

Đến đây, ta bị lỗi. Đây là những folder mà chúng ta không thể nào truy cập vào. Nên mới trả ra những kết quả như thế. Sử dụng thêm câu lệnh `2>/dev/null` để có thể hết lỗi. Bạn có thể xem thêm [tại đây](https://unix.stackexchange.com/questions/581240/what-is-the-use-of-script-2-dev-null-in-the-following-script#:~:text=Save%20this%20answer.-,Show%20activity%20on%20this%20post.,command%20to%20%2Fdev%2Fnull.)

Thử dùng `find` lại với cú pháp mới ta có: 
```
fia@fia:~$ find / -name flag7 2>/dev/null
/usr/bory/hahaha/you_can_not_find_me_here/oops/kkkk/flag7
```
Và giờ mình `cat` thử xem có ra flag hay không 
```
fia@fia:~$ cat /usr/bory/hahaha/you_can_not_find_me_here/oops/kkkk/flag7
FIA{UwU_y0u_f1nd_m3_c0ngr4tul4t10n}
```
#### FIA{UwU_y0u_f1nd_m3_c0ngr4tul4t10n}
---
##  [Flag8] Malmand
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211626576-1b342aa7-482a-48ef-83dc-88491f03744f.png">
</p>

Tại bài này, chúng ta có 2 cách để giải: 

**Cách 1:**
* Dựa vào đề, chúng ta có thể biết rằng, chúng ta cần phải truy cập vào bên trong `task manager` của máy. Sử dụng câu lệnh `ps aux | more` đùng để xem được các tiến trình đang chạy trên hệ thống 

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211627433-f32a7bf3-ccec-4111-bdff-eb553963e06f.png">
</p>

* Ở hàng số 4 từ dưới lên ta có thể thấy tác giả đã đưa flag cho chúng ta `/bin/sh -c echo "Flag8:FIA{SuSSy_pr0c3ss_1s_runn1ng_0n_th3_b4ckgr0und}" & sleep 1m >/dev/null 2>&1`

**Cách 2:**
Dựa vào những kiến thức mà chúng ta có về kiến thức `log file` ta có thể vào bên trong thư mục `/var/log` và sử dụng lệnh `cat` vào thư mục `syslog` hay còn gọi là `system log` nơi lưu `log` của **hệ thống**

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211628250-0cb6526a-e36a-4b9c-8347-b45d0dbdee51.png">
</p>


#### Flag: FIA{SuSSy_pr0c3ss_1s_runn1ng_0n_th3_b4ckgr0und}
----
##  [Flag9] EXEC_ME
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211628586-94278272-211a-41f4-bc5a-c626b5359d6c.png">
</p>

Vào trong file chúng ta có thể thấy đó chính là một file đuôi `sh` thử chạy file thì ta không thể chạy file 

```
fia@fia:~/Flag9$ ./exec_me.sh
-bash: ./exec_me.sh: Permission denied
```

Thử `cat` vào file xem sao
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211629158-009b4c29-edbf-4bcb-820b-f8ad9a27bb18.png">
</p>

Đây là một chương trình =)) có các lớp bảo mật và bạn phải trả lời hết tất cả câu hỏi, nếu trả lời sai thì trả lại "You lose". Nhìn thấy đoạn string ở những dòng đầu tiên làm mình cảm thấy nó giống flag =)) thử `decrypt base64 xem sao`

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211629676-b2b54275-06a6-41a5-bda8-2bca2e3098a0.png">
</p>

Nó là Rickroll =)) không hiểu sao biết nhưng vẫn bấm vào ạ =))

Để ý đoạn code, thấy có những đoạn `authentication` thấy vậy nên mình thử ghép đoạn đó lại thành một, xong thử `decrypt base64`

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211630125-bd18ad4c-e954-4e97-bf60-ad84b46ffa08.png">
</p>

Ghép thêm đầu flag `FIA{` và dấu `}` ta có flag

#### FIA{B4sh_scr1pt_1s_v3ry_1mp0rt4nt}

---
##  [Flag10] Privilege Escalation
<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211630519-71fbb5ab-d01f-4d15-93b2-9c28f66b2c92.png">
</p>

Chal này có thế nói là tốn thời gian rất nhiều, bài này là leo thang đặc quyền, nghĩa là bạn sẽ phải tìm cách truy cập vào những thứ mà mình không thể với tới được. 

Ban đầu mình đã nghĩ tới chuyện là thông qua `Private Key`, nhưng mà kết quả là không thành. Chuyển sang hướng khác, mình bắt đầu đào sâu vào leo thang đặc quyền (Privilege Escalation). Đọc nhiều nguồn tham khảo khác nhau, mình mới biết rằng, để leo thang đặc quyền, chúng ta phải biết **những quyền mà chúng ta có thể sử dụng**

Bắt đầu từ đó, sử dụng câu lệnh `find / -perm -u=s -type f 2>/dev/null`

* /: tìm kiếm từ thư mục gốc
* -perm: các quyền được sử dụng của user nào đó
* -u=s: tìm của root (có thể đổi thành một user nào cũng được)
* -type f: loại file tìm kiếm là *regular file* 

Chúng ta có được những thứ mà chúng ta có thể truy cập vào
```
/usr/bin/chsh
/usr/bin/mount
/usr/bin/newgrp
/usr/bin/at
/usr/bin/umount
/usr/bin/nmap
/usr/bin/chfn
/usr/bin/su
/usr/bin/gpasswd
/usr/bin/sudo
/usr/bin/fusermount
/usr/bin/passwd
/usr/bin/pkexec
```

Để ý thư mục `/usr/bin/nmap` chúng ta có thể leo thang đặc quyền nhờ vào thằng nmap.
Mình có làm một số research thấy có những trang bảo dùng `nmap --interactive`. Nhưng mình cũng đã thử. 
```
fia@fia:~$ nmap --interactive
nmap: unrecognized option '--interactive'
See the output of nmap -h for a summary of options.
```
Thì lỗi này theo như mình tìm hiểu thì những bản nmap ver 5.0 thì mới sử dụng được.
Sau đó mình cũng đã thử tìm thêm những loại leo thang đặc quyền khác như `Path variable` và `restricted shell` =)) nhưng mà nó cũng chả có gì hết :(( Sau đó thì mình nhận được hint: 

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211640533-94147405-131e-4de2-a664-eb5132cc46e2.png">
</p>

Đọc hint xong là biết hướng ban đầu đi là đúng rồi =)) Research một chút xíu, thì mình tìm ra một cái github của một anh người nước ngoài, chỉ về cách leo thang đặc quyền dựa trên những gì mà mình đang có. Link [tại đây](https://gtfobins.github.io/)

Đi vào mục nmap, ở đây có chỉ cách để có thể bypass được quyền root (Nó cũng làm mình mất kha khá thời gian để ngẫm :)) ) Mình có chú ý tới mục `file read`

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211640951-651fd24f-f35d-4461-9390-a3327fff8382.png">
</p>
Thực hiện theo những gì ghi bên trên thì nó lại xuất hiện flag

<p align="center">
  <img class="article-img" src="https://user-images.githubusercontent.com/100250271/211642203-3203abdf-f60b-43c4-8b0f-572042858670.png">
</p>

Ban đầu trước khi ra flag, mình đã read file nhưng ghi sai tên flag10 thành Flag10 nên kết quả trả ra là không có gì =)) nên rút kinh nghiệm mốt nhìn kĩ flag tí xíu @@

#### Flag: FIA{g00d_j0B_you_4r3_4_g00D_H4k3rss!!!}

### Author: P5yDuck