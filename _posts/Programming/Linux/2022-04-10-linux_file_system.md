---
title : Linux File System
date: 2022-04-10 14:53 +0900
categories: [Programming, Linux]
tags: [linux file system, 리눅스 파일 시스템, 파일별 기능, 디렉토리별 기능]
---

## 로그 및 데이터 파일
<hr style="border-top: 1px solid;"><br>

운영 체제 및 서비스의 로그와 각종 문서가 저장되는 위치

+ ```/var/www``` : 웹 문서 및 기타 웹 서버에서 사용되는 파일을 저장.

+ ```/var/lib``` : 시스템의 각종 서비스에서 자료를 저장할 때 사용, 데이터베이스 등이 이에 해당.

+ ```/var/lib/mysql``` : MySQL 데이터베이스의 데이터가 저장됨.

+ ```/var/log``` : 시스템 서비스 등의 로그를 저장할 때 사용, 웹 서버 로그가 보통 여기에 위치.

+ ```/var/cache``` : 캐시 데이터 저장, 삭제되어도 재생성 가능.

+ ```/media``` : 제거 가능한 장치를 마운트할 때 주로 사용하는 디렉토리

+ ```/mnt``` : 기타 파일 시스템을 임시로 마운트할 때 사용하는 디렉토리

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 설정 파일
<hr style="border-top: 1px solid;"><br>

운영 체제 및 서비스를 구성하는 설정 파일

+ ```/etc``` : 운영체제 초기 부팅 시 필요한 최소한의 명령어를 구현하는 프로그램 파일을 저장

+ ```/etc/apache2, /etc/httpd``` : Apache 웹 서버의 설정 정보를 저장

+ ```/etc/nginx, /etc/mysql``` : nginx 웹 서버, mysql 데이터베이스 서버 설정 정보 저장

+ ```/opt/etc``` : 추가적으로 설치된 프로그램의 설정 정보를 저장

+ ```/etc/mysql/my.cnf``` : mysql 데이터베이스 서버의 주 설정파일

+ ```/etc/hostname``` : 현재 시스템의 호스트네임 저장

+ ```/etc/hosts``` : 호스트네임의 실주소를 탐색할 때 사용되는 정적 순람표

+ ```/etc/fstab``` : 현재 시스템에 등록할 파일시스템의 목록 저장

<br><br>

일반 권한으로 접근 가능한 설정 파일

+ ```.htaccess``` : apache 웹 서버에서 웹 문서 디렉터리 내 서버 설정 제어

+ ```~/.bashrc, ~/.profile``` : 사용자 로그온 시 실행되는 셸 명령을 지정

+ ```~/.ssh/authorized_keys``` 
  + 해당 사용자에 로그인할 수 있는 SSH 공개키를 지정
  + 공격자의 SSH 공개키를 추가하면 공격자가 시스템에 로그인할 수 있게됨

+ ```~/.ssh/config```
  + SSH 클라이언트 설정 파일. 
  + 접속할 Host 등을 지정하여 악의적인 Host로 리다이렉트 할 수 있음. 

<br>

**```~/.profile```과 ```~/.ssh/authorized_keys```는 관리자가 로그인 하는 계정과 웹 서버의 계정이 같을 때 사용 가능하다.**

<br><br>

루트 권한으로 공격 시 이용 가능한 파일

+ ```/boot/initramfs-X.Y.Z.img```
  + 부팅 시 초기에 사용되는 파일을 저장한 이미지 파일
  + 부팅 중 시스템 파티션으로 전환되면서 삭제되기 때문에 악성 코드 등이 삽입되면 관리자가 탐지하지 못하는 경우도 있음

+ ```/etc/rc.local``` : 시스템 부팅 시 실행되는 명령을 지정

+ ```/etc/crontab``` : 시스템 부팅 후 주기적으로 실행되는 명령을 지정

+ ```/etc/profile``` : 사용자가 로그인 할 때마다 실행되는 명령을 지정

+ ```/etc/profile.d``` : 사용자가 로그인 할 때마다 실행되는 명령을 지정하는 스크립트를 저장하는 디렉터리

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 장치 및 가상 파일
<hr style="border-top: 1px solid;"><br>

운영 체제를 구성하기 위한 파일, 각각의 파일은 커널 기능과 밀접한 관련이 있음.

+ ```/dev``` : 각종 디스크 및 장치 파일을 제공

+ ```/sys``` : 하드웨어 및 플랫폼에 접근할 수 있도록 함

+ ```/proc``` : 프로세스 및 시스템 정보를 제공하는 가상 파일시스템이 위치

+ ```/proc/sys``` : 운영체제의 동작을 제어할 수 있는 각종 파라미터가 위치

+ ```/proc/self/net, /proc/net``` : 운영체제 네트워크 계층의 다양한 정보 저장

<br>

/proc 파일 시스템 구성 요소
: <a href="https://sonseungha.tistory.com/412" target="_blank">sonseungha.tistory.com/412</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 프로그램 및 라이브러리
<hr style="border-top: 1px solid;"><br>

프로그램의 명령어가 저장된 위치와 이들을 실행하기 위한 라이브러리의 위치

+ ```/bin, /sbin``` : 운영체제 초기 부팅 시 필요한 최소한의 명령어를 구현하는 프로그램 파일을 저장

+ ```/boot``` : 커널이나 부트로더 옵션 등 부팅에 필요한 파일을 저장

+ ```/lib, /lib64, /libx32``` : 운영체제 초기 부팅 시 필요한 최소한의 라이브러리 파일을 저장

+ ```/opt``` : 추가적인 프로그램 저장

+ ```/usr/bin``` : 각종 명령어 및 프로그램 파일을 저장

+ ```/usr/sbin``` : 시스템 관리자가 주로 사용하는 각종 명령어 및 프로그램 파일을 저장

+ ```/usr/lib, /usr/lib64, /usr/libx32``` : 시스템에서 공유되는 라이브러리 파일 저장

+ ```/usr/share``` : 기타 시스템에서 공유되는 라이브러리 파일 저장

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 임시 파일
<hr style="border-top: 1px solid;"><br>

운영 체제 및 서비스에서 사용하는 임시 디렉터리 또는 파일을 의미

+ ```/tmp```
  + 임시 파일을 저장, 시스템 재시작 시 저장한 파일이 삭제될 수 있으며, 용량에 상당한 제한이 있을 수 있음. 
  + 디스크 또는 메모리 상(tmpfs)에 존재할 수 있음. 

+ ```/var/tmp``` : 임시 파일을 저장, 시스템 재시작 시에도 일반적으로 유지됨.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

출처 
: <a href="https://dreamhack.io/lecture/courses/282" target="_blank">dreamhack.io/lecture/courses/282</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
