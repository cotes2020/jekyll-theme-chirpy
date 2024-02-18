---
title : Linux 명령어 2
categories: [Programming, Linux]
tags: [Linux Commands]
---

## file [option] [file name]
<hr style="border-top: 1px solid;"><br>

+ 파일의 종류와 파일 정보 출력

<br>

+ option

  + -b : 지정한 파일명은 출력하지 않고 파일의 유형만 출력

  + -z : 압축된 파일 내용을 출력 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## strings [option] [file name]
<hr style="border-top: 1px solid;"><br>

+ 오브젝트 또는 이진 파일에서 인쇄 가능한 문자열 출력(ASCII 문자를 출력)

<br>

+ option

  + -a : 전체 파일 검색

  + -f : 각 문자열 이전에 파일명 출력

  + ```-min-len (-n min-len)``` 
    + 최소 문자열 길이 지정, 기본 값은 4 , 즉 4줄 이상의 문자열부터 출력하는거임.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## xxd [option] [file name]
<hr style="border-top: 1px solid;"><br>

+ 리눅스 shell 상에서 binary파일의 hexdump를 보여주는 명령어

<br>

+ option

  + -b : dump가 2진수로 출력

  + -c [개수] : 행 당 출력되는 열 개수 설정

    + ex) 4byte 씩 100의 길이만큼 출력 ==> ```xxd –g 4 –l 100 [filename]```

  + -g [개수] : 출력시 group으로 묶이는 byte의 개수 설정

    + ex) 8열로 마지막 80byte만 출력 ==> ```xxd –c 8 –s –80 [filename]```

  + -l [길이] : 설정된 길이 byte만큼 출력

  + -p : 주소나 ASCII 없이 hexdump 내용만 출력

  + -u : hex를 소문자 대신 대문자로 출력

  + -s [+][-] 위치 : 설정된 위치에서부터 hexdump함. 
  
    + ```+```위치는 파일의 시작에서부터, -위치는 파일의 끝에서 부터임. 

  + -i : C언어에서 사용할 수 있는 형식으로 출력

  + -r : 반대로 hexdump를 binary파일로 바꾸어 출력

<br><br>
<hr style="border: 2px solid;">
<br><br>

## base64 [option] [file name] 
<hr style="border-top: 1px solid;"><br>

+ 실행파일이나 zip파일 등을 ASCII 영역의 문자들로만 이루어진 일련의 문자열로 바꾸는 인코딩 방식, 문자열을 base64로 인코드 디코드 해줌

+ option : 

  + -d : 디코드 시켜주고 출력, 이 옵션이 없으면 그냥 인코드 시켜주고 출력

  + -i : 디코딩 할 때, 알파벳 아닌 문자 무시하고 출력

<br><br>
<hr style="border: 2px solid;">
<br><br>

## tr [option] string1 [string2]
<hr style="border-top: 1px solid;"><br>

+ 특정 문자를 삭제 혹은 변환한다.

<br>

+ ```ex) cat data.txt | tr ‘a-z’ ‘A-Z’```

  + data.txt를 출력할 때 소문자를 대문자로 변경

<br>

+ -d : 문자열1에서 지정한 문자를 삭제 후 출력

  + ```ex) cat data.txt | tr –d ‘0-9’``` : data.txt를 출력할 때 숫자 삭제

<br>

+ -s : 문자열2에서 반복되는 문자 삭제

  + ```-s``` 옵션으로 반복되는 문자열 없을 때 원하는 문자로 대체 가능

      + ex) data.txt에서 공백이 하나밖에 없는 상황(반복 안되는 상황)

      + ==> ```cat data.txt | tr –s ‘ ’ ‘@’``` : data.txt.를 출력할 때 공백을 @로 대체


<br><br>
<hr style="border: 2px solid;">
<br><br>

## bzip2 [option] [file name]
<hr style="border-top: 1px solid;"><br>

+ 확장자 .bz2

+ -c : 압축되거나 압축을 푼 파일을 표준출력한다.

+ -z : 파일을 압축 , 압축 시 file name.bz2 파일 생성됨.

+ -d : 압축해제 

<br>

+ bzcat : 압축 파일 내용 보기 명령어. (옵션아님)


<br><br>
<hr style="border: 2px solid;">
<br><br>

## gzip [option] [file name]
<hr style="border-top: 1px solid;"><br>

+ 확장자 : .gz

+ -d : 압축 해제

+ -r : 개별적으로 압축 

+ -rd : 개별적으로 압축 해제


<br><br>
<hr style="border: 2px solid;">
<br><br>

## tar [option] [file name1] [file name2]
<hr style="border-top: 1px solid;"><br>

+ 여러 파일을 묶거나 해제함

+ 파일명1은 결과 파일명, 파일명2는 압축 또는 묶음 파일 

+ 사실 압축프로그램이 아니라 백업용 프로그램임, 용량이 줄어들지 않기 때문임.

<br>

+ options : 

  + -cf archive.tar foo bar : foo와 bar 파일에서 아카이브 파일 생성

  + -tvf archive.tar  # 아카이브 파일에 있는 내용 모두 출력

  + –xf archive.tar  # 아카이브 파일 해제 

  + -cvf [합칠파일] [합칠파일들] : ‘합칠파일‘들을 ’합칠파일‘에다가 합쳐라.

  + -xvf 해제할 파일 : 해제하기

  + -c : 새로운 파일을 만드는 옵션

  + -x (extract) : 압축을 해제

  + -v (view) : 압축이 되거나 풀리는 과정을 출력

  + -f (file) : 파일로서 백업을 하겠다는 옵션


<br><br>
<hr style="border: 2px solid;">
<br><br>

## ssh [option] [IP] 
<hr style="border-top: 1px solid;"><br>

+ usage

  + ```ssh <host address> <option>```
  
  + ```ssh <name>@<host address>``` -> 사용자 ID 추가

<br>
      
+ options :

  + -1 : ssh를 프로토콜 버전 1로 시도

  + -2 : ssh를 프로토콜 버전 2로 시도

  + -4 : IPv4 주소만 사용

  + -6 : IPv6 주소만 사용

  + -F configfile : 사용자 설정 파일(configfile)을 지정한다.

  + -I smartcard_device : 사용자 개인 RSA 키를 저장할 디바이스(smartcard_device)를 지정한다.

  + -i identity_file : RSA 나 DSA 인증 파일(indentity_file)을 지정한다.
  
    ex) ```ssh –i sshkey.private bandit14@localhost```

  + -l [login_name] [IP] == ```name@IP``` : 서버에 로그인할 사용자(login_name)를 지정한다.

  + -p port : 서버에 접속할 포트를 지정한다.

  + -q : 메시지를 출력하지 않는다.

  + -V : 버전 정보를 출력한다.

  + -v : 상세한 정보를 출력한다. 디버깅에 유용하다.

  + -t : 터미널 할당을 강제한다. 

<br>

원격에서 ssh를 이용한 명령어 실행 방법은 뒤에다 “명령어” 해주면 됨.
  
<br><br>
<hr style="border: 2px solid;">
<br><br>

## telnet [option] [host [port]]
<hr style="border-top: 1px solid;"><br>

+ host : 접속할 호스트로 인터넷 주소형식으로 사용

+ port : 접속에 사용할 호스트의 포트를 지정, 초기값은 23번 사용

+ -l 사용자 ID : 텔넷서버 시스템에 접속할 사용자 ID을 지정한다.

+ -a : 현재 사용자를 ID로 사용하여 접속한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## s_client 
<hr style="border-top: 1px solid;"><br>

+ openssl 명령으로 운영중인 웹서버의 SSL인증서 정보를 살펴볼 수 있다 

<br>

+ Usage

  ```openssl s_client –connect [host]:[port] [option]```

<br>

+ option :

  + -connect host:port : 접속할 호스트와 포트, 기본값은 localhost:4433

  + -ssl2,ssl3,tls1,dtls1 : 설정한 프로토콜만 통신

  + -msg : 프로토콜 메시지 출력

  + -cert [file] : client public key 서버인증서

  + -key [file] : client private key 개인키 사용 기본 perm

  + -pass arg : private key를 위한 password을 전달


<br><br>
<hr style="border: 2px solid;">
<br><br>

## nmap
<hr style="border-top: 1px solid;"><br>

+ Network exploration tool and security / port scanner

<br>

+ Usage

  ```nmap [scan type] [option] [host]```

<br>

+ options :

  + -v : 더 자세한 정보 출력
  + -p : 포트 번호 입력 ex) -p31000-32000 : 31000~32000 포트 번호 탐색 

<br>

옵션에 따라 스캔된 호스트의 추가 정보를 출력하는데 그 정보 중 핵심은 'interesting ports table'입니다. 

테이블은 스캔된 포트번호, 프로토콜, 서비스이름, 상태(state)를 출력합니다. 

state에는 open, filtered, closed, unfiltered가 있습니다. 

스캔 결과 출력 시 하나의 상태가 아닌 조합된 값(open|filtered, closed|filtered)을 출력할 수도 있습니다.

<br>

+  open:       스캔된 포트가 listen 상태임을 나타냄

+  filtered:   방화벽이나 필터에 막혀 해당 포트의 open, close 여부를 알 수 없을 때

+  closed:     포트스캔을 한 시점에는 listen 상태가 아님을 나타냄

+  unfiltered: 스캔에 응답은 하지만 해당 포트의 open, close 여부는 알 수 없을 때

<br>

nmap 더 자세히 : <a href="https://ind2x.github.io/posts/nmap/" target="_blank">ind2x.github.io/posts/nmap/</a>

<br>
<br>

### NSE : http-backup-finder
<hr style="border-top: 1px solid;"><br>

NSE : Nmap Script Engine

<br>

```console
nmap -p<port> --script=http-backup-finder --script-args http-backup-finder.url=url host

PORT   STATE SERVICE REASON
80/tcp open  http    syn-ack
| http-backup-finder:
| Spidering limited to: maxdepth=3; maxpagecount=20; withindomain=example.com
|   http://example.com/index.bak
|   http://example.com/login.php~
|   http://example.com/index.php~
|_  http://example.com/help.bak
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## nc [option] [hostname] [port]
<hr style="border-top: 1px solid;"><br>

+ 네트워크 연결상태를 읽거나 쓸 때 사용됨.

+ TCP, UDP 프로토콜을 사용하는 네트워크 연결에서 데이터를 읽고 쓰는 명령어.

<br>

+ option

  + -n : 호스트 네임과 포트를 숫자로만 입력받는다.

  + -v : 더 많은 정보를 얻을 수 있다.

  + -u : TCP 연결대신 UDP 연결이 이루어 진다.

  + -p : local-port를 지정한다. 주로 –l와 같이 사용.

  + -l : listen 모드로 nc를 띄우게 된다. nc를 서버로 이용 시 사용되므로 대상 호스트는 입력하지 않음.

  + -r :여러개의 포트를 지정했을 때 스캐닝 순서를 랜덤하게 한다.

  + -z : 연결을 이루기위한 최소한의 데이터 외에는 보내지 않게 하는 옵션


<br><br>
<hr style="border: 2px solid;">
<br><br>

## ltrace [option ...] [command [arg ...]]
<hr style="border-top: 1px solid;"><br>

+ Trace library calls of a given program.

+ option

  + o, --output=FILENAME write the trace output to file with given name

  + u USERNAME : run command with the userid, groupid of username.

  + ```-s STRSIZE``` : specify the maximum string size to print.


<br><br>
<hr style="border: 2px solid;">
<br><br>

Next 
: <a href="https://ind2x.github.io/posts/linux_command3/">Linux Commands 3</a>

<br><br>
