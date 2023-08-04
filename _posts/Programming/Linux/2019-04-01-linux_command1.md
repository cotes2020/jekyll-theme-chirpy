---
title: Linux 명령어 1
categories: [Programming, Linux]
tags : [Linux Commands, Linux Cheat Sheet]
---

## Linux Cheat Sheet
<hr style="border-top: 1px solid;"><br>

Cheat Sheet
: <a href="http://www.seren.net/documentation/unix%20utilities/Linux_Cheat_Sheet.htm" target="_blank">seren.net/documentation/unix%20utilities/Linux_Cheat_Sheet.htm</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 리눅스 기초 특수기호 문자
<hr style="border-top: 1px solid;"><br>

+ ```>```

  표준 출력 리다이렉션으로 출력 방향을 바꿈.

  + ```ls > test.txt```
    
    출력 내용을 ```test.txt``` 파일에 기록한다. 만약 파일이 없다면 생성한다.
  
  + ```cat > test.txt```
  
    내용을 입력한 후 입력 내용을 ```test.txt``` 파일에 저장한다.

  + ```>>``` 이면 ```ls >> test.txt```는 출력된 내용을 test.txt 파일에 덧붙인다. -> 이어쓰기

<br>

wsl2에서의 문제인지는 모르겠으나, 권한 거부가 떠서 찾아보니 리다이렉션은 sudo로도 안되는 거라고 한다. 

대신 루트로 로그인해서 하거나, tee 명령어가 있다고 한다.
: ```ex) echo test | sudo tee /tmp/foo``` == ```cat > test 후 test 입력```

<br>

+ ```<```

  표준 입력

  + ```cat < test.txt```
  
    ```test.txt```의 내용을 cat 명령어로 읽은 뒤 화면에 출력
  
  + ```<<< here string```을 통해 입력값을 전달해줄 수 있다.
  
<br>

+ ```*```

  모든 문자와 일치하는 와일드 카드 문자

  + ```ls tes*```
  
    ```test.txt, tes/123.txt``` 등 일치하는 모든 파일, 디렉토리(내부)가 출력된다.

  + 파일 이름 자리에 ```*```를 적으면 모든 파일을 뜻함. 
  
    ```ls *``` ==> 모든 파일 보기

<br>

+ ```?```

  하나의 문자와 일치하는 와일드 카드 문자 //길이가 1인 임의의 문자
   
  + ```ls test.tx?```
  
    ```test.txt, test.txx``` 등 하나 일치한 파일을 출력한다.

<br>

+ ```[ ]```

  대괄호 안에 포함된 문자 중 하나라도 일치되는 문자.

  + ```ls a[13]``` : a1 , a3 파일 목록 출력.

<br>

+ ```;```

  명령어 분리자로 한 명령 라인에서 여러 가지 명령을 수행할 수 있도록 함

<br>

+ ```$```

  쉘 변수를 가르킴. 

<br>

+ ```|```

  앞에 나온 명령어의 표준 출력 결과를 다음 명령어의 표준 입력으로 사용
   
  + ```ls –al | grep ^d```
  
    모든 파일을 출력하는데 d로 시작하는 파일을 출력.

<br>

+ ```' ' , " "```

  문자를 감싸서 문자열로 만들고 문자열 안의 특수기호 기능도 없앰.

  작은따옴표는 모든 특수기호, 큰 따옴표는 $,`,\를 제외한 모든 특수기호를 일반문자로 간주

<br>

+ <code>` ` 또는 $()</code>

  shell은 <code>` ` 또는 $()</code>로 감싸인 문자열을 명령으로 해석하게 함.

<br>

+ ```\```

  특수문자 앞에 사용, 특수 문자의 효과를 없애고 일반 문자처럼 처리함.

<br>

+ ```|| , &&```

  한 줄에 여러 명령어 연속 실행
  
  ```&&```은 앞 명령어가 에러가 나지 않아야 뒷 명령어 실행 (Logical AND)
  
  ```||```은 앞 명령어가 에러가 나면 뒷 명령어 실행 (Logical OR)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 기본 명령어
<hr style="border-top: 1px solid;"><br>

+ ```rm-rf <file name>``` : file 강제 삭제

+ ```cp-r (복사할 디렉터리 경로) (붙여넣을 디렉터리 경로)``` : 파일 복사

<br>

디렉터리 생성

+ ```mkdir –p dir01/dir02``` : dir01 디렉터리 안에 dir02디렉터리까지 같이 생성

<br>

파일 종류

+ bin : binary파일 , 실행 파일

+ mnt,tmp: 임시파일

+ etc : 설정파일

<br>

파일색깔

+ 파랑: 디렉터리 파일

+ 하양: 일반 파일

+ 하늘색: 링크파일

+ 녹색: 실행파일

+ 빨강: 압축파일

<br>

파일 경로문자

+ 최상위 디렉터리 : /

+ 현재 디렉터리 : .

+ 상위 디렉터리: ..

+ 홈 디렉터리(root): ~

+ 이전 작업 디렉터리 : -

<br>

경로 구분자 ```/```

디렉터리 사이에서는 ‘/’로 경로 구분

+ ex) 현재 root 디렉터리의 하위에 있는 log 디렉터리로 이동 시

  + ```cd log``` : 하위 디렉터리로 이동 시 에는 / 생략 가능

  + ```cd ./log``` : 현재 디렉터리 안의 log 디렉터리로 이동

  + ```cd log/``` : 파일 이름뒤에 /붙으면 그 파일은 디렉터리란 의미
  
  + ```cd ./log/```

  + ```cd /root/log```

다 같은 의미임.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## man [명령어]
<hr style="border-top: 1px solid;"><br>

페이지 이동 방법

+ e : 한줄 앞으로

+ y : 한줄 뒤로

+ f : 한 window 앞으로

+ b : 한 window 뒤로

+ d : half window 앞으로

+ u : half window 뒤로

+ 이전 페이지 : page up 키 or b

+ 다음 페이지 : page down 키 or 스페이스바

+ 종료 : q

<br><br>
<hr style="border: 2px solid;">
<br><br>

## vi
<hr style="border-top: 1px solid;"><br>

+ ```vi -r file``` : 손상된 파일 회복

<br>

+ ```shift + ↑, ↓``` : 한 페이지 앞, 뒤로 이동 

+ ```Ctrl + i, b``` : 한 화면 위, 아래로 이동

+ ```Ctrl + d, u``` : 반 화면 위, 아래로 이동

+ ```Ctrl + e, y``` : 한 줄씩 위, 아래로 이동

<br>

+ dw : 한 단어 삭제

+ dd : 한 줄 삭제 

  ex) 5dd : 커서가 있는 라인부터 5개의 라인 삭제

+ u : 바로 전에 수행한 명령 취소

+ ```:5, 10``` : 5~10번째 행 삭제

+ ```/name``` : name 문자열 찾기 

+ n : 다음 name으로 이동

+ N : 이전 name으로 이동

<br>

+ ```:w``` : 저장

+ ```:wq``` : 저장하고 vi 종료

+ ```:w``` file : file에 저장

+ ```:q!``` : 저장하지 않고 vi 강제 종료

+ ```:r file``` : file의 내용을 현재 커서가 있는 줄에 출력

+ ```:e file``` : 현재 화면을 지우고 file의 내용 출력

+ ```:5,10 w file``` : 5 ~ 10줄까지의 내용을 file에 저장

<br>

+ ```:set nu``` : 행 번호 보여주기

+ ```:set nonu``` : 행 번호 보여주기 취소

+ ```.``` : 바로 전에 실행한 명령어 재실행

<br>

+ ```:![command]``` -> vi에서 쉘 실행 ex)!ifconfig

+ ```:set shell ?``` -> shell 확인

+ ```:set shell=/usr/bin/bash``` : bash로 쉘 변경 

+ ```:!/bin/bash``` -> bash 쉘 실행

<br><br>
<hr style="border: 2px solid;">
<br><br>

## head
<hr style="border-top: 1px solid;"><br>

+ ```head [option] [file]```

  파일의 앞부분부터 확인, 기본적으로 처음 10행 출력

<br>

+ option

  + -n  : 앞부분에서 num행까지 출력

  + -c : num byte까지의 내용 출력

<br><br>
<hr style="border: 2px solid;">
<br><br>

## tail
<hr style="border-top: 1px solid;"><br>

+ ```tail [option] [file]```

  파일의 끝부분부터 확인, 기본적으로 마지막 10행을 출력

<br>

+ option

  + -n : 앞부분에서 num행까지 출력
  
  + -c : num byte까지의 내용 출력

<br><br>
<hr style="border: 2px solid;">
<br><br>

## more [file]
<hr style="border-top: 1px solid;"><br>

+ 파일을 읽어 화면에 화면 단위로 끊어서 출력하는 명령어. 

+ 이 명령어는 위에서 아래 방향으로만 출력 되어서 지나간 내용을 다시 볼 수 없는 단점이 있음.

<br>

+ 단축키

  + h : more 명령어 상태에서 사용할 수 있는 키 도움말 확인

  + q : more 명령어 종료

  + enter : 1행 아래로 이동

  + space bar, f : 아래로 1페이지 이동

  + b : 1페이지씩 앞으로 이동

  + = : 현재 위치의 행번호 표시

  + /문자열 : 지정한 문자열 검색

  + ```n:/문자열``` : 지정한 문자열을 차례대로 검색

  + v : 현재 열려있는 파일의 현재 위치에서 vi 편집기 실행

<br><br>
<hr style="border: 2px solid;">
<br><br>

## cut [option] [file]
<hr style="border-top: 1px solid;"><br>

+ 데이터의 열을 추출

<br>

+ option :

  + -b : 바이트를 기준으로 추출

  + -c : 문자수를 기준으로 추출

  + -f : 파일의 필드를 기준으로 추출

  + -d : 필드 구분자를 지정, 기본 필드 구분은 TAB

<br>

```
ex) 
/etc/passwd에 root:x:0:0:root:/root:/bin/bash가 있을 때

-> cut -f 1 -d ':' /etc/passwd : ':'을 필드 구분자로 지정, 1열의 내용을 추출 -> root

-> cut -f 3-4 -d ':' /etc/passwd : , 3, 4열의 내용을 추출 -> 0:0
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## diff [option] [file1] [file2]
<hr style="border-top: 1px solid;"><br>

+ 파일 비교 명령어

+ options :

  + -c : 두 파일간의 차이점 출력
  
  + -d : 두 파일간의 차이점을 상세하게 출력

  + -r : 두 디렉토리간의 차이점 출력, 서브디렉토리 까지 비교

  + -i : 대소문자의 차이 무시

  + -w : 모든 공백 차이무시

  + -s : 두 파일이 같을 때 알림

  + -u : 두 파일의 변경되는 부분과 변경되는 부분의 근처의 내용도 출력

<br>

diff3은 3개 파일 비교

<br><br>
<hr style="border: 2px solid;">
<br><br>

## ln
<hr style="border-top: 1px solid;"><br>

+ Link파일을 만드는 명령어

참고 
: <a href="https://jhnyang.tistory.com/269" target="_blank">jhnyang.tistory.com/269</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

### 심볼릭 링크
<hr style="border-top: 1px solid;"><br>

+ 단순히 원본파일을 가리키도록 링크만 시킨 것으로 윈도우에서 '바로가기' 같은 것.

+ 원본파일의 크기와 무관, 원본이 삭제되면 링크파일은 깜박거리면서 원본이 없다는 것을 알려줌.

<br>

+ Usage : ```ln -s 원본 대상```

  + ```ln -s test t``` : test라는 파일의 심볼릭 링크 파일인 t를 현재 디렉토리에 생성

<br><br>
<hr style="border: 2px solid;">
<br><br>

### 하드링크
<hr style="border-top: 1px solid;"><br>

+ 원본과 동일한 내용의 다른 파일을 생성 -> 원본 삭제되어도 링크 파일 속 데이터는 존재함.

+ 원본이 변경되면 링크파일도 변경됨.

<br>

+ Usage : ```ln [option]``` 원본파일 대상파일(디렉토리) 

  + ```ln test.txt t``` : test.txt라는 파일의 하드링크 파일인 t를 현재 디렉토리에 생성

<br><br>
<hr style="border: 2px solid;">
<br><br>

## chmod
<hr style="border-top: 1px solid;"><br>

+ 파일권한 변경

+ 쉽게 쓰려면 8진수 형태, 복잡하지만 기능적으로 좋은 심볼릭 형태가 있음.

<br>

+ option :

  + -R : 하위 디렉토리의 모든 권한 변경

  + -c : 권한 변경 파일내용을 출력

<br><br>
<hr style="border: 2px solid;">
<br><br>

### 8진수 형태
<hr style="border-top: 1px solid;"><br>

+ ```chmod [option] [8진수 퍼미션] [filename]```

<br>

```
777 : 일반적인 8진수 형태
4777 : SetUid 설정, 4000을 더한다.
2777 : SetGid 설정, 2000을 더한다.
1777 : Sticky bit 설정, 1000을 더한다.

8진수 0~7은 아래와 같이 2진수로 표현이 가능하다
0 : 000
1 : 001
2 : 010
3 : 011
4 : 100
5 : 101
6 : 110
7 : 111

위 2진수 세자리는 rwx 세자리와 일치하며 2진수가 1이면 해당 권한을 부여,
0이면 해당 권한을 제거 한다.

ex) chmod 707 test.cnf 
test.cnf 파일에 대해 user, other 은 모두 rwx로 변경하고 group은 모든 권한을 제거한다.

ex) chmod 555 test.cnf 
test.cnf 파일에 대해 user, group, other 모두 rx의 권한을 주고 w의 권한은 제거한다.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

### 심볼릭 형태
<hr style="border-top: 1px solid;"><br>

+  ```chmod [option] [대상] (+/-/=) (rwx) (filename)```

<br>

```
대상
u : user의 권한
g : group의 권한
o : other의 권한
a : 모든 사용자 권한

+ : 해당 권한을 추가한다.
– : 해당 권한을 제거한다.
= : 해당 권한을 설정한데로 변경한다.


ex)　chmod u-x,g+r test.cnf 
test.cnf 파일에 대해 
user는 기존 권한에서 x권한만 제거한다. 나머지 권한은 그대로 유지 된다. 
group은 기존 권한에서 r권한을 추가한다. 나머지 권한은 그대로 유지 된다.


ex)　chmod u=rx,g=-,o=r test.cnf 
test.cnf 파일에 대해 user는 rx 권한만 부여, group는 모든 권한 제거, 
other은 r권한만 부여 한다
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## chown
<hr style="border-top: 1px solid;"><br>

+ 파일과 그룹의 소유권을 변경하는 명령어

<br>

Usage 
: ```chown [option] [변경할 유저명 : 변경할 그룹명] [파일명]```

option 
: ```-R``` => 하위 디렉토리에도 모든 권한 변경 

<br>

+ 명령어

  + ```소유자``` : 소유자만 변경한다.

  + ```:그룹명``` : 그룹만 변경한다.

  + ```소유자:``` : 소유자와 그룹 모두 동일한걸로 변경한다.

  + ```소유자:그룹명``` : 소유자와 그룹을 서로 다른걸로 변경한다. (물론 같은걸 해도 상관없다.)

<br>

파일명에는 설정을 위한 파일명 또는 디렉토리명 이용 , 와일드 카드 이용 가능

```
example

1) chown member1 test.cnf 
   test.cnf 파일에 대해 소유자를 member1로 바꾼다.

2) chown :member1 test.cnf 
   test.cnf 파일에 대해 그룹명을 members1로 바꾼다.

3) chown member1: test.cnf 
   test.cnf 파일에 대해 소유자 및 그룹명을 members1로 바꾼다.

4) chown member1:member2 test.cnf 
   test.cnf 파일에 대해 소유자는 member1, 그룹명은 member2로 바꾼다.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## find [path] [option] [action]
<hr style="border-top: 1px solid;"><br>

+ 주어진 조건을 경로에서 검색하여 파일을 찾는다.

<br>

```
-name [name] : 지정된 이름의 파일을 찾는다.

-user [name] : user 소유의 파일을 찾는다.

-group [name] : group 소유의 파일을 찾는다.

-type [형식] : 지정된 형식의 파일을 찾는다.
  • b : 블록파일
  • c : 문자
  • d : 디렉터리
  • f : 파일
  • l : 링크파일
  • s : 소켓

-perm [mode] : 정확히 같은 권한을 가진 파일을 찾는다.

-perm –[mode] : mode에 있는 권한 중에서 모두 만족하는 것 출력

-perm /[mode] : mode에 있는 권한 중에서 하나만이라도 만족하는 것 출력

-size [+/-]n[bckw] : 지정된 크기의 파일을 찾는다.
  • +n : n보다 크다
  • -n : n보다 작다
  • n : n이다
  • b : 512-byte
  • c : byte
  • k : kilobytes
  • w : 2-byte
```

<br>

+ action

```
-inum number : 지정한 아이노드 번호와 파일을 찾는다.

-print : 표준출력으로 검색된 파일명을 출력한다.

-exec [command] { } \; : 찾은 각 파일에 대해 지정된 명령을 실행한다.

-ok [command] { } \; : 실행 여부를 사용자에게 확인한 후 명령을 실행한다.


2>/dev/null : find 옵션 뒤에 붙이면 에러메세지를 모두 버려주고 
              해당 조건에 정확히 부합하는 내용만을 출력해줌
```

너무 파일이 많고 검색된 내용도 많아서 콘솔화면에 전부 표현할 수 없다면 more을 활용. 쪽단위로 출력할 수 있게 해줍니다.


<br><br>
<hr style="border: 2px solid;">
<br><br>

## grep [option] pattern [file]
<hr style="border-top: 1px solid;"><br>

+ pattern을 file안에서 (옵션에 맞춰) 검색한다.

<br>

+ option

```
-A(-B) num : 일치하는 줄 아래에(위에) 지정한 줄 수(num)만큼의 내용을 더 보여준다.

-c : 일치하는 줄의 수 출력

-C [num] : 일치하는 줄의 위와 아래에 지정한 줄 수 만큼의 내용을 더 출력, 기본은 2줄

-d 디렉터리 : 읽고자 지정한 파일이 디렉터리일 경우 지정한 값을 실행한다.
 d :
    read : 디렉터리를 보통 파일처럼 읽는다. (기본 값)
    skip :  디렉터리를 건너뛴다.
    recurse(-r) : 디렉터리를 포함하여 하위 디렉터리의 모든 파일을 읽는다. 

-i : 대소문자 구별 x

-l : 일치하는 줄의 파일명만 보여주고, 줄의 내용은 출력하지 않는다.

-n : 일치하는 줄의 내용과 해당 줄의 위치를 출력한다.

-v : 지정한 패턴과 일치하지 않는 내용을 보여준다.
```

<br>
<br>

메타문자 | 기능 |      사용 예     | 사용 예 설명 
:--------:|:----:|:----------------:|:------------:
^  |  행의 시작 지시자  |  ‘^love’  |  love로 시작하는 모든 행 출력
$  |  행의 끝 지시자  |  ‘love$’  |  love로 끝나는 모든 행 출력
.  |  하나의 문자와 대응  |  ‘l..e’  |  l다음에 2문자가 오고 e로 끝나는 문자 출력
[] |  []사이의 문자 중 하나와 대응되면 출력  |  ‘[Ll]ove’  | Love나 love 문자 출력
[^]  |  []사이의 문자에 속하지 않는 문자 출력  |  ‘[^A-K]ove’  |  [A-K]ove이외의 문자 출력
x\{m\}  |  문자 x를 m번 반복  |  ‘o\{5\}’  |  문자 o가 5회 연속으로 나오는 모든 행 출력
x\{m,\}  |  문자 x를 적어도 m번 반복  |  ‘o\{5,\}’  |  문자 o가 최소한 5번 반복되는 모든 행 출력
x\{m,n\}  |  m회 이상 n회 이하 반복  |  ‘o\{5,10\}'  |  문자 o가 5회 이상 10회 이하 반복되는 모든 행 출력

<br>
<br>

```  
자주 사용하는 문자 클래스

\d - 숫자와 매치, [0-9]와 동일한 표현식이다.

\D - 숫자가 아닌 것과 매치, [^0-9]와 동일한 표현식이다.

\s - whitespace 문자와 매치, [ \t\n\r\f\v]와 동일한 표현식이다. 
     맨 앞의 빈 칸은 공백문자(space)를 의미한다.

\S - whitespace 문자가 아닌 것과 매치, [^ \t\n\r\f\v]와 동일한 표현식이다.

\w - 문자+숫자(alphanumeric)와 매치, [a-zA-Z0-9_]와 동일한 표현식이다.

\W - 문자+숫자(alphanumeric)가 아닌 문자와 매치, [^a-zA-Z0-9_]와 동일한 표현식이다.
```

<br>

```
메타문자 앞에 역슬래시를 쓰면 메타문자의 특수의미를 무시함, 즉 자체문자로 인식

ex) grep ^\.    : 마침표로 시작되는 줄을 찾음 
    
    grep '\$'   : "$" 문자가 포함된 줄을 찾음 


(1) 패턴을 작성할 때 \(역슬래쉬)가 들어가는 경우(\<, \(..\), x\{m\}) 

(2) 특수문자를 문자자체로 인식하게 하려 \을 사용하는 경우

(3) 패턴에 있어서 파이프( | )를 이용하는 경우

==> 패턴을 따옴표(")나 작은 따옴표(')로 감싸주어야 한다.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## sort [option] [file] 
<hr style="border-top: 1px solid;"><br>

+ 텍스트 파일의 내용을 알파벳 순서대로 정렬

<br>

```
-b : 공백을 무시한다.

-i : 프린트 가능한 문자만 비교

-r : 결과를 역으로 출력

-n : 숫자를 기준으로 정렬

-c : 파일이 정렬되어 있는지 검사

-o : 결과를 지정한 파일에 저장

-u : 필드 내에서 중복되는 값을 제거하고 출력 
    
    ex) abcd가 4줄 중복되면 한 줄만 출력시켜주는 것

-m : 복수의 입력 파일을 병합

-f : 모든 문자를 소문자로 인식
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## uniq [option]
<hr style="border-top: 1px solid;"><br>

+ 중복되는 행을 필터링

<br>

```
-c : 행이 얼마나 중복되는지 계산하여 출력

-d : 중복되는 행의 내용을 한번만 출력

-D : 중복되는 모든 행의 내용을 출력

-N : 첫 행부터 N행까지는 검사 안함

-u : 중복되지 않는 행만 출력


무작위로 있으면 인식 x ----> sort와 uniq 명령어가 보통은 같이 쓰임.
ex) cat data.txt | sort | uniq –u
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

Next 
: <a href="https://ind2x.github.io/posts/linux_command2/">Linux Commands 2</a>

<br><br>
