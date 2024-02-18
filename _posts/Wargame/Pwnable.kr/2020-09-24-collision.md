---
title : Pwnable.kr - collision
categories : [Wargame, Pwnable.kr]
---

## collision
```
Daddy told me about cool MD5 hash collision today.
I wanna do something like that too!

ssh col@pwnable.kr -p2222 (pw:guest)
```
```c
#include <stdio.h>
#include <string.h>
unsigned long hashcode = 0x21DD09EC;
unsigned long check_password(const char* p){
        int* ip = (int*)p;
        int i;
        int res=0;
        for(i=0; i<5; i++){
                res += ip[i];
        }
        return res;
}

int main(int argc, char* argv[]){
        if(argc<2){
                printf("usage : %s [passcode]\n", argv[0]);
                return 0;
        }
        if(strlen(argv[1]) != 20){
                printf("passcode length should be 20 bytes\n");
                return 0;
        }

        if(hashcode == check_password( argv[1] )){
                system("/bin/cat flag");
                return 0;
        }
        else
                printf("wrong passcode.\n");
        return 0;
}
```

## Solution
```
0x21DD09EC을 5로 나누면 0x06C5CEC8임. 근데 이 값에 다시 5를 곱하면 원래의 값이 나오지 않고 4가 부족함.
따라서 payload에는 0x06C5CEC8을 4번 보낸 뒤 이 값에 4를 더한 0x06C5CECC값을 보내주면 됨.
```
```linux
col@pwnable:~$ ./col `python -c 'print "\xc8\xce\xc5\xc6"*4+"\xcc\xce\xc5\x06"'`
daddy! I just managed to create a hash collision :)
```