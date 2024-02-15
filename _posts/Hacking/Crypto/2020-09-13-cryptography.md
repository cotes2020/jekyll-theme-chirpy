---
title : 암호 정리 (Incomplete)
categories : [Hacking, Crypto]
tags : [Crypto, Incomplete]
---

# 암호 기법의 분류
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154783101-9f3189c2-36a3-4b6e-a242-6bc010f185ee.png)


출처 : <a href="https://yjshin.tistory.com/entry/암호학-암호기법의-분류-치환암호-전치암호-블록암호-스트림암호-링크암호-종단간암호-하드웨어암호-소프트웨어-암호" target="_blank">[YJUN IT BLOG]</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>


# 치환암호
<hr style="border-top: 1px solid;"><br>

참고 : <a href="https://blog.naver.com/wnrjsxo/221704381990" target="_blank">blog.naver.com/wnrjsxo/221704381990</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

# Caesar Cipher (seizure cipher)
<hr style="border-top: 1px solid;"><br>

+ monoalphabetic cipher(단일문자 치환 암호)

  + 간단한 치환암호.

  + 평문의 알파벳들을 각각 일정한 거리만큼 떨어진 다른 알파벳으로 치환.

<br>

![image](https://github.com/ind2x/ind2x.github.io/assets/52172169/59d4ab6c-7efe-49e6-a3b1-60291a0b8d37)


<br><br>
<hr style="border: 2px solid;">
<br><br>

# Vigenere Cipher (polyalphabetic cipher)
<hr style="border-top: 1px solid;"><br>

다중문자 치환 암호  
: <a href="https://ko.wikipedia.org/wiki/비즈네르_암호" target="_blank">ko.wikipedia.org/wiki/비즈네르_암호</a>  

<br>

![image](https://github.com/ind2x/ind2x.github.io/assets/52172169/c40f35bf-13aa-4340-bcc9-ab66410ba894)

<br>

![image](https://github.com/ind2x/ind2x.github.io/assets/52172169/08ae283a-30ae-41ff-9fc8-865bb50c76a8)

<br><br>
<hr style="border: 2px solid;">
<br><br>

# 대칭 암호
<hr style="border-top: 1px solid;"><br>

대칭 암호는 암호화와 복호화에 동일한 키를 사용한다. 

이 방식은 비대칭 암호 방식보다 암호화/복호화 속도가 빠르기는 하지만 키 분배를 안전하게 하는 게 가장 어렵다.

<br>

크게 블록 암호와 스트림 암호로 나뉜다.

블록 암호는 고정 크기의 블록 단위로 작업을 수행한다. 이 방식에서는 동일한 키를 사용했을 경우 한 평문 블록이 항상 같은 암호 블록으로 암호화된다.

DES, Blowfish, AES(Rijndael)가 있다.

<br>

스트림 암호는 한 번에 1비트나 1바이트씩 임의의 비트 스트림을 만들어낸다. 이것을 키스트림이라 하는데, 키 스트림ㄹ은 평문과 XOR된다. 

이 방식은 연속적인 데이터 스트림을 암호화하는 데 유용하다.

RC4, LSFR이 있다.

<br><br>

## 블록 암호 
<hr style="border-top: 1px solid;"><br>

블록 암호 방식에서 가장 많이 이용되는 개념에는 혼돈과 확산이 있다.

혼동
: 평문, 암호문, 키 사이의 관계를 감추려고 사용하는 방법을 의미한다.

<br>

확산
: 평문 비트와 키 비트가 가능한 한 많은 암호문에 영향을 주게 함을 의미한다. 

<br>

혼합 암호는 여러 단순 연산을 반복해 이 두 가지 개념을 모두 구현한 방식이다. DES와 AES 모두 혼합 암호 방식이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

# DES 
## 구조
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154691874-bc27327e-4a52-427b-bacd-3686ecd369b6.png)

<br>

+ Block암호(Feistel 구조), 대칭암호

  + Plaintext : 64bit

  + Ciphertext : 64bit

  + Encrypt Key : 64bit -> remove parity bit (8) -> 56bit -> key schedule -> 48 bit  

<br>

+ 과정
  
  1. 64 bit의 평문을 IP(initial permutation, 초기치환)을 해줌.

  2. IP 후 64bit 평문을 32bit씩 L, R로 나눔.

  3. R0를 키 스케줄을 통해 나온 48bit 암호키와 같이 F함수에 들어감. -> 32 bit 생성

  4. L0와 F함수에서 만든 32bit와 XOR연산

  5. 4에서 나온 결과 값은 R1으로, R0은 그대로 L1으로 이동

  6. 위 과정을 16라운드 동안 해주고 마지막 라운드에서는 L, R 위치 변경

  7. 다시 inverse permutation을 해주면 암호문 64bit 생성

<br><br>

### IP(Initial Permutation) table
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154691922-d1f3ca5e-394f-4807-af83-edfd99cd9ae7.png)

<br><br>

## F함수
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154692010-675b0f82-b4ec-4fa0-acdf-417caa71f849.png)

<br>

+ 과정

  1. R 32bit를 Expansion 해주어서 48bit로 Expansion

  2. Expansion R(48bit)와 48bit 암호키 XOR 연산

  3. 2의 결과 값 48bit를 8개의 S-box에 6bit씩 넣어줌.

  4. S-box에서 규칙에 따라 4bit씩 생성 -> 8*4 = 32bit 생성

  5. 32bit를 다시 치환 

  6. 32bit F함수 결과값 생성


<br><br>

### Expansion Permutation table
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154692238-ade9ffc9-0815-4ff4-9538-522e2daf0d0d.png)

<br><br>

### S-Box
<hr style="border-top: 1px solid;"><br>

+ S-Box 규칙

  1. 맨 앞, 뒤 비트를 묶어서 행을 결정 (0~3)

  2. 나머지 4비트를 묶어 열을 결정 (0~15)

<br>

예를 들어, S-Box 1에 100110이 들어왔다하면 
  1. 10 -> 2 (행)
  2. 0011 -> 3 (열)

  -> S-box 1의 2행 3열 값 8 

  -> 8을 다시 2진수로 바꾸면 1000 -> 6bit가 4bit로 변경됨.

<br>

![image](https://user-images.githubusercontent.com/52172169/154692650-6e5311a7-25b0-44b1-ab14-820e3890cd34.png)

<br><br>

## 키 스케줄
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154692683-86e8c66d-c96a-4157-8d90-20d7374d6b4e.png)

<br>

+ 과정

  1. 64bit의 암호키를 PC-1 전치해줌. -> parity bit 제거(8, 16, 24, 32, 40, 48, 56, 64)

  2. 56bit의 키를 Left, Right로 나눠주고 Shift table에 따라 이동

  3. PC-2에서 LR을 합쳐 다시 전치 -> 48bit 키 생성

<br><br>

### PC-1, PC-2
<hr style="border-top: 1px solid;"><br>

![PC1](https://user-images.githubusercontent.com/52172169/154693510-c35123f2-d0ec-482e-99a3-a92e68c73b21.png)

<br>

![PC2](https://user-images.githubusercontent.com/52172169/154693570-121cdeb8-aee8-42ba-bd07-4f5215adab6d.png)

<br><br>

### Shift table
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/154693622-f5e9f03a-8109-4b13-9514-aae75d8135c4.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

# AES
<hr style="border-top: 1px solid;"><br>

Not yet

<br><br>
<hr style="border: 2px solid;">
<br><br>

# RSA
<hr style="border-top: 1px solid;"><br>

공개키 ```{n, e}```가 있고 개인키 ```{p, q}```가 있음.  

<br>

+ RSA 암호화 알고리즘

  + 서로 다른 소수 p, q를 정한 뒤 ```n, φ(n), e, d```를 구한다. 

    + ```n = p * q```
  
    + ```φ(n)=(p-1)(q-1)```

    + ```e : gcd(e, φ(n)) = 1, 1 < e < φ(n)```인 e를 정한다.

    + ```d : ed=1 mod (φ(n)), 즉 φ(n)에 대한 e의 역원을 구한다. 1 < d < φ(n)```

  + C : Ciphertext, M : Plaintext

    + encrypt : ```C ≡ M^e mod (n)```

    + decrypt : ```M ≡ C^d mod (n)```

<br>

<a href="http://factordb.com/" target="_blank">tool</a>을 이용하여 ```{p, q}```를 구할 수 있음.

<br><br>
<hr style="border: 2px solid;">
<br><br>

# 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.crocus.co.kr/341" target="_blank">crocus.co.kr/3410</a> 
: <a href="https://bpsecblog.wordpress.com/amalmot/" target="_blank">bpsecblog.wordpress.com/amalmot/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
