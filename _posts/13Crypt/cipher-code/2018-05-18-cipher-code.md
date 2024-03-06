---
title: Cryptography - Cipher code
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

- [Cipher code](#cipher-code)
  - [playfair Cipher 普莱费尔密码](#playfair-cipher-普莱费尔密码)
  - [Hill Cipher 希尔密码](#hill-cipher-希尔密码)
  - [Vigenere cipher 维吉尼亚密码](#vigenere-cipher-维吉尼亚密码)
  - [Caesar Cipher 凱撒密碼](#caesar-cipher-凱撒密碼)

---


# Cipher code


Classical encryption technique
- substitution 替换
  - playfair cipher: `5x5 metric, 2 pair, same row/column/none`
  - hill cipher: `3x3 metric`
  - Caesar cipher:
  - monoalphabetic cipher
  - polyalphabetic cipher: `any cipher based on substitution, using multiple substitution alphabets`.
    - Vigenère Cipher: `26x26, str x key = cipher`
  - one-time pad
- transposition
  - Rail fence
  - row column transposition

---


## playfair Cipher 普莱费尔密码

- 使用一个关键词方格来加密字符对的加密法，1854年由一位名叫查尔斯·惠斯通（Charles Wheatstone）的英国人发明

![keymatrix-300x292](https://i.imgur.com/AP9SV1C.png)

![encodemsg1](https://i.imgur.com/jaO6QKG.png)


Start
1. Generate the key Square(5×5):
   1. The key square is a 5×5 grid of alphabets that acts as the key for encrypting the plaintext.
   2. Each of the 25 alphabets must be unique
   3. one letter of the alphabet (usually J) is omitted from the table (as the table can hold only 25 alphabets). If the plaintext contains J, then it is replaced by I.
   4. The initial alphabets in the key square are the unique alphabets of the key in the order in which they appear followed by the remaining letters of the alphabet in order.
2. Algorithm to encrypt the plain text:
   1. The plaintext is split into pairs of two letters (digraphs). `"instruments" -> 'in' 'st' 'ru' 'me' 'nt' 'sz'`
   2. **Pair cannot be made with same letter**. Break the letter in single and add a bogus letter to the previous letter. `hello -> he lx lo`
   3. If the letter is standing alone in the process of pairing, then add an extra bogus letter with the alone letter `helloe ‘he’ ‘lx’ ‘lo’ ‘ez’`


Rules for Encryption:
1. If `both the letters` are in the **same column**: Take the letter below each one (going back to the top if at the bottom).
2. If `both the letters` are in the **same row**: Take the letter to the right of each one (going back to the leftmost if at the rightmost position).
3. If neither of the above rules is true: Form a rectangle with the two letters and take the letters on the horizontal opposite corner of the rectangle.


---

## Hill Cipher 希尔密码

希爾密碼是運用基本矩陣論原理的替換密碼，由Lester S. Hill在1929年發明。
每個字母當作26進制數字：A=0, B=1, C=2... 一串字母當成n維向量，跟一個n×n的矩陣相乘，再將得出的結果模26。

c1 = p1p2p3(k1, k21, k31) mod 26
c2 = p1p2p3(k2, k22, k32) mod 26
c3 = p1p2p3(k3, k23, k33) mod 26


---

## Vigenere cipher 维吉尼亚密码

维吉尼亚密码足够地易于使用使其能够作为战地密码。[5]例如，美国南北战争期间南军就使用黄铜密码盘生成维吉尼亚密码。北军则经常能够破译南军的密码。战争自始至终，南军主要使用三个密钥，分别为“Manchester Bluff（曼彻斯特的虚张声势）”、“Complete Victory（完全的胜利）”以及战争后期的“Come Retribution（报应来临）”。

![Vigenère_square_shading](https://i.imgur.com/yb5i5pO.png)

解密的过程则与加密相反。例如：根据密钥第一个字母L所对应的L行字母表，发现密文第一个字母L位于A列，因而明文第一个字母为A。密钥第二个字母E对应E行字母表，而密文第二个字母X位于此行T列，因而明文第二个字母为T。以此类推便可得到明文。

用数字0-25代替字母A-Z，维吉尼亚密码的加密文法可以写成同余的形式：C = (P+K)mod26
解密方法则能写成：P = (C-K)mod26

---


## Caesar Cipher 凱撒密碼

It is a type of substitution cipher in which each letter in the plaintext is replaced by a letter some fixed number of positions down the alphabet.

![Caesar_cipher_left_shift_of_3.svg](https://i.imgur.com/RNeQe4K.png)


- involves replacing each letter in a message with the letter that is a certain number of letters after it in the alphabet.
  - we might replace each A with D, each B with E, each C with F, and so on, if shifting by three characters.
  - We continue this approach all the way up to W, which is replaced with Z. Then, we let the substitution pattern wrap around, so that we replace X with A, Y with B, and Z with C.

- a Caesar cipher with a rotation of r encodes the letter having index k with the letter having index (k + r) mod 26, where mod is the modulo operator, which returns the remainder after performing an integer division.















.
