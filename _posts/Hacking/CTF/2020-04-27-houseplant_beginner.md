---
title: Houseplant CTF Beginner Solution
categories: [Hacking,CTF]
tags: [Houseplant CTF,Crypto]
---

## 1. Tool
Cyberchef : <a href="https://gchq.github.io/CyberChef" target="_blank">https://gchq.github.io/CyberChef</a>

## 2. Crypto used in Houseplant Beginner 1~10 
### Beginner 1 -> base64 decode
```
When Bob and Jia were thrown into the world of cybersecurity, 
they didn't know anything- 
and thus were very overwhelmed. 
They're trying to make sure it doesn't happen to you.
Let's cover some bases first.

cnRjcHt5b3VyZV92ZXJ5X3dlbGNvbWV9 -> base64 decode -> rtcp{youre_very_welcome}
```
### Beginner 2 -> Hex decode
```
Bob wanted to let you guys know that "You might not be a complete failure."
Thanks, Bob.

72 74 63 70 7b 62 6f 62 5f 79 6f 75 5f 73 75 63 6b 5f 61 74 5f 62 65 69 6e 67 5f 65 6e 63 6f 75 72 61 67 69 6e 67 7d
-> Hex decode -> rtcp{bob_you_suck_at_being_encouraging}
```
### Beginner 3 -> Octal decode
```
Fun fact: Jia didn't actually know what this was when they first started out. 
If you got this, you're already doing better than them ;-;

162 164 143 160 173 163 165 145 137 155 145 137 151 137 144 151 144 156 164 137 153 156 157 167 137 167 150 141 164 137 157 143 164 141 154 137 167 141 163 137 157 153 141 171 77 41 175
-> Octal decode -> rtcp{sue_me_i_didnt_know_what_octal_was_okay?!}
```
### Beginner 4 -> Caesar decode(rot 13)
```
Caesar was stabbed 23 times by 60 perpetrators... sounds like a modern group project

egpc{lnyy_orggre_cnegvpvcngr} -> rot 13 -> rtcp{yall_better_participate}
```
### Beginner 5 -> Morse 
```
beep boop

-- .- -. -.-- ..--.- -... . . .--. ... ..--.- .- -. -.. ..--.- -... --- --- .--. ...

Remember to wrap the flag in the flag format rtcp{something} -> From Morse -> MANY_BEEPS_AND_BOOPS 
```
### Beginner 6 -> A1Z26 decode 
```
i'm so tired...

26 26 26 26 26 26 26 26 19 12 5 5 16 9 14 7 9 14 16 8 25 19 9 3 19

*disclaimer: DON'T DO THIS KIDS. only sleep in math.

Remember to wrap the whole thing in the flag format rtcp{} -> zzzzzzzzsleepinginphysics
```
### Beginner 7 -> atbash cipher
```
Don't go around bashing people.

igxk{fmovhh_gsvb_ziv_nvzm} -> rtcp{unless_they_are_mean}
```
### Beginner 8 -> Bacon cipher decode 
```
You either mildly enjoy bacon, think it's a food of the gods, or are vegan/vegetarian.

00110 01110 00100 00000 10011 00101 01110 01110 00011 00011 01110 01101 10011 10010 10011 00000 10001 10101 00100

Remember to wrap the flag in rtcp{}
 Hint! Make sure you use the "complete" alphabet. -> GOEATFOODDONTSTARVE
```
### Beginner 9 -> multiple encryption algorithms in Beginner 1 ~ 8
```
Hope you've been paying attention! :D

Remember to wrap the flag with rtcp{}
 Hint! we stan cyberchef in this household
 Beginner 10.txt 
 ```
```
MmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMGEgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmQgMmQgMmQgMmQgMmQgMjAgMmUgMmQgMmQgMmQgMmQ=
```
```
순서 : base64 decode -> hex decode -> from morse -> binary decode -> A1Z26 -> rot 13 -> Atbash cipher decode -> flag : nineornone
```

## 3. writeup 출처
6 ~ 8번  
<a href="https://vikasgola.github.io/blog/houseplant-ctf-2020" target="_blank">https://vikasgola.github.io/blog/houseplant-ctf-2020</a>  
9번  
<a href="https://ctftime.org/writeup/20318" target="_blank">https://ctftime.org/writeup/20318</a>
