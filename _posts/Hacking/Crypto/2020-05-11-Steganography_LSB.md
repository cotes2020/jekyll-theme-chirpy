---
title : Steganography
categories : [Hacking, Crypto]
tags : [Steganography]
---

## Steganography
<hr style="border-top: 1px solid;"><br>

사진, 음악, 동영상 등의 일반적인 파일 안에 데이터를 숨기는 기술

이미지 파일에 메시지를 숨기는 방법으로 크게 두 가지로 구분하는데 **삽입**과 **변조**가 있음.

<br><br>

### 삽입
<hr style="border-top: 1px solid;"><br>

이미지에 데이터를 삽입하는 

JPEG, PNG, GIF 등의 파일에는 파일의 끝을 알리는 ```EOI(End Of Image) Hex 바이트```가 존재함. 

```Footer Signature``` 라고 하는데, 이런 식으로 ```Header```와 ```Footer Signature```가 있음.    

![image](https://user-images.githubusercontent.com/52172169/167240317-af811458-b3bd-47bd-b2f3-69d6b561ab20.png)  

<br>

사진 출처 
: <a href="https://zzunsik.tistory.com/28" target="_blank">zzunsik.tistory.com/28</a>  

<br>

**이런 ```Footer Signature```, 즉 ```EOI 바이트``` 뒤의 데이터는 무시되므로 이 공간에 데이터를 숨길 수 있음.**

원본 이미지와 데이터를 숨긴 이미지는 겉으로 보기에는 차이가 없음.  

파일의 헤더에도 데이터를 숨길 수 있음.  

파일의 헤더 중 이미지에 영향을 주지 않는 부분이 있는데 이곳에 데이터를 삽입.  

<br><br>

### 변조
<hr style="border-top: 1px solid;"><br>

이미지를 변조하여 데이터를 숨기는 방법.

가장 일반적인 방법이 **최하위 비트인 LSB를 변조하는 방법으로 주로 24비트 이미지 파일에 적용됨.** 
: 24비트(빨강 8 초록 8 파랑 8)  

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처 및 복호화 코드
<hr style="border-top: 1px solid;"><br>

자세한 내용은 여기 
: <a href="https://bpsecblog.wordpress.com/2016/08/21/amalmot_4/" target="_blank">bpsecblog.wordpress.com/2016/08/21/amalmot_4/</a> 

<br>

복호화 코드
: <a href="https://github.com/VasilisG/LSB-steganography" target="_blank">github.com/VasilisG/LSB-steganography</a> 

<br>

온라인 디코더  
: <a href="https://aperisolve.fr/" target="_blank">aperisolve.fr/</a> -> Best tool  
: <a href="https://incoherency.co.uk/image-steganography/#unhide" target="_blank">incoherency.co.uk/image-steganography/#unhide</a>  
: <a href="https://futureboy.us/stegano/compinput.html" target="_blank">futureboy.us/stegano/compinput.html</a>
: <a href="https://stylesuxx.github.io/steganography/" target="_blank">stylesuxx.github.io/steganography/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
