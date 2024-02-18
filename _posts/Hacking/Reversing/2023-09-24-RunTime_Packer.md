---
title : 실행 압축 (Run-time Packer)
categories : [Hacking, Reversing]
tags : [Reversecore, 실행압축]
---

## 실행 압축
<hr style="border-top: 1px solid;"><br>

실행 압축은 PE 파일을 대상으로 파일 내부에 압축해제 코드를 포함하고 있어서 실행되는 순간에 메모리에서 압축을 해제시킨 후 실행시키는 기술이다.

실행 압축된 파일 역시 PE 파일이며, 내부에 원본 PE 파일과 decoding 루틴이 존재한다.

Entry Point 코드에 decoding 루틴이 실행되면서 메모리에서 압축을 해제시킨 후 실행된다.

<br>

원본 EP를 OEP(Original Entry Point)라고 표현하며, 실제 리버싱에서는 일일이 Tracing하진 않고 자동화 스크립트, 노하우 등을 통해 OEP로 바로 간다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Packer
<hr style="border-top: 1px solid;"><br>

패커는 실행 파일 압축기를 말한다.

사용 목적은 PE 파일의 크기를 줄이는 것, PE 파일의 내부 코드와 리소스를 감추기 위한 목적으로 쓰인다. (쓰이나?)

딱 봐도 악의적으로도 쓸 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 프로텍터
<hr style="border-top: 1px solid;"><br>


패커처럼 실행 압축을 해주는 것만이 아니라 리버싱을 막기 위한 다양한 기법이 추가된다. --> 디버깅 매우 어려움

그래서 원본 PE보다 더 커지는 경향이 있다.

사용 목적은 크래킹 방지, 코드 및 리소스 보호

<br><br>
<hr style="border: 2px solid;">
<br><br>