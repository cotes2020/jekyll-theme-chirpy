---
title : Linux 명령어3
categories: [Programming, Linux]
tags: [Linux Commands]
---

## nm [options] objfile
<hr style="border-top: 1px solid;"><br>

+ 목적 파일의 심볼을 보여주는 명령어로 여러 함수의 주소를 찾을 수 있음.

+ 오브젝트 파일에 포함되어 있는 심볼을 알파벳순으로 1행씩 출력.

+ 기본적인 출력 형식은 bsd 형식 : 심볼 값  | 심볼 클래스 | 심볼명

<br>

심볼 클래스 중 U는 정의되지 않은 심볼을 뜻함.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## objdump [options] objfile
<hr style="border-top: 1px solid;"><br>

+ 하나 이상의 오브젝트 파일에 대한 정보 표시

+ option

  + -d : 기계어를 포함해야할 섹션만 disassemble
  
  + -f : objfile 의 각 파일의 헤더 전체로부터의 요약 정보를 표시
  
  + -h : 오브젝트 파일의 섹션 헤더로부터 요약 정보를 표시
  
  + -j name : 지정한 name 섹션 정보만 표시

  + -R : 파일 실행 시 이진 파일의 동적 재배치 항목 표시

  + -s : 지정된 섹션의 모든 내용 표시

<br><br>
<hr style="border: 2px solid;">
<br><br>
