---
title: "Web Browser 웹 브라우저"
# description: ""
categories: [컴퓨터, 인터넷]
tags: [Web]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-05. 07:01
# last_modified_at: 2023-10-05. 07:01
# last_modified_at: 2024-08-29. 21:27
last_modified_at: 2024-09-16. 16:02
---

## Web Browser | 웹 브라우저

---

URL 입력을 통해 Web Page (HTML, CSS)를 다운 받고,  
이를 Rendering 하여 보여주는 GUI 프로그램  
= `원격지` 문서 뷰어  

주소 입력창에 URL Uniform Resource Locator 입력  
→ (일반적으로) [HTTP](/posts/http/)S를 통해 `Resource`(문서/Web Page도 Resource)를 Web Server로 부터 [Get-Read/Post](/posts/get-post/) 하겠다는 것  
→ CMD 명령어 입력하는 거랑 다를 게 없음 (DNS Domain Name System, URL → IP)  
→ Search Query를 입력하면 설정해둔 Search Engine으로 검색  

개발자 도구/네트워크 탭을 통해,  
Web Browser가 Resource를 어떻게 다운로드하고 있는지 확인 가능  

Web Page는 HTML, CSS, JavaScript로 구성됨  

- 다운로드 순서
  1. HTML 문서
  2. HTML 문서가 참조하는 CSS
  3. CSS가 참조하는 폰트?
  4. 렌더링 시작과 동시에, 이미지 (Favicon도 이때)

HTML 문서는 사람이 읽기 어려움  
So, Web Browser는 다음 과정을 통해 읽기 쉽게 보여줌  

1. Parsing 구문 분석을 통해 DOM (자료구조 - 비선형 트리구조) 생성  
2. DOM을 기반으로 Rendering (출력)  

## 메모

---

### 참고

- [널널한 개발자 - 초창기 웹 서비스 구조](https://youtu.be/4Sfned8HLzk?si=_gVz3bwTPSAmk2_v)
- [가장 쉬운 웹개발 with Boaz - 브라우저에 URL을 입력하면 어떤 과정이 진행될까?](https://youtu.be/ipwfEUslfQA?si=PYRBblbYqZD8Bc7u)
