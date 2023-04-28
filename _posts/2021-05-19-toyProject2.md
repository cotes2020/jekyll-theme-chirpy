```
title: Clone Project(2): Care of Legend
author: dongee_seo
date: 2021-05-19
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
```

![post-thumbnail](https://velog.velcdn.com/images/seod0209/post/34f69070-25e2-4a03-bdee-c83bd7a88eaf/quiz.gif)

[Care of Legend Github Repository : Front-end](https://github.com/wecode-bootcamp-korea/19-2nd-CareOfLegend-frontend)
[Care of Legend Github Repository : Back-end](https://github.com/wecode-bootcamp-korea/19-2nd-CareOfLegend-backend)

# 프로젝트 소개

구독 커머스(서브스크립션 커머스: Subscription Commerce), 신문을 구독하듯 한 번의 구매로 특정 상품을 정기적으로 받아볼 수 있는 서비스를 일컫는 말이다.

[&#39;케어오브(care/of)&#39;](https://takecareof.com/)는 영양제를 정기적으로 배송해주는 서브스크립션 커머스중 하나이다.
본 사이트는 고객이 임의로 영양제를 선택하는 것이 아니라, 설문에 답을 한 후 전문의와 약사의 진단을 통해 개인별 맞춤 영양제를 조합해 제공한다. 어떤영양제 먹어야 될지 모르는 사람들을 위한 몸상태 진단 및 진단결과에 맞춰서 영양제 코디네이션 서비스를 제공한다는점이 특히 매력적이었다.
Care of Legend 팀은 케어오브의 특색인 맞춤형서비스에 초점을 두어 클론프로젝트를 진행하였다.
![](<https://images.velog.io/images/seod0209/post/e37e2a75-059b-4a5c-b69c-afeff5736847/Image_from_iOS_(1).jpeg>)

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F8a45afda-b57d-4383-bb5e-2d84885a4cae%2FcareofLogo.png)
아무도 모르게 홍일점....ㅋㅋㅋㅋㅋㅋ

### 작업기간: 2020.04.26 -2020.05.07

### 기술스택

- 협업툴 : Github / Trello / Notion
- 프론트엔드 : React(CRA)
- 백엔드: Django / Python / MYSQL

### 역할분담

- Care of 라는 사이츠는 팀원들 모두에게 생소한 사이트였다. 또한 Front-end는 기존 classs형 컴포넌트가 아닌 함수형 컴포넌트와 styled-component를 사용해야 했기에, 기능구현의 수에 치중 하기 보다는, 기존의 기능구현에 충실하되 새로 배우게된 내용을 이해하고 어떻게 적용할 것인지 더 우선순위를 두기로 하였다.
  1. 소셜로그인
  2. 개인 프로필작성 이후 프로필과 관련한 내용들이 페이지 요소요소마다 나타할것
  3. 퀴즈를 통해 저장된 데이터베이스를 기반으로 영양제를 추천기능
     ::목표: '개인 맟춤형 서비스 제공'이라는 홈페이지의 목적에 초점을 두고 함수형컴포넌트와 스타일컴포넌트를 중심으로 기능구현.
- 이번 프로젝트에서는 Trello는 해야할 일 들이 어떻게 진행되었는지 '진행상황을 확인하기 위한 용도'로사용 하였고, 더 자세한 내용과 공유해야 할 사항들은 Notion으로 협업 도구들의 목적과 기능을 분리하였다.

## 구현 사항

---

### 내가 맡은 구현사항: 네비바/ 메인/ 퀴즈

(1) 네비바

- 스크롤 시 일정 높이부터 고정 및 배경색 변화
- 전체 카테고리에 마우스 오버 시 서버로부터 받아온 카테고리 리스트 드롭다운

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F39c9d4f1-721c-4073-8ae6-a98beec6579d%2Fnav-main.gif)

(2) 메인

- 레이아웃
- 맨 위로 스크롤 버튼

(3) 퀴즈

- 퀴즈 진행: [code](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85)
- 캐러셀
  ![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F1a053325-555f-43a0-853f-5e1d533961ab%2Fquiz.gif)

## 최종 구현 사항

### 👉 [희열님](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85) + [현영님](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85)

🍊** 소셜 로그인(Kakao talk)
🍋 프로필 등록(사진/닉네임)-> 네비바에 변경사항 반영**
![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F5057cabe-34eb-4443-8cc1-5a7f83ddfae5%2Flogin-profile.gif)

### 👉 [민석님](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85) + [정현님](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85) / [형섭님](https://velog.io/@seod0209/Project-3.-Care-of-%EB%AA%A8%ED%8B%B0%EB%B8%8C-%ED%98%91%EC%97%85)

🍊 **상품 리스트**![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F85b7c211-d695-43a1-89c7-75f9c6c9be18%2FproductList.gif)
🍋 **카트**
![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2Fc76f95ff-993e-44e9-bf88-05e2b18c42f6%2Fcart.gif)

## 프로젝트 후기

---

### Notion의 재발견👍🏻

1차 프로젝트 때는 트렐로에 모든 내용을 적어놓고 일을 진행했다면, 2차 프로젝트때는 회의록, 목표상황, 공유해야 할 코드및 데이터 등 백엔드와 프론트의 소통의 창고로 노션의 활용도를 높였다. 트렐로는 최대한 간소화여 일의 진행사항을 체크하고 필요 사항만 적는 용도로 사용하니, 오히려 일의 진행을 한 눈에 파악하는데에 도움되었다.

### 내가 부족했던 점

**:: 너무 조급한걸..**
1차보다 압박감은 덜 했지만, 이번 프로젝트는 생각보다 기능구현을 많이 하지 못했고, 내가 맡은 파트가 백엔드와 데이터 교류가 적어서 그런지 1차 보다 더 조급함을 느꼈던거 같다. 이번에도 최선을 다 했다고 스스로 위안을 해보지만, 내가 목표했던 바를 해내지 못해 아쉬움이 많은 프로젝트이다. (기업협업 끝나고 못다한 기능구현 꼭 해야겠다. 정현님 제가 곧 찾아갑니다 ㅋㅋㅋ)
또한 1차프로젝트는 협업이란게 무엇인지 느낄 수 있었다면, 2차 프로젝트는 내가 정말 알고 하는게 맞는지? 지금 잘 하고 있는게 맞는지? 겨우 이정도 해놓고 자만하고 있는건 아닌지 계속 나 자신에게 질문하게 만드는 프로젝트였다.

### 내가 배운 점

**::양보다 질**

클래스형이 아닌 함수형 컴포넌트사용, scss가 아닌 스타일 컴포넌드 사용, fetch() 대신 axios 라이브러리 사용 등 이번에도 변화가 있었다. 기존에 배운 내용들을 기반으로 약간 변형된 내용이었을 뿐이라고들 하지만,, 여전히 새로운 지식을 받아들일 때면 멘탈대지진이 온다.
그래도 달라진 점은 있다. 이전에는 줄글의 자료를 찾는데 매달렸다면, 이제는 직접 코드를 쳐보고 어떤 변화가 있는를 직접 확인한다는점? 콘솔을 통해 실행여부 등을 확인하려 한다는점? 이 방법말고 다른 방법은 없을까 한 번더 생각해본다는거?
지난 한 달은 아는 게 많아 졌으면 좋겠다는 생각을 했다. 하지만 지금은 지금 알고 있는 지식들이 제대로 알고 치는 코드였으면 하는 생각이 많이 든다.
이럴때 쓰는거 맞죠? Do the thing right🤦🏻‍♀️ ㅋㅋㅋㅋㅋ

자바스크립트 공부를 놓지 말아야겠다. 하핳. 앞으로 어떤 방향성을 가지고 공부해야할지. 왜 그 공부해야하는지 계속 생각하며 공부해야겠다.
