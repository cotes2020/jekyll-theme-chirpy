---
title: "간단한 날씨 위젯 코딩 #1 [검색창과 검색, 위치 버튼 생성 / Java Script 미니 프로젝트]"
categories:
- MiniProject
- WeatherWidget
tag: [MiniProject, JS, HTML]
author_profile: false
sidebar:
    nav: "docs"
search: true
---

안녕하세요! 블로그를 만들고 첫 게시물이네요 ㅎㅎ

9월에 전역을 하고, 복학하기 전에 어떤 활동을 할까 생각을 해보았고, 자바 스크립트로 간단한 프로젝트들을 만들어보자고 결심했습니다.

제가 해 볼 첫 번째 프로젝트는 날씨 위젯 코딩입니다😉

***


## 구현 목표

1. 위치 검색 기능과 사용자 위치를 기반으로 현재 날씨 출력
2. 날씨에 맞는 이미지 출력 기능 (화창한 날씨에는 ☀️, 비가 오는 날씨에는 🌧️ 등등)
3. 기온, 습도, 풍속 출력 기능



먼저 **index.html / script.js / style.css** 3개의 파일과 **assets**라는 폴더를 만들어주고, assets에 404 에러 사진을 하나 넣어줬습니다.

검색, 위치 아이콘을 넣으려면 Font Awesome 사이트에서 자신의 Kit을 만들고 본인 코드에 붙여 넣어야 합니다!

[Font Awesome Kit 코드 가져오기](https://blog.naver.com/dlwndbs5/223575661946) ⬅️이 글을 참고해서 킷 생성을 먼저 해주세요.<br>

***

## 검색창과 검색, 위치 버튼 생성

먼저 검색창을 만들어주고, 그 옆에 검색 버튼, 위치 버튼을 생성해주도록 하겠습니다!

아래는 코드입니다🥳

***

### index.html

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <meta http-equiv='X-UA-Compatible' content='IE=edge'>
    <title>Weather Widget</title>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel='stylesheet' type='text/css' media='screen' href='style.css'>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="search-box">
                <input type="text" placeholder="Search" class="input-box">
                <button class="fa-solid fa-magnifying-glass"></button>
                <button class="fa-solid fa-location-dot"></button>
            </div>
        </div>
    </div>
    <script src='script.js'></script>
    <script src="https://kit.fontawesome.com/당신의 킷 주소.js" crossorigin="anonymous"></script>
</body>
</html>
```

기본 html 코드에서 title을 바꾸고, Font Awesome Kit 스크립트를 입력해주었습니다!

그리고 검색 아이콘과 위치 아이콘을 Font Awesome에서 찾아서 넣어줬습니다.



#### 구현 결과

![image-20241215184011882](../assets/img/posts/2024-12-15-weather_widget_1/image-20241215184011882.png){:style="border:1px solid #ababab; border-radius: 10px;"}

실행 결과입니다. html만 코딩하니 너무 초라합니다...😭

***

### style.css

css 코딩을 통해서 아이콘들을 보기 좋게 변경해주었습니다.

이해가 쉽게 가도록 주석을 통해 코드 설명하겠습니다😉

```css
/* 모든 요소 */
* {
    margin: 0; /* 기본 마진: 0 */
    box-sizing: border-box; /* 요소의 너비와 높이에 패딩과 테두리를 포함 */
    border: none; /* 기본 테두리 제거 */
    outline: none; /* 포커스 시 표시되는 테두리를 제거 */
    font-family: sans-serif; /* 글꼴 설정 */
}

/* body 요소 */
body {
    min-height: 100vh; /* 최소 높이를 뷰포트 높이의 100% */
    display: flex; /* flexbox 레이아웃 */
    justify-content: center; /* 수평 방향, 가운데 정렬 */
    align-items: center; /* 수직 방향, 가운데 정렬 */
    background-color: #000; /* 배경 색상 설정 */
}

/* 컨테이너 */
.container {
    width: 400px; /* 너비 400픽셀 */
    height: min-content; /* 컨테이너 높이 = 컨텐츠의 최소 높이 */
    background-color: #fff; /* 배경 색상 설정 */
    border-radius: 12px; /* 모서리를 12픽셀 둥글게 */
    padding: 28px; /* 내부 여백을 28픽셀로 */
}

/* 검색 박스 */
.search-box {
    width: 100%; /* 너비: 부모 요소의 100% */
    height: min-content; /* 검색 박스의 높이 = 컨텐츠의 최소 높이 */
    display: flex; /* flexbox 레이아웃 */
    justify-content: space-between; /* 수평 방향, 요소들을 균등한 간격으로 정렬 */
    align-items: center; /* 수직 방향, 가운데 정렬 */
    gap: 12px; /* 요소들 사이 12픽셀 간격 */
}

/* search-box 입력 필드 */
.search-box input {
    width: 84%; /* 너비: 부모 요소의 84% */
    font-size: 20px; /* 글꼴 크기 */
    text-transform: capitalize; /* 입력되는 텍스트의 첫 글자를 대문자로 설정 */
    color: #000; /* 글씨 색상 */
    background-color: #e6f5fb; /* 배경 색상 설정 */
    padding: 12px 16px; /* 내부 여백: 상하 12픽셀, 좌우 16픽셀 */
    border-radius: 14px; /* 모서리를 14픽셀 둥글게 */
}

/* placeholder */
.search-box input::placeholder {
    color: #000; /* 플레이스홀더 텍스트 색상 설정 */
}

/* 버튼 */
.search-box button {
    width: 46px; /* 버튼 너비 설정 */
    height: 46px; /* 버튼 높이 설정 */
    background-color: #e6f5fb; /* 배경 색상 설정 */
    border-radius: 50%; /* 모서리를 50% 둥글게 => 원형으로 */
    flex-shrink: 0; /* flexbox 레이아웃에서 버튼의 크기를 줄이지 않도록 설정 */
    cursor: pointer; /* 마우스 커서를 포인터로 변경 */
    font-size: 20px; /* 글꼴 크기 설정 */
}

/* 버튼 (마우스 올렸을 때) */
.search-box button:hover {
    color: #fff; /* 마우스를 올렸을 때 버튼의 텍스트 색상: 흰색 */
    background-color: #ababab; /* 마우스를 올렸을 때 버튼의 배경 색상: 회색 */
}
```

#### 구현 결과

![image-20241215185740037](../assets/img/posts/2024-12-15-weather_widget_1/image-20241215185740037.png){:style="border:1px solid #ababab; border-radius: 10px;"}

제법 보기 좋아진 것 같습니다!

다음 포스팅에서는 날씨 이미지를 넣어보겠습니다. 수고하셨습니다~!
