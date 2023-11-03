---
title: 이미지를 다른 페이지로 넘겨주기 대작전(blob 과 base64)
date: 2023-08-18
categories: [troubleshooting]
tags: [base64, blob, react]
---

## 🤔 Problem

<div markdown="block">
<div markdown="block" style="width: 50%; float: left; clear: right; padding: 0 2% 0 0; ">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/4.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/4.png)
</div>

<div markdown="block" style="width: 50%; float: left; clear: right; padding: 10% 0;border-left: 1px solid lightgray; ">

1. 사용자가 인증하기 버튼을 누르면 사진에 exif에 기록되어있는 위도, 경도가 해당 장소와 일치하는지를 검증하는 페이지

2. 사용자가 이미지를 데이터베이스에 게시하기 전에 사진의 크기를 조절하는 페이지
</div>
</div>
<div markdown="block" style="clear: both;">
</div>
<br/>

프론트엔드 담당자 분들이 서로 다른 로직과 UI를 사용하는 페이지를 연결하고자 하는 상황이었다.

다른 컴포넌트 간 이미지 주소를 전달해야만 하는 상황이었는데, 지속적으로 실패 중이라고 하셨다.

브라우저 단에서 넘겨주는 것이 실패중이라면 AWS s3에 먼저 올리고 해당 사진을 받아와 수정하는 방식을 먼저 제시 드렸으나,

API 요청 횟수를 늘리지 않으면서 비용을 절약해보고 싶다고 하셨고, 알맞은 방법이 있는지 함께 찾아보기로 했다.

Context API 에 blob uri나 base64를 담아 보내다가 실패하는 중이라고 하여 같이 투입되어 문제를 슈팅해보았다.

## 🌱 Solution

### 1차 시도 - blob uri 을 context API 로 넘겨주기(실패)

<div markdown="block" style="width: 80%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/1.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/1.png)
</div>

**"Blob URI는 임시적인 URL이며, 해당 URI는 생성된 후 페이지가 새로고침 될 때마다 변경될 수 있습니다."**

유튜브 영상과 gpt 에게 질문하여 페이지가 새로고침 될 때 Blob URI는 무효화되기 때문에 사용할 수 없게 된다는 걸 깨달았다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/UgCCR-ZBP8w" title="What is a blob URI?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### 2차 시도 - 로컬스토리지에 Base64 를 담아 사용하기(실패)

스택오버플로우를 탐험하면서 context API에 이미지를 담았다는 답변을 찾을 수 없었기에 나는 local storage 측으로 방향을 틀었다. 메가바이트 크기 정도의 이미지를 담으려면 context API의 용량이 부족한가 싶었기 때문이다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/AUOzvFzdIk4" title="Storing Objects with Local Storage in JavaScript" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/2.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/2.png){:width="50%"}

LocalStorage 에 Base64로 인코딩하여 담아 보내는 것을 성공한 줄 알았으나..

### 결과

정훈님과 확인해본 결과 어떤 사진은 모바일에서는 확인되나 웹에서는 파일이 깨지기도 하고…
또 용량이 큰 사진을 넣으면 안보이길래 여러 크기와 확장자의 파일로 시험을 해보았는데, 그 어떤 규칙도 없이 **랜덤하게** 표시가 되거나 안되거나 했다.

안정성이 없는 방식을 고치는 것은 우리의 영역을 넘어섰다는 걸 깨닫고 처음 제시된 방식인 aws s3에 게시 후 PUT 하는 방식을 사용하기로 했다.. ^^

비록 문제 해결에는 실패했지만, blob 객체와 base64 에 관하여 팀원들과 함께 자세히 알아 볼 수 있어 굉장히 의미있는 트러블슈팅이었다.

<br/>

<div markdown="block" style="width: 80%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/3.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-18-blob-base64-react/3.png)
</div>

눈물의 정훈님

<br/>

## 📎 Related articles

| 이슈명                             | 링크                                                                                                                                                                                                                                                           |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 스택오버플로우 로컬스토리지와 blob | [https://stackoverflow.com/questions/67754721/can-i-store-image-as-blob-to-localstorage-or-somewhere-so-it-can-used-in-another](https://stackoverflow.com/questions/67754721/can-i-store-image-as-blob-to-localstorage-or-somewhere-so-it-can-used-in-another) |
