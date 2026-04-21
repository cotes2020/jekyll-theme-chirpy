---
title: "Unity PlayFab 에러"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/20240827-140647.jpg"

date: 2022-11-16. 11:38
last_modified_at: 2024-08-29. 22:29
---

글 계승, 병합.  
`2022-11-16-PlayFab-GooglePlay-Error: 🌔 Unity PlayFab, Google Play 연동 에러`  
`2022-11-18-PlayGamesPlatform: 🌔 PlayFab, Google Play 연동 중 PlayGamesPlatform 를 찾지못하는 경우`  
`2023-02-15-PlayFab-Memory-Leak-Error: 🌔 Unity PlayFab, A Native Collection... 에러`  

## PlayFab + PlayGameServices 0.11.01 ?

---

PlayGameServices 0.11.01 버전 기준,  
예전 버전 자료들을 바탕으로 PlayFab과 Google Play 로그인 연동을 위한 코드 작성 중,

PlayGamesClientConfiguration 과,  
PlayGamesPlatform.Instance.Authenticate(); 등을 찾지 못하는 에러 발생  

PlayGameServices 0.11.01 버전에 구글 플레이 V2 를 사용하게 되면서, 위 내용들이 더이상 지원하지 않는 코드가 됨.  
PlayGameServices ReadMe와 [업그레이딩 문서](https://github.com/playgameservices/play-games-plugin-for-unity/blob/master/UPGRADING.txt)를 참고해 새 버전의 코드를 작성할 수는 있음.  

다만,  
구글 플레이를 통해 PlayFab에 로그인 하고자 하는 경우,  
구글 플레이 서버 인증 토큰을 포함하여 로그인 리퀘스트를 보내야 하는데  

기존 버전에서 토큰을 불러오기위해 사용했던 **PlayGamesPlatform.Instance.GetServerAuthCode();** 가,  
V2 에서 **PlayGamesPlatform.Instance.requestServerSideAccess()** 로 바뀌게 되면서 원하는 값을 받아올 수 없는 것 같음  

때문에 로그인에 계속해서 실패  

→ 해결:  

23/06/01 기준, 아직 해결 방법을 찾지 못함  
0.11.1 버전이라면, 어쩔 수 없이 0.10.14 버전으로 다운그레이드하여 사용 (참고 링크 2, PlayFab 답변 참고)  

[참고 링크 1](https://github.com/playgameservices/play-games-plugin-for-unity/issues/3141)  
[참고 링크 2](https://community.playfab.com/questions/61120/googleoauthnoidtokenincludedinresponse-when-loggin.html)  
[PlayFab + PlayGameServices 참고](https://stealnewspaper.tistory.com/2)  

## PlayGamesPlatform Missing

---

PlayFab과 Google Play 로그인 연동을 위한 코드 작성 중,

PlayGamesPlatform.Instance.Authenticate(); 에서 에러 발생  
PlayGamesPlatform 을 찾지못함.

Assembly 'Assets/ExternalDependencyManager/Editor/1.2.167/Google.IOSResolver.dll' will not be loaded due to errors:

[참고 링크](https://github.com/googlesamples/unity-jar-resolver/issues/441)

→ 해결:  

빌드 세팅이 Window 플랫폼으로 설정되어 있었음  
Android로 바꿔주니 에러 사라짐  

## A Native Collection has not been disposed, resulting in a memory leak. Enable Full StackTraces to get more details

---

에러 로그가 자꾸만 뜬다.  
Play Mode 가 멈춘다거나, 게임 플레이에 이상이 생긴다거나 하는 건 아니지만, 신경쓰인다.  

→ 해결:  

[참고](https://community.playfab.com/questions/65805/a-native-collection-has-not-been-disposed-resultin-1.html)  

Assets\PlayFabEditorExtensions\Editor\Scripts\PlayFabEditorSDK\PlayFabEditorHttp.cs  
파일 내용 일부분에, 참고 링크에 Rick Chen이 남긴 코드를 추가한다.  

이것만의 문제는 아닌지, 가끔 똑같은 에러 로그가 발생하고 있긴 하지만,  
그 빈도가 확연하게 줄어들었다.  

## Google Play Login, ERROR 403 access_denied

---

(PlayFab 관련은 아니지만)  

→ 해결 :
Google Cloud Console - OAuth 동의 화면 - 테스트 사용자 추가  

[참고](https://jeeu147.tistory.com/91)
