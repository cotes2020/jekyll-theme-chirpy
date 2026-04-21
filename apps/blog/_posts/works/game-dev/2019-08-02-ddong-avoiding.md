---
title: "똥 피하기"
# description: ""
categories: [작업물, 게임 개발]
tags: [작업물, 게임 개발, 유니티]
image: "/assets/img/post/works/ddong-avoiding/190802-000013.png"
hidden: true

date: 2019-08-02. 18:40:00
# last_modified_at: 2023-05-10 13:23: 사진 찾기, 글 마크다운 문법으로 수정
# last_modified_at: 2023-08-26. 10:54
# last_modified_at: 2023-11-17. 01:50
last_modified_at: 2024-08-29. 22:21
---

{% include custom/common/old-post.html %}

여담  

작년부터 올해까지, 똥 피하기 게임 개발을 중단하기까지 의 이야기  

처음으로 만들어 본 게임, 좀 부르기 뭐한 똥 피하기.  
작년에 조금 만들고 방치했던 것을, 올해 게임 개발 공부를 다시 시작하면서 프로젝트를 꺼내봤습니다.  

​프로젝트를 열어보지 않는 동안, 유니티 버전을 이리저리 바꾸다보니 콘솔창에는 빨간 에러들로 가득했습니다.  

첫 게임을 차마 삭제하기는 싫어서, 한 번 고쳐보기로 마음먹었습니다.  
에셋들을 없에보고, 유니티 버전도 다시 바꿔보고, 많은 시간을 보낸끝에 결국 에러들을 모조리 없앴습니다.  

건강해진 프로젝트를 이리저리 만지작거리다보니, 문득 순수한 마음으로 플레이스토어에 올려보고 싶었습니다.  
작년 초, 구글에 몇 만원주고 플레이스토어 개발자 등록를 미리 해뒀기때문에, 앱 등록은 쉬울것이라 생각했습니다.  

유튜브에서 하라는 대로, 열심히 따라했습니다. 순조롭게 진행되는 줄 알았습니다.  
하지만 APK 파일을 업로드하니, 갑자기 빨간 오류가 뜹니다.  

플레이스토어 정책이 바꿔면서, 기존에는 게임이 32비트만 지원해도 출시가 가능했던 것이,  
이제는 32/64비트 지원, 그리고 앱 번들까지 준비해야 한답니다.  

이게 무슨소리인가 싶어 검색해봤습니다.  

[기존 앱에 App Bundle 적용하기](https://eso0609.tistory.com/)  
흠.. 완벽히 이해하지는 못했지만, 아무튼 저런거랍니다.  

![KakaoTalk_20190802-172723076](/assets/img/post/works/ddong-avoiding/190802-000000.png)
![KakaoTalk_20190802-172723413](/assets/img/post/works/ddong-avoiding/190802-000001.png)
![KakaoTalk_20190802-172723962](/assets/img/post/works/ddong-avoiding/190802-000002.png)

유튜브에서 하라는 대로, 힘들게 따라했습니다.(안드로이드 NDK에서 해맴)  

결국 등록심사 신청에 성공했고, 앱 출시로 이어졌습니다!  

![SE-44778eed-8525-416e-9700-eccff336d00b](/assets/img/post/works/ddong-avoiding/190802-000003.png)
![KakaoTalk_20190802-172724628](/assets/img/post/works/ddong-avoiding/190802-000004.png)
![KakaoTalk_20190802-172722136](/assets/img/post/works/ddong-avoiding/190802-000005.png)

덕분에 친구들의 영혼가득 리뷰들을 잔뜩 받았습니다..  

![SE-f3e358b0-1abc-4cd8-af42-7c669763fce9](/assets/img/post/works/ddong-avoiding/190802-000006.png)
![SE-3826f3aa-8155-42e6-bdeb-40821dd936cd](/assets/img/post/works/ddong-avoiding/190802-000007.png)
![KakaoTalk_20190802-172721635](/assets/img/post/works/ddong-avoiding/190802-000008.png)

... 만 얼마지나지않아 이용약관을 넣지않았다고 삭제됬습니다. ㅜㅜㅜㅜ  

구글 콘솔에 ( 앱 등록하려고 할 때 ) 이용약관을 넣으라고 경고한걸 무시한 탓입니다.  
게임 출시하시려는 분들은 조심하시기 바랍니다.  

이번에는 새로 리메이크를 해보려다가, 금방 그만뒀습니다. 똥 피하기는 이제 지겨웠기 때문입니다.  
너무 '처음' 에 신경썼나봅니다.  

... 라는 이유로 똥 피하기 게임은 더 이상 만들지 않게되었습니다.  
사실 놓아준지 좀 지나긴했지만, 게으름피우다 글 쓰는게 늦어졌습니다.  

아래는 플레이 영상입니다.  
게임 실행 후 녹화를 했어야하는데, 바탕화면에서 녹화를 시작해서 세로화면으로 녹화됬네요.  

{% include embed/youtube.html id = "gbjy5KcnnGU" %}
{% include embed/youtube.html id = "ieyDD-Ot4pE" %}​

영 좋지않은 퀄리티에 재미까지없는 이 게임을 하고 싶으신 분은 없겠지만, 그래도 한 번 해보고 싶으시다면 [링크](https://drive.google.com/file/d/1--B2vzoravEZ85nsVj7hXdvNQDiFt4Jn/view?usp=sharing)에서 다운을 받으시면 되겠습니다.  
겸사겸사 유니티 프로젝트 파일도 넣어봤습니다. 코드는 보지말아주세요. 부끄  

AvoidingDDong 폴더는 위에서 소개하던 게임의 유니티 프로젝트 파일  
AvoidingDD 폴더는 리메이크 하던 유니티 프로젝트 파일  
AvoidingDDong.Apk는 그동안 테스트하면서 만들어온 테스트버전  
AvoidingDD.Apk는 앱 출시를 시도했던 버전  
( AvoidingDD.Apk가 안되면 테스트 버전으로.. )  

아이디어  

학교에서 나름 틈틈히 기획했던 것들  
![SE-586de145-4ea7-453e-91af-488725559048](/assets/img/post/works/ddong-avoiding/190802-000009.jpg)
![KakaoTalk_20190802-180345284](/assets/img/post/works/ddong-avoiding/190802-000010.jpg)
![KakaoTalk_20190802-180346814](/assets/img/post/works/ddong-avoiding/190802-000011.jpg)

시커먼 노트에 연한 샤프로 끄적끄적한 것을, 화질 안좋은 카메라로 찍으면 나오는 사진.  
그닥 영양가있는 기획은 아니라 그냥 글로 적겠습니다.  

요약하면 똥 피하기 게임의 업그레이드 버전, 똥피하기++  
기존 똥 피하기 게임에 여러가지 컨텐츠? 를 더 집어넣어 봤습니다.  

1. 좌우만 움직이는 조작에 점프를 더함
2. 위 뿐만아니라, 좌/우, 아래에서 날라오는 투사체
   - 피해야하는 투사체 말고도 아이템, 골드도 날라옴
3. 다소 혐오스러운? 똥 대신 여러 투사체 및 배경의 스킨을 만들었습니다.
   - EX) 중세 성 스킨을 꼈다면, 바위 / 화살 / 검 모양의 투사체
   - EX) 해적선 스킨을 꼈다면, 폭탄 / 해골 / 보물 모양의 투사체
   - EX) 판타지 스킨을 꼈다면, 슬라임 / 파이어볼 / 열매 모양의 투사체
   - EX) 할로윈 스킨을 꼈다면, 박쥐 / 호박 / 사탕 모양의 투사체
   - EX) 설산 스킨 - 눈덩이, 현대 스킨 - 총알 / 로켓
4. 여러가지 게임 모드
   - 위에서 떨어지는 투사체를 피하는 클래식 모드
   - 위에서 떨어지는 투사체를 피하면서 탑을 오르는 탑 모드
   - 플레이어 주위만 조금 보이는 안개 모드 ( 할로윈 모드 )
   - 테트리스 블럭이 떨어지는 테트리스 모드
   - 투사체 등 보스의 공격을 피하면서, 바닥에 있는 폭탄을 밟아 보스를 쓰러뜨리는 보스 모드
   - 온라인으로 한 명은 투사체를 던지고, 한 명은 피하는 공/수 모드
   - 서로를 밀치고, 방해하면서 최후의 1인을 가리는 서바이벌 모드 ( 배틀로얄 )

---

![0017](/assets/img/post/works/ddong-avoiding/190802-000012.png)
![0017](/assets/img/post/works/ddong-avoiding/190802-000013.png)
![0017](/assets/img/post/works/ddong-avoiding/190802-000014.png)
![0017](/assets/img/post/works/ddong-avoiding/190802-000015.png)
![0017](/assets/img/post/works/ddong-avoiding/190802-000016.png)
