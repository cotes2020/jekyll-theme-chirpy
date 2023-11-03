---
title: 엘리스 1차 프로젝트 회고
date: 2023-04-21
categories: [reviews]
layout: post
tags: [elice]
---

# 링크모음

팀노션
[https://www.notion.so/Car-Car-Car-406e6320166945aea1137606331d5897](https://www.notion.so/Car-Car-Car-406e6320166945aea1137606331d5897?pvs=21)

팀깃랩
[https://kdt-gitlab.elice.io/sw_track/class_04/web_project/team02/project](https://kdt-gitlab.elice.io/sw_track/class_04/web_project/team02/project)

팀피그마

[https://www.figma.com/file/LUKbstQZGyvnseuLvpdPU1/Elice_SW_Project_Team2?node-id=34-2&t=2ZMtCSozarIMCF7o-0](https://www.figma.com/file/LUKbstQZGyvnseuLvpdPU1/Elice_SW_Project_Team2?node-id=34-2&t=2ZMtCSozarIMCF7o-0)

# 소감

처음으로 다른 사람들과 개발 협업 프로젝트를 진행해 보았다.

1차 스터디로 진행했던 개인 프로젝트 진행과는 몇몇 다른 점이 있었다.

## 1. **협업으로 따라오는 깃의 심화 실력 필요**

git을 본격적으로 사용하기 시작했다. 그 전까지는 내 개인 저장소에서만 clone 후 형상관리하는 용도로만 사용했는데, 이제 나의 커밋과 브랜치가 다른 사람들의 작업에도 반영이 되다보니 merge를 할때 마다 에러메시지가 매번 뜨면 덜컥 식은땀이 흐르고… 겁이 나고…

첫 merge 는 백엔드 팀원들과 엘리스랩에 모여서 셋이서 순서대로 진행했는데 마지막 순서였던 나에게 충돌이 많이 일어나서 1시간동안 다함께 씨름했다.
너무 긴장해서 마치 영화처럼 마지막 엔터를 칠땐 마치 영화처럼 이마에서 땀방울이 뚝 하고 노트북에 떨어졌다. 심리적 압박으로 인해 실시간으로 수명이 깎이고 있는게 느껴졌다.

- 무서운 느낌표와 무서운 에러메시지

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/1.gif)

![20171227_5a4396f205dd2](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/2.png)

그 다음날 바로 유튜버 엘리님의 git 강의를 구매해서 봤는데, 와 진작 볼걸 하는게 생각이 들었다. 다른 무료 강의들에선 듣지 못했던 깔끔한 자료와 현업에서 필요한 팁(a.k.a : 협업에 필요한 팁)이 이번 프로젝트에서 내 수명을 지켜주었다.

특히 내가 제일 무서워했던 rebase 와 confilct 메시지를 git stash 로 keep 해둘수 있단 걸 배워서 정말 좋았다. 이번엔 깃랩을 써서 github 기능을 써보진 못했지만 fork 라는 기능도 안전하게 협업할 수 있는 방법 같아서 꼭 써보고 싶어졌다.

- 깃 필기, 강의 사이트
  ![20171227_5a4396f205dd2](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/4.png)
  ![20171227_5a4396f205dd2](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/5.png)

엄청난 브랜치의 흔적 ㄷㄷㄷ

## 2. **WSL 세팅**

깃을 쓰면서 터미널을 어쩔 수 없이 많이 쓸수 밖에 없어졌는데, 내 터미널이 너무 못생기고 가독성이 안좋아서.. 참을수가 없었다. 그 이유는…

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/6.png)

**첫번째로**,
git 을 사용할때 내가 어느 브랜치에 있는지 바로바로 확인이 안되는 점이 너무 불편했다. 아직 초보자라서 신경 쓸 점이 한 두가지가 아닌데 급하게 다른 사람과 함께 모니터를 봐야하는 상황이 생기면 허둥지둥하느라 내가 어느 브랜치에 있는지 확인하는 과정이 너무 번거로웠다.

**두번째로**,
우리 팀원들은 6명중 4명이 Mac을 사용하고 있었는데 사소한 명령어가 내 윈도우 터미널에서 돌아가지 않아 불편한 일이 계속 생겼다. 그럴때마다 빌게이츠와 현피 뜨고 싶은 심정이 강하게 들었다.

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/7.png)

**세번째로**,
난 순수 미술을 전공했다. 시각적으로 아름다운 것을 보는걸 좋아하고, 조형적 감각이 뛰어남을 자타공인으로 인정받았다. 그런 나에게 이렇게 심미적인 감수성이 부족한 글자 뭉치들을 하루종일 보는 것은 제법 고문이었다.

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/8.png)

눈을 찌르고 싶어지는 화면

그래서 예전에 시도하다가 wsl 설치에서 실패했던 노마드코더 윈도우 세팅강의를 다시 시도했다. 작년에는 따라하고 싶어도 컴퓨터 지식이 너무 부족하니까 wsl 설치에서 계속 오류가 나서 보류했었다. 올해는 엘리스를 수강하면서 wsl를 미리 설치해두었기 때문에 수월하게 zsh 설치부터 진행할 수 있었다.

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/9.png)

안녕하세요 니꼴라스입니다

확실히 wsl, zsh, powerlevel10k 까지 설치하고 나니 현업 개발자들이 하루종일 컴퓨터를 볼 수 있는 이유를 알 수 있게 됐다.
터미널에 내가 어느 브랜치에 있는지도 실시간으로 보이고, cd 나 git branch 를 입력하고 tab을 누르면 자동으로 목록을 띄워주고 이동할 수도 있어졌다. 거기다 귀여운 깃랩이나 깃허브 아이콘까지 띄워주니 정말 살거같았다. 내 안의 미술가를 지켜줘서 고마워요 zsh, powerlevel10k 제작자 분들😭

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/10.png)

**wsl 을 설치하면 인싸도 될 수 있다.**

![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/11.png)

윈도우에선 깔 수 없었던 재밌는 패키지들을 많이 설치할 수 있게됐다.

기존까지는 choco, npm 두가지 패키지 관리자만 써봤는데 이번엔 apt라는 새로운 패키지 관리자를 사용해보면서 gui보다 cli가 더 편하다는게 어떤 말인지 이해할 수 있게 됐달까.

윈도우를 쓸때 항상 exe 파일이나 설치파일을 브라우저에서 다운로드한 뒤 설치프로그램을 실행하는게 상당히 귀찮았는데 이제 엔터만 치면 모든 프로그램을 설치할 수 있어졌다.
아마 컴퓨터를 더 좋아하게 될거같다!

==============================================

> > 2023.05.14 추가

후..^^
![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/12.png)

> > 2023.09.25 추가

혹시 이 글을 보고있는 개발자 지망생들이 있다면 꼭 맥북을 사자

## 3. AWS S3

상품 기능을 맡아본 김에 예전부터 관심있었던 이미지 호스팅 서비스를 사용해보기로 했다.

백엔드 담당 코치님께 보통 어떤 서비스를 많이 쓰는지 여쭤봤더니 보통 S3를 많이 쓴다고 하셔서 ‘요즘 AWS가 그렇게 핫하다던데!’ 하고 나도 한번 발을 살짝 담궈볼까 하고 맛을 봐봤다.

1.  **aws의 ux 디자인.. 이게 최선인걸까?**
    처음 가입하고 사이트를 살펴보면서 처음 든 생각은 이래서 개발자들에게 시험까지 보게 하는구나 였다. 해외 사이트여서 그런지 aws 콘솔의 ux 디자인은 그다지 좋아보이지 않았다..^^ 번역도 절대 잘 되어있다고 할 수 없었고.. 공식문서 한번 읽고 바로 개발 시작할 수 있는 수준이 아니었다. 괜히 aws에서 게속 시험 준비 무료로 시켜준다고 꼬드긴게 아니었다.
    내가 써보니 무료로 가르쳐주지 않으면 “에잇 안써!” 하고 개발자들이 안 쓸거 같았다.
    물론 aws의 장점은 저렴한 가격이라고 하니 aws에게 사용성이 1순위로 투자할 부분은 아닌거 같지만.. 그래도 너무 했다. 번역과 ux가 잘되어있었다면 기능 구현시간이 절반 이상 줄었을거 같다.
2.  **IAM(Identity and Access Management)**
    전에 gcp를 살짝 써보면서 IAM이라는 글자를 봤던거 같은데 여기서 만나니 반가웠다. 그 때는 생성해 보진 않았는데 aws에서는 필수로 생성해야만 서비스를 이용할 수 있어서 공부하며 만들어보았는데 앞으로도 꽤 도움이 될 거 같았다.
    원격에 있는 컴퓨터를 돈을 주고 빌리는 것이다 보니, 권한을 소홀하게 관리하거나 다 풀어놔 버리면, 지구 반대편 누군가가 내 저장소로 비트코인을 채굴하고.. 프로모션을 돌려서.. 1000만원 고지서가 날아올수도 있다고한다. (코치님의 실제경험담을 들었다 😭) 보안이란 글자가 나오면 나는 왠지 영화에 나오는 해커처럼 멋있어 보여서 드릉드릉하는데 그런 점에서 재미있게 공부해볼 수 있었다.
3.  **AWS CLI**
    2번에서 배운 리눅스를 실전에서 열심히 써봤다. 일단 내 linux 컴퓨터에 python3, pip로 aws cli를 설치하고 access-key-ID, secret-access-key 를 적어주면 된다. 이 부분이 너무 오래 걸려서 코치님들께도 여쭤봤는데 aws가 정말 개복치마냥 예민한 친구라 하나라도 다르면 적용이 되지 않는다고 했다 🥹 그래서 결국 새벽까지 싸우다가 내가 이겼다 크크

    [aws](https://www.notion.so/aws-f3f5a4699a244164917190e1081a90ac?pvs=21)
    <br/>
    [뉴맨코드](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/newcarcar_collection.json)

    <br/>

    [이미지업로더.js](https://github.com/YubinShin/carcar/blob/dev-BE/back/src/utils/aws-uploader.js)

    ```bash
    sudo mkdir ~/.aws
    sudo vim ~/.aws/credentials
    ```

4.  **aws-sdk**

    ```
    const AWS = require("aws-sdk");
    const multer = require("multer");
    const multerS3 = require("multer-s3-transform");
    ```

    이렇게 npm 패키지를 설치하고 로직을 짜주니 내 aws s3 버킷에 사진들이 주르륵 올라가있었다. 만드는 과정을 엄청 헤맸지만 다 만들고 나니 아주 뿌듯했다.

    앞으로 내 클라우드에 무거운 이미지들을 직접 올려놓지 않아도 된다니!

    경로 때문에 고생할 일이 없다니!

    이것이 바로 그 Making the better world가 아닐까 ^^

    [드라마 실리콘밸리 : Making the world better place~ 를 외치는 몇십명의 개발자들ㅋㅋㅋ](https://www.youtube.com/watch?v=B8C5sjjhsso)

    ![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/13.png)

    ![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/14.png)

    버튼 1개만 누르면 로직에 맞춰서 이미지 98개가 db 데이터와 매핑되면서 자동으로 업로드된다?!

    POSTMAN의 CLI 심화버전인 newman과 결합하니 나는 더이상 데이터 업로드가 두렵지 않아졌다. 😆

## 4. 예상치 못했던 미완성..

프로젝트 발표 전날, 원래 계획은 분명 엘리스랩에 모여서 같이 테스트를 해보는 날이었는데 왠지 아직도 FE 파트 분들이 분주했다. 아직 DOM 코드를 다 못 짜셨다고 하셔서 도와드리겠다고 하고 브랜치를 받아와서 확인해봤다.일단 요청한 페이지 하나를 얼른 만들어서 드리고 또 어떤 게 안되있는지 확인해봤는데 api 연결이 아예 안된 파트도 있었고, dom도 완성되지 않은 페이지가 대부분이었다.
이게 무슨 일이지..?! 분명 우리팀은 매일 오전 10시에 6명이 모여서 데일리 스크럼도 진행했고, 팀장님을 비롯해 팀원들 대부분이 항상 엘리스랩에서 같이 코딩을 했다고 들었다. 나는 내 개인 기능 구현을 충실히 완료한 후 프로젝트 기한이 남은 3일간 매일, 같이 상품파트를 페어로 진행하는 FE 분께 도와드릴 부분이 없는지 여쭤봤었다. 😭

그런데 발표전날에 이 사태를 알게되다니 허걱.. 팀장을 따로 불러서 현 상황을 알고 있는지 상의한 후에 일단 엘리스랩이 닫는 밤 10시가 30분 정도 남았으니 일단 프론트분들도 작업 중지하고 merge 하기로 했다.

이야기를 들어보니 지금까지 한번도 다같이 dev-FE에 merge 한 적이 없어서 충돌이 엄청나게 일어나고 있었다.

일단은 프로젝트는 미완성으로 결론이 났고, 헤어지기 전에 FE 분들께 “고생많으셨습니다 그런데 어쩌다가 이렇게 되었을까요…?” 라고 물어보니 이유로는 1) 협업경험이 많이 없음, **2) 다른 분들도 계속 바빠 보이셔서 질문이 있어도 드리기 어려웠고, 자신의 작업에도 바빠서 서로의 작업 상황을 알기가 어려웠다.** 로 정리되었다.

FE 분들은 새벽까지 회의실을 빌려서 작업된 부분까지 merge를 진행하셨다고 했고, 팀장님이 부재해서 내가 대신 VM에 FE분들의 마지막 커밋을 배포해드렸다. 🥹유빈님 덕분에 프론트파트 배포가 되었다고 정말 감사하다는 FE 팀원분들을 보며, 완성한 상태로 서버에 올라갔다면 얼마나 좋아하셨을까 싶기도해서 마음이 아팠다.

발표 후 곰곰히 생각해보면서 어떻게 하면 이런 사태가 다음엔 일어나지 않을까 하다가, 최근에 봤던 엘리님의 CI/CD 영상이 떠올랐다. 그 중 특히 **CI** 부분!

### **CI (Continuous Integration : 지속적인 통합)**

[https://www.youtube.com/watch?v=0Emq5FypiMM](https://www.youtube.com/watch?v=0Emq5FypiMM)

    ![Untitled](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-04-21-elice-1st-project-review/15.png)

영상을 보면서 너무 당연하게 그렇지 그렇지 하고 넘어갔던 **주기적인 머지** 부분..😭
주기적으로 머지를 안할수도 있나? 하고 안일하게 생각하던 그 부분이 바로 내 옆에서 일어나고 있었던 문제였던 것이다.

위에 적어둔 FE 분들의 미완성 사유는 바로 이 부분에서 비롯되었다고 생각한다.
**2) 다른 분들도 계속 바빠 보이셔서 질문이 있어도 드리기 어려웠고, 자신의 작업에도 바빠서 서로의 작업 상황을 알기가 어려웠다.**

주기적으로 전체 작업상황에 개개인의 코드가 반영되면 다른 사람의 파트와 내 코드가 잘 상호작용되고 있는지 자연스레 확인할 수 있다. 따로 질문을 하지 않아도 된다. 내 컴퓨터에서 바로 돌려보면 되니까~!

반면, BE 파트는 발표 때 질문이 들어올만큼 MR을 빈번하게 했다. 하루에 3~4번은 항상 있었던거 같다. 내 짐작으론 아마 그래서 작업이 예상보다 3일이나 일찍 끝나지 않았나 싶다.

아마 이 부분이 이번 팀프로젝트에서 얻은 가장 귀한 경험이 아닐까 싶다.

이번엔 팀원으로 참여했지만 곧 2차 스터디때는 나를 믿고 따라와주는 스터디원들을 이끌어야한다.
그때는 꼭.. 빈번한 병합을 내가 진두지휘 해드리리.

==============================

2023.05.15 추가

ㅎㅎ 1주차 FE분들이 작업을 60~70프로 진행하신 상태이나 아직 한번도 PR을 하지 않았다고 하시길래 바로 **머지대작전** 미팅을 잡아서 같이 rebase 도 해결해드리고 1시간만에 후딱 병합해드렸다. 다음날까지 만족도가 높으셔서 아주 뿌듯했다. BE분들과는 이미 쓴맛을 봤던 내가 작업 1일차 부터 PR 해주세요😋 하면서 따라다녀서 그런지 rebase 충돌이 한번도 안나고 아주 수월하게 automatically merge 되고 있어서 아주 뿌듯하다 ^^

정말 값진 경험이었다.
