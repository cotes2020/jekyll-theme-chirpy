---
title: "마인크래프트 플러그인 개발 - #0 세팅, 프로젝트 만들기"
date: 2021-08-31
categories: Programming Java MinecraftPlugin
---

**확인하세요!**: 이 튜토리얼은 2021년 8월 31일, <u>Minecraft 1.17.1</u>, <u>JDK-16</u>, <u>CraftBukkit-1.17.1</u> 을 기반으로 작성되었습니다.
{: .notice}
{: .text-center}

## 시작
---

일반적으로 **플러그인(Plugin)**은 프로그램에 새 기능을 추가하기 위해 만들어집니다.  
우리는 마인크래프트 서버에 새로운 기능을 추가하기 위해, 마인크래프트 플러그인을 만들어 봅시다.

마인크래프트는 **자바(Java)** 라는 프로그래밍 언어로 만들어졌습니다.  
때문에 마인크래프트 버킷도 **자바** 로 만들어졌고, 저희가 만들게 될 플러그인 역시 **자바** 를 이용해 만들게 됩니다.

그렇다면 먼저 **자바** 를 설치해 봅시다.  

마인크래프트 1.17 이전까지는 **자바 7** 또는 **자바 8** 을 사용했지만,  
마인크래프트 1.17 버전부터는 **자바 16** 을 사용합니다.  
**[[참고 - 마인크래프트에 자바 16 도입, 무엇이 달라지나?](https://thecraftdaily.com/ko-kr/minecraft-java-edition-now-uses-java-16/)]**

이 튜토리얼에서는 마인크래프트 1.17.1 버전을 다루기 때문에,  
이 **[[링크](https://www.oracle.com/java/technologies/javase-jdk16-downloads.html)]**에서 자신의 컴퓨터 운영체제에 맞는 Installer (.exe 파일)를 받아 **자바 16** 을 설치해주시면 되겠습니다.  
저 같은 경우, Window 10 을 사용하고 있기 때문에 Windows x64 Installer 를 통해 자바를 설치하였습니다.


`Java` 를 모두 설치했다면, 이제 `Java` 를 다룰 통합개발환경(IDE)를 설치해 봅시다.





저는 Visual Studio Code를 통해 개발해보도록 하겠습니다.

![0001](https://user-images.githubusercontent.com/55438621/131517438-80a49f3a-cca4-440f-bcdb-c9d27ab320f6.png)
![0002](https://user-images.githubusercontent.com/55438621/131517379-bbed3d8a-bc83-4a3b-9c9f-7c2d9de2aec4.png)
![0003](https://user-images.githubusercontent.com/55438621/131517383-718cb6cb-8f12-4499-b3dd-a42b52189615.png)
![0004](https://user-images.githubusercontent.com/55438621/131517385-1eac675d-7c81-41bf-9a7b-4396d18e9ec8.png)
![0005](https://user-images.githubusercontent.com/55438621/131517392-bc36ca18-6994-4001-96ed-22270767e621.png)
![0006](https://user-images.githubusercontent.com/55438621/131517394-71329fae-0579-4794-af04-b2c67035a211.png)
![0007](https://user-images.githubusercontent.com/55438621/131517397-94e52bba-08e1-43c9-bac2-3b951538a218.png)
![0008](https://user-images.githubusercontent.com/55438621/131517400-382f8e4a-eca9-4926-b099-e7b0181b22e2.png)
![0009](https://user-images.githubusercontent.com/55438621/131517401-600a45d4-6c8f-4033-ae6c-875133c74201.png)
![0010](https://user-images.githubusercontent.com/55438621/131517402-1842d06c-a598-49fc-ba0b-18916f608c07.png)
![0011](https://user-images.githubusercontent.com/55438621/131517404-53c6d55c-e628-460e-99b0-071d14bee506.png)
![0012](https://user-images.githubusercontent.com/55438621/131517406-54a1bf30-c315-4ac4-952c-fa53b88d4b6e.png)
![0013](https://user-images.githubusercontent.com/55438621/131517408-c5d3aa4b-dd11-477b-b141-f45920c71237.png)
![0014](https://user-images.githubusercontent.com/55438621/131517410-4f473850-4807-4906-9980-6db1f329150d.png)
![0015](https://user-images.githubusercontent.com/55438621/131517412-69cb9594-fc23-46be-9136-53a37441c96e.png)
![0016](https://user-images.githubusercontent.com/55438621/131517414-f6239061-87ac-4ff9-be12-4ed4b0dd532b.png)
![0017](https://user-images.githubusercontent.com/55438621/131517416-c829f9a5-a017-41c1-9bb7-a84756a013a1.png)
![0018](https://user-images.githubusercontent.com/55438621/131517419-e430d8b5-8ea1-4046-a7a5-2c900913cf75.png)
![0019](https://user-images.githubusercontent.com/55438621/131517422-b8e0bfe7-3d48-4dd1-ad64-ceadc70fa0ca.png)
![0020](https://user-images.githubusercontent.com/55438621/131517423-ac69f0e7-f434-41ce-91cb-042b60da2e6e.png)
![0021](https://user-images.githubusercontent.com/55438621/131517427-9d7454cf-8161-46a3-9d20-09ed805a32e1.png)
![0022](https://user-images.githubusercontent.com/55438621/131517429-4f178d29-c025-46d7-be7f-6e4d2204e2e9.png)
![0023](https://user-images.githubusercontent.com/55438621/131517430-93185bb9-dbf1-426e-8239-8ffa3f2619c7.png)
![0024](https://user-images.githubusercontent.com/55438621/131517432-457095bb-23b2-4333-8af2-184cfd612c64.png)
![0025](https://user-images.githubusercontent.com/55438621/131517435-3b037f93-bc57-4229-b9ca-3f659525efd6.png)
![0026](https://user-images.githubusercontent.com/55438621/131517437-bdc66c40-6e32-4025-93ef-a83e1a84904e.png)


## Hello Blog!
---