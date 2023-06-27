---
title: Show Seconds in Windows10 Taskbar Clock
date: 2023-06-27 12:52 +0900
category: [Environment Settings]
tag: [Windows10]
---

Windows10에서 화면 오른쪽 아래에 있는 작업표시줄 시계에 초 단위까지 보이도록 하는 설정이다.

1. regedit을 검색하여 레지스트리 편집기를 실행한다.

2. <kbd>HKEY_CURRENT_USER\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Advanced</kbd>로 이동한다.

3. <kbd>ShowSecondsInSystemClock</kbd> 항목이 없다면 만들고 값을 1로 설정

4. 컴퓨터 재부팅

### Ref.

<https://lightinglife.tistory.com/entry/Windows-%EA%BF%80%ED%8C%81-%EC%9C%88%EB%8F%84%EC%9A%B0-%EC%9E%91%EC%97%85%ED%91%9C%EC%8B%9C%EC%A4%84-%EC%8B%9C%EA%B3%84-%EC%B4%88-%EB%8B%A8%EC%9C%84-%ED%91%9C%EC%8B%9C-%EB%B0%A9#:~:text=4%20%EC%9E%91%EC%97%85%ED%91%9C%EC%8B%9C%EC%A4%84%EC%9D%98,%EB%A1%9C%20%ED%91%9C%EC%8B%9C%ED%95%A0%20%EC%88%98%20%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.>