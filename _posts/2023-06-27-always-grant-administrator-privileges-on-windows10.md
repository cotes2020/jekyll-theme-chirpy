---
title: Always Grant Administrator Privileges on Windows10
date: 2023-06-27 13:03 +0900
category: [Environment Settings]
tag: [Windows10]
---

Windows10에서 아무런 추가 절차 없이 항상 관리자 권한을 부여하여 프로그램을 실행시키도록 하는 설정이다.

1. regedit을 검색하여 레지스트리 편집기를 실행한다.

2. <kbd>HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System</kbd>으로 이동한다.

3. <kbd>EnableLUA</kbd> 항목이 없다면 만들고 값을 1로 설정

4. 컴퓨터 재부팅

### Ref.

<https://planelover.tistory.com/307>