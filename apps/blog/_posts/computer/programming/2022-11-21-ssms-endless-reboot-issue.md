---
title: "SSMS 설치 시 계속해서 리부트를 요구하는 문제"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2022-11-21. 16:35
# last_modified_at: 2022-11-21. 16:35
---

## SSMS 설치 시 계속해서 리부트를 요구하는 문제

---

'microsoft odbc driver 17 for sql server a previous installation required a reboot' 가  
아무리 컴퓨터를 껐다켜도 해결되지 않는다면,  

레지스트리 편집기에서,  
`컴퓨터/HKEY_LOCAL_MACHINE/SYSTEM/CurrentControlSet/Control/Session Manager` 에 위치한  
PendingFileRenameOperations 값을 비워주고 다시 설치를 시도한다.  

[참고 링크](https://stackoverflow.com/questions/62261436/how-to-fix-endless-reboot-loop-installing-microsoft-odbc-driver-17-message-a)
