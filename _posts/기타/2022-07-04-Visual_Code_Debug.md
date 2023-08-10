---
layout: post
title:  "Visual Code 디버깅 모드 "
date:   2022-07-04 22:17:24
categories: [Forensic, Browser]
---

# Visual Code Debug 모드 사용하기 


![Untitled (1)](https://user-images.githubusercontent.com/46625602/177163058-7ac1bcd7-d31f-4465-979e-add47280198d.png)

인자값 전달하고 싶으면 `launch.json` 파일을 만들어서 그 안에 `args["인자","인자2"]`, 이런 식으로 넣어 주면 된다. 

디버깅 실행 단축키는 ``F5`` 다

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 현재 파일",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args" : ["export","-f","password.csv", "--password_crack"],
        }
    ],
    "variablePresentation":{
        "function": "hide",
        "special": "hide",
    }
}
```

---
**[참고자료]**

* 