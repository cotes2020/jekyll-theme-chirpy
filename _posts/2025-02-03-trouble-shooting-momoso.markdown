---
title: "[Troubleshooting] docker에서 react-scripts를 찾지 못하는 문제"
author: kwon
date: 2025-02-03T23:00:00 +0900
categories: [toubleshooting]
tags: [docker, react]
math: true
mermaid: false
---

# 🚫 현상
- docker-compose로 react 프로젝트를 실행하려 할 때 아래와 같은 문제 발생

```bash
react_frontend  | 
react_frontend  | > app@1.0.0 start
react_frontend  | > react-scripts start                                              
react_frontend  |                                                                    
react_frontend  | sh: react-scripts: not found                                       
react_frontend exited with code 127
```

---


# 💡원인

- Dockerfile에 명시한 `npm install` 이 실행되지 않고 생략되는 것을 확인(`Cached`)
- local의 node_modules가 container 내부에 영향을 주는 것으로 판단
---


# 🛠 해결책

- `.dockerignore`에 node_modules를 추가
    
    ```
    npm-debug.log
    node_modules
    build
    Dockerfile
    .git
    *.log
    *.env
    *.tmp
    *.bak
    *.swp
    *.DS_Store
    ```
    
    - 효과는 없었음
- docke-compose에 volume을 명시하여 node_modules가 contianer에서 독립적으로 관리될 수 있도록 수정
    
    ```yaml
    ...
      frontend:
        build: ./Frontend
        environment:
          - CHOKIDAR_USEPOLLING=true  # 파일 변경 감지를 위한 설정
        ports:
          - "3000:3000"
        volumes:
          - ./Frontend:/app  # 로컬 파일 시스템을 컨테이너에 마운트
          - /app/node_modules  # 로컬의 node_modules가 container 내에 적용되지 않도록
        networks:
          - app_network
    ...
    ```
---


# 🤔 회고

- 해결하기 까지 정말 오래 걸렸다. node_modules를 지우고 하더라고 한 번만 잘 되고 계속 안됐었는데 mount된 volume에 local 파일이 영향을 줘서 생기는 문제라고는 생각을 못했다.
- 어찌 보면 기본적인 compose 작성법일 수 있는데 기초 공부가 조금 부족하지 않았나 반성하게 되는 계기가 되었다.
---


# 📚 Reference

- From GPT
    
    ### **`/app/node_modules`**
    
    - `node_modules` 폴더를 컨테이너 내에서 별도로 관리하여, 로컬의 `node_modules`과 충돌하는 것을 방지합니다.
    - 이 설정이 없으면 `./Frontend/node_modules`가 컨테이너 내부 `/app/node_modules`로 덮어씌워져서, **컨테이너 내에서 설치된 패키지가 정상적으로 작동하지 않는 문제**가 발생할 수 있습니다.