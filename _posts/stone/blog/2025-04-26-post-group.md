---
title: "Post Group | 글 분류"
description: "이 블로그에서 사용하는 글 분류 규칙"
categories: [블로그]
tags: [블로그]
image: "/assets/img/background/20230112-151539.jpg"

date: 2025-04-26. 14:57 # Init (Blog.md에서 분리)
# last_modified_at: 2025-04-26. 23:35 # 중복 카테고리 수정
# last_modified_at: 2025-05-01. 14:23 # Tree
# last_modified_at: 2025-05-02. 11:50 # 정리
last_modified_at: 2025-06-24. 23:10 # blog를 stone 하위 분류로, Tree
---

## 머리말

---

이 블로그에서 사용하는 글 분류 규칙  

## 카테고리

---

Chirpy Theme의 Category tab에서 표현되는 카테고리 최대 깊이는 2.  
깊이 3 부터는 외부에서 바로 알아채기 어렵다.  

수필, 일기: [일기, 생각, 기록]  
수필, 생각: [생각, 이론, 전략]  
전에는 Milestone에서 따와, Stone이라 불렀다. (하룻돌, 달돌, 삶돌)  

```shell
├─computer
│  ├─algorithm
│  ├─data-structure
│  ├─graphics
│  ├─internet
│  ├─programming
│  ├─software
│  └─system
├─stone
│  ├─blog
│  ├─dairy
│  ├─library
│  └─think
├─witch-mendokusai
│  ├─dev-log
│  ├─game-design
│  └─world
└─works
```

## 태그

---

지엽적인 태그는 추가하지 않기, 최소 두 곳 이상에서 쓰이는 것  
나보다는 블로그를 보는 사람을 위한  

## 메모

---

### 주의사항

- 블로그 빌드 시
  - 카테고리 이모티콘 빠진채로 설정되는데, 이때 중복된 카테고리 없도록 주의

### Regex

- `categories: \[.*?\]`
- `_posts/computer/algorithm/**/*.md`
- `(\d{6})_(\d{6})`
- `$1-$2`

### 기록

- 250222
  - `Figma` 공부를 시작, 메모 글을 작성하기 위해 새로운 카테고리 `Design`을 만듦.
  - 근데 `Game-Engine` - `CG`랑 범위가 조금 겹치는 것 같아, `Game-Engine`으로부터 `Computer-Graphics`를 분리하고, `Computer` 카테고리의 주 카테고리로 승격.
  - `Computer-Graphics`에 `Design` 카테고리를 포함시키는 것으로.

- 250315: `Game-Engine`을 `Programming`에 병합

- 250423: `Memo`를 `LifeStone`에 병합
  - `Word.md`의 경우, 나에게 있어 특별한 의미가 있거나, 그런 의미를 가지고 싶은 단어들로 구성하도록 수정

- 250426: 대규모 정리

- 250501 ~ 250502: 대규모 정리
  - 카테고리 구조 수정
  - 이모티콘 제거
    - 🌑🌒🌓🌔🌕🌖🌗🌘🌚-💫🫧
    - 🌱🪴🌴🏝️-🗿🪨
    - 🍉🍊🍍🍌🍋🍐🥑🍋‍🟩🍈🥥🫐🍇-📀💿
    - 🫐-📀💿
    - 📀-📀💿

- 2025-06-24. 23:10 blog를 stone 하위 분류로
