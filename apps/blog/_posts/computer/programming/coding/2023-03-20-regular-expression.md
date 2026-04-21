---
title: "Regular-Expression | 정규표현식"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-03-20. 14:21
# last_modified_at: 2023-04-03. 14:23
# last_modified_at: 2024-09-29. 17:49
last_modified_at: 2024-10-24. 16:24 # Regex101.com 추가
---

## Regular-Expression | 정규표현식

---

```plaintext
/[[0-9]{1,}:[0-9]{1,} ((AM)|(PM))/]Papyrus:
```

> `//` 슬래시는 `\\` 역슬래시로 바꿔써야 함.

디스코드에 정리했던 글을,  
블로그에 그대로 옮기려고 복사하였는데,  

글 앞에 \[14:21 PM\] Papyrus 이런 식으로  
보낸 시간과 닉네임이 함께 딸려와서 곤란했다.  

한 문단 정도면 모르겠는데,  
수 백줄 정도 되는 글이라 일일이 수정하기가 어려웠다.  

평소 글을 쓸 때 VSCode를 이용하는데,  
검색 기능에 정규표현식을 사용할 수 있다는 것을 알게 되었고,  
이를 이용해 내가 찾고자 하는 모든 경우의 수에 대해 검색할 수 있었다.  

## 메모

---

- [Regular Expressions](https://regex101.com/)
- `(/posts/)(*?/)`, `$1\L$2`

### 참고

- [참고 0](https://hamait.tistory.com/342)
- [참고 1](https://regexr.com/)
- [참고 2](https://stackoverflow.com/questions/41409872/invalid-escape-in-pattern-html-javascript)

### 기록

- [240929](https://github.com/mascari4615/KarmoPlayground/commit/6357c7bc5790591e05296a259d8c5a45e6810d27)
  - [Past](/posts/past/) 글을 정리하면서 정규표현식을 사용했다.
  - C# 코드를 간단하게 짰다.
  - 텍스트 파일을 읽어와서 정규표현식을 이용해 특정 문자열을 찾아내고, 이를 원하는 모양으로 바꿨다.
  - 코파일럿한테 몇 가지 예시 던져주고 정규식 만들어달라고 하니 편하다.
