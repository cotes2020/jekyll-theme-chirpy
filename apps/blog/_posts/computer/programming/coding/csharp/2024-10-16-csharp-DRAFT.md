---
title: "C#"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [CSharp]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-10-16. 06:40
# last_modified_at: 2024-10-16. 07:10 # Init
# last_modified_at: 2024-10-21. 15:42 # string 생성자 count
# last_modified_at: 2025-03-15. 08:37 # 메모
# last_modified_at: 2025-04-28. 18:50 # 다중 변수 for문
last_modified_at: 2025-05-28. 21:58 # +Q, +메모
---

## Q

---

- IEnumerable
- 리플렉션
- GC

## 메모

---

- Struct 인터페이스 상속
- [자동 구현 된 반복자는 IEnumerable < T > 및 IEnumerator < T > 이며 최적화를 위해 첫 번째 GetEnumerator () 에서 자신을 반환합니다. 2번째 이후는 자신을 new해 다른 것을 돌려준다.](https://x.com/neuecc/status/1843568471768215622)
- [](https://x.com/_danuel_/status/1823337950832382124)
- [.Net 9.0 LINQ 성능 개선 사항](https://news.hada.io/topic?id=17327)
- `string.String(char c, int count)`
- `$"{someVar:N0}"` - Number Comma, `$"{someVar:D2}"` - Leading Zero
- 최대한 this 많이 안쓰는 게 좋을 듯?
  - 그냥 동일한 변수 명이 있는 것이 실수 만들기 좋은 것 같음
  - 변서 이름 유사하게 짓지 않기. 헷갈린다 이거.
- params T[] -> 배열로 받아서 사용할 수 있음, 효과음 랜덤에 사용했음 `params SfxType[] sfxTypes`
- enum안에도 `#region` 가능
- <https://stackoverflow.com/questions/1658557/multiple-initialization-in-c-sharp-for-loop> 한 for문 안에 여러 변수 넣기, 혹은 튜플 넣기
- nameOf 연산자 컴파일 타임에 문자열로 바뀜
