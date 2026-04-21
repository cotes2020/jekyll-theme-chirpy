---
title: "Coding Convention | 코딩 컨벤션"
# description: ""
categories: [컴퓨터, 프로그래밍, Convention]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

# Coding-Convention, Readable-Code

date: 2023-02-17. 10:38
# last_modified_at: 2024-09-05. 04:47
# last_modified_at: 2024-10-16. 08:36
# last_modified_at: 2025-03-15. 09:42 # 메모
last_modified_at: 2025-05-28. 22:05 # +메모
---

2024-09-05. 04:47: 글 추상화.  
`2023-02-17-The-Art-Of-Readable-Code / 🌒 읽기 좋은 코드가 좋은 코드다`  

## 머릿말

---

### _

코드는 이해하기 쉬워야 한다  
코드는 다른 사람이 그것을 이해하는 데 들이는 시간을 최소화하는 방식으로 작성되어야 한다  

적은 분량으로 간결한 코드를 작성하는 것이 좋은 목표이긴 하지만,  
(이해하기 위한 코드량이 절대적으로 적은 것이니)  

분량이 적다고 해서 항상 더 좋은 것은 아니다  
(주석, 설명 변수는 분량을 늘리지만 이해 비용을 줄여준다)  

## Surface-Level 에서의 개선

---

1. 이름에 정보를 담아내라
   - 구체적인 단어 선택 → 추상적/중의적/무의미한 단어 피하기, 의도한 정보를 정확히 전달하는
     - Get → Fetch, Download, Compute(Get은 관행적으로 가벼운 접근자)
     - Size, Length → Height, NumNodes, MemoryBytes, Chars, CountSize/CountElements(Size는 관행적으로 일정한 시간을 소비, i.e. O(1))
     - Filter → Select, Exclude
     - Clip → Truncate
     - Stop → Kill, Pause
   - Temp 같은 보편적 이름 피하기
     - Temp: 짧게 임시적으로만 존재, 임시적 존재 자체가 변수의 가장 중요한 용도일 때
     - 루프 반복자: i j k 보다는, 인덱스로써 쓰인다면 컨테이너 이름과 인덱스 접두문자를 같게 한다던지
   - 세부 정보 붙이기
     - 단위: a_chars, a_secs, a_mb, a_kbps, hex_id,
     - 속성(내용이 중요하다면): plaintext_password, html_utf8, min/max
     - 경계: Start/Stop → First/Last(경계포함), Begin/End(경계포함/배제)
     - Boolean: is, has, can, should + 긍정형(부정형은 이해가 느리다)
   - 변수는 작은 설명문, 이름 길이 알잘딱으로
     - 좁은 범위에서는 짧은 이름, 넓은 범위에서는 긴 이름이 좋다
     - 약어/축약형: 새로 합류한 사람이 이름이 의미하는 바를 이해할 수 있다면, doc, str
     - 불필요한 단어 제거: Convert^ToString, Do^ServeLoop
   - 코딩 표준, 이름 포맷팅으로 의미 전달

2. 미학
   - 줄바꿈, 열 맞추기(?)
   - 함수로 불규칙성 정리(모양 통일, 중복 코드를 간결하게)
   - 일관성 (있는 패턴), 의미있는 순서, A B C 로 언급했다면, B C A 금지
   - 코드(선언문)를, 문단/블록으로, 논리적 영역(주제/순서)에 따라 성격 구분

3. 주석
   - 코드/이름에서 빠르게 유추할 수 없는 내용 (새로운 정보가 아니더라도 '빠르게' 유추할 수 없다면)
   - 나쁜 코드/이름에 주석을 달지 말고, 코드/이름을 고치기 (좋은 코드 > 나쁜 코드 + 좋은 주석)
   - 생각을 기록하기 (감독의 설명, 의도)
     - 이 코드는 A 해서 개쩜
     - 이 코드는 A 하고 1분 뒤에 폭파됨
     - 이 코드는 A 때문에, B 해줘야함. 주의
     - 이 코드는 A 부분에 하자가 있음. 근데 난 안고침 ㅋㅋ
       - **TODO**: 더 빠른 알고리즘을 쓰셈
       - **TODO**: JPEG말고 다른 이미지 포맷도 처리할 수 있어야 함.
       - **TODO**, todo, FIXME, maybe-later, HACK, XXX
     - 상수 값이 { 범위/조건 } 이면 된다, { 통계/고찰/사실 } 때문에 이게 좋다
     - 코드를 읽은 사람의 입장이 되기, 읽는 사람이 ? 할 부분
     - 큰 그림 알리기: 파일/긴 함수에서 크게 설명하고, 각 조각이 어떻게 맞춰지는 지
     - 어려우면 그냥 생각을 적어내기
       - 아 A 너무 어렵네 그냥 B 써야겠다
       - → 주의: 이 코드는 A 대신 B 를 사용했기에 @ 부분은 처리하지 않는다. 그렇게 하는 것이 어렵기 때문이다.

## 폴더/파일 구조/네이밍

---

- [유니티 폴더 네이밍 규칙](https://x.com/U2tyDragon/status/1771204321226498417)

## 메모

---

- '읽기 좋은 코드가 좋은 코드다', '개발자의 글쓰기'
- Enum
  - 요소가 몇 개 없다면, 줄바꿈 없이 한 줄에 몰아 쓸 수도 있다.
  - `NONE = -1`, `TOTAL = 마지막`
  - Define partial class
    - `\{ Enum class \{ Enum What~ \} \}`
- ~Service (Manager 대신?)
