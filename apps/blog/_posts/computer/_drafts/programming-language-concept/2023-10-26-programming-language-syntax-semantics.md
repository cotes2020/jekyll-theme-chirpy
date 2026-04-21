---
title: "프로그래밍 언어 - 구문론과 의미론"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-26. 13:32
# last_modified_at: 2023-10-27. 09:17
# last_modified_at: 2023-11-24. 10:51
last_modified_at: 2023-12-08. 10:08
---

@ 구문론과 의미론

## Syntax, Semantics

---

Syntax 구문론 (형태, 모양, 문법)  
→ The `Form` of expressions, statements, and program units  

Semantic 의미론  
→ The `Meaning` of expressions, statements, and program units  

i.e. while (bool) statement  
→ 위 명령어의 `문법` 그리고 `의미`  

## 언어 기술의 문제점 - Programming Language Description

---

프로그래밍 언어를 간명하게/이해하기 쉽게 기술하는 것은 필수적이지만, 어렵다.  

왜 Why, 언어 기술 독자가 다양함

- 초기 평가자: 언어 기술의 명료성 중요
- 구현자: 언어 기술의 완전성과 정확성 중요
- 사용자: 언어 참고 메뉴얼의 제공

### Syntax와 Semantics는 서로 밀접한 관련

언어 설계가 잘되었다면,  
문장의 문법(Syntax)에서 의미(Semantics)가 곧바로 보여야  

### Syntax를 기술하는 것이 Semantics를 기술하는 것보다 쉽다

@ 정석: 명확하고 공통적으로 받아들여지는 기술 양식(방법)  

언어의 Syntax: 정석이 있음  
언어의 Semantics: 정석이 없음  

## Syntax 정의의 문제점 (해결 과제)

---

@ 컴파일러 만들 때, Lex (Lexeme 구분해주는) & Yacc (Parser)  

### Lexeme, Token

- Lexeme - 어휘 항목
  - (형식적 문법으로) 코드 최소 구분 단위 (Syntax 기준)
  - 언어의 형식적(Formal) 기술(Description)에서 Lexeme은 포함되지 않음
    - → 언어의 Syntax (혹은 Grammar)의 기술에 포함되지 않음

- Token
  - 의미적으로 구분되는 최소 단위
  - 어휘 항목에 대한 한 부류
    - i.e. C, 6개의 토큰 (Identifier, Keyword, Constant, String literal, Operator, Separator)

어휘 항목 → 그룹들 → 대표(분류) by 토큰  

i.e. x + y = 10
`Lexeme` - `Token`  
x, y - Identifier  
\+ - Addition Operator  
= - Equal Sign  
10 - Int Literal  

@ Parse 트리를 만드는데, 안 만들어지면 문법 오류  

프로그래밍 언어 정의 방법
→ 언어 인식기, 언어 생성기

언어 인식기
→ 정의된 문법으로부터 언어 L을 정의하고
→ 주어진 문자열이 L에 포함되는지 판단
→ 컴파일러의 어휘 분석기(Lexical Analyzer)와 구문 분석기(parser)에서 사용  

언어 생성기  
→ 정의된 문법으로부터 언어 L을 생성하는 장치  

## 구문 기술의 형식적 방법

---

@ Syntax Description, Formal  

Grammar: 구문 기술의 형식적 언어 생성 매커니즘  

Chomsky Hierarchy  
Type-0, Unrestricted Grammar  
Type-1, Context Sensitive Grammar  
Type-2, Context Free Grammar  
Type-3, Regular Grammar  

### 프로그래밍 언어와 문맥 자유 문법 Context-Free

- 토큰들의 형태는 정규문법으로 기술 가능 Regular Grammar
  - by Lex(Lexeme Analyzer Program)
- Syntax는 몇 가지 사항만 제외하면 Context-Free 문법으로 기술 가능

@ Meta Language - 다른 언어를 기술하는 언어  

- BNF, Backus-Nour Form 형식  
  - 메타 언어, 구문 구조 추상화
  - Context-Free 문법과 거의 동일  
  - Bakus: John Bakus, ALGOL 58 기술  
  - Nour: Peter Naur, ALGOL 60 기술 위해 수정  

@ LHS - Left Hand Side  
@ RHS - Right Hand Side  

i.e. \<assign\> → \<var\> = \<expression\>  

- Rule 규칙, Production 생성
  - LHS → RHS, 연결(유도) 하는 것  
  - LHS: 정의하려는 추상화
  - RHS: 정의 - Token, 어휘항목, 다른 추상화

@ Terminal 끝난, 변하지 않는  

Nonterminal Symbol: 추상화된 대상, 여러 정의 가능  
Terminal Symbol: 규칙에 포함된 어휘 항목과 토큰  

i.e.  
\<if_stmt\> →  
if (\<logic_expr\>) \<stmt\> | if (\<logic_expr\>) \<stmt\> else \<stmt\>  

가변 길이의 리스트를 표현할 때 (BNF에서는) Recursive 재귀 사용  
i.e. \<ident_list> → identifier | identifier, <ident_list>  

## 문법과 유도

---

@ U 중간고사 출제: 문법이 모호하다는 것은 어떤 의미인지, 주어진 문법과 문장을 가지고 설명하시오.  

문법: 언어를 정의하기 위한 생성 장치  

- 유도(대체) Derivation
  - 문장을 생성하기 위해 일련의 규칙을 적용하는 것

- 시작기호 Start Symbol
  - 최초의 시작점을 나타내는 특정 논터미널
  - 프로그램 전체를 나타냄, 일반적으로 \<program\>

시작 기호부터 정의된 문법을 이용하여 문장 유도  
Loop Until 어떠한 논터미널도 포함하지 않을 때까지  

i.e.  

주어진 배정문: begin A = B + C; B = C end  

```BNF
<program> → begin <stmt_list> end
<stmt_list> → <stmt> | <stmt> ; <stmt_list>
<stmt> → <var> = <expression>
<var> → A | B | C
<expression> → <var> + <var>, <var> | <var> | <var>
```

주어진 배정문: begin A = B * A + C end  

```BNF
<assign> → <id> = <expr>
<id> → A | B | C
<expr> → <id> + <expr> | <id> * <expr> | (<expr>) | <id>
```

최좌단 유도 - Leftmost Derivation  
→ 왼쪽 논터미널부터 유도(대체)  

Parse Tree  
→ Subtree는 문장에 포함된 추상화의 사례  

문법의 모호성  
→ 문장의 Parse Tree 수 > 1 (어떤 유도든 간에)  

컴파일러는 구문 구조로 의미를 파악하기 때문에,  
문장이 여러 파스 트리로 구축되면 의미를 결정 불가능  

- 해결책
  - 파서 설계자가 비문법적 올바른 파스 트리를 구성  
  - 재작성
  - 연산자 우선 순위: 낮은 곳에 위치한 항목들로 먼저 계산
  - 연산의 결합 규칙 :
    - 동일 우선 순위 연산자들 중 어떤 연산자가 먼저 계산되는지
    - i.e. A + B - C, 좌결합 우선 → \+

좌결합 규칙  
좌순환적 표현은 좌결합 규칙을 기술한다  
LHS가 RHS 시작 위치에 나타나는 경우  
중요한 구문분석 알고리듬을 허용하지 않음
좌순환 제거하던지, 비문법적 요소로 컴파일러에서 지원

우결합 규칙
우순환적 표현은 우결합 규칙을 기술한다  
LHS가 RHS의 오른쪽 끝으로 나타나는 경우  

\<factor>→ \<exp> ** \<factor> | \<exp>  
\<exp> → (\<expr>) | id  

- is then else를 위한 모호하지 않은 문법
  - Dangling Else

i.e. Ada BNF 규칙  
\<if_stmt\> →  
if (\<logic_expr\>) \<stmt\> |
if (\<logic_expr\>) \<stmt\> else \<stmt\>  

→ \<stmt\> → \<if_stmt\> 시 모호성 발생  
if (\<logic_expr\>) if (\<logic_expr\>) \<stmt\> else \<stmt\>  

So, 많은 언어에서, else 문은 이전에 매칭되지 않은 가장 가까운 if/then에 매칭  
→ 문법으로는 matched, unmatched로 나눠서 작성  

- 확장 BNF (EBNF)
  - 서술 능력 그대로,가독성과 작성력 ↑
  - ~가 생겼다

화살표 대신에 콜론이 사용되고, RHS는 다음 줄에 표현
수직바(|)를 사용하여 여러 개의 RHS를 구별하는 대신에, 각 RHS를 단순히 별개의 줄로 표현
대괄호를 이용하여 선택 사항을 나타내는 대신에 아랫첨자 opt가 사용
소괄호 안에 포함된 요소들의 리스트에서 선택 사항을 표현하기 위해 | 기호를 사용하는 대신에 단어 "one of"가 사용

- 문법과 인식기
  - 문맥 자유 문법이 주어진 경우, 해당 언어를 인식하는 인식기 구축 가능
    - Yacc: 구문 분석 생성기
