---
title: "프로그래밍 언어 - 어휘 분석, 구문 분석"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-26. 13:37
last_modified_at: 2023-12-08. 10:09
---

## 언어를 구현하는 세 가지 접근

---

- Compilation
  - 고급 언어 → 기계 코드 번역, by Compiler
  - C, C++, Cobol
- Pure Interpretation
  - 고급 언어 → 해석 후 실행, by SW Interpreter 해석기
  - HTML에 포함된 JS
- Hybrid Implementation
  - 고급 언어 → 중간 코드 → 해석 후 실행
  - Java, Python, .Net
  - Just In Time - JIS 컴파일러로 성능 향상

모두 어휘 분석기, 구문 분석기 사용  

Parser - 구문 분석기는 형식적 기술에 기반  
BNF 사용 → 문맥 자유 문법  

대부분 컴파일러, 어휘/구문 분석을 따로 실행  
어휘 분석: Lexeme, Token 분리 - 작은 규모의 언어 구조 처리  
구문 분석: 식, 문장, 프로그램 단위 - 큰 규모의 언어 구조 처리  

- 왜 Why, 어휘/구문 분석 따로?
  - 단순성
    - 분리 → 복잡성 완화
    - 어휘가 구문보다 단순
  - 효율성
    - 어휘 분석 오래 걸려서, 따로 최적화
  - 이식성
    - 어휘 분석기: 플랫폼 종속, 파일 Read 과정에서 입력 버퍼 사용
    - 구문 분석기: 플랫폼 독립일 수 있음

### Compilation 과정

1. C Source File (Temp.c)
   - By C Preprocessor ↓
2. 전처리된 C Source File (Temp.l)
   - By C Compiler ↓
3. Assembly Code File (Temp.S)
   - By Assembler ↓
4. Object Code File (Temp.o)
   - By Linker ↓
5. Executable Code File (Temp)

### Compile 과정 (컴파일러 내부 과정), Lex, Yacc

1. Source Program
2. Lexical Analyzer
   - 코드 해석, Token 분해
   - Scanner, Tokenizer
   - Source Code → Tokens, By `Lex`
3. Syntax Analyzer
   - Tokens → Syntax Tree, By `Yacc`
   - Rule 항목(Like BNF)으로 부터 Parser 생성
   - Parser
     - 구문 분석
     - 구문 구성 성분의 위계 관계 분석 → 문장 구조 결정
4. Semantic Analyzer
5. Intermediate Code Generator
6. Code Optimizer
7. Code Generator
   - Syntax Tree → Generated Code

모든 과정에 Symbol-Table Manager가 컴파일러를 도움  
Error 발생 시 Error Handler가 처리  

Lexical Analyzer  
유한 오토마타로 작성할 수 있지만, 어렵고 복잡하기 때문에 Lex 이용  

실습  
Linux (VM), Lex, Yacc, Lex File Format, Python Tokenize Module ~  

## 어휘 분석

---

@ U 중간고사 출제: 주어진 파싱 테이블을 가지고, 'inout state ~' 을 완성하고, LL파서보다 LR파서가 좋은 이유를 설명하시오.  

- 어휘 분석
  - 주어진 문자열(프로그램)에서 특정의 문자 패턴과 일치하는 부분 문자열을 찾는 행위
    - 어휘 분석기는 패턴 매칭기라고 불림
  - 어휘 분석은 구문 분석에 선행함
    - 기술적으로 어휘 분석기는 구문 분석기의 일부

어휘 (Lexme): 문자들을 모아서 구성한 논리적 그룹  
토큰 (Token): 어휘들 분류를 위한 부류(Category)  
토큰화 (Tokenize): 어휘를 토큰으로 분류  

초기 어휘 분석기,  
소스 프로그램 Read → 토큰화 → 결과 (Lexeme, Token) 파일 Create  

오늘날 어휘 분석기,  
구문 분석기의 부프로그램으로써,  
구문 분석기가 어휘 분석기 호출 → 토큰화 결과 받음  
→ 한 번 호출에 하나의 토큰화 결과 받음  

- 어휘 분석기 역할
  - 주석 제거
  - 심볼 테이블 구축
  - 어휘 에러 탐지 및 통보
    - i.e. 부동 소수점 잘못 사용 등

- 어휘 분석기 구성 방법
  - 정규 표현식을 이용하여 언어의 토큰 패턴에 대한 형식적 기술을 작성
    - Lex
  - 언어의 토큰 패턴을 정의하는 상태 전이도 (State transition diagram)을 설계하고 이를 직접 구현
  - 언어의 토큰 패턴을 정의하는 상태 전이도를 설계하고 이 상태도에 대한 테이블-구동 (Table-driven) 구현을 직접 구성

- 상태 전이도 (상태도, state diagram)
  - 유향 그래프(directed graph)
  - 노드는 상태 이름을 그 레이블로 가지고, 아크는 상태들 간의 전이를 야기하는 입력 문자들을 레이블로 가짐
  - 유한 오토마타(finite automata)라 불리는 수학적 기계의 한 유형
    - 정규 언어를 인식하게 설계

- 프로그래밍 언어의 토큰들은 정규 언어이고, 어휘 분석기는 유한 오토마타

- 어휘 분석에 필요한 상태 전이도는 매우 복잡함

- 어휘분석기는 심볼테이블을 구축
  - 심볼테이블: 이름(Identifier)들로 구성된 데이터베이스 역할

## Parse

---

Parsing  
주어진 구문을 분석 하는 과정 - Syntax Analysis  

Parser  
주어진 프로그램의 구문 분석을 담당  
주어진 프로그램의 Parse Tree를 구성  

Parse Tree  
번역을 위한 기반으로 사용  

### 구문 분석의 목적 (Parser 역할)

- 입력 프로그램을 검사하여 구문적으로 올바른지를 판단
  - 오류가 발견되면 진단 메시지를 생성 및 복구
  - 최대한 많은 Error 발견
- 구문적 오류가 없는 프로그램에 대해서는 완전한 파스 트리를 구축

### Parser 분류

Top-Down, Bottom-Up Parse

- Top-Down 하향식
  - Root Node로부터 Leaf Node로 Parse Tree 생성
  - Leftmost Derivation 최좌단 유도 같은 순서
    - 재귀-하향파싱, LL 파서

- 재귀-하향 파싱
  - 재귀적인 Subprogram 부프로그램으로 구성
  - 하샹식 순서로 Parse Tree 구축
  - EBNF 구축에 적합
  - 문법의 각 NonTerminal에 대해 한 개의 부프로그램을 갖는다
    - 입력 문자열이 주어질 때, 부프로그램에서 해당 논터미널을 루트 노드로 가지며, LeafNode들이 그 입력 문자열과 매칭되는 ParseTree를 추적
  - 전역 변수 nextToken: 다음 번 토큰을 의미
    - 파싱할 때 항상 다음 토큰을 미리 본다

- LL 파서
  - L 왼쪽에서 시작하며, L 좌측 유도 방식으로 파싱
  - 약점
    - Left Recursion 좌순환
      - 직접 좌순환: A → A + B
        - A가 자기 자신 호출, Stack Overflow
      - 간접 좌순환: A → BaA, B → Ab
        - 결국 A가 자기 자신을 호출하는 부분 발생
      - 상향식 파싱 알고리듬은 이런 일 없음
  - 하향식 파서는 최좌측 논터미널에 의해 생성되는 첫번째 토큰만을 사용, 파서가 입력의 다음번째 토큰에 기반하여 항상 올바른 RHS를 선택할 수 있는가가 하향식 파서의 구축에서 중요
    - Pairwise Disjoint Test 집합쌍 공통 테스트로 검증 가능

- Pairwise Disjoint Test 집합쌍 공통 테스트
  - 주어진 문법으로 Top-Down Parsing이 가능한지 판단하는 테스트
  - Top-down parsing의 절차는 lookahead 값을 이용하여 올바른 RHS를 결정하는데 이때 집합쌍 불일치 테스트를 만족하지 못하면 올바른 RHS를 선택할 수 없다.

- Bottom-Up 상향식
  - Leaf Node로부터 Root Node로 Parse Tree 생성
  - Rightmost Derivation 최우단 유도의 역순
    - 이동-감축 알고리듬, LR 파서

- 상향식 파서의 파싱 문제
  - 식 id+id*id 에 대한 최우단 유도
  - 상향식 파싱은 최우단 유도의 역순으로 수행
    - 이전 단계 문장을 얻기 위해 상응하는 LHS로 재작성되는 RHS
  - 상향식 파서의 역할
    - 이전 문장 형태를 만들기위한 특정 규칙(handle)을 발견하는 것

- 이동-감축(Shift-Reduce) 알고리즘
  - 모든 상향식 파서를 구축하는데 활용되며 스택을 이용하여 구축
  - 상향식 파서의 입력은 토큰 스트림이며 출력은 발견된 문법 규칙
  - 동작
    - 이동(Shift) - 다음번째 입력 토큰을 스택으로 이동
    - 감축(Reduce) - 스택의 꼭대기에 위치한 RHS를 상응하는 LHS로 변경

@ ~ 푸시다운 (아래에서 확인)  

- LR 파서
  - 좌측(L)에서 시작하여 우측(R)유도 방식으로 파싱
  - 장점
    - 상대적으로 작은 파서코드와 파싱 테이블로 구성
    - 모든 프로그래밍 언어에 대한 파서를 생성할 수 있다.
    - 왼쪽에서 오른쪽 순서로 검사가 가능하므로 조기에 구문 오류를 감지할 수 있다.
    - LL 파서로 처리 가능하면, LR 파서도 처리 가능
      - @ LL 단점 (Left Recursion, Stack Overflow) 없음
      - @ LL 상위 호환인데 코드도 작음
  - 단점
    - 파싱 테이블을 수작업으로 구축하기가 어렵다.
      - Yacc, 문법을 입력받아서 파싱테이블을 자동으로 생성
      - @ LL 상위 호환인데 단점도 극복 가능

- LR 파싱 테이블
  - Action과 Goto로 구성
  - Action
    - 파서의 행동을 기술하고 있음
    - 행은 상태기호를 열을 터미널 기호를 가짐
    - 파서의 꼭대기에 현재 상태를 저장하고 다음번 입력 터미널을 보고서 무엇을 해야 하는지를 판단
      - 이동(Shift): 다음번째 입력을 스택으로 이동
      - 감축(Reduce): 스택 꼭대기의 상태를 LHS로 감축
      - Accept: 파싱을 성공적으로 완료
      - 오류: 오류 처리루틴을 호출
  - Goto
    - 행은 상태기호를 열은 논터미널 기호를 가짐
    - 감축이 되고난 후 (핸들이 스택에서 제거되고 새로운 논터미널이 스택에 저장 됨을 의미) 어떤 상태 기호가 저장되어야 하는지를 나타냄

- 푸시다운(Pushdown) 오토마타 (Context-Free 문법에 대한 인식기)
  - 프로그래밍 언어에 대한 모든 파서는 푸시다운 오토마타
  - 재귀-하강, 이동-감축 파서도 푸시다운 오토마타에 해당

### Parse Algorithm 복잡도

임의의 모호하지 않는 문법에 대한 파싱은 O(n^3)  
다양한 기법을 활용하여 상업적 파서의 복잡도 O(n)으로 줄임  

---

Program과 Process  

Program: Code(Text), Data  
Process: Code(Text), Data + Stack, Heap  

메모리에 프로그램 그대로 올라가고,  
메모리에 동적으로 Stack과 Heap 할당  

Code(Text): 컴파일된 코드  
Data: External, Static, 전역변수  
Stack: 매개변수, 함수 호출 위치 (돌아갈 곳), 지역변수  
Heap: 메모리 할당, new 등  

4기가 가상공간 (32bit Linux 기준)  
1기가 OS/커널 영역 + 3기가 유저 영역(프로세스가 들어가는 곳)  

모든 프로세스는 자기가 3기가 유저 영역에 혼자 존재하는 줄 앎  
멀티 프로세스는 OS/커널 영역에서 유저 영역에 있는 프로세스를 스위칭 하는 것  

Data  
Symbol Table에서 구분한 전역변수, Static 변수가 들어감  
변수 선언 → Symbol Table에서 구분 → Data 영역에 저장  
이 때 변수 이름 I.E. x 는 사라지고, 이는 메모리 주소가 대신함  

Stack  
→ 스택(시스템) 위-아래, 스택(자료구조) 아래-위  
→ Main 함수부터 쌓기 시작  

Stack에 데이터 쌓다가

BRK 넘어가면  
Segmentation Fault Error  

---

### 컴파일과 실행 단계  

@ 언어 디자인 단계  
@ 컴파일러 구현 단계  

- Editor or IDE (Edit Time): 1. Write Source Codes
  - Source codes (.c), Headers (.h)
- Preprocessor (Build) : 2. Preprocess
  - Included files, replaced symbols
- Compiler (Compile Time)(Build): 3. Compile
  - Object codes (.obj, .o)
- Linker (Link Time)(Build): 4. Link Edit
  - By Static Libraries (.lib, .a) → Excutable Code (.exe)
- Loader (Load Time)(Run): 5. Load
  - By Shared Libraries (.dll, .so)
- CPU (Run Time)(Run): Execute
  - By Input → Output

Linker 전: Static 정적, 프로그램  
Linker 후: !정적, 프로세서  

Preprocess  
gcc -E -P main.c  

Compile  
gcc -S main.c  
gcc -c main.c  

Link  
gcc main.o -o main  
