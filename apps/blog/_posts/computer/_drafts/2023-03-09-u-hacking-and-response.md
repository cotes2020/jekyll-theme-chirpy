---
title: "해킹과 대응 기술 과목"
# description: ""
categories: [컴퓨터, 🌚Computer-General]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-03-09. 15:01
last_modified_at: 2023-04-11. 15:02
---

VMWare 환경에서 Kali-1 Kali-2, 두 개 혹은 하나  
Windows XP 7 업데이트 안된 환경 설치  

XSS CSRP  
XSS SSRF  
CSS XSS  

---

### 블록암호운영모드

---

- 질문
  - 다음은 무엇을 설명한 것인가?
    - 각각의 평문 블록은 암호화되기 전에 이전 암호문 블록과 XOR 연산한다.  
    - 첫 블록의 경우 이전 암호문 블록이 존재하지 않기 때문에 IV (Initialization Vector)가 사용된다.  
    - 암호화 할 때 이전 블록 암호화 결과에 의존하기 때문에 병렬화가 불가능하지만, 복호화의 경우 각 블록을 복호화한 다음 이전 암호화 블록과 XOR 연산하여 복구할 수 있기 때문에 병렬화가 가능하다.  
  - 다음은 무엇을 설명한 것인가?  
    - Ci = Ek (Pi + Ci-1), C0 = IV (+: XOR ftn, Ci: i번째 암호문, Pi: i번째 평문, IV:초기벡터, Ek:암호화, Dk:복호화)  
    - Pi = Dk(Ci) + Ci-1, C0 = IV  

- 암호화/복호화 단계별 과정  
- 암호화/복호화 과정에서 병렬처리가 가능한가?  
  - 근거  
- 복호화할 때 왜곡된 비트 정보가 삽입된다면  
- 재전송 공격이 가능한가?
  - 근거

---

- Block Cipher Mode of Operation
  - ECB Electronic CodeBook
    - 평문 블록을 암호화한 것을 그대로 암호블록으로 사용한다
    - 마지막 블록에 패딩이 필요하다
    - 복호화 알고리즘 필요
    - *평문 블록의 패턴이 암호 블록에 그대로 나타난다. 반복 (재전송) 공격에 취약하다
    - *한 개의 블록만 해독되면 나머지 블록도 해독이 되는 단점이 있다 (Brute-Force Attack, Dictionary Attack)
    - 평문 블록의 순서를 바꾸어 공격 가능하다.
    - 병렬 처리가 가능하다.
  - **CBC Cipher-Block Chaining
    - 암호화 알고리즘의 결과를 평문 블록과 XOR하고 나서 암호화를 수행한다.
    - 각 암호문 블록은 이전 평문 블록들의 영향을 받게 된다.
    - *처음 평문 블록을 위해 초기벡터 Initialization Vector가 존재한다  
    - 평문의 마지막 블록에 패딩이 필요하다 (패딩 오라클 공격 가능)
    - *평문에서 1비트의 오류가 생기면 오류가 전파된다
    - 평문의 패턴이 암호문에 나타나지 않는다
    - SSL/TLS에 사용된다
    - 복호화에서 병렬처리가 가능하다
    - 복호화 알고리즘이 필요하다
    - IPSec, AES-CBC 등에서 사용
  - CFB Cipher FeedBack
    - 1단계 앞의 암호문 블록을 암호 알고리즘의 입력으로 사용
    - IV 필요
    - 패딩 불필요
    - *재전송 공격이 가능
    - 복호화 알고리즘이 필요없다
    - CBC모드와 마찬가지로 암호화는 순차적이고, 복호화에서 병렬 처리를 할 수 있다.
  - OFB Outout FeedBack
    - IV 필요
    - 패딩 불필요
    - 암호화/복호화 전에 IV로 사전 준비 가능
    - 복호화 알고리즘이 필요 없다
    - 암호화 함수는 키 생성에만 사용되며, 암호화/복호화 방법이 동일하여 암호문을 한 번 더 암호화하면 평문이 나올 수 있다.
    - 병렬 처리를 할 수 없다.
  - *CTR CounTeR
    - 1씩 증가하는 카운터를 암호화하여 키 스트림을 만드는 스트림 암호
    - *카운터의 초기값은 난수로 만드는 비표 Nonce와 1식 증가하는 블록 번호로 이루어져 있다
    - 오류의 확산이 없다
    - 병렬 처리가 가능하다
    - 패딩이 필요 없다
  - CTS, GCM  
    - GCM Galois Counter Mode  
    - CCM Counter with CBC-MAC  
    - CTS Ciphertext Stealing  
    - XTS mode is similar to CBC-CTS in operation  

---

| | FCB | *CBC | CFB | OFB | CTR |
| 초기화 벡터 | X | O | O | O | - |
| 병렬 처리 | 암호화/복호화 | 복호화 | 복호화 | X | 암호화/복호화 |
| 패딩 | O | O | X | X | X |
| 데이터 형태 | Block | Block | Stream | Stream | Stream |
| 특성(기능) | 간단하며, 패턴이 반복된다. 재전송, 패딩오라클 공격이 가능하다. (해커)암호문 블록의 순서를 뒤섞거나, 삭제/복제 한다. | 암호문 블록이 파손되면 2개의 평문 블록에 영향을 준다. 패딩오라클, 초기화 벡터 공격이 가능하다. | 재전송 공격이 가능하다. 초기화 벡터 공격이 가능하다. | 비트 단위의 오염된 암호문에 대응되는 평문 비트에만 오류가 발생하다. | 카운트를 이용한다 (비트, 블록 번호). 1비트가 오염된 암호문에 대응되는 평문 블록의 1비트에 영향을 주고, 오류 전파는 없다. |
| 적용분야 | | AES-CBC, IPSec, Kerberos V5, SSL/TLS | | | AES-CTR |
| | | | 비동기 | 동기 | 동기 |

---

P1 평문 (n비트 블록1) (64, 128, 256, ...)  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
C1 암호문 (n비트 블록1)  

C1 암호문 (n비트 블록1)  
|  
Key → Block Cipher Decryption  
|  
P1 평문 (n비트 블록1) (64, 128, 256, ...)  

- ECB 운영모드 (Electronic CodeBook)  
  - #문제가 많다, 사용을 안한다  
  - 짧은 메시지에 적합하며, 오류 전파는 없다
  - 마지막 블록에 패딩이 필요할 수 있따
  - *블록간 독립성이며, 발생하는 오류가 다른 블록에 영향을 주지 않는다.  
  - 기밀성 낮고, 재전송 공격에 취약하며, 암호화/복호화가 병렬적으로 수행된다.
  - 평문의 블록 패턴과 암호문의 블록 패턴이 동일하게 유지된다.

---

P1 평문 (n비트 블록1) (64, 128, 256, ...)  
|  
X ← (IV 초기벡터) = ㄱ  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
X → = ㄴ  
|  
C1 암호문 (n비트 블록1) (#손상될 경우)  

P2 평문 (n비트 블록2) (64, 128, 256, ...)  
|  
X← ㄴ  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
X → = ㄷ  
|  
C2 암호문 (n비트 블록2)  

- CBC Cipher Block Chaining  
  - #문제 나오면 CBC가 답인 경우가 많다! 그만큼 경쟁력이 있다는 것  
  - 각각의 암호 블럭이 영향을 준다. 블록간의 연관성이 존재한다.  
  - *암호화할 때 평문에 손상이 있다면 다음 단계로 파급된다. 복호화할 때도 동일한 효과가 발생한다.  
  - 보안성이 높고, 초기값이 필요하다.  
  - MAC으로 블록을 검사하고, 마지막 블록의 전부 혹은 일부를 MAC으로 사용한다.  

- 평문에서 손상이 발생한다면  
  - 암호화 과정에서 이상한 (다른) 암호가 나온다  
- 암호문이 손상된다면  
  - 복호화 과정에서 이상한 (다른) 평문이 나온다  
  - 그 다음 복호화 단계에도 영향을 준다 (현재 암호가 다음 복호화 단계에서 키로 쓰이니까)  
  - 그 다음 복호화 단계에도 영향을 줄까? ㄴㄴ  

---

IV (초기 벡터) - Random  
n비트 Shift 레지스터 (64, 128, 256, ...)  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
n비트 암호문  
암호문에서 일부 비트만 (r) 선택하여 키 k1로 사용한다. i.e. 8Bit  
|  
X← r비트 평문 P1  
n비트 암호문 좌측에서 r비트를 선택하여 암호키 k1으로 사용한다  
|  
r비트 암호문 C1  

| C1이 r비트 왼쪽으로 Shifting  
n비트 Shift 레지스터 (64, 128, 256, ...)  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
n비트 암호문  
암호문에서 일부 비트만 (r) 선택하여 키 k2로 사용한다. i.e. 8Bit  
|  
X← r비트 평문 P1  
n비트 암호문 좌측에서 r비트를 선택하여 암호키 k2으로 사용한다  
|  
r비트 암호문 C2  

- CFB Cipher Feedback  
  - 키 스트림을 이용한다  
  - 오류 전파가 있다  
  - 데이터 처리효율이 낮다. n>r (8 or 16 Bit, 주로 8비트 쓴다고 함)  
  - (암호문 C1이 손상된 경우) P1과 P2 복호화 까지 영향을 미친다. 그러나 암호문 C2에는 무영향이므로, P3를 복호화 할 때 영향을 미치지 않는다.  
  - 암호화/복호화 과정에 동일한 알고리즘을 사용한다  

- 평문에서 손상이 발생한다면  
  - 암호화 과정에서 이상한 (다른) 암호가 나온다  
- 암호문이 손상된다면  
  - 복호화 과정에서 이상한 (다른) 평문이 나온다  
  - 그 다음 복호화 단계에도 영향을 준다 (현재 암호가 다음 복호화 단계에서 Shift 됨)  
  - 그 다음 복호화 단계에도 영향을 줄까? ㄴㄴ  

---

IV (초기 벡터) - Random  
n비트 Shift 레지스터 (64, 128, 256, ...)  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
n비트 암호문  
암호문에서 일부 비트만 (r) 선택하여 키 k1로 사용한다. i.e. 8Bit  
|  
X← r비트 평문 P1  
n비트 암호문 좌측에서 r비트를 선택하여 암호키 k1으로 사용한다  
|  
r비트 암호문 C1  

| C1이 r비트 왼쪽으로 Shifting  
n비트 Shift 레지스터 (64, 128, 256, ...)  
|  
Key → Block Cipher Encryption (DES, AES, SEED, ...)  
|  
n비트 암호문  
암호문에서 일부 비트만 (r) 선택하여 키 k2로 사용한다. i.e. 8Bit  
|  
X← r비트 평문 P1  
n비트 암호문 좌측에서 r비트를 선택하여 암호키 k2으로 사용한다  
|  
r비트 암호문 C2  

- OFB Output FeedBack  
  - CBC, CFB 오류 전파를 제거한다  

- 평문에서 손상이 발생한다면  
  - 암호화 과정에서 이상한 (다른) 암호가 나온다  
- 암호문이 손상된다면  
  - 복호화 과정에서 이상한 (다른) 평문이 나온다  
  - 그 다음 복호화 단계에 영향을 주지 않는다  
    - 현 단계 암호와는 상관없는, Key 값이 다음 복호화 단계에서 Shift 됨  

---

- CTR (CounTeR)  
  - 1씩 증가해가는 카운터를 암호화해서 키스트림을 만들어 내는 스트림 암호이다.  

- CFB만 알고 가자
- OFB, CFB 차이는 Output, Cipher

---

크리덴셜 (자격증명)  

해커가 크리덴셜을 가지고 무엇을 하나  
I.E. 스피어피싱  

F/W vs Proxy  
F/W I.E. 경호원 (많을 수록, 내부에도)  
Proxy I.E. 부동산 중계 업소?  

보안  

포기 / 보험 / 넘기기  

암호학적 해시함수  
I.E. PW, (+ SALT 이후 암호화)  
I.E. 키 합의 프로토콜  

패딩  
블록크기보다 데이터가 부족할경우 임의의  

아핀 암호  
곱셈과 덧셈 암호  
근데 곱셈 Matters  

---

- DES Data Encryption Standard

- Biba 모델  
- BLP 모델  
- 만리장성 모델 - Biba + BLP?  

- 고객 - 은행 - 인증기관

[참고](https://m.blog.naver.com/sdug12051205/221575582613)  

- DAC MAC RBAC  

- SSL/TLS
- IPSec
- VPN
- TOR The Onion Touter
- Zero Trust - SDP

- Block Cipher

- SET, SET 이중 서명  

- Internet Banking 의 작동원리
- BLock Cipher mode of operation - CBC
- 구매자, 판매자, 대행사 (사과를 온라인 쇼핑몰에서 구매하고자 할 때)
- 공개키 암호화(공개키,개인키)에서 핵심
- IPSec VPN의 Tunnel 생성 과정 - IKE에서 대칭키를 교환하는 과정
- MITM, Replay Attack, ARP Spoofing , TCP Session Hijacking
- 해킹 공격 및 실습

- 양방향 (기밀성)
  - 대칭키 (데이터 암호화)
    - Steam 방식 (True Rabdin) = OTP, (Phudo Random 의사 무작위 추출= RCA)
    - Block 방식 = 페이스탈 DES SEED, SPN ARS ARIA?
  - 비대칭키(부인방지, 서명, 메시지인증,대칭키교환, PKi Public Key )암호화 및 전자서명
    - 인수분해 RSA PCP Radian?
    - 이산대수 D.H?
    - 타원곡선 ECC?

- 일방향 (무결성)
- Hash 압축성 계산용이성 일방향성 충돌회피성
  - MAC 키 써서
  - MDC 키 적용안하는 일방향성

- 디페 헬만 키 교환 알고리즘

[SSL/TLS](http://wiki.gurubee.net/display/SWDEV/SSL+%28TLS%29)  

- 암호화 통신을 하기 위한 대칭키 전달을 기반으로 통신 과정을 설명하라
  - 인증서는 누구나 만들 수 있지만 브라우저는 공인 인증기관 목록에 포함된 기관에서 발급한 인증서만을 신뢰한다
  - 1Trusted root CA(Certificate Authority) store, 2Browser에 사전 설치되어 제공된다. 3Browser에서 지정한 보안 및 인증 표준을 준수하고 감사를 받는다.  

- SSL/TLS
  - Site, 인증기관에 Site의 공개키(암호화용) 전송 (인증 요청)
  - 인증기관, Site의 공개키를 인증기관의 개인키(암호화용)로 암호화하여 사이트 인증서 제작  
  - 인증기관, 인증서를 Site에 전송, Browser에 인증기관의 공개키(복호화용) 내장
  - User, Site에 접속 요청
  - Site, Site의 인증서 전달
  - User, Browser에 내장된 인증기관의 공개키로 Site의 인증서 복호화
  - User, Site의 공개키(암호화용) 획득  
  - User, Site의 공개키로 User의 대칭키 암호화 후 전송
  - SIte, Site의 대칭키로 User의 대칭키를 복호화

[개인키 공개키, 대칭키 비대칭키](https://spidyweb.tistory.com/310)  

- 공개키 암호화 (공개키,개인키)에서 핵심은?
  - 송신자의 공개키로 암호화(잠금)하는 행위는 기밀성을 확보한다
  - 송신자의 개인키로 암호화(잠금)하는 행위는 시그니처(전자서명), 인증, 부인방지, 무결성을 확보한다
  - 공개키 암호화와 전자서명의 차이
    - 공개키 암호화에서 수신자의 공개키는 평문을 암호화하는것이고, 수신자의 개인키는 암호문을 복호화한다
  - 전자서명에서 송신자의 개인키는 평문을 사용하여 서명하기위한 것이고, 송신자의 공개키는 수신자가 서명이 올바른지 검증한다

- SSL/TLS, IPSec-VPN

- IPSec Internet Protocol Security
  - Host와 host 사이, 보안 Gateway 사이 (Network <→ Network),보안 Gateway와 host 하이 (Network <→ Host)에 보안 Tunneling 을 형성하여 데이터 흐름을 보호한다

- VPN Virtual Private Network
  - 개인들이 공동으로 네트워크를 이용하면서 이느와 암호화 Tunneling 기술을 이용한 가상적인 시설 보안 네트워크로서 저비용, 보안성, 익명성이 제공된다
  - Gateway 위치에 설치되므로 방화벽과 같은 위치에 설치된다 → Trends 통합장비 (VPN+방화벽)
  - Anywhere Anytime AnyDevice → SSL, VPN, IPSec

- S-HTTP (Secure-HTTP), SSL/TLS, IPSec
  - Interface는 System Module 간 통신 및 정보 교환을 위한 통로로 사용되므로 보안 기능을 갖춰야 한다
  - Interface는 Network, Application, DB 영역에 각각 적용한다
  - Network에서 송신 및 수신 간 Sniffing 등을이용한 데이터 탈취 및 변조 위협에 대응하기 위해 Traffic에 대한 암호화를 설정한다. 암호화는 Interface Architecture 에 따라 S-HTTP, SSL/TLS, IPSec 등이 적용된다
- S-HTTP
  - 특성
    - Web 상에서 네트워크 트래픽을 암호화하는 방법으로 Clinet와 서버간에 전송하는 모든 메시지를 암호화하여 전송하는 보안 프로토ㅗㄹ
    - HTTP 세션으로 주고 받는 자료에 대한 암호화, 전자서명, 보안기능을 제공
  - QSI Layer
    - Application
  - 범위
    - Web에 한해서만 보호된다
    - 트랜젝션(기밀성), 메시지(무결성), 발신자 언증, 부인 방지, 접근 통제
  - 인증 방식
    - 클라와 서버 각각 인증서가 필요하다 (상호인증)
  - 인증서
    - 클라 인증서를 보낼 수 있다
  - 접근성 (응용)
    - -
  - 암호화 단위
    - 메시지 단위 (메시지 기반 프로토콜)
  - URL
    - shttps://
- SSL/TLS
  - 특성
    - 전송계층과응용계층 사이에서 클라와 서버간의 암호화, 상호인증, 무결성을 보장하는 보안 프로토콜
    - ㅡㄹ라와 서버강의 상호인증, 암호방식에 대한 협상을 한다
    - 핸드쉐이크로 인증하고, 인증 확인을 하므로(날)사칭하는 것을 방지한다
    - 정보 유출과 악성코드 유입경로로 역이용될 수 있따
  - asd
    - Transport
  - asd
    - Telnet, FTP 등의 Application Protocol
    - Online Shopping
    - WebBrowser에서 SSL VPN을 연결한다
  - asd
    - 클라의 인증이 선택적
    - One or Two way 인증 - 인증서 이용
  - ㅁㄴㅇ
    - 서버만이 인증할 수 있다
  - ㅁㄴㅇ
    - 분산 환경 접속, 원격 근무자 (웹 이용)
  - ㅁㄴㅇ
    - 서비스 단위, 브라우저에 의존한다
  - https://
- IPSec
  - ㅁㄴㅇ
    - 네트워크 계층에서 무결성과 인증을 보장하는 인증헤더 AH와 기밀성까지 보장하는 암호화ESP을 이용한 층단간 보안서비스를 제공하는 네트워크 프로토콜
    - IP 패킷 단위의 데이터 변조 방지, 은닉 기능을 제공
    - Router 간 안전한 정보를 교환
    - 두 장비간 논리전 커넥션 구성 → H/w S/W 필요
  - ㅁㄴㅇ
    - Network
  - asd
    - IP에 보안기능을 추가
    - ~? ㅂㅎ안
    - VPN을 이용한 보안 채널 구성
  - ~
  - ~
  - ~

~~

---

@0411  

- 전자상거래
  - 메일 주고 받는 것도 전자상거래
  - 돈 주고 물건사고 하는게 전자상 거래 아니였어?!!
  - 좀 더 추상화시켜보면..
  - 실제로 돈을 주고 물건사고 하는 게 아니다
  - 돈을 주라는 데이터, 물건을 보내라는 데이터를 주고 받는 것
  - 다시, 데이터를 주고 받는 것

- SSL/TLS 취약점?
  - 서로 다른 버전의 SSL/TLS 간의 통신 시, 낮은 버전 기준으로 통신 (높은 버전이 낮은 버전으로 낮춤)

- ESP ?
  - 인증 기밀성 무결성 암호화 ~ 뭐 다 있음 !!
  - 와우 이거 쓰자
  - 근데.. 무거움 = 느림

- IPSec
  - Transport Mode 에서 ESP 는 IP Payload 를 암호화하지만 IP Header는 암호화하지 않는다
  - IP HEADer 까지 암호화? = Tunnel Mode

@0509

핵티비즘?  
종교/신념에 따라 공격  

- TCP 신뢰성 3가지  
  - 패킷 순서가 정확한지?
  - 중간에 손실된 패킷이 없는지?
  - 손실된 패킷의 재전송 요구가 있는지?

SYN Flooding, TearDrop  

POP3(110) VS IMAP4(143) ?  

- DNS 취약성
  - UDP(53) 기반으로 불안정한 Protocol, Why?
  - Query 시 인증을 수행하지 않는다
  - 공격자가 local에 존재하므로 실제 DNS 서버보다 빠르게 응답할 수 있다
  - Client는 DNS Query를 수행한 후 먼저 응답한 IP 수용

DNS Spoofing, Pharming  

클라이언트 밑에 숨는,  
클라이언트가 이용하는 DNS 서버에 들어가는  

독약을 넣느다 = Poisioning

- 방화벽의 한계점, 방화벽 3단계
  - 내부공격에 대해 어떤 보호도 제공할 수 없다
  - 네트워크상에 백도어를 통해서 들어오는 무분별하거나 권한 없는 엑세스에 대해 보호할 수 없다
  - 바이러스와 악긔적인 코드에 대해서 완벽한 보호를 할 수 업다
  - 어떤 방화벽도 *
  - ㅡ

Genmask  
Netmask  

[http://www.ktword.co.kr/index.php](http://www.ktword.co.kr/index.php)  
[https://hipolarbear.tistory.com/36](https://hipolarbear.tistory.com/36)  
