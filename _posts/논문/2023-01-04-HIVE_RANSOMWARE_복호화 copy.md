---
title: A Method for Decrypting Data Infected with Hive Ransomware
author: JOSEOYEON
date: 2023-01-04 14:23:02
categories: [Algorithm, Cryptography]
tags: [LEA]
---


# A Method for Decrypting Data Infected with Hive Ransomware

학회: Journal of Information Security and Applications

<br/>

2022 KDFS 챌린지 대회에서는 Hive Ransomware에 감염된 PC를 분석하는 문제가 나왔습니다. <br/>
문제를 풀면서 KISA에서 배포한 Hive Ransomware 복호화 도구를 사용하였고, 마스터  키를 복호화 하는 방법이 궁금해서 읽어봤습니다.

<br/>

## Introduction

랜섬웨어는 사용자 컴퓨터 내의 파일을 암호화 하고, 복호화를 대가로 금전을 요구하는 사이버 범죄에 사용됩니다. 최근 많은 랜섬웨어는 대칭 암호로 사용자 데이터를 암호화 하고, 대칭 암호에 사용된 마스터 키를 비대칭 암호로 암호화 하는 하이브리드 암호화 방식을 사용한다. 따라서 공격자의 개인키를 얻지 못하면 암호화된 파일의 파일을 해독하기 어렵습니다. 
<br/>
대부분의 랜섬웨어들은 파일 암호화에 AES, DES 등을 사용하지만, 특정 랜섬웨어는 파일 암호화에 자체 개발 암호화 알고리즘을 사용합니다. 
<br/>
자체 개발한 랜섬웨어를 구성할 때, 암호화 방식을 잘못 구성하면 암호학적 취약점이 발생합니다. 

## Hive Ransomware analysis

저자들은 Hive 랜섬웨어 분석에 있어서 IDA pro v7.6, Vmware 를 사용했습니다. <br/>
Hive 랜섬웨어는 Go Language로 작성되었으며 UPX 패커가 적용되어 있. <br/>

### Encryption process

하이브 랜섬웨어의 동작과정은 8단계로 설명한 그림이다. <br/>
![image](https://user-images.githubusercontent.com/46625602/210491231-6d4f0e7e-0e6f-4829-96da-0acb51d7b172.png)

8 단계를 세부적으로 설명하면 

### 1. Generating a master key

0xA00000 bytes(10MiB) random data 생성 후 마스터키로 사용(파일 암호화에 사용할 keystream 생성에 사용)<br/>

### 2. Encrypting the master key
RSA-2048-OAEP public key(hive Ransomware 실행 파일에 fixed 됨) 로 마스터키 암호화 후 저장<br/>
* Hive Ransomware administrator privileged : C:\\
* Hive Ransomware not administrator privileged : C:\Users\\AppData\Local\VirtualStore
* encrypted master key name : “base64url_encoded_string.key.hive”

특정 서비스들을 종료한다. <br/>
![image](https://user-images.githubusercontent.com/46625602/210491541-28c323fa-da0f-456a-a971-52bc894b7d8b.png)

### Creating batch files

랜섬웨어와 볼륨쉐도카피 본을 삭제하고 자기 자신(.bat) 파일을 삭제하는 .bat 파일을 만든다. <br/>

* hive.bat : Ransomware 실행 파일 삭제, removes itself
* shadow.bat : deletes volume shadow copy (VSC) files, removes itself 

### Creating ransom notes

각 폴더에 HOW_TO_DECRYPT.txt 생성한다. <br/>
이 노트에 포함된 URL, login information 로 암호화 할 수 있는 방법을 알려준다. <br/>
![image](https://user-images.githubusercontent.com/46625602/210491570-eeb60427-e96d-4230-91e2-8a97f733ce29.png)

### Encrypting files

C:\Users\Windows 에 있는 .lnk, exe, file 들을 제외한 모든 파일들을 암호화(10 개의 스레드 생성)<br/>
C:\Users\Program Files(x86), C:\Users\Program Files, and C:\Users\ProgramData<br/>
* 관리자 권한 실행 : 해당 경로에 암호화 파일 생성 후 원본 파일 삭제  
* 관리자 권한 없이 실행 : C:\Users\\AppData\Local\VirtualStore에 파일 복사후 암호화(암호화/원본 파일 동시 존재) 

### Destroying the master key 

파일 암호화 후 마스터 키를 할당 해제 한다. <br/>
메모리 스냅샷에서 마스터키를 찾을 수 있지만, 메모리에 마스터키가 잔류할 가능성은 낮다. <br/>

### Cleaning disk

C:\\ or C:\Users\<User_name>\AppData\Local\VirtualStore 경로에 의미 없는 데이터 생성/삭제 반복 하는데, <br/>
암호화된 파일 원본 데이터를 디스크 영역에 잔존하는 것을 방지하기 위함이다. 

### File Encryption process

Hive 랜섬웨어는 마스터키로부터 암호화키 스트림(EKS)를 생성한다. <br/>

파일을 암호화 하는 공식은 단순 키 스트림과 XOR 한 것과 같다. <br/>
* 암호화 파일 = 키 스트림(EKS) XOR 개별 원본 파일 <br/> 

Hive 랜섬웨어의 특징 중 하나는 파일 전체가 아니라 일부분을 암호화 한다는 것이다. <br/>

파일을 암호화 하는 과정은 2단계로 나누어 지는데, <br/>

1. 마스터 키로부터 키 스트림을 추출하는 단계 <br/>
2. 파일을 암호화 하는 단계 <br/>
 
이다. <br/>
각 단계를 살펴 보면 <br/>

### 1. Extracting two keystreams from the master key

각 파일 암호화 프로세스는 마스터키에서 두개의 키 스트림을 추출하는 것 부터 시작한다 <br/>
두개의 키 스트림은 마스터 키에서 두개의 임의의 오프셋을 선택해서 생성이 되는데 <br/>
선택한 오프셋에서 각각 0x100000바이트(1MiB) 및 0x400바이트(1KiB)를 추출합니다. <br/>
오프셋은 수학 패키지의 랜드 함수에서 각각 8바이트의 두 개의 난수(R1, R2)를 사용하여 아래와 같이 계산됩니다. <br/>
* R1 and R2는 8 byte 의 난수  암호화된 파일 이름에 넣음 
* Keystream1 offset (SP1) : R1 % 0x900000 (0x100000 size keystream)
* Keystream2 offset (SP2) : R2 % 0x9FFC00 (0x400 size key keystream)

키 스트림 오프셋을 선택한 후, 두 개의 키 스트림이 그림에 지정된 대로 추출됩니다. <br/>
키 스트림은 하나의 파일 암호화 프로세스가 완료될 때까지 계속 사용되는데 개별 파일 마다 다른 오프셋이 선택되고, 추출된  스트림은 다릅니다. <br/>
파일이 암호화될 때, R1과 R2는 base64url로 인코딩한 후 리틀 엔디언 형식의 파일 이름으로 저장되어 복호화 시에 사용합니다. <br/>
파일 이름은 다음 규칙을 사용하여 생성됩니다. <br/>

* `원래 파일 이름.base64url(MD5(암호화_master_key)` 

![image](https://user-images.githubusercontent.com/46625602/210492126-cb4b0f28-03a8-4746-9d00-466bb97436fe.png)

### 2. Encrypting a file

* Encrypted data[i]← Data[i] ⊕ Keystream1[i%0x100000] ⊕ Keystream2[i%0x400]
* EKS[i]← Keystream1[i] ⊕ Keystream2[i%0x400] (i← 0,1,· · · ,0xFFFFF) 
* Encrypted data← Data[offset] ⊕ EKS[offset%0x100000]
* NBS(Non-encrypted data block size)은 파일 크기에 따라 가변적 

그림처럼 암호화/non encrypt block 을 번갈아 가면서 생성합니다. <br/> 
암호화 블록 사이즈는 0x10000 사이즈로 고정되어 있고, non-encryption block 은 파일 사이즈에 따라 가변적입니다.  <br/>
파일 사이즈에 따른 non-encryption block 은 파일 크기에 따라 가변적이고, 오른쪽 표는 파일 크기에 비례해서 NBS 계산 방법을 정리한 표입니다. <br/>

이때 오프셋은 0에서 시작하여 순차적으로 1씩 증가하지만 연속적이지는 않습니다. 오프셋은 총 0x1000회 증가한 다음 일정량만큼 점프합. <br/> 
오프셋이 점프하는 NBS(암호화되지 않은 데이터 블록 크기)는 암호화할 파일의 크기에 따라 달라지며 표 3에 표시된 대로 계산됩니다.<br/>

![image](https://user-images.githubusercontent.com/46625602/210492418-bdb94118-3822-4a5f-88f5-3f3c35646035.png)

## Hive Ransomware decryption methodology

이 섹션에서는 암호화 취약성을 사용한 하이브 랜섬웨어의 파일 암호 해독 방법에 대해 설명합니다.  <br/> 
하이브 랜섬웨어는 마스터 키에서 파일 암호화를 위한 두 개의 키 스트림을 추출하며, 각 파일 암호화 시작 시 한 번씩 생성됩니다.  <br/> 
두 키 스트림은 EKS를 생성하는 데 사용되며, EKS는 XOR를 사용하여 데이터를 암호화합니다.  <br/> 
EKS는 무작위로 보이지만 EKS를 생성하기 위한 키 스트림은 다양한 파일을 암호화할 때 부분적으로 재사용됩니다.  <br/> 
암호화 알고리즘은 XOR 연산이며, EKS를 생성하는 알고리즘도 XOR이므로 키 스트림을 쉽게 추측할 수 있는데 이런 특성을 이용하여 마스터 키를 복구하는 알고리즘을 제안합니다. <br/>  
암호화된 파일에서 XOR 연산으로 구성된 방정식을 얻고, 이 방정식을 풀어서 마스터 키를 찾습니다 <br/> 
방정식을 구하는 방법은 다음 조건 중 하나를 필요로 합니다.  <br/> 

* 일반적으로 저장하는 Program File, Program File(x86), Program Data 폴더에  대한 내용 인터넷 다운
* 첨부파일 백업, 동기화 및 다운로드 사용 


이 논문에서 설명하는 방법은 두 조건 중 적어도 하나를 만족하는 조각화된 EKS를 가능한 많이 수집하여 마스터 키를 복원하는 것을 목표로 합니다. <br/> 
두 조건 중 하나가 충족되면 EKS를 수집하고 마스터 키를 복구할 수 있습니다. <br/> 

### Method for restoring the Hive Ransomware master key

코어 암호화 알고리즘에서 알 수 있듯이 EKS는 Keystream2를 반복하여 1,024회 연결한 후 Keystream1과 XOR 처리하여 생성됩니다. <br/>
EKS는 두 개의 키 스트림으로 생성되지만 마스터 키의 특정 오프셋 데이터입니다. <br/>
원본 파일과 감염된 파일을 사용하여 최대 1MiB의 연속 EKS를 획득할 수 있다는 것은 마스터 키의 0x100000 및 0x400 오프셋으로 구성된 0x100000 XOR 방정식을 얻을 수 있다는 것과 같습니다. <br/>
Keystream2가 반복적으로 사용됨에 따라, 반복되는 오프셋을 기준으로 세트를 분할하면, 0x100000 XOR 관계는 1,024 크기의 1,024 세트로 분할됩니다.<br/>
각 집합이 하나의 동시 방정식을 형성하기 때문에, Keystream2의 1바이트를 추측하면, 0x400 값도 방정식을 풀어서 결정되게 됩니다. <br/>
하나의 EKS를 사용하는 경우, 하나의 바이트를 추측하여 마스터 키의 최대 1,025 바이트 값을 결정할 수 있습니다. <br/>
따라서, SP2+N 하나의 바이트만 추측하면 1024개의 블록 추측 가능하다는 말입니다. <br/>

1,025 바이트는 1 바이트에 따라 결정되므로 실제 마스터 키를 찾으려면 256 케이스가 필요합니다. <br/>
그러나 EKS를 생성하기 위해 1,025바이트 중에서 2바이트를 선택하면 XOR에서 추측한 바이트가 제거되고, 이 방법을 사용하여 수집되지 않은 EKS를 생성할 수 있습니다.<br/> 
이 말은 어차피 나중에 복호화 할 때 하나의 게싱된 바이트로 부터 추측된 모든 바이트는 원본 마스터 키 바이트와 동일한 값의 차이가 나고 이 차이는 짝수배의 xor 연산으로 날아가기 때문에 결론적으로 복호화에 아무런 영향을 미치지 않습니다. <br/>
따라서 0~255의 수 중 아무런 바이트로 추측을 해도 무방합니다. <br/>

![image](https://user-images.githubusercontent.com/46625602/210492720-86e9ca73-35ba-4ea3-aa10-815276143c31.png)

1MiBEKS 한 세트에서 추측할 수 있는 데이터 양은 1,025바이트입니다.<br/>
이제 두 세트의 경우를 고려해볼 때, 두 개의 독립된 집합에서 각 값을 선택하여 EKS를 생성할 때, 확률 1로 추측 값의 삭제가 수행되지 않습니다.<br/>

### 복호화 알고리즘 설계 

복호화 할때는 3개의 알고리즘이 필요합니다. 
* 1. FILE NAME 으로부터 2개의 랜덤 오프셋 R1||R2 추출 및 SP1, SP2 연산  
    * (base64url(MD5(Encrypted_master_key(16 bytes))||R1||R2)) 
* 2. FILE SIZE 에 따라 NBS(non-encrypted data block size) 결정되므로 NBS 를 구하는 알고리즘 
* 3. 추출된 EKS 들을 엮어서 게싱하는 코드 

1번과 2번에 해당하는 코드 입니다. 

![image](https://user-images.githubusercontent.com/46625602/210492749-02766183-0a3d-49d4-bf27-b85724ce3b51.png)

감염 파일은 부분적으로 암호화 되어 있고, 암호화된 부분과 암호화 되지 않은 부분이 번갈아 가면서 존재합니다. 암호화되는 부분의 사이즈는 0X1000 으로 일정합니다. 그러나 파일 사이즈에 따라 암호화 되지 않는 사이즈인 NBS 는 상이합니다. <BR/>
추출된 암호화 문으로부터 원본 파일을 XOR 하면 두개의 키 스트림을 연산한 결과를 알 수 있습니다. <br/>
![image](https://user-images.githubusercontent.com/46625602/210492798-5ef43326-603e-45de-9449-253c0567ffec.png)

추출된 키를 엮어 마스터 키를 연산하는 알고리즘 입니다. 
![image](https://user-images.githubusercontent.com/46625602/210492903-99c39b00-cb92-44d0-8471-d060f6b41658.png)


### Experiments

저자들이 제안한 기법을 적용하여 얼마나 효율적으로 마스터키를 복호화 할 수 있는지에 대한 실험입니다. <br/>
윈도우 7, 10(x86, x64)을 하이브 랜섬웨어에 감염시킨 후 실험이 진행되었으며, 두가지 상황을 가정했습니다. <br/>
* 첫 번째 실험은 무작위로 생성된 데이터 세트를 사용하여 마스터 키 복구 속도를 결정
* 두 번째 실험에서는 실제 환경을 고려하여 데이터 백업 없이 온라인으로 다운로드 가능한 데이터만 검색하여 사용

Python에서 만든 PoC(Proof-of-Concept) 코드를 사용하여 이 프로세스를 수행했으며, 이를 통해 마스터 키가 복구되었다. 마스터 키 복구 실험의 첫 번째 실험 결과는 다음과 같습니다. 왼쪽 그림을 보면 파일 사이즈에 따라 복구되는 키 사이즈가 달라지는데, 이는 파일 사이즈에 따라 적용되는 NBS 영역이 상이하기 때문입니다. 오른쪽 표는 파일 사이즈에 따라 복호화 시에 필요한 파일 개수를 정리한 표 입니다. 

![image](https://user-images.githubusercontent.com/46625602/210492963-8dc24a71-75eb-4ea7-9b42-65d18f9aea4b.png)

하이브 랜섬웨어 마스터키 복구 실험은 파일이 500KB 미만일 때 키의 90% 이상을 복구하기 위해 다수의 파일을 사용해야 했으며,  
1MiB보다 큰 파일에서 EKS를 수집했을 때 Hive 마스터 키의 90% 이상을 100~200개의 파일로 복구할 수 있었습니다.<br/>
일반적으로 파일 크기가 클수록 수집할 수 있는 EKS의 양이 크지만 경우에 따라 EKS의 양이 줄어들 수 있습니다. <br/>
우리의 실험 결과 0x280보다 큰 파일을 사용할 때 EKS를 효과적으로 획득할 수 있었다.<br/>

아래 그림은 마스터 키 복구 및 파일 복호화 실험 결과와 복구되지 않은 영역을 시각화한 것입니다. <br/>
실험 1-2는 마스터 키의 복구되지 않은 영역과 파일 암호화에 사용되는 (R1, R2)를 모두 보여줍니다. 실험 3-5는 (R1, R2)를 제외한 마스터 키의 복구 및 복구되지 않은 영역만 보여줍니다.<br/>
하이브 랜섬웨어는 시큐어 랜덤을 사용하기 때문에 (R1, R2)가 마스터 키 오프셋 전체에 고르게 분포되어 있다. 따라서 마스터 키의 복구 속도와 전체 파일의 암호 해독 속도는 비례해야 합니다.<br/> 
대부분의 경우 마스터 키의 처음 부분과 마지막 부분이 복구되지 않았다는 것을 발견했습니다. 마스터 키의 처음과 마지막 부분을 복구하려면 SP1이 0x0, 0x900000이거나 이에 가까운 감염된 파일의 원본 파일이 필요합니다. 복구된 마스터 키를 사용하여 암호화된 데이터를 복구한 결과는 다음과 같습니다. <br/> 

마스터 키는 약 72%의 파일을 해독하는 데 92%가 성공했고, 마스터 키는 약 82%의 파일을 해독하는 데 96%가 성공했으며, 마스터 키는 약 98%의 파일을 해독하는 데 성공했습니다.<br/>

![image](https://user-images.githubusercontent.com/46625602/210493000-ff546dcf-4c32-4db5-9b7f-d7c3b58515a9.png)

## Conclusions


* Hive Ransomware의 암호화 알고리즘 동작 과정 분석
* Hive Ransomware의 마스터 키 생성 및 저장 방식 확인 
* Hive Ransomware은 파일마다 다른 임의의 키 스트림으로 데이터를 XOR 처리하여 파일 암호화 확인
* 공격자의 개인키 없이 암호화된 파일 복호화 방식 제안 
* 논문에서 제안한 방법을 활용하여 암호화 키 스트림을 생성하는데 사용된 마스터 키의 95% 이상 복구 
* 복구실험 결과 제시 


---

**Refference:**

* Kim, Giyoon, et al. "A Method for Decrypting Data Infected with Hive Ransomware." arXiv preprint arXiv:2202.08477 (2022).
[https://arxiv.org/abs/2202.08477](https://arxiv.org/abs/2202.08477)
* [KISA 에서 배포한 HIVE 랜섬웨어 복구 도구](https://seed.kisa.or.kr/kisa/Board/128/detailView.do)
