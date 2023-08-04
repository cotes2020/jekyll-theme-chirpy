---
title : "Data Structure : Binary Tree"
categories : [Study, Data Structure]
tags : [Binary Tree, Binary Search Tree, Thread Binary Tree, Balanced Binary Search Tree, Heap]
---

## 이진 트리
<hr style="border-top: 1px solid;"><br>

트리의 모든 노드 차수(자식 노드 개수)를 2 이하로 제한하여, 전체 트리의 차수가 2 이하가 되도록 정의한 것

다음과 같은 특징이 있음.

1. 노드가 n개인 이진 트리는 항상 간선이 n-1개이다.

2. 높이가 h인 이진 트리가 가질 수 있는 노드 개수는 최소 (h+1)개에서 최대 (2^h+1-1)개이다. (h >= 0)

<br>

![image](https://user-images.githubusercontent.com/52172169/167242912-97179e4d-62f1-4287-a08b-21f98259a4c3.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>


## 순차 자료구조로 구현한 이진트리
<hr style="border-top: 1px solid;"><br>

이진 트리를 1차원 배열로 표현 시 노드 번호를 배열의 인덱스로 사용.

인덱스 값은 1부터 시작. (root 노드의 인덱스는 1)

<br>

![image](https://user-images.githubusercontent.com/52172169/167242926-87436c90-6e18-48ed-bd3a-48bd7e24a19f.png)

![image](https://user-images.githubusercontent.com/52172169/145019062-b2820a62-b00c-4e58-a0b6-3ea73cda8a84.png)

![image](https://user-images.githubusercontent.com/52172169/167242945-f01110ca-3c2d-4bcb-bd2b-97f9d94c6722.png)

<br>

+ i 번째 노드의 부모 노드 : i//2

+ i 번째 노드의 왼쪽, 오른쪽 자식 노드 : 2*i, 2*i+1

+ 루트 노드 : 1

<br>

![image](https://user-images.githubusercontent.com/52172169/167242962-8ae821bf-2204-43df-a642-9881a45b4d32.png)

<br>

일차원 배열로는 쉽게 구현 가능하며, 인덱스 규칙에 따라 부모 노드와 자식 노드를 쉽게 찾을 수 있음.

하지만 편향 이진 트리와 같은 트리는 메모리 낭비가 심해지게 됨.

<br><br>
<hr style="border: 2px solid;">
<br><br>


## 연결 자료구조로 구현한 이진트리
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167242974-21fa81d3-aa08-4af8-a004-d57543ed90dd.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/167242988-10471f9d-905a-4c31-96df-c1670e6c6bc4.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>


## 이진트리 순회
<hr style="border-top: 1px solid;"><br>

순회란 모든 원소를 빠트리거나 중복하지 않고 처리하는 연산을 의미함.

이진 트리는 비선형 자료구조이므로 현재 노드를 처리한 후에 어떤 노드를 처리할지 결정하는 기준을 정해 놓은 순회 연산이 필요함.

<br>

순회를 위해 수행할 수 있는 작업은 세 가지로 정의할 수 있음.

+ D : 현재 노드를 방문하여 처리
+ L : 현재 노드의 왼쪽 자식 노드로 이동 (왼쪽 서브 트리)
+ R : 현재 노드의 오른쪽 자식 노드로 이동 (오른쪽 서브 트리)

<br>

위의 세 가지 작업 순서에 따라 세 가지로 구분 할 수 있음.

1. 전위 순회 : DLR 
2. 중위 순회 : LDR
3. 후위 순회 : LRD

<br><br>

### 전위 순회 (Preorder Traversal, DLR)
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243063-d593a89e-192f-4b86-9a45-c3e7c1fec864.png)

<br><br>

### 중위 순회 (Inorder Traversal, LDR)
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243072-bcfdf4c1-4c71-478c-8738-ade794eecb6d.png)

<br><br>

### 후위 순회 (Postorder Traversal, LRD)
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243091-34b0bdb3-9819-4d58-9388-6dec2626e86a.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/167243099-71529970-78e7-4e81-9ff7-e0973494e975.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Thread 이진 트리
<hr style="border-top: 1px solid;"><br>

이진 트리는 부모 노드와 자식 노드의 이진 트리 기본 구조가 각 레벨에서 순환적으로 반복하여 전체 트리가 구성되는 구조임.

따라서 각 노드에서의 순회 연산을 재귀호출 방식을 이용해 순환적으로 반복하여 전체 트리에 대한 순회를 처리함.

하지만, 재귀호출 방식은 알고리즘이나 함수 구현은 간단하지만 성능에서 보면 비효율적일 수 있음. 

<br>

**따라서 재귀호출 없이 순회할 수 있도록 수정한 이진트리가 스레드 이진 트리임.**

**스레드 이진 트리**는 자식 노드가 없는 경우에 링크 필드를 널 포인터 대신 순회 순서상의 다른 노드를 가리키도록 설정한 것. 

이런 링크 필드를 스레드라고 함.

<br>

![image](https://user-images.githubusercontent.com/52172169/167243131-ceb82e71-aa6a-453c-8b28-34b5db41c6f6.png)

<br>

+ 현재 노드 직전에 처리한 노드, 즉 선행자에 대한 포인터는 왼쪽 ----> ```(isThreadLeft)```

+ 현재 노드 다음에 처리할 노드, 즉 후행자에 대한 포인터는 오른쪽 ----> ```(isThreadRight)```

+ isThread 필드는 링크 필드가 자식 노드에 대한 포인터인지, 스레드가 저장되어 있는지 구별하기 위한 태그.

  + isThreadLeft 필드가 true ----> left 링크 필드는 선행자를 가리키는 스레드가 됨.

  + isThreadLeft 필드가 false ----> left 링크 필드는 왼쪽 자식 노드를 가리키는 포인티가 됨.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 이진 탐색 트리 (BST)
<hr style="border-top: 1px solid;"><br>

이진 탐색 트리는 이진 트리를 탐색용 자료구조로 사용하기 위해 원소 크기에 따라 노드 위치를 정의한 것임.
: 이진 탐색 + 이진 트리 = 이진 탐색 트리

<br><br>

### 이진 탐색 트리 정의 
<hr style="border-top: 1px solid;"><br>

1. 모든 원소는 서로 다른 유일한 키(고유 값)를 갖는다.

2. 쪽 서브 트리에 있는 원소들의 키는 그 루트의 키보다 작다.

3. 오른쪽 서브 트리에 있는 원소들의 키는 그 루트의 키보다 크다.

4. 왼쪽 서브 트리와 오른쪽 서브 트리도 이진 탐색 트리이다.

<br>

즉, ```왼쪽 노드 < 루트 노드 < 오른쪽 노드```

![image](https://user-images.githubusercontent.com/52172169/167243163-955fd259-99b8-42fd-a4f2-31c8cf6f831a.png)

<br><br>

### 이진 탐색 트리의 탐색 연산
<hr style="border-top: 1px solid;"><br>

키 값이 x인 원소를 탐색하는 경우 탐색은 항상 루트 노드에서 시작.

<br>

1. 키 값과 루트 노드 값 비교
 
    1-1. 키 값 == 루트 노드 값 -> 성공

    1-2. 키 값 > 루트 노드 값 -> 오른쪽으로 이동.

    1-3. 키 값 < 루트 노드 값 -> 왼쪽으로 이동.

<br>

2. 서브 트리로 가서 해당 노드를 루트 노드로 하여 1번 반복.

<br>

![image](https://user-images.githubusercontent.com/52172169/167243182-e1b57515-df18-455e-8a8b-5b4a4466c94e.png)

<br><br>

### 이진 탐색 트리의 삽입 연산
<hr style="border-top: 1px solid;"><br>

삽입하려면 이진 탐색 트리에 같은 원소가 있는지 먼저 확인해야 함. (특징 1번)

+ 탐색 성공 시 삽입 연산은 하지 않음.

+ 탐색 실패 시 삽입 연산을 수행하며, 삽입 위치는 실패가 발생한 현재 위치임.

<br>

![image](https://user-images.githubusercontent.com/52172169/167243189-114ae840-87af-4162-acca-397a4d41d843.png)

<br><br>

### 이진 탐색 트리의 삭제 연산
<hr style="border-top: 1px solid;"><br>

삭제하려면 이진 탐색 트리에서 삭제할 노드의 위치를 탐색해야 함.

삭제할 노드는 자식 노드 수에 따라 후속 처리를 해줘야 함.

+ 삭제할 노드가 단말 노드인 경우 -> 차수 = 0

+ 삭제할 노드가 자식 노드를 한 개 가진 경우 -> 차수 = 1

+ 삭제할 노드가 자식 노드를 두 개 가진 경우 -> 차수 = 2

<br>
<br>

#### 삭제할 노드가 단말 노드인 경우
<hr style="border-top: 1px solid;">

![image](https://user-images.githubusercontent.com/52172169/167243204-6f32e1a9-c5de-47c2-acb9-3cc2f99a2e33.png)


그냥 삭제하면 됨.

<br><br>

#### 삭제할 노드가 자식 노드를 한 개 가진 경우
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243210-c982ab27-b731-4bb2-bce5-b05e6d1d6eef.png)


노드 삭제 후 자식 노드로 그 자리를 채우면 됨.

<br><br>

#### 삭제할 노드가 자식 노드를 두 개 가진 경우
<hr style="border-top: 1px solid;"><br>

이 경우에는 직계 자식 노드뿐만 아니라 트리 전체에서 찾아야 함.

노드가 삭제되고 자손 노드에게 자리를 물려준 후에도 BST가 유지가 되어야 함.

<br>

후계자 노드는 

1. 왼쪽 서브 트리에 있는 노드들의 키 값보다 커야 하고
2. 오른쪽 서브 트리에 있는 노드들의 키 값보다 작아야 함.

<br>

따라서 **삭제 노드의 왼쪽 서브 트리에서 가장 큰 자식 노드로 채우거나, 오른쪽 서브 트리에서 가장 작은 자식 노드로 채워야 함.**

<br>

![image](https://user-images.githubusercontent.com/52172169/150636318-89c11e63-b774-4242-89ab-f3886342e06a.png)

![image](https://user-images.githubusercontent.com/52172169/150636306-ec6b6cd9-f272-4a83-b26e-c30134b5d5f8.png)


<br><br>
<hr style="border: 2px solid;">
<br><br>


## 균형 이진 탐색 트리
<hr style="border-top: 1px solid;"><br>

이진 탐색 트리에서 좌우 균형이 잘 맞으면 탐색의 성능이 높아짐.

이러한 원리로 이진 탐색 트리에 왼쪽 서브 트리 높이와 오른쪽 서브 트리 높이에 대한 균형 조건을 추가하여 정의한 트리를 **Balanced Binary Search Tree**라 함.

대표적으로 AVL 트리, 레드블랙 트리가 있음.

<br>

레드블랙트리 
: <a href="https://zeddios.tistory.com/237" target="_blank">zeddios.tistory.com/237</a>

<br>

AVL 트리 (Adelson-Velskii, Landis Tree)는 대표적인 균형 이진 탐색 트리임.

AVL 트리는 각 노드에서 왼쪽 서브 트리의 높이 HL과 오른쪽 서브 트리 높이 HR의 차이가 1 이하인 균형 트리임.

AVL 트리의 특징은 아래와 같음.

<br>

* 왼쪽 서브 트리 < 부모 노드 < 오른쪽 서브 트리

* HL-HR 인 노드의 균형 인수```(BF, Balance Factor)```를 관리함.

* 각 노드의 균형 인수로 ```-1, 0, 1``` 값만 가지게 함으로써 왼쪽 서브 트리와 오른쪽 서브 트리의 균형을 항상 유지.

<br>

![image](https://user-images.githubusercontent.com/52172169/167243245-8d745aaf-483d-42e9-8687-e2298ae8b6dc.png)

<br>

**AVL 트리의 예**

![image](https://user-images.githubusercontent.com/52172169/167243251-ee60117f-61d4-451e-8bf5-610b8bfe1930.png)

<br>

**비AVL 트리의 예**

![image](https://user-images.githubusercontent.com/52172169/167243267-8811035e-8c11-417c-bd9e-9a0e64092231.png)

<br><br>

### AVL 트리 회전 연산
<hr style="border-top: 1px solid;"><br>

AVL 트리가 불균형을 이룬면 균형으로 맞춰줘야 함.

즉, 삽입 삭제 후 균형 인수를 확인하여 균형을 맞추는 재구성 작업이 필요. 이 작업은 회전 연산을 통해 이루어짐.

불균형 유형에는 4가지가 있음. (LL, RR, LR, RL)

<br><br>

#### LL 유형
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243312-5b9b2949-bf69-4156-ac53-ecf456417f14.png)

<br>

#### RR 유형
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243317-f539846b-e336-4fdb-acde-e3c3a11f475c.png)

<br>

#### LR 유형
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243322-ab7ac2af-b691-4cea-a2b9-497b93951145.png)

<br>

#### RL 유형
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/167243331-c9f4194b-cb3e-4597-bdd9-be10949f98f5.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>


## 히프 (Heap)
<hr style="border-top: 1px solid;"><br>

완전 이진 트리 (Complete Binary Tree)
: 높이가 h이고, 노드 수가 n개일 때 노드 위치가 1번부터 n번까지의 위치가 포화 이진 트리와 일치하는 트리.

<br>

힙은 완전 이진 트리에 있는 노드 중에서 키 값이 가장 큰 노드나 키 값이 가장 작은 노드를 찾기 위해 만든 자료구조.

<br>

+ 키 값이 가장 큰 노드를 찾기 위한 히프 -> 최대 히프 (Max Heap)

+ 키 값이 가장 작은 노드를 찾기 위한 히프 -> 최소 히프 (Min Heap)

<br>

![image](https://user-images.githubusercontent.com/52172169/167243340-2223e3c5-9bde-465c-948a-25be3b7fef40.png)

<br>

+ 최대 힙은 항상 부모 노드 >= 자식 노드인 완전 이진 트리

+ 최소 힙은 항상 부모 노드 <= 자식 노드인 완전 이진 트리

<br>

**힙이 아닌 트리 예시**  

![image](https://user-images.githubusercontent.com/52172169/167243355-9757dc3d-e377-4a7c-b8a4-2228234440d4.png)

<br><br>

### 힙의 삽입 연산
<hr style="border-top: 1px solid;"><br>

완전 이진 트리의 형태 조건을 만족하기 위해서 현재의 마지막 노드의 다음 자리를 확장하여 자리를 생성 후 임시로 저장.

현재 위치에서 최대 힙(최소 힙)인지에 따라 부모 노드와의 값을 비교 후 자리 확정.

<br>

![image](https://user-images.githubusercontent.com/52172169/167243371-e884e081-1e33-4bce-aae2-4594f8493258.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/167243390-2e25db34-e497-4650-9e02-e723812c4b85.png)

<br><br>

### 힙의 삭제 연산
<hr style="border-top: 1px solid;"><br>

힙에서 삭제는 언제나 루트 노드에 있는 원소를 삭제하여 반환함.

중요한 것은 루트 노드 삭제 후에도 완전 이진 트리의 형태와 노드의 키 값에 대한 히프의 조건이 유지되어야 함.

<br>

1. **루트 노드 삭제.**

2. **완전 이진 트리 형태 유지를 위해 마지막 노드 삭제.**

3. **삭제한 마지막 노드를 루트에 임시 저장.**

4. **키 값의 관계가 최대(최소) 힙이 되도록 조정.**

5. **임시로 루트에 옮겨논 값과 왼쪽, 오른쪽 자식 노드와 값 비교.**

6. **가장 큰(작은) 값과 자리 swap.**

7. **다시 자식 노드와 비교 후 6번 실행.**

<br>

![image](https://user-images.githubusercontent.com/52172169/167243425-e951ee71-58f6-4f24-b549-50cbda7bd2ad.png)

<br>
<br>

### 힙의 구현
<hr style="border-top: 1px solid;"><br>

+ 부모 노드의 인덱스 : i//2

+ 왼쪽 자식 노드 인덱스 : i*2

+ 오른쪽 자식 노드 인덱스 : i*2+1

<br>

![image](https://user-images.githubusercontent.com/52172169/167243465-d2b77cfc-3eed-416e-beb0-1ba62d9202a1.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>


## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://foxtrotin.tistory.com/184?category=635012" target="_blank">foxtrotin.tistory.com/184?category=635012</a>  
: <a href="https://foxtrotin.tistory.com/187?category=635012" target="_blank">foxtrotin.tistory.com/187?category=635012</a>  

<br>

BST 
: <a href="https://foxtrotin.tistory.com/190?category=635012" target="_blank">foxtrotin.tistory.com/190?category=635012</a>  

<br>

AVL 
: <a href="https://foxtrotin.tistory.com/191?category=635012" target="_blank">foxtrotin.tistory.com/191?category=635012</a>  

<br>

Heap 
: <a href="https://foxtrotin.tistory.com/205?category=635012" target="_blank">foxtrotin.tistory.com/205?category=635012</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>
