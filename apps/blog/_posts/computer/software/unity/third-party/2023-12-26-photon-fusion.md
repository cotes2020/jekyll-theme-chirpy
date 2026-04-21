---
title: "Photon Fusion 2"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/20240827-140647.jpg"

date: 2023-12-26. 18:05
# last_modified_at: 2024-01-02. 06:31
# last_modified_at: 2024-01-07. 08:29
last_modified_at: 2024-08-29. 22:29
---

## 메모

---

### NetworkRunner

- 포톤 매니저
  - 포톤과 관련된 설정, 조작을 하기 위한
- INetworkRunnerCallbacks
  - 포톤 이벤트를 받기 위한 인터페이스, 매개변수에 runner가 들어감

- runner.Spawn (네트워크 Instantiate)
- runner.Despawn (네트워크 Destroy)

### NetworkObject

- 동기화 되기 위한 모든 오브젝트는 NetworkObject가 달려있어야 함
- 특별한 기능은 없고, 단순히 네트워크 상에서 누군지 구별하는 ID를 제공하는 컴포넌트

### NetworkTransform

- Transform을 동기화 시키는 Class
- 이를 상속 받으면 마찬가지로 Transform을 동기화

- Interpolation
  - 네트워크 상의 이유로 움직임이 뚝뚝 끊겨보이는 것을 보간시켜주는
  - 때문에 최상위/부모 오브젝트가 아니라, 자식 오브젝트(Mesh, Sprite 등 시각적인 표현)를 타겟으로 지정해줘야 함

### NetworkBehaviour

- FixedUpdateNetwork를 사용할 수 있음

### FixedUpdateNetwork

- 시뮬레이션 틱, `예측`과 `실제 연산`, 재예측 등 한 프레임에 여러 번 실행 될 수 있음
- Update, FixedUpdate 대신 사용
- (Update, FixedUpdate는 유니티 이벤트니까 여러 번 실행되면 어색하니까)
- GetInput(out 입력 구조체)로 입력을 받아올 수 있음

- Fusion2
- Proxies 더 이상 기본적으로 FixedUpdateNetwork를 실행하지 않음
- `Runner.SetIsSimulated(Object, true)`로 설정 가능 (Spawned에서 실행하는 게 베스트)

### Runner.Spawn

- `onBeforeSpawned`은 Server/Host에서만 동작하기 때문에, Network 속성을 변경하는 것에 쓸 것

### Spawned

- `if (Object.HasStateAuthority == false) return;`와 함께 쓴다면, VRChat에서 Start-IsMaster 같이 마스터(Host)가 딱 한 번 실행하는 명령 가능
  - 게임 상태, 오브젝트 상태 초기화

### PlayerRef

- 플레이어를 구분 짓는
- NetworkObject Spawn 시, 이 오브젝트의 입력 권한을 누가 가질지 PlayerRef로 설정 가능
  - NetworkBehaviour의 FixedUpdateNetwork, GetInput에서 어떤 입력 데이터를 가져올 지

### TickTimer

- 처음 생성된 시각 기준 Timer
- TickTimer.CreateFromSeonds(Runner, 시간)
- FixedUpdateNetwork에서 timer.Expired(Runner)로 체크하고 Despawn

- Awake는 네트워크랑 동기화 되지않기 때문에 Awake에서 초기화하면 안됨
- 오브젝트 생성 시 콜백 함수로 초기화해야 함 (그래야 처음 생성될 때 동기화)

### 물리 연산

- Rigidbody, NetworkRigidbody
  - 성능 상의 이유로 dll(어셈블리)로
- NetworkTransform 마찬 가지로 Interpolation
- NetworkTransform를 붙이지 않음
- 성능이 안좋아서 사용 자제

### 입력

- OnInput을 통해 입력을 받음 (INetworkRunnerCallbacks)
  - INetworkInput을 상속 받는, 사용자 정의 입력 데이터 구조체를 정의해야 함 (i.e. NetworkInputData)
  - OnInput에서 입력 데이터 구조체를 만들어 값을 설정하고, `input.Set(구조체);` 로 할당

- 입력을 받아올 때도 이를 받아와야 함

- 왜 이렇게 하느냐
  - 서버가 이동을 시켜야 하기 때문
    - 네트워크는 기본적으로 시간 지연이 발생한다
    - 한 명이 모든 권한을 가지는 서버가 되고, 클라는 서버에 데이터를 보내 서버가 처리를 하도록 하는데,
    - 클라가 공격 버튼을 눌렀다
    - 서버에서 승인을 받아야 하는데
    - 가는데 2틱, 돌아오는데 2틱 -> 4틱 후에 공격 하는 게 보여짐 (지연)
    - 클라이언트 사이드 예측을 사용
      - 2틱 정도 걸릴거라 예상하고 2틱 이후 미리 로컬로 공격
    - 클라는 이동에 관한 권한은 없고, 입력에 관한 권한만 있음
      - 서버가 이동 시킴
    - 클라가 이동 시킨 것은 예측

### 예측

- 예를 들어 총알이나 공
- 모든 이동을 서버로 검증 받으면 뚝뚝 끊겨 보일 것
- 하지만 앞으로 계속해서 갈 것 같음 -> 딜레이 만큼 예측
- 딜레이 이후 서버로부터 이동 데이터가 도착한 이후 다시 또 딜레이 만큼 예측, 반복
- 참고: 예측할 때 마다 FixedUpdateNetwork가 실행, FixedUpdateNetwork를 기반으로 예측

### NetworkBool

- 내부적으로 어셈블리 마다 불 바이트 크기가 다르다?
- 통신할 때 바이트 크기가 다르면 문제가 될 수 있어서
- 크기가 같은 것이 보장되도록, NetworkBool
- 다른 데이터는 상관 없음

### Network 어트리뷰트

- 프로퍼티 동기화
- StateAuthority 플레이어 기준
- OnChanged = nameof(함수 이름)
  - FixedUpdateNetwork는 예측, 실제 연산, 재예측 등 여러 번 호출 될 수 있음
  - OnChanged는 프레임 기준으로 변화를 감지
    - 한 프레임 사이에 True, False, True로 변화했다면, 한 프레임 사이에 값이 변화하지 않았음으로 OnChanged가 호출되지 않음
- VRChat과 마찬가지로, StateAuthority(VRChat에선 Owner)가 아닌 플레이어가 값을 변경하면, 로컬로만 동작하고, 언제든지 StateAuthority(Owner)의 값으로 동기화 될 수 있음

- Fusion 2
- 프로퍼티의 Get/Set을 커스텀 코드로 대신하기 때문에, Get/Set을 기준으로 변화를 감지하면 안되고, 별도의 Setter 구현도 로컬로만 동작함
- `Spawned`에서 `ChangeDetector` 변수에 `GetChangeDetector(ChangeDetector.Scoure.SimulationState)`를 할당하고, `Render`에서 아래와 같이 변화를 감지

```cs
public override void Render()
{
    foreach (var change in _changeDetector.DetectChanges(this))
    {
        switch (change)
        {
            case nameof(변수):
                // 변화 시 수행할 명령
                break;
        }
    }
}
```

- 이러한 동작은 여러 번 실행될 수 있고 (예측 이후 틀린 예측을 고치려고 한 번 더 실행된다던지), 스킵 될 수 있고 (데이터가 도달하기도 채 전에 원래 값으로 돌아온다던지), 누락될 수 있음 (Packet Drop)
- RPC보다 빠른 시점에 변화를 감지할 수 있음 (값이 변화한 네트워크 Tick 이후 바로)

- Spawned가 실행되는 시점에는 이미 Network 어트리뷰트가 붙은 속성들이 모두 동기화 된 이후

### Render

- 시각적인 계산은 Render에서 Like LateUpdate
- FixedUpdateNetwork는 예측, 실제 연산, 재예측 등 여러 번 호출 될 수 있기 때문에, 네트워크가 필요없는 시각 적인 연산은 중복 연산으로 자원 낭비가 될 수 있음

### Remote Procedure Calls, RPC

- 원격 Procedure/함수 콜
- 엄격하게 동기화 되지 않음, 틱도 사용 안함
  - 지연되거나, 로스, 손실될 수 있음
  - Update에서 그냥 보냄
    - `if (Input && Object.HasInputAuthority) RPC_SendMessage("Sans");`
- 크게 중요하지 않은 것, Like 하스 감정표현

### ETC

- 위는 호스트 모드
- Shared Mode: 각자 이동 권한을 가짐
- Predictive Spawn: 클라에서 Spawning도 예측
- Lag Compensation: 지연 보상, 클라 기존으로 판정 (핵 문제)
- Area Of Interest: Interest Area에 있지 않는 오브젝트는 동기화하지 않는
