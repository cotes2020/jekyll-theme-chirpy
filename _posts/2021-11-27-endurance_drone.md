---
title: 곤충 시각 기관 모방 센서와 인공신경망을 적용한 실내 드론 시스템
author: JeongHyun Kim, JaeMin Kim, JeongHyun Nam, SangYoon Kim, Daniel Kwak
date: 2021-11-27 01:15:00 +0900
categories: [Exhibition, Others]
tags: [post, drone, ai, rtos, embedded, arm, tensorflow, kalman, pid, rl] # TAG names should always be lowercase, 띄어쓰기도 금지
---

<style>
  h1 {
    color: red;
  }

  h2 {
    font-weight: bold;
    color: chartreuse;
  }

  img {
    display: block;
    margin: auto;
  }

  .caption {
    display: block;
    color: grey;
    margin-bottom: 1rem;
    text-align: center;
    font-size: 0.85rem !important;
  }
</style>

## 작성중인 초안입니다!!

## 연구 동기

4차 산업혁명 시대에 드론은 기존 산업과 유기적으로 접목되어 사람을 대신하는 경우가 많아지고 있습니다.
예를 들어, 재난현장에서는 사람이 직접 들어갈 수 없는 건물 내부를 수색하는 역할을 해내고 있습니다. 이를 위해서는 비좁은 실내에서도 자유롭게 비행할 수 있게 작은 크기를 가지면서 사용할 수 있는 정보가 제한된 실내서도 사용할 수 있는 항법 장치가 필요 합니다.
이러한 경우엔 SLAM등의 카메라와 인공신경망을 사용하여 경로를 추정하게 되지만, 상당한 전력과 처리성능을 필요로 하기 때문에 실내 드론에 적용하는 것에는 한계가 있습니다.
본 연구에서는 이러한 문제를 해결하기 위해 GPS, 카메라 등의 장치 대신 초파리의 시각 구조망을 모방한 넓은 시야각을 가진 밝기 센서 배열과 효과적으로 장애물을 감지할 수 있는 거리센서를 이용해 저전력으로도 충돌회피 및 지형지물을 파악하고 목표 지점까지 주행하는 실내 드론 시스템을 개발하고자 합니다.

## 독창성 및 차별성

- 현재 상용화된 대부분의 드론은 실외용으로 설계되어 실내에서 사용이 어려운 GPS 등의 무선 신호 기반 항법 장치나 고성능 영상 처리 기반 경로 탐색 시스템을 갖추고 있습니다. 따라서 저전력 및 저성능으로 작동되어야 하는 실내용 드론에 적합한 저차원, 저용량의 센서 데이터를 활용한 경로 탐색 방식을 사용하여 기존보다 경량화된 장애물 회피 및 경로 탐색 방식을 구현할 수 있습니다.
- [선행 연구(Bardienus P. Duisterhof 외 7명, 2019)](https://arxiv.org/abs/1909.11236)는 어두운 실내 공간에서 밝은 목표물을 찾아가는 제한적인 상황에서의 시나리오를 가정했습니다. 본 연구는 선행 연구를 기반으로 보다 보편적인 실내 공간에서 적용시키기 위해, 특정 점멸 패턴을 발산하는 물체를 식별하여 경로를 탐색할 수 있도록 발전시키는 것을 목표하고 있습니다.
- 목표물을 찾는 밝기 센서에 지정 과제로 제시된 곤충의 시각 기관을 모방한 배열구조로 센서를 배치하여 보다 넓은 시야각을 확보함으로써 선행 연구에 비해 보다 발전된 성능의 경로 탐색 시스템을 구현할 수 있을 것으로 기대합니다.
- 임베디드 시스템 위에서 인공신경망을 기반으로 한 추론을 진행하는 의사 결정 방식은 최근에 대두되고 있는 새로운 방식입니다. 저용량, 저성능(Cortex-M4, 80MHz, 256KB)의 시스템에서 RTOS를 탑재한 채로 **실시간** 요구 사항을 충족하면서 추론을 진행하는 것 또한 기존의 전통적인 시스템에서의 결정론적인 의사 결정 방식에 대비해 차별화되고 있습니다.

## 연구 방법

본 연구는 저성능 프로세서 (Cortex-M4)와 제한된 관측 센서 (거리 4개, 밝기 1개)의 환경에서 POMDP를 통해 암실에서 장애물을 회피해 밝은 전등을 향해 비행하는 소형 드론 제어 전반에 대해 연구한 선행 연구를 기반으로 이것의 방법론을 차용하여 예산을 지원하는 교내 사업단 지정 주제에 맞게 발전시키는 방식으로 진행했습 니다.

본 연구에서는 앞서 설명한 목적과 차별성을 달성하기 위해 아래 세 가지에 대한 연구를 진행했습니다.

- 6개의 거리 센서와 겹눈 구조로 배치된 9개의 밝기 센서를 관측 입력으로 사용하는 POMDP 모델 설계와 이것을 기반으로 위의 목적을 달성할 수 있는 DQN 알고리즘을 Tensorflow Lite를 통해 구현
- 구현한 모델과 알고리즘을 평가(보상)하는 함수와 평가 방법 설계, 테스트 시뮬레이션 계획 및 실제 실험 환경 조성
- 위의 거리, 밝기 센서와 드론 비행에 필요한 자세 제어 센서에서 받은 입력값으로 강화학습 추론 알고리즘을 실행하여 그 출력을 실제 비행에 반영하는 Nuttx RTOS 및 Cortex-M4 프로세서 기반 비행 제어 시스템

## 팀 구성 및 역할

### 지도

- **문준** (전기공학전공, 교수) : 지도교수
- **이명훈** (전기정보통신기술연구소, 박사후연구원) : 팀장
- **이승준** (16기, 한국항공우주연구원) : 산업체 멘토

### 연구

- **김정현** (25기, 전기공학전공) : 연구 총괄, 강화 학습, 회로 설계, 임베디드 프로그래밍(RTOS)
- **김재민** (26기, 기계공학부) : 강화 학습 시뮬레이터 제작, 제어 공학 (칼만 필터)
- **남종현** (26기, 융합전자공학부) : 회로 설계, 회로/PCB 설계, 제어 공학 (PID 제어), 하드웨어 제작
- **김상윤** (25기, 신소재공학부/다중:융전) : 신호처리(FFT)
- **곽다니엘** (24기, 생체공학전공) : 백업 비행 컨트롤러 연구(DJI 社 NAZA)

## 연구 내용

**연구의 내용이 여러 방면에 다각도로 걸쳐져 있어 분야별로 챕터와 연구자를 구분하여 서술했습니다.**

### 강화학습 시뮬레이터 제작 **(김재민)**

AirSim은 Unreal Engine을 기반으로 하는 드론, 자동차 등을 위한 시뮬레이터로 자율 주행 차량을 위한 딥 러닝, 컴퓨터 비전 및 강화 학습 알고리즘을 실험할 수 있습니다. 시뮬레이션에서 차량과 상호 작용할 수 있도록 여러가지 API를 노출하는데, 이러한 API를 사용하여 이미지 검색, 상태 가져오기, 차량 제어 등을 수행할 수 있습니다. API는 RPC를 통해 노출되며 다양한 언어를 통해 액세스할 수 있습니다.

본 연구에서는 강화학습 Agent를 학습시키기 위해 실제 드론을 사용하는 것은 매우 번거롭고 현실 세계의 노이즈로 인해 데이터의 품질이 더욱더 안 좋아지기 때문에 (강화학습의 수집된 데이터셋은 보통 통제된 시뮬레이터 상태에서도 학습이 용이하지 않음) 시뮬레이터를 통해 데이터셋을 수집하고 학습하기 위하여 사용했습니다.

학습의 효율을 위해 데이터셋을 수집할 때 9개의 다른 방(Room)을 하나의 맵에 구성하고 9개의 방을 광원,장애물, 드론 초기 위치 세가지 변수를 변형하며 경우의 수를 만들어 여러 조합으로 맵을 구성하여 과적합(Overfitting)이 발생하지 않도록 고려했습니다.

#### 맵 전경

<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_map.png">
<p class="caption">맵 전경, 9개의 방(Room)이 보입니다.</p>
<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_room.png">
<p class="caption">방의 내부 예시, 광원이 왼쪽, 장애물이 가운데, 스폰 장소가 왼쪽임을 확인할 수 있습니다. </p>
또한 일정간격으로 빛나는 광원의 제작을 위해 블루프린트에서 위와 같은 이벤트그래프를 구성하였습니다.
예를 들어 전방 3개의 밝기 센서는 간단한 영상 처리를 통해 시뮬레이팅 했습니다.

#### 밝기 센서 영상처리 예시

<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_normal.png">
<p class="caption">과정 1. 시뮬레이터에서 가져온 원본 카메라 사진</p>
<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_bw.png">
<p class="caption">과정 2. 흑백으로 원본 사진을 변환</p>
<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_thumbnail.png">
<p class="caption">과정 3. 3개의 센서의 시야(ROI)를 시뮬레이트 하기 위해 가로 3픽셀로 썸네일라이징</p>

#### 광원 프로그래밍

연구의 목적 상 광원은 일정한 주파수로 점멸하는 것을 가정하므로, 언리얼 엔진의 광원을 아래와 같이 프로그래밍하여 점멸하게 하였습니다.

<img src="/assets/img/post/2021-11-27-endurance_drone/airsim_light.png">
<p class="caption">프로그래밍한 이벤트 그래프</p>

### IMU 센서와 칼만 필터를 이용한 자세 추정 (사원수 기반) **(김재민)**

드론의 기본적인 비행을 위해서, IMU 센서를 이용해 드론의 현재 자세를 추정하는 알고리즘을 직접 구현하였습니다. 아래는 구체적인 구현을 위한 수식 설명입니다.

#### 자이로 센서

자이로 센서(각속도계)는 가속도과 지자기 센서보다 훨씬 높은 정확도를 가지지만 일종의 편향을 동반하기 때문에 일정 주기마다 편향 값을 보정하는 작업이 필수적입니다. 이를 위해 상보 필터를 적용하였습니다.
<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/imu_quad.jpg">

<p class="caption">사원수를 이용한 자이로 센서의 각도 표현 방법</p>

#### 가속도 센서

아래와 같은 수식을 현재 각도를 얻을 수 있습니다. yaw축의 각도는 알 수 없으며 roll, pitch 축의 각도만 추정할 수 있습니다.
<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/imu_acc.jpg">

<p class="caption">3축 가속도로부터 중력가속도를 기준으로한 각도 계산 방법</p>

#### 칼만 필터

얻어온 가속도, 자이로 센서 각도를 기반으로 상보 필터를 한번 거친 후 칼만 필터를 통해 정확한 자세를 추정하게 됩니다.

<img width="250px" src="/assets/img/post/2021-11-27-endurance_drone/imu_kalman.jpg">
위와 같은 변수를 정의합니다. 각각 아래와 같은 의미를 띕니다.
- A: x의 변수간 관계를 통해 과거와 현재사이의 물리적인 수식을 행렬로 정리해 놓은 것
- Q, R: 잡음
- H: 위치를 센서로 측정하여 기준값으로 잡아 칼만필터를 연산하고 싶으면 H=[1 0], 반대로 속도를 측정하여 연산 하려면 H=[0 1]입니다.

이 변수들을 통해 아래와 같은 연산을 진행합니다. (A와 H는 상태 공간 방정식)

- 초기값 선정: 이전 스텝에서의 결과값이 그 다음 스텝에서의 초기값(x, P)으로 사용
- 추정값과 오차 공분산 예측: 현재상태의 x,P를 추측하는값을 계산
  <img width="200px" src="/assets/img/post/2021-11-27-endurance_drone/imu_kalman1.jpg">
- 칼만 이득 계산
  <img width="300px" src="/assets/img/post/2021-11-27-endurance_drone/imu_kalman2.jpg">
- 추정값 계산: 위에서 구한값들과 센서로 측정한값(z)만을 가지고 현재상태의 값을 추정
  <img width="300px" src="/assets/img/post/2021-11-27-endurance_drone/imu_kalman3.jpg">
- 오차 공분산 계산
  <img width="200px" src="/assets/img/post/2021-11-27-endurance_drone/imu_kalman4.jpg">
- 다시 초기값 선정 과정으로 돌아가 반복

### 센서 제어 **(남종현, 김정현)**

본 연구에서는 드론 자세 감지를 위한 IMU 센서, 광원을 감지하기 위한 광원 센서, 장애물 회피를 위한 거리 센서를 제어하여 값을 읽어와야 합니다. 이 단원에서는 센서 제어에 대한 자세한 세부 사항을 설명합니다. 연구에서는 사용되지 않은 GPS 센서와 거리센서의 UART 프로토콜도 참고용으로 기술하였습니다.
<img width="400px" src="/assets/img/post/2021-11-27-endurance_drone/sensor_diagram.jpg">

<p class="caption">센서들이 장착되는 개요도</p>

#### IMU 센서

IMU 센서로는 InvenSense 社의 MPU9250을 사용했습니다. 본 연구에서는 이 센서와 함께 부가 전원 회로를 구현한 모듈인 GY-9250을 사용했습니다. MPU9250은 3축 가속도, 3축 각속도를 제공하며 3축 지자계를 읽어낼 수 있는 아사히 카세이社의 AK8963를 Slave 장치로 내장하고 있으며 특정 방식을 이용해 접근할 수 있습니다. MPU9250 센서의 상세 명세는 아래와 같습니다.

- MPU-9250과 AK8963을 MCM 방식으로 통합
- I2C, SPI 통신 방식
- 16비트 ADC를 통해 ±250 ~ 2000°/sec의 각속도와 ±2 ~ 16g의 범위의 가속도를 측정 가능
- 내장된 AK8963센서로 ±4800uT의 자기장 측정 가능
- 하드웨어 LPF와 DMP 방식 내장

본 연구에서는 이 센서를 아래와 같은 설정셋을 통해 사용했습니다.

- 가속도 측정 범위 ±2g, 각속도 측정 범위 ±250°/sec : 정숙하게 움직이는 시나리오를 가정하여 최대한 고해상도를 설정했습니다.
- SPI 통신 방식, 400kHz 클럭 : I2C보다 빠른 통신 속도를 보장하기 위해 SPI 통신 방식을 사용하였습니다. 필요다면 수MHz 클럭 속도로 통신할 수 있습니다.
- 가속도와 각속도 측정에 대하여 LPF를 5Hz로 설정 : 프로토타입의 특성상 IMU 센서가 브레드보드에 장착되어 있으며, 따로 물리적 댐퍼도 없어 진동에 매우 취약한 상태에서, 진동때문에 너무 큰 오차가 발생하는 상황이였습니다. (10~20도의 자세 각도 오차) 이때문에 소프트웨어적으로 해결하고자 극단적일 수도 있는 범위의 필터링을 진행하였고 정숙한 실내 비행상태에서 양호한 반응 특성을 얻어낼 수 있었습니다.
- 지자계 비활성화 : AK8963 통신 구현 실패 (통신 프로토콜 이해 부족 및 시간 부족에서 기인)로 인해 절대 yaw각 보정을 포기하고 지자계 센서를 비활성화 한 상태입니다. 대신 진동으로 생기는 yaw각 적분 오차를 완화시키기 위해 일정 각도 이하의 각도 변화(현재 0.3도)는 각도 산정에 반영하지 않고 드랍합니다. 정숙한 상황을 가정한 극단적인 대책 중 하나입니다.

이렇게 얻어온 값을 위에서 설명한 상보 필터를 통해 조합하여 각도를 산출한 후 칼만 필터를 통해 현재 자세를 추정합니다.
<img width="650px" src="/assets/img/post/2021-11-27-endurance_drone/imu_diagram.jpg">

<p class="caption">자세 추정 방식 요약 다이어그램</p>

##### 구현

이 부분의 구현 전체 소스는 [여기](https://github.com/Dictor/hamstrone-drone/blob/master/mpu9250.c)를 참고하세요.

```c
uint8_t initRegister[INIT_REGISTER_COUNT][2] = {
  {MPUREG_PWR_MGMT_1, BIT_H_RESET},
  {MPUREG_PWR_MGMT_1, 0x01},
  {MPUREG_PWR_MGMT_2, 0x00},
  {MPUREG_ACCEL_CONFIG, BITS_FS_2G},
  {MPUREG_ACCEL_CONFIG_2, BITS_DLPF_CFG_10HZ},
  {MPUREG_GYRO_CONFIG, BITS_FS_250DPS},
  {MPUREG_CONFIG, BITS_DLPF_CFG_10HZ},
  {MPUREG_INT_PIN_CFG, 0x12},
  {MPUREG_USER_CTRL, 0x30},
  {MPUREG_I2C_MST_CTRL, 0x0D},
  {MPUREG_I2C_SLV0_ADDR, AK8963_I2C_ADDR},
  {MPUREG_I2C_SLV0_REG, AK8963_CNTL2}, // ak reset
  {MPUREG_I2C_SLV0_DO, 0x01},
  {MPUREG_I2C_SLV0_CTRL, 0x81},
  {MPUREG_I2C_SLV0_REG, AK8963_CNTL1},
  {MPUREG_I2C_SLV0_DO, 0x12},
  {MPUREG_I2C_SLV0_CTRL, 0x81}
};
```

```c
for (int i = 0; i < INIT_REGISTER_COUNT; i++) {
  SPIWriteSingle(HAMSTRONE_GLOBAL_SPI_PORT, MPU9250_SPI_MODE, initRegister[i][0], initRegister[i][1]);
  mpudebug("initMPU9250: init reg %d = %d", initRegister[i][0], initRegister[i][1]);
  usleep(1000);
}
```

초기화를 위한 레지스터 설정값. 위의 배열을 순회하며 전원 인가후 배열의 값대로 레지스터를 설정합니다.

```c
uint8_t data[21];
if (SPIRead(HAMSTRONE_GLOBAL_SPI_PORT, MPU9250_SPI_MODE, MPUREG_ACCEL_XOUT_H | READ_FLAG, 21, data) < 0) {
  mpudebug("readMPU9250: read error");
  return ERROR_READ_FAIL;
}
mpudebug("readMPU9250: read ok");
```

MPU9250의 레지스터 값을 읽어와서

```c
value[10] = ((int16_t)data[6] << 8) | data[7];
ret->accX = ((float)value[0] / MPU9250_ACCEL_COEFFICIENT);
```

상, 하 바이트를 결합하고 이를 부동소수점으로 캐스팅한 후 단위를 가질 수 있게 변환합니다.

#### 밝기 센서

밝기 센서로는 SNA 社의 SO6203을 사용했습니다. 본 연구에서는 광원을 감지하기 위해 드론 전방에 3개를 독립적인 시야각을 가지게 장착해서 사용합니다. 상세 스펙은 아래와 같습니다.

- 730nm에서 피크치를 가지는 광 감도, 대략 350~1100nm의 감지 범위
- I2C 방식으로 통신
- 16비트 ADC 내장

본 연구에서는 백색광의 세기를 사용합니다. 별도의 필터링은 없으며 신경망에 입력하기 위한 양자화만 진행합니다.

##### 구현

이 부분의 구현 전체 소스는 [여기](https://github.com/Dictor/hamstrone-drone/blob/master/bright_distance_sensor.c)를 참고하세요.

```c
for (int c = chanStart; c <= chanEnd; c++)
{
  if (TCA9548SetChannel(HAMSTRONE_GLOBAL_I2C_PORT, c) < 0)
    errcnt++;
  if (I2CWriteRegisterSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_SO6203, HAMSTRONE_CONFIG_SO6203_EN, 0b00001011) < 0)
  errcnt++;
}
```

`EN` 레지스터를 설정해 센서를 활성화합니다.

```c
if (I2CReadSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_SO6203, HAMSTRONE_CONFIG_SO6203_ADCW_H, &valueh) < 0)
  errcnt++;
if (I2CReadSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_SO6203, HAMSTRONE_CONFIG_SO6203_ADCW_H + 1, &valuel) < 0)
  errcnt++;
result[c] = (valueh << 8) | valuel;
```

값 레지스터를 읽어 밝기를 계산합니다.

#### 거리 센서

거리 센서로는 Benewake 社의 TFmini-S를 사용했습니다. 레이저를 사용한 LIDAR 방식으로, 드론이 비행 중 장애물을 감지하고 회피하기 위해서 사용합니다. 상세 스펙은 아래와 같습니다.

- 측정 거리 범위 0.1m ~ 12m (90% 반사율의 대상에 대해서)
- 정확도 6cm (0.1 ~ 6m) 또는 ±1% (6m ~ 12m)
- 해상도 1cm
- 프레임률 100Hz
- 통신 방식 UART 또는 I2C

본 연구에서는 좀 더 정확한 거리 계산을 위한 변수까지는 사용하지 않고 단순 거리 값만 사용합니다. 별도의 필터링은 없으며 신경망에 입력하기 위한 양자화만 진행합니다.

##### 구현

이 부분의 구현 전체 소스는 [여기](https://github.com/Dictor/hamstrone-drone/blob/master/bright_distance_sensor.c)를 참고하세요.

```c
uint8_t data[7];
for (int c = chanStart; c <= chanEnd; c++) {
  if (TCA9548SetChannel(HAMSTRONE_GLOBAL_I2C_PORT, c) < 0)
    errcnt++;
  I2CWriteSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_TFmini, 0x01);
  I2CWriteSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_TFmini, 0x02);
  I2CWriteSingle(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_TFmini, 0x07);
  usleep(1000);
  if (I2CRead(HAMSTRONE_GLOBAL_I2C_PORT, HAMSTRONE_CONFIG_I2C_ADDRESS_TFmini, 7, data) < 0)
    errcnt++;
    //  data[0]=isValid? [2]=distl [3]=disth [4]=strengthl [5]=strengthh [7]=rangetype
  result[c] = (data[3] << 8) | data[2];
}
```

값 레지스터를 읽어 거리를 계산합니다.

#### I2C 통신 (TCA9548) 멀티플렉서

위에 설명된 I2C 통신을 사용하는 센서(밝기, 거리)들은 프로세서의 하나의 I2C 버스에 연결되어 통신을 진행하게 됩니다.
다만, 밝기 센서와 거리 센서들이 각각 동일한 I2C 주소를 가지고 있어 별다른 조치 없이는 하나의 버스에서 통신할 수 없습니다.
프로세서의 I2C 버스 숫자도 제한되어 하나의 버스에서 중복된 주소의 장치들을 사용하기 위해 I2C 주소 멀티플렉서인 TCA9548을 사용 했습니다.

<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/tca9548.jpg">
<p class="caption">TCA9548의 연결 예시도</p>

TCA9548은 프로세서의 I2C 버스를 주(main) 버스로 삼고, 최대 8개의 장치가 I2C 버스를 부(slave) 버스로 삼아 종 9개의 I2C 버스를 제공합니다. I2C 통신은 간단하게 요약해서 특정 주소의 레지스터를 쓰거나, 읽는 2가지 행위로 요약할 수 있는데, 이때 TCA9548은 장치 주소 + 레지스터 주소 + 레지스터 값의 조합의 메세지를 쓰기 전에 현재 통신이 진행될 부 버스를 선택할 수 있게 TCA9548의 레지스터에 버스 번호를 쓴 뒤, 메세지를 쓰는 방식으로 멀티플렉싱을 구현합니다.

<img width="750px" src="/assets/img/post/2021-11-27-endurance_drone/tca9548_msg.jpg">
<p class="caption">TCA9548을 이용할 때 메세지 예시</p>

##### 구현

앞선 구현 예시에서 `TCA9548SetChannel` 함수를 확인하실 수 있었습니다. 이 함수의 내부 구현은 아래와 같이 간단한, 위에서 설명한 추가적인 정보를 작성하는 내용입니다.

```c
int TCA9548SetChannel(int fd, uint8_t chan) {
  return I2CWriteRegisterSingle(fd, HAMSTRONE_CONFIG_I2C_ADDRESS_TCA9548, HAMSTRONE_CONFIG_TCA9548_CHAN, 1 << chan);
}
```

##### SPI, I2C 통신 개요도

<img width="500px" src="/assets/img/post/2021-11-27-endurance_drone/sensor_conn.jpg">
<p class="caption">SPI, I2C를 이용하는 센서의 연결 개요도</p>

#### GPS

범용적인 용도를 위해 GPS의 출력인 NMEA 프로토콜을 현재 위치의 위도 및 경도를 파싱하는 루틴을 작성해두었습니다만, 현재 사용되고 있진 않습니다. 테스트는 GY-GPS6MV2 센서로 진행되었으며 올해 초에 실외용 드론에 탑재될 용도로 개발하게 되었습니다.

GY-GPS6MV2 센서는 `$GPGGA`, `$GPGSV`, `$GPRMC` 등의 데이터 포맷을 전송하는데 `$`기호를 기준으로 새로운 데이터가 시작되기 때문에 `$`문자를 받아들이면 기존의 데이터를 초기화하고 `GPGGA`, `GPGSV`, `GPRMC`등의 5자리 문자열을 인식합니다.각각의 문자열에 따라 가지고 있는 데이터가 다르기 때문에 경우에 따라 알맞은 데이터를 추출할 수 있도록 코드를 작성하였습니다. 센서로부터의 데이터는 큐에 누적되며 파싱을 진행합니다. 메세지가 완성되지 않은 경우 대기하며 큐에 데이터가 쌓일 때 까지 대기합니다.

##### 구현

이 부분의 구현 전체 소스는 [여기](https://github.com/Dictor/hamstrone-drone/blob/master/gps.c)를 참고하세요.

```c
if (dataReceive[2] == 'R' && dataReceive[3] == 'M' && dataReceive[4] == 'C' && commaCnt == 12) {
    assembleCnt++;
    for (i = 0; i < gpsType.num - 1; i++)
    {
        int k = 0;
        char assembleData[15] = {
            0,
        };
        if (i == 1 || i == 3 || i == 5)
        {
            for (j = gpsType.Element[i]; j < gpsType.Element[i + 1] - 1; j++)
            {
                assembleData[k] = dataReceive[j];
                k++;
            }
            convert = atof(assembleData);
            if (i == 1)
            { // UTC
                convert *= (int)100;
                HAMSTRONE_WriteValueStore(11, (uint32_t)convert);
            }
            else if (i == 3)
            { // Latitude
                convert *= (int)100000;
                HAMSTRONE_WriteValueStore(12, (uint32_t)convert);
            }
            else if (i == 5)
            { // Longitude
                convert *= (int)100000;
                HAMSTRONE_WriteValueStore(13, (uint32_t)convert);
            }
        }
    }
} else if (dataReceive[2] == 'G' && dataReceive[3] == 'G' && dataReceive[4] == 'A' && commaCnt == 14) {
    assembleCnt++;
    for (i = 0; i < gpsType.num - 1; i++)
    {
        int k = 0;
        char assembleData[15] = {
            0,
        };
        if (i == 7 || i == 8)
        {
            for (j = gpsType.Element[i]; j < gpsType.Element[i + 1] - 1; j++)
            {
                assembleData[k] = dataReceive[j];
                k++;
            }
            convert = atof(assembleData);
            if (i == 7)
            { // Number of Satellites used for Calculation
                convert = (int)convert;
                HAMSTRONE_WriteValueStore(14, (uint32_t)convert);
            }
            else if (i == 8)
            { // HDOP
                convert *= (int)100;
                HAMSTRONE_WriteValueStore(15, (uint32_t)convert);
            }
        }
    }
}
```

#### 마이크로프로세서 (STM32L432KC)

드론에 탑재된 마이크로프로세서는 ST 社의 STM32L432KC 프로세서를 사용했습니다. 상세 명세는 아래와 같습니다.

- ARM 32비트 Cortex-M4 80MHz(100DMIPS) CPU (FPU, ART 탑재)
- 256KB 플래시, 64KB SRAM
- 26개의 IO, 11개의 타이머, 12비트 ADC와 DAC
- USB, SAI, 2개의 I2C, 3개의 USART, 2개의 SPI, CAN

<img width="550px" src="/assets/img/post/2021-11-27-endurance_drone/cpu_io.jpg">
<p class="caption">CPU의 IO 연결표</p>

### PID 제어 **(남종현)**

<img width="650px" src="/assets/img/post/2021-11-27-endurance_drone/pid.jpg">
<p class="caption">PID 제어기의 블록 다이어그램</p>

PID 제어기는 비례 적분 미분 제어기의 줄임말로, 대표적인 폐루프 제어기입니다. 제어하고자 하는 대상의 출력값을 측정하여 정상 상태 값(목표 값)과 비교하여 오차를 계산하고, 이 오차값을 이용하여 제어에 필요한 출력 값을 계산하는 구조로 되어 있습니다.
상보필터와 칼만필터로 계산한 현재 자세를 입력으로 받아 roll, pitch의 각도에 대해 오차를 수정하기 위한 출력 값을 제공하는 방식으로 제어를 진행합니다.
PID 제어기의 특성은 3개의 비례, 적분, 미분 이득치(게인)으로 결정되는데 이 값에 따라 정상 상태에 수렴하는 특성이 크게 바뀌기 때문에 자세 제어의 핵심이라고 할 수 있습니다. 이 이득치 값들을 자동 또는 결정하는 공식도 존재하지만 본 연구에서는 여러번의 테스트를 진행하며 수동으로 최적의 값을 찾고자 했습니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/gwXdtai-zcc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<p class="caption">초기 PID 게인 테스트 도중의 실험 영상, LPF 및 칼만 필터를 적용하지 않고 낮은 이득치를 사용하여 제어가 매우 불안정 했습니다. (제어 값의 경향은 맞으나 과감쇠 시스템처럼 수렴하지 못하고 오버 슈팅이 발생하며 진동함)</p>

현재 최신적으로 사용하고 있는 이득치는 비례 이득 3, 적분 이득 2, 미분 이득 0.5입니다. (최종이 아님)
<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/pid_left.png">
<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/pid_right.png">

<p class="caption">프로세서가 계산하여 ground 프로그램에 송출하고 있는 PID 제어 출력값 (각각 좌, 우로 드론을 기울였을때, 대칭적으로 모터 출력의 대소 관계가 변하는 것을 확인할 수 있습니다. (사진상의 6~9 항목))</p>

#### 구현

`updatePID` 함수에 현재 측정 각도를 입력하면 출력 제어값이 반환됩니다.

```c
double kP[PID_DIMENSION] = {3, 3};
double kI[PID_DIMENSION] = {2, 2};
double kD[PID_DIMENSION] = {0.5, 0.5};

void updatePID(double AngX, double AngY, double *result)
{
    double degree[PID_DIMENSION] = {
        0.0,
    };
    degree[0] = AngX;
    degree[1] = AngY;

    static double prevInput[PID_DIMENSION] = {
        0.0,
    };
    static double controlI[PID_DIMENSION] = {
        0.0,
    };
    double controlP[PID_DIMENSION], controlD[PID_DIMENSION], dInput[PID_DIMENSION], error[PID_DIMENSION], desired[PID_DIMENSION] = {10.0,};
    double time = 0.01;
    int i;

    for (i = 0; i < 2; i++)
    {
        error[i] = desired[i] - degree[i];
        dInput[i] = degree[i] - prevInput[i];
        prevInput[i] = degree[i];

        controlP[i] = kP[i] * error[i];
        controlI[i] = kI[i] * error[i] * time;
        controlD[i] = -kD[i] * dInput[i] / time;

        result[i] = controlP[i] + controlI[i] + controlD[i];
    }
}
```

### 회로/PCB 설계 **(김정현, 남종현)**

#### 전원부 (PTH08080)

11.1V (3셀 LIPO) 배터리에서 회로에 공급될 5V 전원을 공급하기 위해 TI 社의 스위칭 레귤레이터 모듈인 PTH08080W를 사용했습니다. 상세 명세는 아래와 같습니다.

- 섭씨 85도에서 2.25A 출력
- 4.5 ~ 18V 입력 전압
- 0.9 ~ 5.5V 출력 전압 (조정 가능)
- 93% 효율

<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/pth08080.png">

<p class="caption">PTH08080의 표준 사용 방법</p>

#### 회로

위에서 설명한 모든 센서와 처리 칩셋, 마이컴을 통합하여 한 회로로 표현하면 아래 사진과 같습니다.
<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/circuit.png">

<p class="caption">회로도</p>

#### PCB

위의 회로도를 드론에 탑재하기 적합한 소형 PCB에 올리면 아래 사진들과 같습니다.
<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/gerber.png">

<p class="caption">PCB 패턴도</p>

<img width="450px" src="/assets/img/post/2021-11-27-endurance_drone/pcb.png">

<p class="caption">실제로 제작된 견본 PCB</p>

현재 제작한 견본 PCB는 여러가지 회로 실수와 풋프린트 오류가 발견되어 지속적으로 수정, 제작중입니다.

### 하드웨어 제작 **(남종현, 김재민)**

테스트에 사용된 견본 드론 하드웨어는 아래 사진과 같습니다. PCB에 오류로 회로가 동작하지 않아 임시로 브레드보드에 조립한 모습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/drone_top.jpg">

<p class="caption">상단에서 바라본 견본 드론 하드웨어</p>

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/drone_front.jpg">

<p class="caption">정면에서 바라본 견본 드론 하드웨어</p>

### 임베디드 프로그래밍 (RTOS) **(김정현)**

본 연구에서는 다양한 기능들이 임베디드 시스템에서 유기적으로 작용할 필요가 있으므로 Nuttx RTOS(실시간 운영체제)를 채용했습니다. Nuttx 운영체제는 아파치 재단에서 인큐베이팅중인 프로젝트로 광범위한 아키텍쳐를 지원하며 POSIX 표준을 만족하기에 많이 사용하는 데스크탑 환경과 비슷한 코딩을 할 수 있습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/nuttx.jpg">

<p class="caption">NuttX 구조 다이어그램</p>

본 연구에서 작성한 드론 펌웨어는 User Layer에서 User Application(다이어그램의 Other Application)으로서 작동하며 하드웨어 접근은 HAL과 드라이버를 통해 POSIX API로서 추상화되어있습니다. 예를 들어 UART 통신은 표준 파일처럼 `read`, `write` 메서드를 통해 스트림과 작용하며, I2C와 SPI 통신은 `ioctl` 메서드를 통해 작용합니다. 태스크 제어도 `usleep`등의 표준 메서드로 컨텍스트 스위칭이 쉽게 가능합니다. NuttX는 본 연구에서 사용한 L432KC 프로세서에 대해 대부분 주변장치에 대한 Character Driver를 제공하고 있습니다. 본 연구에서 일부 미흡한 부분에 대해 NuttX에 대해 기여를 한 부분도 [존재](https://github.com/apache/incubator-nuttx/pull/2837)합니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/task.jpg">

<p class="caption">드론 펌웨어의 태스크와 리소스 점유 관계</p>

idle 태스크는 진입 함수(main)이 실행되는 태스크고, 초기화후에 실제 기능 태스크들에 CPU 실행권을 무조건적으로 양보합니다. 3개의 태스크들이 각자의 기능을 처리하며, UART 장치는 두 개 이상의 태스크에서 동시 접근 가능성이 있어 세마포어로 동기화됩니다. Value Store (배열로 구현된 중앙 데이터 저장소)는 읽기, 쓰기 태스크가 명확히 분리되어 있어 별도의 동기화는 없습니다.

### 지상국 **(김정현)**

지상국은 위에서 언급된 Value Store을 랩탑에서 원격으로 확인하고 지령을 내리기 위한 프로그램입니다. golang으로 작성된 백앤드가 HamsterTongue 프로토콜 형식의 시리얼 신호를 파싱하여 Websocket을 통해 vueJS로 작성된 랩탑의 Web UI에 값을 표출합니다. Value Store의 값을 표출, 현재 자세를 그래프로 표현, Signal Verb를 가진 메세지들을 표현하는 3개의 기능이 존재합니다.

#### HamsterTongue 프로토콜

경량 메세지를 송수신하기 위하여 통일된 프로토콜로 통신하게 됩니다. 메세지의 상세 구조는 아래와 같습니다.

```
Name        Marker  Length  Verb    Noun    Payload     CRC
Size(byte)  1       1       1       1       ~255        1
            <-              Message Length              ->
                            <-      Length              ->
```

- Marker : 메세지의 시작을 알리는 바이트, 0xFF 고정
- Length : Verb ~ CRC까지의 길이를 알리는 바이트, 0~255
- Verb : 메세지의 종류
- Noun : 메세지의 변수
- Payload: 데이터 (실질적으로 최대 252바이트)
- CRC: Verb ~ Payload 까지의 1바이트 CRC 검증 값

##### 구현

C언어로 구현된 송수신 구현

```c
uint8_t *HAMSTERTONGUE_SerializeMessage(HAMSTERTONGUE_Message *msg)
{
	uint8_t *res = malloc(HAMSTERTONGUE_GetMessageLength(msg));
	res[0] = HAMSTERTONGUE_MESSAGE_MARKER;
	res[1] = HAMSTERTONGUE_GetMessageLength(msg) - 2;
	res[2] = msg->Verb;
	res[3] = msg->Noun;
	memcpy(res + 4, msg->Payload, msg->PayloadLength);
	res[4 + msg->PayloadLength] = 0; //CRC
	return res;
}

HAMSTERTONGUE_Message *HAMSTERTONGUE_ReadMessage(int fd)
{
	HAMSTERTONGUE_Message *msg;
	uint8_t buf[64], msgpayload[256];
	uint8_t buildStatus = 0, buildLen, msglen, msgverb, msgnoun, msgcrc; // 0 = not STX yet, 1 = not Len yet, 2 = payload reading, 3 = complete
	uint8_t readlen = HAMSTERTONGUE_Read(fd, buf, 64);
	if (readlen > 0)
	{
		for (int i = 0; i < readlen; i++)
		{
			if (buildStatus == 0)
			{
				if (buf[i] == HAMSTERTONGUE_MESSAGE_MARKER)
				{
					buildStatus = 1;
				}
				else
				{
					continue;
				}
			}
			else if (buildStatus == 1)
			{
				msglen = buf[i];
				buildStatus = 2;
				memset(msgpayload, 0, 256);
				buildLen = 0;
				break;
			}
			else if (buildStatus == 2)
			{

				buildLen++;
				if (buildLen == 1)
				{
					msgverb = buf[i];
				}
				else if (buildLen == 2)
				{
					msgnoun = buf[i];
				}
				else if (buildLen > 2 && buildLen <= msglen - 1)
				{
					msgpayload[buildLen - 3] = buf[i];
				}
				else if (buildLen == msglen)
				{
					msgcrc = buf[i];
					buildStatus = 3;
				}
			}
			else if (buildStatus == 3)
			{
				uint8_t payloadLen = msglen - 3;
				msg = HAMSTERTONGUE_NewMessage(msgverb, msgnoun, payloadLen);
				msg->Payload = malloc(payloadLen);
				memcpy(msg->Payload, msgpayload, payloadLen);
				return msg;
			}
		}
	}
}
```

Go 언어로 구현된 메세지 수신 구현

```golang
func decodeMessage(msgchan chan *hamsterTongueMessage, sendchan chan []byte) {
	defer ValueMutex.Unlock()
	for {
		select {
		case msg := <-msgchan:
			globalLogger.WithFields(logrus.Fields{
				"length":  msg.Length,
				"verb":    msg.Verb,
				"noun":    msg.Noun,
				"payload": msg.Payload,
			}).Debugf("serial message income")
			switch msg.Verb {
			case hamstertongue.MessageConstant["Verb"]["Heartbeat"]:
			case hamstertongue.MessageConstant["Verb"]["Value"]:
				ValueMutex.Lock()
				for i := 0; i < 16; i++ {
					Value[strconv.Itoa(i)] = binary.LittleEndian.Uint32(addArrayPadding(msg.Payload[i*4:i*4+3], 4))
				}
				ValueMutex.Unlock()
			case hamstertongue.MessageConstant["Verb"]["Signal"]:
				data, err := json.Marshal(generalMessage{
					Type: "signal",
					Data: []interface{}{msg, string(msg.Payload)},
				})
				if err != nil {
					globalLogger.WithField("error", err).Errorln("error caused while making message")
					continue
				}
				sendchan <- data
			}
		}
	}
}
```

### 강화학습 **(김정현)**

본 연구에서 가정하고 있는 문제는 실제 상황을 전부 관측할 수 없고, 부분적인 관측치로 상황을 예측하여 해결하는 POMDP(부분 관찰 마르코프 의사결정 과정) 문제 입니다. 이 문제는 강화학습을 통해 상황을 확률적으로 모델링하고 상황에 대한 정답과의 근접 정도인 보상치를 최대화할 수 있게 행동을 인공 신경망으로 행동을 선택할 수 있게 하는 가치 함수를 통해 풀어내게됩니다. 변수로서 표현하면 아래와 같이 표현할 수 있습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/pomdp.jpg">

- O : 실제 상황에 대한 부분적인 관측치의 집합
- A : 가능한 행동에 대한 집합
- R : 보상치
- S : 실제 상황 (POMDP 문제에서는 O로 가정함)
- γ : 할인율 (미래와 현재 보상치의 시간 경과에 따른 가치를 결정)

O는 센서의 관측치를 간단히 양자화하여 입력하면, 가치 네트워크 (인공 신경망으로 구현된 가치 함수의 일종)을 통해 가장 확률이 높은 A가 도출되는 방식으로 동작합니다. 학습시에 R은 학습 환경이 목표와의 거리에 대하여 입력시켜주게 되며 γ은 0.95(95%)입니다.

강화학습 알고리즘은 DQN을 사용합니다. 간단하게 설명하면 전통적인 결정적 가치함수 대신에 인공 신경망을 기반으로한 가치 네트워크를 행동을 결정하는 정책으로서 사용하는 방식입니다. DQN의 특성중 하나인 하이퍼 파라미터와 보상 정책은 아래 사진과 같습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/dqn.jpg">

<p class="caption">하이퍼 파라미터와 보상 정책</p>

이러한 변수와 정책으로 학습시킨 저희의 가치 네트워크의 테스트 예시입니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/dqn_test.jpg">

<p class="caption">가치 네트워크 테스트 결과</p>

비교적 쉬운 시작 위치인 1~3번 장소에서는 양호한 성공률과 평균 episode 길이(소요 시간에 대한 선형적인 수치, 환산하면 대략 1에피소드당 2~3초)를 보여주지만 광원을 찾기 어려운 환경인 4번 장소에 대해서는 성공하지 못하는 모습을 보여줍니다. 현재 이러한 문제를 보정하기 위해 더 다양한 학습 장소에서의 학습을 진행하고 있습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/dqn_sim.gif">

<p class="caption">가치 네트워크 테스트 시뮬레이션 예시</p>

쉬운 환경에서는 금방 광원에 도착하지만, 어려운 환경에서는 행동을 반복하며 방황하는 모습을 관찰할 수 있습니다.

#### 이식

이렇게 만들어진 가치 신경망을 마이크로 프로세서에 이식하기 위하여 `Tensorflow Lite for microcontroller` 프레임워크를 사용하게 됩니다. 텐서플로우의 API 경량화 및 정적 할당 파생형입니다. 현재 C++과 C 펌웨어의 결합 단계에서 시행 착오를 진행중이며 현재 시점에선 미완성입니다.

### 신호처리(FFT) **(김상윤)**

고속 푸리에 변환(FFT)은 이산 푸리에 변환(DFT)의 기본적인 구현에서 반복해서 수행하는 계산을 한 번씩만 수행하도록 하여 효율성을 높이는 방법입니다. 본 연구에서는 환경 광원과 일정 주파수로 점멸하는 목표 광원을 구분하기 위하여 밝기 센서의 감지 값에 사용합니다.

FFT는 신호의 길이 N이 N=2^d와 같이 2의 거듭제곱 형태라고 가정합니다. DTF는 전체 주파수 함수값을 구하는 계산량은 입력 신호의 원소 수 N의 제곱에 비례하는 데 이렇게 짝수와 홀수 원소로 분류하여 변환을 수행하면 원소 수가 N/2인 신호를 두 번 변환하는 것이 되어 계산량이 2*(N/2)*(N/2)에 비례하게 됩니다.

#### 구현

FFT는 현재 단계에선 아직 구현되있지 않습니다.

<img width="100%" src="/assets/img/post/2021-11-27-endurance_drone/fft.png">

<p class="caption">FFT 코드 예시</p>

## 연구 결과

현재 진행중인 프로젝트라 아직 만족스러운 결과가 도출되진 않았습니다. 차후 연구 진행 계획은 아래와 같습니다.

- RL 모델 포팅 : ~ 21년 12월
- 실 하드웨어 제작 및 실험 : ~ 22년 1월
- 결과 정리 및 발표 : ~ 22년 2월

## 부록

### 코드

프로젝트의 모든 산출물 코드는 오픈소스입니다.

- 강화학습 : https://github.com/Dictor/endurance-rl
- 강화학습 환경 : https://github.com/Dictor/endurance-environment
- 드론 펌웨어 : https://github.com/Dictor/hamstrone-drone
- NuttX OS : https://github.com/Dictor/incubator-nuttx
- 지상국 : https://github.com/Dictor/hamstrone-ground
- 회로도 및 PCB : https://github.com/Dictor/endurance-circuit

### 라이센스

본 문서는 CC BY-SA (저작자표시-동일조건변경허락)에 따라 이용할 수 있습니다
대부분 자체 저작물이나, 일부 포함된 저작물은 각 저작권자의 소유입니다.

### 감사의 인사

회로 설계와 PCB 설계에 많은 도움을 주시고 매 달 회의도 참석해주시고, 대전에서 족발도 사주신 16기 이승준 선배님께 다시 한번 감사드립니다. 강화학습에 관련하여 많은 인사이트를 제공해주신 문준 교수님과 이명훈 교수님께도 감사 인사 전합니다. 팀원들과 도움주신 바라미 회원님들에게도 정말 감사드립니다!
