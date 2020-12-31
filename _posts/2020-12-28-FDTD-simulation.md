---
title: FDTD-simulation
author: Hwang Hyeon Jo
date: 2020-12-28 22:50:00 +0900
categories: [Exhibition,2020년]
tags: [post,hwanghyeonjo,fdtd]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# 서론
안녕하세요 융합전자공학부 3학년 황현조입니다. 저는 FDTD simulation을 주제로 전시회를 준비했습니다. 

전자장 수업을 들었던 사람이라면 수업시간에 E 또는 H wave가 하나의 매질에서 다른 매질로 옮겨가면서 반사파를 내보내고.. 몇몇 파들은 통과하고.. 반사파랑 입사파랑 다시 합쳐지고.. 그런 애니메이션을 다들 보셨을 것입니다. 그걸 만들었습니다. 정확히는 그렇게 움직이게 만드는 원리에 대해서 공부하고 움직이는걸 실험해 봤습니다.

# 이론

Finite Difference Time Domain의 약자입니다.

E field와 H field를 컴퓨터 시뮬레이션 하는데 쓰이는 가장 기본적인 model로 orthogonal grid에 한정되어 있다는 단점이 있습니다. 또 High Q System에서는 시뮬레이션 시간이 오래 걸리기도 합니다.

Yee Algorithm을 사용했습니다. 그냥 우리가 알고있는 Maxwell equation에 Finite Difference Approximation을 더한 것입니다.

Maxwell equation이야 흔히 다들 알고 있는 대로

<img src="/assets/img/post/2020-12-28-FDTD-simulation/Maxwelleq.JPG" width="90%">

와 같고 Finite Difference Approximation은 (CDS)

<img src="/assets/img/post/2020-12-28-FDTD-simulation/CDS.JPG" width="90%">

와 같습니다.

편의를 위해 notation을 정의하겠습니다.

<img src="/assets/img/post/2020-12-28-FDTD-simulation/notation.JPG" width="90%">

이제 Maxwell equation에 CDS(central diffrence scheme)을 적용하면 

<img src="/assets/img/post/2020-12-28-FDTD-simulation/Result.JPG" width="90%">

이렇게 나오고 뒤에 식을 변형하면 
<img src="/assets/img/post/2020-12-28-FDTD-simulation/Result2.JPG" width="90%">

이렇게 나옵니다.

이 식의 의미는 leap frog time-stepping이라고 할 수 있습니다. 
앞의 식과 뒤의 식이 계속 연결되면서 다음 영역과 시간대의 E값과 H값을 알 수 있게 쭉쭉 전개됩니다.

아래는 3D로 이 식을 풀었을 떄의 결과 입니다.

<img src="/assets/img/post/2020-12-28-FDTD-simulation/3D-1.png" width="70%">

<img src="/assets/img/post/2020-12-28-FDTD-simulation/3D-2.png" width="70%">

# 실행

1D코드와 3D코드 모두 제 깃헙에 있습니다.

### 1D FDTD


코드 중간에 to make the plot move를 주석을 빼고 74~80번줄을 주석처리 하면 꿈틀 거리는 그래프를 볼 수 있습니다.(멋 있 습 니 다)

다음 아래 3개의 그래프는 이론값과 FDTD를 적용했을 때의 E wave를 비교한 것입니다. 빨강이 이론값, 파랑이 FDTD를 적용했을 때의 그래프 입니다. dt = 1.67*1e-12로 700번 돌렸고 각각 time step이 0, 100, 200, ... 700일 때의 그래프 입니다. PEC로 boundary를 설정해서 반사파가 오는 것을 확인 할 수 있습니다. 

<img src="/assets/img/post/2020-12-28-FDTD-simulation/1Dresult-1.png" width="80%">

<img src="/assets/img/post/2020-12-28-FDTD-simulation/1Dresult-2.png" width="80%">

<img src="/assets/img/post/2020-12-28-FDTD-simulation/1Dresult-3.png" width="80%">

### 3D FDTD

3D도 똑같은 과정을 거쳤으나 마지막에 dft시켜서 resonance frequency를 확인해서 실제 mode들과 같은지를 확인했습니다. PEC box를 가정하고 한 점(3,3,3)에서 Pulse를 만들어서 (12,3,10)지점에서 그 결과를 확인했습니다. 아래는 그 결과입니다.
<img src="/assets/img/post/2020-12-28-FDTD-simulation/3Dresult.JPG" width="80%">


### 왜 dft시키냐 fft 시켜라 더 빠르지 않냐

읽고 있는데 아직 덜 읽었는데 이렇다고 합니다.(Why the DFT is Faster Than the FFT for FDTD
Time-to-Frequency Domain Conversions, C.M.Furse,1995)

궁금하시면 논문을 줄테니 읽고 저한테 알려주세요.
<img src="/assets/img/post/2020-12-28-FDTD-simulation/dftfft.JPG" width="80%">



# 추가

이 프로젝트는 Gedney의 Introduction to the Finite Difference Time Domain을 한학기동안 공부하면서 만들었습니다. 아직 공부가 완벽하게 되지 않아서 미숙한 점이 많습니다. 지적해주신다면 감사하겠슴니다. 

또 안테나 분야 졸업생 선배님 계시면 인생 조언 부탁드립니다... 

감사합니다.