---
title: "Particle"
# description: ""
categories: [컴퓨터, 그래픽]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-05-28. 06:01 # Init
last_modified_at: 2025-05-28. 06:15 # +메모 from career-learning
---

## Particle

---

### CPU Particle

- 정확하게 의도된 움직임과 파티클 간 부모 자녀 상속 컨트롤 용이
- 개수가 적으면, GPU 파티클보다 효율적
- 다루는 것이 많이 수월, GPU 파티클보다

### GPU Particle

- 수십만, 수백만 파티클을 표현해도 성능 비용이 저렴한 편
  - 많은 파티클에 대한 물리표현, 적은 비용으로 가능
- 개별 형태에 대한 표현이 좀 더 추상적인 형태로 표현이 가능하다? (잘 이해 못함)

### 비교

![_](https://64.media.tumblr.com/282c50e933fcc92330924399b78396d9/d3727fd838d93d13-f7/s1280x1920/25844483bd95cf9464d3e65cdfc71fb2e98b1d8e.png)

메모리, 대역폭 등 여러가지 요인에 의해서 CPU와 GPU 파티클의 스윗스팟의 임계점은 보통 1만개  
[참고](https://eullee.tumblr.com/post/184693453455/gpu속도의-내막-파티클-1만개-안쪽에서는-cpu만-쓰는것과-별-차이가-없을거에요)  

### 메모

- 파티클 시스템 렌더러.bakeMesh(mesh,카메라) -> Graphics
- 파티클, Trails
- 파티클 시스템 -> 버텍스 컬러 바꿈
  - 그래서 파티클 시스템에서 버텍스 컬러 받아야 함
  - v2f, appdata 에서 float4 color : COLOR, 이것을 frag에서 사용
- a practical tutorial to hack and protect unity games  

#### 키워드

- Particle Live

#### 참고

- ['X, Irua': GPU 파티클](https://x.com/kitsuneAnCr/status/1926887724658606337)
