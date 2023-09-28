---
layout: post
title: Latency Metrics P90, P95, P99
date: 2023-08-19 12:32 +0900
category: [Background]
tag: [Network]
---

### P90, P95, P99 Latency

latency
: 서비스에서 어떤 작용에 의한 결과가 나오기까지 걸리는 시간. 게임에서는 '랙'이라고도 부른다. 네트워크에서는 패킷이 도착하는데 걸리는 시간을 말한다. 

P90 latency
: 서비스를 수행하면서 발생하는 latency를 오름차순으로 정렬했을 때 앞에서 90퍼센트에 해당하는 값을 말한다. 다시 말해, 90퍼센트의 latency는 P90 latency보다 작다.

같은 원리로 **P95 latency**와 **P99 latency**도 있다.

### 의미

P90 latency|서비스를 이용하면서 대다수의 유저가 느꼈을 대기 시간이다.<br>앱의 전반적인 성능을 끌어올릴 때 모니터링한다.
P95 latency|더 복잡한 기능을 사용하는 유저가 느꼈을 대기 시간이다.
P99 latency|latency 분포의 꼬리가 얼마나 긴지를 평가한다.

### Ref.

<https://www.linkedin.com/pulse/day-5-mastering-latency-metrics-understanding-p90-p95-nguyen-duc#:~:text=P99%20(99th%20percentile)%3A%20The,a%20small%20percentage%20of%20users.>