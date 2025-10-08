---
title: "Playlist-Randomizer | 유튜브 재생목록 랜덤재생 사이트"
# description: ""
categories: [컴퓨터, 인터넷]
tags: [Web]
image: "/assets/img/background/kururu-lab.jpg"

date: 2021-12-16. 09:18
# last_modified_at: 2024-08-29. 22:15
last_modified_at: 2024-10-20. 12:41 # 유튜브 정리
---

## 유튜브 재생목록 셔플 기능 문제점

---

1. 셔플 알고리즘 문제인지, 같은 영상이 같은 패턴으로 나올 때가 있음

2. 재생목록에 영상이 많으면 어떤 영상을 재생할 때 그 주변 영상만 불러오는데, 이때 셔플 기능을 사용하면 그렇게 불러온 주변 영상들 중에서만 랜덤 재생함
   - 예를 들어, 100개의 영상이 있는 재생목록에서 1번째 영상을 재생하면 한 번에 1~50번째까지 50개의 영상만 불러오고, 셔플 기능을 이용하면 이렇게 불러온 50개의 영상들 중에서 셔플이 반복되는 식. 나머지 뒷 부분의 영상들은 랜덤 대상이 되지 않는다.

## Playlist-Randomizer

---

[http://www.playlist-randomizer.com/](http://www.playlist-randomizer.com/)
{: .text-center}

- 위 사이트를 이용하여 해결
- 재생목록 ID를 넣어 랜덤으로 재생하는 방식

- 설정한 재생목록이 캐시에 저장되어, 나중에 다시 들어갔을 때 또 입력할 필요가 없음
- 여러 재생목록을 한 번에 넣는 기능도 있음 (여러 재생목록을 한 번에 셔플)
