---
title: "Unity | DOTween"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-03-15. 08:31 # Init
# last_modified_at: 2025-04-19. 01:08 # toUniTask
last_modified_at: 2025-05-28. 21:30 # +메모
---

## 머리말

---

## 메모

---

### _

- Tween -> Coroutine (Var)
- tween.toUniTask
- Kill -> StopCoroutine
- tr.DOMove(targetPos, moveTime).SetEase(Ease.OutExpo)
- DOTWeen.To(c => 시작값, x => 값 변경 작업, 목표값, 시간)
- DOTween VS Anim
  - 상태가 있는 Anim이라면... DOTween이 좋을 지도?
  - Anim으로 하려면 (int)State SetInt SetTrigger로 해야하는데
  - DOTween은 Switch가 가능하니까
  - ... 그냥 일단 그대로 두자 (<- 후담, 결국 수정함)
- DOTween.Init (logBehavior: ".ErrorOnly)
- DOTween.Play(object_GameObjectX);
- PrimeTween
- DOVirtual
