---
title: "Unity | Resource, AssetBundle, Addressable"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-12-02. 19:03 # Init
# last_modified_at: 2025-03-15. 09:33 # Unity-Asset -> Unity-Resource
last_modified_at: 2025-08-01. 22:10 # Addressable: 플랫폼 별 빌드
---

## Resources

---

- ['_':](https://blog.naver.com/sorang226/223792661583)

## AssetBundle

---

AssetBundle로 불러오는 방법으로는 씬에 포함된 스크립트가 불러와지지 않는다.  

AssetBundle로 씬을 묶어오면, 최상위 오브젝트들의 순서가 뒤죽박죽이 되어버린다. (일정하지 않는다?) (알파벳 순으로 정렬된다?)  

또 에디터에서 불러올 때랑, 런타임에서 불러올 때랑 다르게 불러와진다.  

## Addressable

---

- AddressableDownloadRequest
- 플랫폼 별로 빌드 해야 함
  - 플랫폼마다 셰이더 다르게 컴파일 되다 보니까
  - C# Script도 Scripting Define Symbol에 따라 달라지긴 할 듯
  - 그 외 다른 점 있나? 기획 상 다른 것 말고
