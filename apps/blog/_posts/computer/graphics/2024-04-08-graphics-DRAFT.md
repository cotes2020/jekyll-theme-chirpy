---
title: "Graphics"
# description: ""
categories: [컴퓨터, 그래픽]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-08. 07:59
# last_modified_at: 2024-08-29. 21:46
# last_modified_at: 2024-11-13. 07:47 # Init
last_modified_at: 2025-06-10. 22:57 # ~정리, 네이밍: DirectX -> Graphics
---

## Q

---

- `그래픽스 API`
- DirectX / OpenGL, 뭐든 "CG 이론을 알고 있다"를 어필할 수 있는

## 메모

---

{% include embed/youtube.html id = "NTvhVxSC_80" %}

- [Direct X 프로그래밍 학습에 대한 조언](https://megayuchi.com/2019/04/18/direct-x-프로그래밍-학습에-대한-조언/)
- 콘솔모드 테트리스 or 뱀 (뭐든 간에)
- GDI로 헬로 월드
- -> Device Context에 대한 학습
- Direct Draw로 화면에 점 찍어보기
  - -> 필요없는데, 왜?
- 비트맵 제어를 배울 수 있음
- 포인터에 능숙해짐.
- 그래픽 하드웨어 다루는 기초
- 이미지 데이터 (텍스처) 에서의 Pitch의 개념?
- Width * bytes in format과 Pitch 는 뭐가 다른가?
- 이미지 데이터에서 Padding이 고려된 한 row의 길이
  - 왜 Padding이 왜 고려되어야 해?
  - -> 성능 - 그 메모리가 그래픽 하드웨어에 1:1 맵핑 되어야 하는 주소일 수도 있으므로...
  - 그래픽스도 결국은 하드웨어 구조를 알아야함
  - 픽셀도 Padding이 있을수도
- Direct Draw로 간단한 게임 만들어보기
  - 이미지 다루는 법
  - 마우스와 키보드 이벤트 처리
  - COM 다루는 방법 = win32/ MS의 프로그래밍 스타일 이해하기
  - 네트워크 프로그래밍 -> windock + 멀티스레드 + IOCP
  - 게임 프로그래밍 -> D3D11 -> D3D12
- 괜찮은 dx11책 하나 사셔서 삼각형 그리기
  - 모델 로딩해서 보여주기 (FBX 라이브러리가 있으니 그걸 사용해도 될듯)
  - 셰이더 이것저것 작성해보기 (어차피 dx11 일부?)
- 그래픽스 파이프라인을 얼마나 이해하고 있는지, 원하는 셰이더를 상용엔진에서 적용시킬 줄 아는지  
- dx11 공부, 최적화 기법 적용  
- 기본적인 공간벡터, 행렬, 삼각함수
- 화면 상에 선이나 원을 찍는 기본적인 알고리즘 (DDA등)
- Concave한 폴리곤인지 판별하는 부분인 Primitive 내용, 핵심인 3D Viewing, 회전 스케일링 나오고 Clipping
- Visually detection하는 여러가지 방법들, Shading 들어가면 광원이랑 물체 두개 사이만 판별하는 flat, goraud, phong, blinn-phong shading, 모든 물체에 광원을 실시간으로 적용하는 Global illumination인 레이트레이싱을 넘어서 texture, color
- 학부 과정: 3차원 공간 만드는 것
- PBR, RT, spatial structure, fluid, procedural animation 등
- 파이프라인 쭉 훑고 텍스쳐링이랑 알파블렌딩 해보면 충분?
- ['OpenGL-Tutorial': 렌더링 기초?](https://www.opengl-tutorial.org/)
