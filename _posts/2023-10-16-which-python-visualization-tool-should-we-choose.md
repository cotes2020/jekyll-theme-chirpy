---
layout: post
title: Which Python Visualization Tool Should We Choose?
date: 2023-10-16 08:43 +0900
image:
    path: /posts/title/pyviz.png
    alt: 개략적으로 나타낸 파이썬 시각화 도구 관계도
---
[pyviz](https://pyviz.org/index.html)는 현재 존재하는 모든 파이썬 시각화 오픈소스 라이브러리를 소개하는 페이지이다. 일부 목적에 특화된 시각화 도구로 분류하여 github 스타 수, 이용자 수, 라이센스 종류나 스폰서 등 다양한 정보를 일목요연하게 볼 수 있다. 추가로 다양한 라이브러리간에 서로 장단점을 비교 분석하는 글, 튜토리얼 등 읽어볼 거리를 제공한다.

다음은 [책 『Scientific Visualization: Python + Matplotlib』](https://inria.hal.science/hal-03427242/)의 Introduction 부분에 나왔던 간단한 지침이다. 최신정보와 새로운 라이브러리들은 pyviz 사이트에서 확인하는 것이 좋다.

jupyter에서 브라우저를 통해 상호작용하는 시각화를 원한다면?

![bokeh logo](bokeh.png){: width="200"}
_bokeh_

desktop에서 매우 큰 데이터의 3D 시각화를 원한다면?

![vispy logo](vispy.png){: width="200"}
_vispy_

![mayavi logo](mayavi.png){: width="200"}
_mayavi_

직관적이고 빠르며 아름다운 시각화를 원한다면?

![seaborn logo](seaborn.png){: width="200"}
_seaborn_

![altair logo](altair.png){: width="200"}
_altair_

지리적 정보를 활용한 시각화를 원한다면?

![cartopy logo](cartopy.png){: width="200"}
_cartopy_

### 참조

[책 『Scientific Visualization: Python + Matplotlib』 Introduction](https://inria.hal.science/hal-03427242/)