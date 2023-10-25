---
layout: post
title: Matplotlib Elements
date: 2023-10-18 08:46 +0900
image:
    path: /posts/title/matplotlib.png
    alt: matplotlib 로고
---

# Matplotlib 들어가기

matplotlib
: 파이썬 시각화 도구 중 하나이다. pandas, seaborn 등 다양한 타 라이브러리의 기반이 된다.

matplotlib은 사용하기에 초보자도 어렵지 않게 입문할 수 있으나 대략적인 구조를 알아두지 않으면 디버깅하기도 어렵고 공식 문서도 알아보기 어려우며 내가 원하는 자유자재로 커스터마이징하기도 어렵기 때문에 이번에 한 번 깔끔하게 정리하여 이 다음부터 공식 문서를 보고 공부할 수 있는 정도의 수준을 만드는 것을 목표로 하였다.

# Matplotlib의 구조

## 간단한 예시

![matplotlib의 구성 요소](elements.png)
_matplotlib의 구성 요소_

matplotlib은 Figure, Axes, Axis, Spines, Grid, Markers, Title 등 다양한 구성요소를 계층적으로 구성해놓고 플롯팅한다. 복잡해보이지만 사용자가 이를 하나하나 생성할 필요는 없다. 대부분 디폴트값을 통해 자동으로 생성된다. 수정하고자 하는 구성 요소가 있다면 해당 객체에 접근하여 프로퍼티를 고쳐주는 방식으로 원하는 그래프를 플로팅할 수 있다.

가장 간단한 예시 코드이다.

```python
plt.plot(range(10))
plt.show()
```
{: file='implicit.py'}
![implicit 결과](result1.png){: width="320"}_implicit.py의 결과_

위 실행결과를 보면 Figure, Axes, Line, Major tick, Spines, Major tick label이 자동으로 생성된 것을 알 수 있다. plt.plot을 호출하면 내부적으로 일어나는 일이다.

만약 여기서 그래프의 크기를 변경하고 싶다고 하자. Figure 객체에 접근하여 figsize 프로퍼티를 수정하면 된다.

```python
fig = plt.figure(figsize=(6,6))
ax = plt.subplot()
ax.plot(range(10))
plt.show()
```
{: file='explicit.py'}
![explicit 결과](result2.png){: width="320"}_explicit.py의 결과_

위 코드는 Figure, Axes까지는 직접 생성하고, Figure 객체를 생성할 때 figsize 프로퍼티를 지정한 형태이다.

또한 `plt.subplots`를 통해 더 컴팩트하게 표현할 수 있다.

```python
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'aspect':1})
ax.plot(range(10))
plt.show()
```

>`plt.plot`과 `ax.plot`은 뭐가 다른걸까? `plt.plot`은 가장 마지막으로 생성된 ax의 plot을 호출하는 편의용 래퍼 함수이다. 그래서 많은 경우에 `plt.plot`을 쓰나 `ax.plot`을 쓰나 차이가 없다. 하지만 가능한 한 `ax.plot`을 써서 명시적으로 표현해주는 것이 코드 스타일 상으로 좋다.
{:.prompt-tip}

## Matplotlib의 구성 요소

[**Figure**](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure)

* 가장 최상위 계층에 있는 구성 요소
* `plt.figure`, `plt.subplots` 함수로 생성한다.
* `figsize`, `facecolor`, `suptitle` 등 다양한 설정을 할 수 있다.
* `savefig` 메소드를 호출하여 파일로 저장할 수 있다.

[**Axes**](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes)

* Figure 바로 아래 계층에 있는 구성 요소
* `plt.subplot`, `plt.subplots`, `Figure.add_axes` 함수로 생성한다.
* 하나의 Figure는 0개 이상의 Axes를 가진다.
* 각각의 Axes는 하나의 데이터를 표현하는 렌더링 영역을 나타낸다.

[**Axis**](https://matplotlib.org/stable/api/axis_api.html#matplotlib.axis.Axis)

* Axis는 Spine에다가 major/minor tick, major/minor/axis label 등 부가적인 정보를 나타내는 역할을 한다.
* 하나의 Axes는 2개의 Axis를 가진다. 가로 방향, 세로 방향 하나씩 각각 xaxis, yaxis를 가진다.

[**Spine**](https://matplotlib.org/stable/api/spines_api.html#matplotlib.spines.Spine)

* 데이터가 표시될 영역의 경계를 나타내는 역할을 한다.
* 하나의 Axes는 4개의 Spine을 가진다. Spine은 Axes의 렌더링 영역을 사각형으로 둘러싸는 선 형태로 표현된다.
* `color`, `position`, `visible` 등 다양한 설정을 할 수 있다.

[**Artist**](https://matplotlib.org/stable/api/artist_api.html#matplotlib.artist.Artist)

* 화면에 그려질 수 있는 모든 것은 Artist 객체를 상속한다.
* Figure, Axes, Axis, Spine 모두 Artist를 상속한다.
* 그 외 Text, Line2D, Patch 등도 Artist를 상속한다.
* 모든 Artist는 두 개 이상의 Axes에 동시에 속할 수 없다.

## 그래픽의 최소 단위

matplotlib의 구성 요소를 화면에 어떻게 렌더링 할지를 지정하기 위한 그래픽의 최소 단위가 다음과 같이 정의되어있다. 여기에는 아래 총 3가지가 존재한다.

[**Patch**](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch)

* marker나 bar에 쓰이는 크고 작은 그래픽, 원이나 사각형같은 도형까지를 포함한다.
* 예를 들어 facecolor로 그려지는 배경색은 Patch가 그리는 것이다.

[**Line**](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D)

* 선이다. tick이나 Spine을 렌더링할 때 사용된다.

[**Text**](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)

* 글씨를 쓸 때 쓰인다.

이러한 최소 단위들은 각각 다양한 성질을 가진다. (facecolor, edgecolor, shadows, outline, antialiased, transparency 등등) 하지만 일반적으로 직접 조작하지 않고, 이걸 가지고 있는 앞서 본 matplotlib의 구성 요소가 가진 메소드를 호출하여 조작한다. (set_facecolor, set_edgecolor 등등)

## 예시 코드

### 1. tick label 조작

Axes 객체에서 xaxis에 접근하여 tick label 목록을 얻은 뒤 폰트를 굵게 바꿔 화면에 출력하는 예시이다.

```python
fig, ax = plt.subplots(figsize=(5,2))
for label in ax.xaxis.get_ticklabels():
    label.set_fontweight('bold')
plt.show()
```
![result3](result3.png){: width="320"}_실행 결과_

### 2. multi axis

Axis축을 여러 개 만들 수도 있다.

```python
fig = plt.figure()
ax = fig.add_subplot()
ax2 = ax.twinx()
ax.plot(range(10))
ax.set_facecolor((1.0, 0.0, 0.0, 0.3))
ax2.set_ylim(ax.get_ylim())
plt.show()
```
![result4](result4.png){: width="320"}_실행 결과_

twinx는 xaxis를 공유하는 Axes를 복제하고 xaxis, patch를 visible하지 않게 하여 마치 축이 여러 개인 것처럼 표현될 수 있도록 해주는 함수이다.

>위 코드에서 `ax2.set_facecolor`를 통해 배경색을 지정해도 실제로는 배경색이 바뀌지 않는다. 왜냐하면 `ax2`는 `twinx`함수로부터 생성될 때 `patch`가 `visible=False`로 지정되었는데 배경색을 그리는 주체가 patch이기 때문이다. 따라서 `ax2.patch.set_visible(True)`를 해야 배경색이 바뀐다.
{:.prompt-tip}

## zorder

모든 그래픽의 최소단위는 zorder라는 float값을 가진다. zorder는 가상의 깊이를 나타낸 것으로, 무엇이 앞에 오고 무엇이 뒤에 갈지를 결정한다.

```python
from matplotlib.patches import Circle
fig, ax = plt.subplots(figsize=(6,6))
circles = [
    Circle((0.4, 0.7), 0.1, facecolor='#ddddaa', zorder=5), # 위, 왼쪽 원
    Circle((0.5, 0.7), 0.1, facecolor='#aadddd', zorder=6), # 위, 가운데 원
    Circle((0.6, 0.7), 0.1, facecolor='#ddaadd', zorder=7), # 위, 오른쪽 원
    Circle((0.4, 0.3), 0.1, facecolor='#ddddaa', zorder=7), # 아래, 왼쪽 원
    Circle((0.5, 0.3), 0.1, facecolor='#aadddd', zorder=6), # 아래, 가운데 원
    Circle((0.6, 0.3), 0.1, facecolor='#ddaadd', zorder=5), # 아래, 오른쪽 원
]
for circle in circles:
    fig.add_artist(circle)
plt.show()
```

![result5](result5.png){: width="320"}_실행 결과_

zorder가 작을수록 먼저 그려진다. 따라서 뒤에 있는 것처럼 보여진다. zorder의 기본 값은 다음과 같다.

Figure | Axes(spines, ticks & labels) | Patches | Lines | Text | inset axes & legend
0      | 0                            | 1       | 2     | 3    | 5

# 백엔드

matplotlib의 백엔드
: `plt.show()`가 호출된 순간 화면에 실제로 그림을 그리는 렌더러와 상호작용할 수 있는 인터페이스를 말한다. matplotlib이 지원하는 백엔드는 종류가 다양하고 각각의 장단점이 있다.

현재 어떤 백엔드를 사용중인지 확인해보자.
```python
import matplotlib as mpl
print(mpl.get_backend())
```

원하는 백엔드로 언제든 바꿀 수 있다. 단, `matplotlib.pyplot`을 import 하기 전에 해야한다.
```python
import matplotlib as mpl
mpl.use('백엔드 이름')
```

백엔드는 static backends와 interactive backends로 분류된다.

[static backends](https://matplotlib.org/stable/users/explain/figure/backends.html#static-backends)
: 인터페이스가 없어서 상호작용할 수 없는 백엔드이다. Agg, PDF, PS, SVG, PGF, Cairo 등이 있고 그래픽 표현 방식(raster / vector)이 다르다.

![static backend](static_backend.png){: width="320"}_static backend (Agg)_

[interactive backends](https://matplotlib.org/stable/users/explain/figure/backends.html#interactive-backends)
: 인터페이스가 있어서 상호작용할 수 있는 백엔드이다. 인터페이스의 종류는 Qt, GTK, Tk, Web 등이 있고, 렌더러인 Agg, Cairo 등과 결합되어 작동한다. 예를 들어 QtAgg, GTK4Agg, GTK4Cairo, WebAgg 등이 있다.

![interactive backends](interactive_backend.png){: width="320"}_interactive backends (WebAgg)_

인터페이스를 통해 그래프의 특정 부분을 확대/축소/이동하고 파일으로 저장할 수 있다.

## 그래픽 표현 방식

그래픽 표현 방식은 raster grphic 방식과 vector graphic 방식이 있다.

raster graphic
: 렌더링 결과를 2차원 픽셀에 정보를 담아 저장하는 방식이다. 확대할 경우 화질이 안좋아질 수 있다.

raster graphic은 파일 확장자가 png, jpg, tiff 등인 경우가 해당된다.
raster graphic으로 결과를 표시할 수 있는 렌더러는 AGG, Cairo가 있다.

vector graphic
: 점, 선, 도형의 좌표와 색상 등의 정보를 저장하여 표현하는 방식이다. 확대하더라도 화질이 전혀 안좋아지지 않는다.

vector graphic은 파일 확장자가 pdf, svg, ps 등인 경우가 해당된다.
vector graphic으로 결과를 표시할 수 있는 렌더러는 PDF, PS, SVG, PGF, Cairo가 있다.

> 렌더러의 종류에 상관없이 `savefig`을 통해 그래프를 사진으로 저장할 땐 파일 확장자명에 따라 그래픽 표현 방식이 결정된다. 예를 들어 Agg는 raster 그래픽 렌더러이지만 `filename.svg`로 저장하면 vector 그래픽으로 저장된다.
{:.prompt-tip}

## Ipython을 사용한다면 알아둘 것

### [matplotlib_inline](https://pypi.org/project/matplotlib-inline/)

Ipython을 사용하는 경우 백엔드 이름이 'module://matplotlib_inline.backend_inline'으로 설정되어 있을 수 있다.
* 이것은 `matplotlib_inline`이라는 라이브러리의 백엔드이다.
* 지금까지 살펴본 것과 달리 non builtin backend이면서 Ipython 전용 백엔드이다.
* Ipython의 셀 사이사이에 렌더링 결과를 바로바로 출력하는 역할을 한다.

### [matplotlib과 관련된 매직 커멘드](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

Ipython에서는 python과 달리 `%`로 시작하는 매직 커멘드를 실행시킬 수 있다.
* `matplotlib`을 설치할 경우 이와 관련된 매직 커멘드가 추가되는데, 다른 백엔드를 사용하다가 matplotlib_inline 백엔드를 선택하고 싶다면 `%matplotlib inline`을 쓰면 된다.
* 물론 기존의 방식인 `mpl.use('module://matplotlib_inline.backend_inline')`처럼 써도 된다.
* 그외 다른 백엔드를 사용하기위한 다양한 매직 커멘드도 존재한다.

### [ipympl](https://matplotlib.org/ipympl/)

matplotlib_inline 백엔드는 static backend에 속한다. Ipython에서 interactive backend를 사용하고 싶다면?

* `ipympl`을 설치한 뒤 `%matplotlib ipympl`으로 백엔드를 지정해준다.
* `mpl.use()'module://ipympl.backend_nbagg')`으로 지정해줄 수도 있다.

![ipympl](ipympl.png){: width="320"}_ipympl_

### matplotlib_inline와 인터렉티브 모드

matplotlib에는 [인터렉티브 모드](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.isinteractive.html#matplotlib.pyplot.isinteractive)를 on/off 할 수 있는 기능이 있다. 하지만 matplotlib_inline 백엔드를 선택했을 경우 인터렉티브 모드에 관계없이 항상 non-interactive 하게 작동한다.

# 해상도

## 파일으로 저장하기

dpi (dots per inch)
: 프린터가 1인치 당 표현할 픽셀의 수. matplotlib에서는 기본값 100으로 설정된다.

지금까지 figure의 크기를 지정했던 것은 모두 단위가 inch이다. `savefig`와 같은 함수를 통해 그래프를 저장할 경우 dpi를 반영하여 이미지의 크기가 정해진다.

```python
fig, ax = plt.subplots()
ax.plot(range(10))
fig.savefig('plot.png')
```
이 경우 `plot.png`의 크기는 (640, 480)이 된다. figure의 크기는 기본값 (6.4, 4.8)을 가지고, dpi는 기본값 100을 가지기 때문이다.

dpi는 figure의 프로퍼티이다.
```python
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.plot(range(10))
fig.savefig('plot.png')
```
이 경우 `plot.png`의 크기는 (1200, 1200)이 된다. figure의 크기는 (4, 4)이고 dpi는 300이기 때문이다.

> Ipython을 쓰는 경우 셀 출력으로 볼 수 있는 그래프도 dpi에 따라 확대/축소가 가능하지만 정확히 픽셀 수가 `savefig`를 했을 때와 일치하지는 않는다.
{:.prompt-tip}

> vector graphic 형식으로 저장한 경우 확대/축소를 자유롭게 해도 이미지의 품질이 떨어지지 않기 때문에 원본 크기를 고집할 필요가 없다. 하지만 vector graphic 형식으로 저장하더라도 벡터화가 불가능한 요소(예를 들어 image)가 있을 수 있기 때문에 적절한 dpi와 이미지 크기를 지정해주어야 한다.
{:.prompt-tip}

## 화면에 실제 크기로 표시하기

ppi
: 모니터의 1인치당 픽셀의 수.

dpi와 ppi를 알면 화면에 표시되는 크기도 정확히 계산할 수 있다. [디지털 자](https://www.piliapp.com/actual-size/cm-ruler/)를 직접 구현해보자.

ppi를 계산하기 위해 사용하고 있는 모니터가 몇 인치인지와 해상도 정보는 주어져야 한다. 

```python
def ruler(monitor_inch, monitor_pixel, size_cm, dpi):
    '''
    2차원 디지털 자를 matplotlib으로 그립니다.

    Args:
        monitor_inch (float): 사용하고 있는 모니터 인치 수. ex) 24, 27, 32, ...
        monitor_pixel (tuple[int, int]): 사용하고 있는 모니터 해상도. ex) (1920, 1080)
        size_cm (tuple[int, int]): 출력할 자의 가로 길이와 세로 길이. ex) (5, 5)
        dpi (int): dot per inch. ex) 72, 96, 100, 300, 600, ...
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.transforms as transforms
    import math

    inch = 2.54
    fig = plt.gcf()
    fig.set_facecolor('lightgrey')
    fig.set_dpi(dpi)

    ppc = math.hypot(*monitor_pixel) / monitor_inch / inch
    
    fig.set_size_inches(size_cm[0] * ppc / dpi, size_cm[1] * ppc / dpi)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='None')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_tick_params('both', labelsize='x-small', direction='in', pad=-8)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_tick_params('both', labelsize='x-small', direction='in', pad=-8)
    for label in ax.xaxis.get_ticklabels():
        label.set_verticalalignment('top')
    for label in ax.yaxis.get_ticklabels():
        label.set_horizontalalignment('left')
    ax.text(0.5, 0.4, 'cm', ha='center', va='center', size='x-small')
    ax.grid(linestyle='--', linewidth=0.5)

    n = int(size_cm[0]) + 1
    ax.set_xlim(0, size_cm[0])
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels([''] + list(range(1, n)))

    markersize = ax.xaxis.get_ticklines(True)[0].get_markersize()
    for line in ax.xaxis.get_ticklines(True)[2::9]:
        line.set_markersize(1.5 * markersize)

    n = int(size_cm[1]) + 1
    ax.set_ylim(size_cm[1], 0)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([''] + list(range(1, n)))

    markersize = ax.yaxis.get_ticklines(True)[0].get_markersize()
    for line in ax.yaxis.get_ticklines(True)[2::9]:
        line.set_markersize(1.5 * markersize)

    bbox = transforms.Bbox([[0, 0], fig.get_size_inches()]).padded(0.1)
    
ruler(27, (1920, 1080), (29.7, 21.0), 96)
```

내가 사용하는 모니터는 27인치에 1920, 1080 해상도이고, A4용지와 똑같은 크기의 29.7, 21.0으로 만들어보았다. dpi는 많이 쓰이는 96으로 잡아보았다.

![ruler](ruler.png){: width="320"}_실행 결과_

만약 27인치 1920, 1080 해상도 모니터를 사용한다면 실제 자를 가져와서 모니터에 길이를 대 보면 일치하는 것을 볼 수 있다.

> ms word가 화면에 표시하는 크기는 실제 인쇄물의 크기와 상관 없다고 한다. 다시 말해, 제어판 설정으로 확대 비율을 100%로 설정하고 word 내 확대 비율도 100%로 맞추어도 실제 종이를 모니터 스크린에 가져다 대고 비교해보면 크기의 차이가 있을 수 있다. [참고 링크](https://answers.microsoft.com/en-us/msoffice/forum/all/why-is-100-on-the-screen-not-100-of-the-page/b97c0177-9f12-4b69-aa36-29d3fb168821)
{:.prompt-tip}

## 글자크기 단위 포인트(pt)

pt
: 글자 크기를 나타낼 때 쓰이는 길이 단위이다. 1pt는 1/72인치이다.

글자 포인트가 같더라도 폰트 종류에 따라 폰트 크기가 달라 보일 수 있다. 글자의 정확히 어떤 부분이 몇 포인트인지에 관한 기준이 있는 것이 아니기 때문이다. 평균적으로 라틴어 기반 언어의 대문자 높이는 포인트 크기의 70%이고 소문자 x의 높이는 50%이다.

> 웹 브라우저에서는 실제 길이 단위(인치)는 중요하지 않고 픽셀 수가 중요하기 때문에 CSS에서는 dpi가 96인 것으로 간주하여 1pt에 4/3픽셀인 것으로 계산된다. [참고 링크](https://fonts.google.com/knowledge/glossary/point_size)
{:.prompt-tip}

> 예외적으로 latex에서는 1pt에 1/72.27인치이다. [참고 링크](https://www.overleaf.com/learn/latex/Lengths_in_LaTeX#:~:text=a%20point%20is%20approximately%201/72.27%20inch)
{:.prompt-tip}

# 예시 코드

```python
def curve():
    ''' 데이터 생성용 함수 '''
    n = np.random.randint(1,5)
    centers = np.random.normal(0.0,1.0,n)
    widths = np.random.uniform(5.0,50.0,n)
    widths = 10*widths/widths.sum()
    scales = np.random.uniform(0.1,1.0,n)
    scales /= scales.sum()
    X = np.zeros(500)
    x = np.linspace(-3,3,len(X))
    for center, width, scale in zip(centers, widths, scales):
        X = X + scale*np.exp(- (x-center)*(x-center)*width)
    return X

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

n_series = 50
fig, axes = plt.subplots(1, 3, figsize=(18, 18), dpi=72, sharey=True)
fig.set_tight_layout(True)

for axes_idx, ax in enumerate(axes):
    
    # spine 설정
    for spine in ax.spines.values():
        spine.set_visible(False)

    # xaxis 설정
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_tick_params(width=2, length=7)
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontsize('xx-large')
    # 위 for문을 안 쓰고 아래 함수를 호출하여도 같은 동작이 수행됨
    # ax.xaxis.set_tick_params(labelsize='xx-large')

    # yaxis 설정
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(1, n_series + 1)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('Serie %d'))
    ax.set_ylim(0, 53)
    for ticklabel in ax.yaxis.get_ticklabels():
        ticklabel.set_va('bottom')
    for tickline in ax.yaxis.get_ticklines():
        tickline.set_visible(False)
    # 위 for문을 안 쓰고 아래 함수를 호출하여도 같은 동작이 수행됨
    # ax.yaxis.set_tick_params(left=False)

    colors = mpl.colormaps['Spectral'](np.linspace(0, 1, n_series))
    zorders = np.linspace(4, 3, n_series * 2)

    # 아래부터 그리는 요소들은 zorder를 직접 지정해주었음.
    for i in range(50):
        y = curve()
        x = np.linspace(-3, 3, len(y))

        # 선그래프 그리고 색칠하기
        ax.plot(x, y * 3 + i + 1, color='black', zorder=zorders[i * 2])
        ax.fill_between(x, i + 1, y * 3 + i + 1, color=colors[i], zorder=zorders[i * 2 + 1])

        # 그래프 오른쪽에 작게 별표시 있는 것 그리기
        n_star = np.random.randint(4)
        ax.text(3, i + 1, '*' * n_star, ha='right', va='bottom',
                fontsize='large', zorder=zorders[i * 2] + 1)

    # 가운데를 가로지르는 점선 그리기
    line = mpl.lines.Line2D([0, 0], [0, 53], zorder=5, color='black', linestyle='--')
    ax.add_artist(line)
    # 위 Line2D를 직접 만들어서 등록하는 작업 대신 아래 함수를 호출하여도 같은 동작이 수행됨
    # ax.axvline(0, zorder=5, color='black', linestyle='--')
    # 위 함수는 vertical line을 그리는 함수이고 horizontal line을 그리는 함수는 axhline임.

    # 제목 지정
    ax.set_title(f'Value {axes_idx+1}', x=0, y=1, ha='left', va='top',
                 fontsize='xx-large', fontweight='bold')
```

![example](example.png)_실행 결과_