---
layout: post
title:  "08 주차 Intensity Transformations and Spatial Filering II"
date:   2020-05-25 11:27:39
categories: [ImageProcessing,Histogram Processing]
tags: [Histogram Processing]
---

# Histogram Processing

어떤 디지털 이미지의 강도가 [0,255]를 가질 때, 

분산된 히스토 그램으로 나타날 수 있는데 

![image](https://user-images.githubusercontent.com/46625602/85114543-92e99500-b254-11ea-814f-9bf7ae6652a9.png)

rk는 히스토 그램의 강도고, <br/>
nk는 rk라는 강도를 가진 이미지의 수다.<br/>
즉 다시 나타내면 `h(rk) = nk`라고 나타낼 수 있다. 

근데 픽셀의 개수 그대로 사용하면 y축의 개수가 천차 만별이 되기 때문에 서로 다른 크기의 이미지들을 비교할 때는 정규화 해서 사용해야 한다.
<br/>
따라서 위의 값에서 nk를 전체 픽셀의 개수만큼 나누어 줘서 확률로 사용해버린다. 
<br/>
당연히 확률을 사용했기 때문에 다 더하면 1이 된다.

히스토그램은 공간 도메인 기법 중에서 가장 기법이 되며 이미지를 향상을 위해서 히스토그램을 이용한다.  (이미지의 통계)

이미지의 통계 뿐만 아니라 다른 이미지 프로세싱을 사용할 때 판단의 기준이 히스토그램이 될 수도 있다. 
<br/>
히스토그램은 만드는 것도 쉽다
<br/>

다음 그림은 각각의 이미지를 히스토그램으로 나타낸 것이다.

어두운거, 밝은거, 낮은 대조, 높은 대조

![image](https://user-images.githubusercontent.com/46625602/85114971-e6a8ae00-b255-11ea-87e9-b65850c802ed.png)

x 축은 rk, y 축은 nk/n*m 이다
<br/>

* 대조가 낮은 것은 한쪽에 히스토 그램이 몰려 있다는 것을 알 수 있고
* 대조가 높은 것은 넓게 퍼져 히스토그램이 형성되어 있다.

<br/><br/>

**Histogram Equalization**

히스토그램 평활화 이미지의 강도 값은 연속적인 값이 주어진다는 가정 하에서 이루어 진다.<br/>
이미지 강도를 r 이라고 하고 0~(L-1)255다.<br/>
0은 블랙 L-1은 흰색이다 <br/>
s(출력 이미지 강도) = T(r(입력 이미지 강도))

다음과 같은 3가지 가정이 있다. 

![image](https://user-images.githubusercontent.com/46625602/85115993-97fc1380-b257-11ea-856c-448c1876cd8a.png)

A) 입출력 이미지에 대한 함수는 단조 증가 
B) 위의 조건에 역함수도 단조증가 한다. 즉 입력 강도의 범위와 출력 강도의 범위가 같다. 또한 1:1 관계가 되어 역함수가 존재 할수 있도록 하기 위해 C 조건인 엄격한 단조 증가 조건을 만족해야 한다. 
C) 엄격한 단조 증가

반올림 하면 다른 값들을 찾을 수 있는데, 

이때 두개의 다른 x 값이 같은 y값을 가질 때가 있다. 이때 1:1 조건을 만족 시키지 못 할 수 도 있다. 왜냐하면 반올림해서 같은 값을 가질 때가 이렇게 되는데, 이렇게 되면 반올림 되는 정수 값을 다시 넣어도 다시 정수로 1:1 대응이 되기 때문이다.

r은 강도 값, 

pr 은 r 의 강도값을 가지는 pixel의 개수


p(r) 입력 이미지 p(s) 는 출력 이미지 분포 

![image](https://user-images.githubusercontent.com/46625602/85129450-19f73700-b26e-11ea-8eb5-d60dcca04451.png)


![image](https://user-images.githubusercontent.com/46625602/85129758-a275d780-b26e-11ea-9375-f7c442ac51b6.png)

함수에 대한 적분이니까 누적 증가이다. 절대로 앞의 값보다 역전된 값이 나올 수가 없다. <BR/>
위의 식은 CDF라고 불린다.<BR/>
앞에다가 L-1 을 곱하는 것은 다 더하면 최대 값이 1에서 255로 된다. 그래서 값의 범위를 맟춰 주려고 곱하는 것이다. 
<BR/>

![image](https://user-images.githubusercontent.com/46625602/85130823-9ee35000-b270-11ea-950b-56e04cff1acd.png)

앞에서 맨 위의 미분 식을 가져 왔다. 

S = T(r)에다가 미분을 각각 취하면 이런 식이 되괴 T를 불면 그 다음 식이 나온다.

각 식을 불면 이렇게 나오고 위의 미분 식에다가 나온 식을 대입하면 아래의 식이 나오다. 결과적으로 P(S)는 1/L-1 이 나온다. 즉 S가 무슨 값이 나오든 상수 값인 1/L-1 이 나온다

CDF 를 취하면 다음과 같은 상수 함수가 나온다.

![image](https://user-images.githubusercontent.com/46625602/85131098-216c0f80-b271-11ea-9f80-0315caf4da53.png)

높은 대조의 이미지를 가지기 위해서는 전 범위의 강도를 가지고 있어야 하고, 

이 픽셀들의 강도 분포가 상수 함수의 양상과 비슷해야 하기 때문이다.


---

**Refference:**

* 08주차 강의자료(Chapter 08 Intensity Transformations and Spatial Filering II)
* [이미지 평균화](https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html)
* [이미지 코드](https://webnautes.tistory.com/1043)