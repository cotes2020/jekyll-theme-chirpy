```
title: `<canvas>` 와 `SVG`
author: dongee_seo
date: 2021-03-27
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
```

> HTML5에서는 `<canvas>` 와 `SVG`를 이용하여 기존에 플래시, 실버라이트로 했던 그래픽 처리를 대체할 수 있다.

## 1 . Canvas

1-1. 의의

- Canvas는 자체가 그래픽을 구현 하는 것이 아닌 웹문서에 그래픽 구현을 위한 영역을 정의하고
  실제로는 JavaScript를 이용해 Canvas API로 만든 코드를 이용해 그래픽을 구현한다.
- 픽셀기반의 즉시 모드 그래픽 시스템으로 매 프레임마다 다시 화면을 그린다. 비트맵![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F42dc7e1c-d781-4888-86bb-a537a76b9c30%2Fdownload.png)
- 브라우저 창안에 어떤것이든 그려내고 인터렉트한 이미지 편집도 가능하고 동영상제어도 가능하다.
  별도의 플러그인을 다운로드 할 필요 없이 실시간 그래프, 애니메이션, 대화형 게임을 사용자에게 제공할 수 있다.
  1-2. 사용법
  html 문서에 다음과 같이 `<canvas>`를 삽입한다.

```null
<canvas id="test" width="150" height="150"></canvas>
```

위의 요소를 js를 통해 찾아 제어하는 것이 canvas 의 기본적인 동작이다.

- 선분, 사각형, 원, 문자, 이미지 추가하는 method 존재
- 그래픽을 그리기 위해 **JavaScript를 사용**해야 함

```null
const canvas = document.getElementById("number");       //id로 canvas element에 접근
const ctx = canvas.getContext("2d");                    //2d 그래픽을 그릴 수 있는 메서드를 지닌 HTML5 오브젝트
const x = 32;                                           //원점 x 좌표
const y = 32;                                           //원점 y 좌표

const radius = 30;                                      //반지름
const startAngle = 0;                                   //0도
const endAngle = Math.PI * 2;                           //360도

ctx.fillStyle = "rgb(0,0,0)";                           //채우기 색 설정
ctx.beginPath();                                        //새로운 경로 생성, 경로 생성 후 그리기 명령들은 경로를 구성하고 만드는데 사용
ctx.arc(x, y, radius, startAngle, endAngle);            //x,y위치에 원점을 두고, radius 반지름을 가지고, startAngle에서 시작하여 endAngle에서 끝나며 주어진 방향(옵션 기본값은 시계방향)으로 호 그림
ctx.fill();                                             //경로의 내부를 채워서 내부가 채워진 도형을 그린다. 열린도형은 자동으로 닫히므로 closePath()호출 필요 X

ctx.strokeStyle = "rgb(255,255,255)";                   //윤곽선의 색 설정
ctx.lineWidth = 3;                                      //윤곽선의 두께
ctx.beginPath();                                        //새로운 경로 생성
ctx.arc(x, y, radius, startAngle, endAngle);            //원 생성
ctx.stroke();                                           //윤곽선을 이용하여 형 그림

ctx.fillStyle = "rgb(255,255,255)";                     //채우기 색 설정
ctx.font = "32px sans-serif";                           //문자 폰트 크기, 글씨체
ctx.textAlign = "center";                               //문자 정렬
ctx.textBaseline = "middle";                            //베이스 라인 정렬 설정
ctx.fillText("1", x, y);                                //주어진 x,y 위치에 주어진 "1" 텍스트를 채움, 최대폭 maxwidth 옵션값
```

## 2. SVG(Scalable Vector Graphic)

2-1. 의의

- W3C에서 1999년 개발한 XML기반의 오푼 표준의 벡터 그래픽 파일 형식이다.
- 모양 기반의 유지ㅣ 모드 그래픽 시스템으로 화면에 그릴 오브
- ![](<https://velog.velcdn.com/images%2Fseod0209%2Fpost%2Fac8db047-3b5a-4135-84c5-924d5c12d9d5%2Fdownload%20(1).png>)SVG는 점과 점사이의 계산을 통해서 그리는 그래픽임.

출처: [https://superfelix.tistory.com/604](https://superfelix.tistory.com/604) [☼ 꿈꾸는 도전자 Felix !]

2-2. 사용법
2-2-1 ``<img>`에 추가

`<img src="/resources/images/test.svg"/>`

2. CSS Background

- class에 아래와 같이 css background로 넣어주고,
  html 태그 요소의 class로 넣어줌
  css의 `background: url('/resources/images/test.svg');`

3. SVG를 직접 HTML에 inline으로 삽입
   \_아래처럼 html body태그 안에 바로 삽입

```null
  <body>

    <svg>

        <rect x="0" y="0" width="100" height="100"></rect>

    </svg>

  </body>
```

4. object태그를 이용

```null
 <object data="/resources/images/test.svg" type="image/svg+xml"></object>
```

## 3. 적합한 상황: Canvas vs. SVG

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F90597869-46e8-4d35-904c-c06e7f19b278%2Fcanvas_svg_---.png)

- Canvas: 작은표면, 많은 수의 개체가 필요한 부분, 전체적으로 많은 부분이 변경되어야 할 때 적합
- SVG: 더 넓은표면, 적은 수의 개체, 기존의 객체위주로 변경될 떄, 확대 또는 축소시에 깔끔한 이미지가 필요할 때.

> 출처
>
> - [https://www.nextree.co.kr/p9307/](https://www.nextree.co.kr/p9307/)
> - [https://developer.mozilla.org/ko/docs/Web/API/Canvas_API/Tutorial/Basic_usage](https://developer.mozilla.org/ko/docs/Web/API/Canvas_API/Tutorial/Basic_usage) -[https://superfelix.tistory.com/604](https://superfelix.tistory.com/604)
