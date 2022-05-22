---
title: CSR project(create-react-app)를 SSR(Next.js)로 옮기기 (3.31~4.08)
author: Beanie
date: 2022-04-09 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Next]
mermaid: true
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

처음에 Catty 웹앱은 리액트 CSR로 개발되었다. CSR로 개발한 이유라면 SSR이 익숙치 않았고, 좀 더 빨리 프로토타입을 개발하고 싶어서 정도였던 거 같다.
그러나 SEO 대응하면서 이슈가 발생했고 Next.js로 옮겨가야하나 생각했지만 일이 너무 커질 거 같아 CSR에서 SEO 비스무리하게 할 수 있는 다양한 툴들로 나름 SEO 최적화를 해보려고 하였다.

하지만...결국 다 포기하고 Next.js로 옮겨가게 되었다...ㅎ 기존의 리액트 프로젝트를 Next로 옮기는데 1주일 정도 소요되었다. 이번 글에서는 갈아엎기 까지 CSR을 고수하기 위해서 한 여러 시행착오와 결국 Next.js 로 옮기며 겪은 다양한 에러에 대해서 남겨보려고 한다.

## react-snap과 react-helmat 라이브러리
&nbsp;

먼저 CSR에서 react-snap이라는 라이브러리를 사용할 수 있다. [https://byseop.netlify.app/csr-seo/](https://byseop.netlify.app/csr-seo/)
그렇지만 블로그를 참고해서 다 적용을 하니까 재밌는 문제가 발생하였다.

Catty 처음 진입 로직은 다음과 같이 설계되어 있는데,

<div class="mermaid">
  graph LR;
  A(랜딩페이지)-->B[자동로그인 확인];
  B-->C(로그인 페이지);
  B-->D(메인페이지);
</div>

메인페이지에 진입하기 전에 랜딩페이지의 DOM을 생성하고 이를 없애지 않은 채로 메인페이지에 들어가다보니 메인페이지의 배경화면으로 랜딩페이지의 DOM이 남아있는 문제가 생겼다. 이외에도 CSS가 많이 깨지고, 무엇보다 react-snap이 최근 2년 동안 업데이트를 하지 않은 정지된 라이브러리 같아 그냥 이런식말고 다른 방법을 찾기로 결정했다.

&nbsp;

## webpack과 babel 설정을 통해 Next.js 없이 SSR 구축
&nbsp;

그러던 중 [React로 Next.js 처럼 Server-side-rendering 구현하기](https://minoo.medium.com/next-js-%EC%B2%98%EB%9F%BC-server-side-rendering-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-7608e82a0ab1) 라는 미디엄 포스팅을 발견하였고 괜찮은 거 같아 프로젝트에 적용해보았다. 적용하는 건 이 블로그 포스팅을 따라한 것이 전부라 따로 적지는 않을 것이다.

### window is not defined 에러

그렇게 다 설정을 완료했는데 window, document, self 등 객체에 접근할 때 `not defined` error가 발생하였다.

window 객체는 브라우저의 요소들과 자바스크립트 엔진, 그리고 모든 변수를 담고 있는 객체이다. 인터넷 브라우저를 보면 위에 탭들도 있고, 주소창도 있고, 즐겨찾기, 툴바 등이 있다. 그 다음부터는 이제 웹사이트가 표시된다. 여기서 브라우저 전체를 담당하는 게 window 객체이고, 웹사이트만 담당하는게 document 객체라고 볼 수 있다.

이런 window, document 브라우저 객체는 말그대로 **브라우저**의 객체이기 때문에 서버에는 존재하지 않는다. 따라서 SSR으로 이 브라우저 객체에 접근하려고 하면 에러가 발생한다. 이런 에러를 해결하기 위한 방법은 다음 포스팅([SSR에서 window 객체가 없다고 할 때]()) 에 정리해두었다. Catty 웹앱도 이런 방법들을 이용하여 에러를 해결하려고 했지만, Catty 웹앱은 외부 라이브러리를 사용해서 useEffect에 넣지 못하는 상황이 있었다. 그래서 Next.js의 dynamic 기능 처럼 특정 component에서만 SSR을 끄고 싶었지만, 지금 webpack 구성에서 맞는 방법을 찾지 못하였다(..) 그래서 결국 특정 컴포넌트에서 SSR을 끌 수 있게 하는 dynamic 기능이 있는 Next.js로 옮겨가게 되었다.

&nbsp;

## React.js 프로젝트를 Next.js로 옮기기
&nbsp;

결국 돌고돌아 Next.js로 옮겨가게 되었다. 지금 프로젝트 폴더에서 Next.js을 import하면 예상치 못한 충돌이 발생할 수 있을 거 같아, 새롭게 Next.js로 프로젝트 폴더를 만들어서 기존 코드 파일을 하나씩 옮기기로 하였다.

### Next.js란?

먼저 시작하기 전에 Next.js를 한 번 짚고 넘어갈 것이다. [Next.js](https://nextjs.org/)는 React 프레임워크이다. Next.js는 크게 다음의 기능을 지원한다.
* 다이나믹 라우트를 지원하는 직관적인 페이지 기반의 라우팅 시스템
* 각 페이지 마다 기본적으로 사전 렌더링, SSG(Static Generation)과 SSR(Server-Side Rendering)을 지원한다.
* 빠른 페이지 로딩을 위한 자동 코드 스플릿팅
* 최적화된 프리페칭을 이용한 클라이언트 사이드 라우팅
* Built-in CSS과 Sass 지원, 그리고 다른 CSS-in-JS 라이브러리 지원
* 개발 환경에서의 빠른 리프레시 지원
* 서버리스 환경에서 API를 구축하기 위한 API라우트
* 넓은 확장성


이처럼 많은 기능을 제공하고 있는 Next.js는 세계적으로 유명한 기업들을 포함하여 수많은 웹사이트와 웹 애플리케이션에서 사용되고 있다. Next.js 세팅하는 법은 검색해보면 많이 나와서 건너띄고 Next.js 사용하면서 특징적이라고 생각하는 부분만 따로 정리하겠다.

### pages와 Next/Link

Next.js를 사용하면서 다음과 같이 폴더 구조를 짰다.

```text

// 어플리케이션에 사용되는 정적 파일들
public
  └── asset
        ├── images
        └── icons

// 어플리케이션의 기능에 사용되는 소스들
src
  ├── component                    // 컴포넌트들 모음
  │      ├── common                // 공통적으로 사용되는 컴포넌트들
  │      └── 페이지별 컴포넌트          // 각 페이지에서 사용되는 컴포넌트
  ├── containers                   // 컨테이너(store값이 전달된 컴포넌트)들 모음
  │      ├── common                // 공통적으로 사용되는 컨테이너
  │      └── 페이지별 컨테이너          // 각 페이지에서 사용되는 컨테이너
  ├── store                        // redux store 관련 파일 모음
  │      ├── modules
  │      └── index.js
  ├── utils                        // util 함수 모듬
  │      ├── hooks
  │      └── 기타 util 함수들
  ├── theme                       // 테마 관련 파일 모음
  │      ├── globals.css
  │      └── Home.modules.css
  └── styles                      // 스타일 관련 파일 모음
         ├── globals.css
         └── Home.modules.css

// 페이지를 담당하는 컴포넌트(폴더구조로 url 결정)
pages
  │                         // (❗ Nextjs에서는 Routing 시스템이 파일/폴더 구조로 되어있다.)
  ├── index.tsx
  ├── _app.tsx              // 각 페이지별로 공통적으로 쓰는 부분에 대한 리펙토링을 해주는 곳
  │                         // (index.js파일과 같은 역할, 모든 페이지에서 쓰는 스타일, 레이아웃 을 넣어 주기)
  ├── _document.tsx         //  meta태그를 정의 및 전체 페이지의 구조를 만들어준다
  │                         // (index.html파일과 같은 역할, html body와 같은 기본 태그들의 속성지정하여 어플리케이션의 구조를 만들어 주는 파일)
  ├── api                   // axios를 이용하여 서버와의 통신 로직을 담당
  └── product
        └── [id].tsx
```


다른 부분은 기존과 비슷한데, pages라는 폴더가 새로 생겼다. 이 페이지 폴더가 쉽게 말해 Next.js의 라우팅 시스템을 구성한다.
pages의 index.tsx 파일은 'localhost:3000'에 접속했을 때 뜨는 화면이 되고, pages의 'sample-folder/sample-file.tsx'의 파일은 'localhost:3000/sample-folder/sample-file'에 접속해서 뜨는 화면이 된다.

페이지 라우팅은 `<Link href="/경로">`를 사용하면 된다.

```html
import Link from 'next/Link';

<Link href="/about">    // pages 내 about 파일로 라우팅 됨(파일명 = 경로)
  <a> 이동하기 </a>
</Link>
// (만약 <a>태그를 사용하지 않는다면 해당 컴포넌트가 onClick props를 전달받아서 실행할 수 있게 해야한다)
```

### Redux

기존에 react 프로젝트에서 Redux를 사용했기 때문에 Next.js에서도 계속해서 Redux를 사용하였다. 먼저 Redux를 사용하기 위해 필요한 라이브러리들을 설치해줘야한다.
이 때, Next에서 redux를 사용하기 위해서는 next-redux-wrapper가 필요하기 때문에 함께 설치해준다.

```javascript
npm install redux react-redux next-redux-wrapper
```

그리고 다음의 블로그 [https://kir93.tistory.com/entry/NextJS-Redux-Toolkit-next-redux-wrapper](https://kir93.tistory.com/entry/NextJS-Redux-Toolkit-next-redux-wrapper-%EC%84%A4%EC%A0%95%ED%95%98%EA%B8%B0)를 참고해서 추가 설정을 했다.

### dynamic

Next.js에서도 마찬가지로 서버사이드에서는 브라우저 객체를 사용할 수 없기 때문에 `window is not defined` 에러가 뜬다. 따라서 브라우저 객체를 사용하는 컴포넌트에서는 SSR을 꺼두어야 하는데, [SSR에서 window 객체가 없다고 할 때]() 글에 정리했듯이 다음과 같이 SSR을 꺼서 에러를 방지할 수 있다.

```html
import dynamic from 'next/dynamic'

const ComponentsWithNoSSR = dynamic<{props:type;}>( // typescript에서 props를 전달할때 interface를 정의해줍니다.
  () => import('./components/Component'), // Component로 사용할 항목을 import합니다.
  { ssr: false } // ssr옵션을 false로 설정해줍니다.
);

const App = () => {
  return(
    <div>
    	<Components/> // 해당 컴포넌트는 SSR로 불러올 수 있습니다.

    	<ComponentsWithNoSSR/> // 해당 컴포넌트는 ssr:false이기 때문에 서버사이드 렌더를 하지않습니다.
    </div>
  )
};
```

### next/Image

기존에 이미지는 `<img>` 태그를 사용하여 추가하였지만, Next.js에서 이렇게 사용하면 warning을 띄운다. 물론 warning을 무시하고 계속 사용해도 서비스가 잘 돌아가긴 하지만 next/Image가 Next.js에서 이미지를 표시하는 데 최적화되어 있다고 하여 next/Image로 다 수정하였다. next/Image는 기존 `<img>` 태그와 거의 비슷한데, width/hight를 명시해주거나, layout props에 'fill'이라고 명시해줘야 한다.

```html
import Image from 'next/Image'

<div width='80px' height='80px'>
  <Image
    src={sampleImage}
    onError={onHandledError}
    alt='sample image'
    layout='fill'
    objectFit="cover"
    unoptimized
    loader={loaderProp} />
</div>

<Image
  src={sampleImage}
  alt='sample image'
  width={64}
  height={64} />
```

### SEO

처음에 SSR을 쓰려고 했던 근본적인 이유 중 하나였던 SEO를 마지막으로 설정해주었다. SEO는 `next/head` 라이브러리를 사용하면 쉽게 설정할 수 있다. 다음에 `next/head`로 SEO를 설정한 코드를 정리하였다. 이 SEO 컴포넌트를 SEO를 설정하고 싶은 컴포넌트에 추가하면 SEO 설정이 끝난다!

```html
import Head from 'next/head'

export default function SEO({ title, siteTitle, description, creator, image, summary }) {
  return (
    <Head>
      <title>{`${title} | ${siteTitle}`}</title>
      <meta name="description" content={description} />
      <meta property="og:type" content="website" />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:site_name" content={siteTitle} />
      <meta property="og:type" content={"website"} />
      <meta property="og:image" content={image} />
      <meta property="twitter:card" content="summary" />
      <meta property="twitter:creator" content={creator} />
      <meta property="twitter:title" content={title} />
      <meta property="twitter:description" content={description} />
      <link rel="image_src" href={image} />
      <meta itemProp="image" content={image} />
      <meta property="fb:pages" content="Catty, my knowledge base" />
    </Head>
  )
}
```

&nbsp;

***

참고 내용 출처 :
[https://velog.io/@kirin/Next.js-%ED%8E%98%EC%9D%B4%EC%A7%80-%EA%B5%AC%EC%A1%B0](https://velog.io/@kirin/Next.js-%ED%8E%98%EC%9D%B4%EC%A7%80-%EA%B5%AC%EC%A1%B0)
[https://www.zerocho.com/category/JavaScript/post/573b321aa54b5e8427432946](https://www.zerocho.com/category/JavaScript/post/573b321aa54b5e8427432946)

