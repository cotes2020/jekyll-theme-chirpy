---
title: Node.js & NPM 버전 최신화
date: 2025-01-25 01:00:00 +09:00
categories: [Fronte-End]
tags: [Node.js, npm]
---

# Node.js & NPM 버전 최신화

# 23.07 - 23.10
<details>
  <summary>서칭</summary>
    
  ```jsx
  ---
  강제로 캐시 삭제
  $ sudo npm cache clean -f
  npm WARN using --force I sure hope you know what you are doing.
  ---
  ```
  
  [NPM Audit으로 패키지 취약점 관리하기](https://medium.com/bagelcode/npm-audit으로-패키지-취약점-관리하기-e42e0a418960)
  
  [Node.js 12에서 16로 한번에 업데이트 적용후기 - 환경 세팅편](https://velog.io/@djunnni/Node.js-12에서-16로-한번에-업데이트-적용후기-환경-세팅편)
  
  [nodejs 와 node-sass 버젼 호환 : Node Sass does not yet support your current environment](https://juntcom.tistory.com/167)
  
  [node.js와 NPM버전 업그레이드로 발생한 에러 해결하기](https://kang-ji.tistory.com/entry/nodejs와-NPM버전-업그레이드로-발생한-에러-해결하기)
  
  [Node.js 업데이트 후 node-sass 오류](https://velog.io/@nine9/Node.js-업데이트-후-node-sass-오류)
  
  [Sass Loader Error: Invalid options object that does not match the API schema](https://stackoverflow.com/questions/58184549/sass-loader-error-invalid-options-object-that-does-not-match-the-api-schema)
  
  ![Untitled](/assets/img/2025-01-25/2025-01-25-Nodejs_version_1.png)
  
  ![Untitled](/assets/img/2025-01-25/2025-01-25-Nodejs_version_2.png)
  
  # Node 18
  
  ```js
  ---
  package.json
  
  NODE_OPTIONS=--openssl-legacy-provider 추가
  
  "scripts": {
      "serve": "npm run build:plugin_local && cross-env NODE_OPTIONS=--max_old_space_size=8192 NODE_OPTIONS=--openssl-legacy-provider BUILD_TYPE=pages vue-cli-service serve --mode localhost --open --hot",
  }
  ---
  ```
  
  ```yaml
  ---
  terminal
  
  npm i node-sass@8
  export NODE_OPTIONS=--openssl-legacy-provider
  ---
  ```
  
  ```js
  ---
  .npmrc
  
  node-options="--openssl-legacy-provider"
  ---
  ```
</details>

<br>

# **기획 의도**

**현재** fo-display-pc-ui와 fo-display-mo-ui의 노드버전 v10.16.3(2019.10 ver)이 너무 오래 되어 추후에 생길 성능 최신화 및 모듈 에러를 방지하기 위한 최신화 작업 테스트를 로컬에서 선행해보기 위함.

# **작업 우선순위**

1. LTS버전을 기준으로 순차적 업그레이드
2. 업그레이드로 인한 문제 발생시 문제 해결
3. 문제 해결이 어려울 시 minor버전 순차적 다운그레이드

# **작업 과정**

우선, 현재 업무하는 환경에 영향을 주지 않기 위해 따로 작업용 폴더에 프로젝트를 클론하여 환경을 구성했습니다.

그 다음 우선순위에 따라, 현재 노드버전인 10.16.3 이후 LTS버전(v10, 12,14, 16)을 nvm을 이용하여 순차적으로 올리며 `rm -rf node_modules` , `npm i`, `npm run serve`를 반복 수행했습니다.

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_3.png)

# **문제 해결**

1. **node & node-sass & 버전 호환 이슈**
2. **node-sass & sass-loader 버전 호환**
3. **sass-loader 버전 변경에 따른 옵션 변화**

### **node & node-sass 버전 호환 이슈**

기존의 환경설정에서 별다른 변화없이 node v14.21.3까지는 이상 없이 노드 버전을 업그레이드 할 수 있었으나, v16.20.1 버전부터는 node-sass 모듈과 node의 버전 호환 이슈가 있었습니다. 그래서 node-sass를 6버전으로 설치하려 하였으나, vue와 webpack 버전 이슈로 인해 `npm uninstall webpack`, `npm i --save-dev webpack@4`하여 webpack

v4.46.0 설치 진행 후 `npm uninstall node-sass`,  `npm i --save-dev node-sass@6` 하여 6버전을 설치하여 다시 빌드를 진행했습니다.

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_4.png)

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_5.png)

### **node-sass & sass-loader 버전 호환**

node-sass 설치를 정상적으로 마치고 빌드를 실행한 결과, 
Node Sass version 6.0.1 is incompatible with ^4.0.0 || ^5.0.0. 와 같은 에러 문구를 만났습니다.

이는 node-sass 와 sass-loader 버전 호환 문제로 node-sass 버전을 낮추는 해결책을 많이 사용하지만, 이는 node 및 관련 의존성 모듈의 버전 업그레이드의 취지에 맞지

않기 때문에 sass-loader의 버전을 업그레이드하여 문제를 해결하는 방향으로 결정했습니다. 그래서 호환되는 버전을 찾다 v10.4.1 버전이 호환되는 것을 찾았습니다.

### **sass-loader 버전 변경에 따른 옵션 변화**

sass-loader 또한 버전이 업그레이드됨에 따라 loaderOptions 에 변동 사항이 있었습니다.

v8의 경우

data 옵션이 prependData 로 바뀌었고

v9이상 버전의 경우

prependData 에서 additionalData로 바뀌었습니다.

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_6.png)

# 최종 결과물

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_7.png)

![image.png](/assets/img/2025-01-25/2025-01-25-Nodejs_version_8.png)