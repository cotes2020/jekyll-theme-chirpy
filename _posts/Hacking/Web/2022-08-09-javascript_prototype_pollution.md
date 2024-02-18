---
title: Javascript prototype pollution
date: 2022-08-09 01:19  +0900
categories: [Hacking, Web]
tags: [javascript prototype pollution, environment pollution]
---

## 자바스크립트 프로토타입?
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://www.howdy-mj.me/javascript/prototype-and-proto/" target="_blank">howdy-mj.me/javascript/prototype-and-proto/</a>
: <a href="http://www.tcpschool.com/javascript/js_object_prototype" target="_blank">tcpschool.com/javascript/js_object_prototype</a>

<br>

자바스크립트는 프로토타입 기반(prototype-based)의 객체 지향 언어이다.

프로토타입이란, 다른 객체에 공유 property를 제공하는 객체라고 한다.

자세하게 설명하면 다음과 같다.

<br>

자바스크립트의 모든 객체는 **프로토타입(prototype)**이라는 객체를 가지고 있습니다.

모든 객체는 그들의 **프로토타입**으로부터 **프로퍼티와 메소드를 상속**받습니다.

이처럼 자바스크립트의 **모든 객체는 최소한 하나 이상의 다른 객체로부터 상속**을 받으며, 이때 상속되는 정보를 제공하는 객체를 프로토타입(prototype)이라고 합니다.

<br>

객체에는 ```prototype``` property가 있다. ```ex) Object.prototype, Date.prototype```

prototype 프로퍼티를 통해 현재 존재하고 있는 프로토타입에 새로운 프로퍼티나 메소드를 손쉽게 추가할 수 있다.

흠.. 아직 모르겠다..

<br>

```javascript
const a = {'a':'b'};
a.a // b
```

<br>

위의 코드를 보면 a 객체의 프로퍼티 a를 불러온다.
 
이 때, 내부적으로 ```[[Get]]```을 호출하는데, 만약 a 객체에 없는 프로퍼티를 호출한다면 ```[[Prototype]]```을 호출한다고 한다.

```[[Prototype]]```을 따라가 프로퍼티를 검색을 하고, 최종적으로 최상위 프로토타입 객체인 ```Object.prototype```까지 검색을 한다.

<br>

prototype 프로퍼티랑 ```__proto__```가 있던데 정리 해보면 다음과 같다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## prototype과 ```__proto__``` 차이점
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/183582649-e347d9ee-c0d5-4a37-b31a-b784b560d516.png)

<br>

위의 사진을 보면 생성자 함수, 객체, 프로토타입은 연결되어 있다고 한다.

예를 들어, Person 함수를 정의를 해줬다고 한다면, Person 함수의 객체가 생성되는데 Person 함수의 객체의 Prototype 객체도 같이 생성된다는 것이다.

<br>


<br>

Person 객체의 prototype 프로퍼티는 Person의 prototype 객체를 가리키고, Person의 prototype 객체는 constructor 라는 프로퍼티를 생성해 Person 객체를 가리킨다.

constructor는  Prototype Object와 같이 생성되었던 함수를 가리키고, ```__proto__```는 Prototype Link(```[[Prototype]]```)을 가리킨다.

<br>

이제 Person 객체의 객체(인스턴스)를 생성했다고 가정하면 인스턴스에 ```__proto__``` 라는 프로퍼티가 생긴다.

<br>

![image](https://user-images.githubusercontent.com/52172169/183588092-f6c0706d-cab4-4099-baa2-b72de6a0f812.png)

<br>

이 객체(인스턴스)의 ```__proto__``` 프로퍼티는 Person 객체의 prototype 객체를 가리킨다.

즉, ```__proto__```는 객체가 생성될 때 조상이었던 함수의 Prototype Object를 가리킨다.

그러니까, **```__proto__``` 프로퍼티도 해당 객체의 prototype 객체를 가리키고, 객체의 prototype 프로퍼티도 해당 객체의 prototype 객체를 가리키는 것**이다.

<br>

둘의 차이를 정리하면 **prototype 프로퍼티는 함수 객체만** 가지고 있는데, **```__proto__``` 프로퍼티는 모든 객체**가 가지고 있다.

또한 단순히 ```__proto__```라고 하는 것은 ```Object.prototype```을 가리키는 것이다.

왜냐하면, 위에서 설명했듯이 **자바스크립트의 모든 객체는 ```Object.prototype``` 객체를 프로토타입으로 상속**받는다.

자바스크립트에 내장된 모든 생성자나 사용자 정의 생성자는 바로 이 객체를 프로토타입으로 가진다.

Object.prototype 객체는 어떠한 프로토타입도 가지지 않으며, 아무런 프로퍼티도 상속받지 않는다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## ```__proto__``` pollution
<hr style="border-top: 1px solid;"><br>

이제 ```__proto__```에 대해 알았다.

그럼 이 ```__proto__```를 오염시키는 것은 어떤 것인가?

```__proto__```는 ```Object.prototype```과 같다고 했고, 이를 오염시키는건 모든 객체에 어떠한 값을 추가해주거나 변경을 시켜준다는 의미와 같다.

<br>

예를 들면 아래와 같은 코드로 확인할 수 있다.

<br>

```javascript
const a = {};
const b = {'key':'val'};

a.key == undefined
b.key == 'val'

b.__proto__.pollution = 'polluted'
a.pollution == 'polluted'
```

<br>

a에는 아무런 프로퍼티가 없었고, b에는 key 라는 프로퍼티를 넣어줬다.

두 객체는 아무런 상관이 없는데, a에서 접근을 하면 polluted가 출력된다.

<br>

그럼 ```Object.prototype == __proto__```에서도 들어가 있는가? 에 대한 답은 들어가 있다.

즉, 애초에 ```Object.prototype == __proto__```에 어떤 프로퍼티를 생성해주거나, 기존의 프로퍼티 값을 변경해줄 수 있다면, 모든 객체에서 이를 사용할 수 있게 된다.

<br>

이러한 취약점이 발생하는 부분이 여럿 있는데, 그 중 3가지를 보면 객체의 속성 추가, 객체의 병합, 객체의 복사가 있다.

이러한 과정에서 prototype pollution이 일어날 수 있다. 자세한 내용들은 출처에서 확인

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Environment Pollution RCE
<hr style="border-top: 1px solid;"><br>

prototype pollution 취약점을 통해 환경변수에 등록된 값도 오염시킬 수 있다.

node.js를 보면 process.env가 있는데, 이것은 환경변수 값이다.

<br>

만약 자식 프로세스를 생성하는 ```child-process``` 모듈을 사용한다고 가정한다.

요약하면 child-process를 불러오는 과정에서 환경변수를 복사를 해오는데, 환경변수에는 ```NODE_OPTIONS```라는 변수가 있다.

이 변수에는 인자가 있는데, ```--require [module]``` 인자를 설정을 해주면, NodeJS 프로세스 실행 시 module을 먼저 불러와서 실행시킨 다음 프로세스를 실행한다.

따라서 만약 환경변수를 오염시킬 수 있다면, child-process를 불러오는 동시에 메소드인 exec나 spawn으로 RCE 또는 리버스쉘을 할 수 있게 된다.

**자세한건 아래 출처에서 보기, 정리가 매우 잘 되어있음**

<br>

child-process에 관한 건 아래 링크에서 확인
: <a href="https://backback.tistory.com/362" target="_blank">backback.tistory.com/362</a>
: <a href="https://nodejs.org/api/child_process.html" target="_blank">nodejs.org/api/child_process.html</a>

<br>

추가로 AST Injection이라고 있는데, 같이 봐야 할 듯 하다. 나중에 정리해야 겠다. (원본, 번역본)
: <a href="https://blog.p6.is/AST-Injection/" target="_blank">blog.p6.is/AST-Injection/</a>
: <a href="https://kiristhome.notion.site/AST-injection-5c8af1c530854fcb8d73e0c73525b711" target="_blank">kiristhome.notion.site/AST-injection-5c8af1c530854fcb8d73e0c73525b711</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

proto, prototype
: <a href="https://www.howdy-mj.me/javascript/prototype-and-proto/" target="_blank">howdy-mj.me/javascript/prototype-and-proto/</a>
: <a href="https://velog.io/@h0ngwon/Javascript-proto-vs-prototype-차이" target="_blank">velog.io/@h0ngwon/Javascript-proto-vs-prototype-차이</a>
: <a href="http://www.tcpschool.com/javascript/js_object_prototype" target="_blank">tcpschool.com/javascript/js_object_prototype</a>
: <a href="https://velog.io/@adam2/자바스크립트-Prototype-완벽-정리" target="_blank">velog.io/@adam2/자바스크립트-Prototype-완벽-정리</a>
: <a href="https://medium.com/@bluesh55/javascript-prototype-이해하기-f8e67c286b67" target="_blank">medium.com/@bluesh55/javascript-prototype-이해하기-f8e67c286b67</a>

<br>

prototype pollution
: <a href="https://jjy-security.tistory.com/30" target="_blank">jjy-security.tistory.com/30</a>
: <a href="https://www.hahwul.com/cullinan/prototype-pollution/#prototype-pollution" target="_blank">hahwul.com/cullinan/prototype-pollution/#prototype-pollution</a>
: <a href="" target="_blank"></a>
: <a href="" target="_blank"></a>

<br>

prototype pollution rce
: <a href="https://blog.p6.is/prototype-pollution-to-rce/" target="_blank">blog.p6.is/prototype-pollution-to-rce/</a>
: <a href="https://hackerone.com/reports/878181" target="_blank">hackerone.com/reports/878181</a>
: <a href="https://slides.com/securitymb/prototype-pollution-in-kibana" target="_blank">slides.com/securitymb/prototype-pollution-in-kibana</a>
: <a href="https://research.securitum.com/prototype-pollution-rce-kibana-cve-2019-7609/" target="_blank">research.securitum.com/prototype-pollution-rce-kibana-cve-2019-7609/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
