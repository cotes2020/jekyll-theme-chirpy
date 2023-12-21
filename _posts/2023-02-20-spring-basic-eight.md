---
title: Spring Basic[의존관계 자동 주입]
date: 2023-02-20 23:12:00 +0800
categories: [Spring-Basic, 의존관계 자동 주입]
tags: [Spring]
---

# 의존관계 자동 주입
의존관계 자동 주입은 객체 간의 관계를 코드에서 직접 설정하는 번거로움을 덜어주고,       
유지 보수성을 향상 시키며 생산성을 높이기 위한 매커니즘입니다.      
이를 통해 코드의 모듈성이 강화되고, 객체의 독립성이 증가하여 재사용성이 향상된다는 이점이 있습니다.         
또한 스프링 컨테이너 객체간의 의존성을 관리하므로 개발자는 주로 비즈니스 로직에만 집중할 수 있어        
코드의 복잡성을 간소화하며 시스템의 유연성과 확장성을 제공하는 장점이 있습니다.

## 의존관계 주입 방법
주입 방법엔는 생성자 주입, 수정자 주입, 필드주입, 일반 메서드 주입 총 4가지 방법이 있습니다.        
다음각 각 방법에 대한 설명입니다.

### 1. 생성자 주입
이름 그대로 생성자를 통해서 의존관계를 주입 받는 방법입니다.        
```java
@Component
public class BeanServiceImpl implements BeanService {
 private final BeanRepository beanRepository;
 @Autowired
 public BeanServiceImpl(BeanRepository beanRepository) {
 this.beanRepository = beanRepository;
 }
}
```
생성자 주입은 다음과 같은 특징이 존재 합니다.
- 생성자 호출시 1번만 호출 됩니다.
- 불변, 필수 의존관계에 사용됩니다.

### 2. 수정자 주입
setter라 불리는 필드의 값을 변경하는 수정자 메서드를 통해서 의존 관계를 주입하는 방법입니다.
```java
@Component
public class BeanServiceImpl implements BeanService {
 private BeanRepository beanRepository;
 @Autowired
 public void setBeanServiceImpl(BeanRepository beanRepository) {
 this.beanRepository = beanRepository;
 }
}
```
수정자 주입은 다음과 같은 특징이 존재 합니다.
- 선택,변경 가능성이 있는 의존관계에 사용합니다.
- 자바 빈 프로퍼티 규약의 수정자 메서드 방식을 사용하는 방법입니다.
#### Java Bean 프로퍼티 규약
- 필드의 값을 직접 변경하지 않고, setXxx, getXxx 메서드를 통해서 값을 읽거나 수정하는 규칙

### 3. 필드 주입
이름 그대로 필드에 바로 주입하는 방법이다.      
하지만 외부에서 변경이 불가능해, 테스트 하기 힘들다는 치명적인 단점이 존재한다.     
그 외에도 의존성 숨김, 순환 의존성 문제등 다른 문제점도 존재하여 사용하지 않는걸 추천한다.      

### 4. 일반 메서드 주입
일반 메서드를 통해서 주입을 받는 방법이다.
```java
@Component
public class BeanServiceImpl implements BeanService {
 private BeanRepository beanRepository;
 @Autowired
 public void init(BeanRepository beanRepository) {
 this.beanRepository = beanRepository;
 }
}
```
다음과 같은 특징이 존재한다.
- 한번에 여러 필드를 주입 받을 수 있다.
- 일반적으로 잘 사용하지 않는다.

## 결론
최근에는 스프링을 비롯한 DI(Dependency Injection) 프레임워크에서 생성자 주입을 강력히 권장하고 있습니다. 이는 수정자 주입이나 필드 주입 대신 생성자 주입을 선호하는 이유가 있습니다.        

생성자 주입은 의존 관계가 한 번 설정되면 애플리케이션 종료까지 변경되지 않는 경우가 대부분입니다.       
이로써 의존 관계의 불변성을 보장할 수 있으며, 객체 생성 시 한 번만 호출되므로 이후 호출되는 일이 없습니다. 또한, 생성자 주입을 사용하면 해당 클래스의 의존 관계를 외부에서 주입받기 때문에 테스트 용이성이 높아집니다.        

또한, 생성자 주입은 불변성을 지킬 수 있어 객체를 안정적으로 사용할 수 있습니다.         
final 키워드와 함께 사용되어 불변한 객체를 생성할 수 있기 때문에, 코드의 안정성과 유지보수성을 높일 수 있습니다. 따라서 항상 생성자 주입을 선택하는 것이 좋습니다.      