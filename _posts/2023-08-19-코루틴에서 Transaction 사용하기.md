---
title: 코루틴에서 Transaction 사용하기
authors: jongin_kim
date: 2023-08-19 01:48:00 +0900
categories: [데이터베이스]
tags: [kotlin ,spring, transaction]
---
# 코루틴에서 Transaction 사용하기

## Spring `@Transactional` 동작 원리
---
- `@Transactional` 어노테이션은 스프링 프레임워크에서 제공하는 기능으로, 데이터베이스 트랜잭션을 관리하기 위해 사용된다. 
- 이 어노테이션을 메서드에 적용하면, 해당 메서드가 실행될 때 스프링은 트랜잭션을 시작하고, 메서드가 예외 없이 정상적으로 실행되면 트랜잭션을 커밋(commit)하고, 예외가 발생하면 트랜잭션을 롤백(rollback)시킨다.
* `@Transactional` 어노테이션의 동작 원리는 프록시 패턴과 AOP (Aspect-Oriented Programming)를 기반으로 하고. 프록시는 대상 객체를 감싸서 대신 호출을 가로채고 필요한 부가 작업을 수행할 수 있는 중간 역할을 한다. 
* 스프링에서 `@Transactional` 어노테이션은 이러한 프록시 기술을 활용하여 동작한다

### @Transactional 어노테이션이 적용된 메서드가 호출되면, 스프링은 다음과 같은 과정을 거쳐서 트랜잭션을 수행한다.
1. 프록시 생성
   - `@Transactional` 어노테이션이 적용된 빈(Bean)의 메서드를 호출할 때, 스프링은 해당 빈의 프록시 객체를 생성한다. 이 프록시 객체는 실제 메서드 호출을 가로채서 트랜잭션 관리 작업을 수행한다.
2. 트랜잭션 시작
   - 프록시 객체는 메서드 호출 전에 트랜잭션을 시작하고, 데이터베이스 연결을 하고 데이터베이스 트랜잭션을 시작한다.
3. 메서드 실행
   - 프록시 객체는 실제 메서드를 호출하고, 이때 메서드 내부에서 데이터베이스 관련 작업이 수행된다.
4. 트랜잭션 관리
   - 만약 메서드가 예외 없이 정상적으로 실행되면, 프록시 객체는 트랜잭션을 커밋하고, 예외가 발생하면 트랜잭션을 롤백한다. 
5. 트랜잭션 종료
   - 트랜잭션 커밋 또는 롤백 후, 프록시 객체는 트랜잭션을 종료하고 데이터베이스 커넥션을 반환한다.

> 이렇게 프록시와 AOP를 통해 `@Transactional` 어노테이션이 적용된 메서드의 트랜잭션 관리가 이루어지게 된다.


## 트랜잭션 정보는 어디에 저장하고 있을까?
---
스프링 프레임워크는 `@Transactional` 어노테이션과 관련된 트랜잭션 정보를 스프링 컨텍스트 내부에서 관리한다. 이때 트랜잭션 정보는 주로 `스레드 로컬(Thread Local)`을 사용하여 현재 실행 중인 스레드에 저장된다.

일반적으로 `@Transactional` 어노테이션을 사용한 메서드가 호출되면, 스프링은 다음과 같은 작업을 수행한다.

1. 트랜잭션 정보 저장
   - 호출된 메서드의 `@Transactional` 어노테이션에서 정의한 트랜잭션 속성을 읽어와 스레드 로컬에 트랜잭션 정보를 저장한다 (트랜잭션의 격리 수준, 전파 동작, 읽기 전용 등)
2. 프록시 생성 및 호출
   - `@Transactional` 어노테이션이 적용된 메서드를 호출할 때, 스프링은 해당 빈의 프록시 객체를 생성하고 실제 메서드 호출을 프록시로 위임한다. 이 프록시 객체는 트랜잭션 관련 작업을 수행하면서 메서드 실행을 감싸게 된다.
3. 메서드 실행 및 트랜잭션 관리
   - 프록시 객체가 실제 메서드를 호출하면 메서드 내부에서 데이터베이스 작업 등이 수행되고, 이때 스레드 로컬에 저장된 트랜잭션 정보를 활용하여 트랜잭션을 수행한다.
4. 트랜잭션 종료
   - 메서드가 예외 없이 정상적으로 실행되면 트랜잭션을 커밋하고, 예외가 발생하면 트랜잭션을 롤백한다. 이때 스레드 로컬에서 저장된 트랜잭션 정보는 제거된다.

> 스프링의 트랜잭션 관리는 다양한 방식과 설정 옵션을 제공한다. 
> 이를 통해 개발자는 트랜잭션의 격리 수준, 전파 동작, 제한 시간 등을 조절할 수 있다.



## 코루틴 스코프 안에서는?
---
- 코루틴은 기본적으로 스레드 로컬(Thread Local)을 사용하지 않으며, 코루틴이 만든 경량 스레드마다 고유한 코루틴 컨텍스트를 가진다. 당연히 코루틴 컨텍스트에서는 트랜잭션 관련 정보에 접근할 수 없다.
- 그러므로 아래 코드는 TransactionSynchronizationManager.isCurrentTransactionReadOnly (read-only)가 false다.
```kotlin
@Transactional(readOnly = true)
suspend fun findOne(id: Long) {
    return runBlocking(Dispatchers.IO) {
        userRepository.findByIdOrNull(id)?.apply {
            print(TransactionSynchronizationManager.isCurrentTransactionReadOnly()) // false
        }
    }
}
```


### 코루틴에서 read-only 트랜잭션을 사용하는 방법? (읽기 작업)
- 코루틴 워커 스레드에서는 트랜잭션 정보를 알 수 없으니
- read-only 트랜잭션을 사용하고 싶다면 코루틴 스코프 안에서 별도의 트랜잭션을 사용해야한다.
- ex) `transactionTemplate`

```kotlin
private val transactionTemplate: TransactionTemplate,

return runBlocking(context) {
    transactionTemplate.execute {
        transactionTemplate.isReadOnly = true
        userRepository.findByIdOrNull(id).apply {
            print(TransactionSynchronizationManager.isCurrentTransactionReadOnly()) // true
        }
    }
}
```

### 코루틴에서 쓰기 작업?
- 보통 스프링은 트랜잭션을 수행하기 위해서 커넥션 풀에서 커넥션을 가져오고 이를 스레드 로컬에 저장하고 있는다. 획득한 커넥션을 활용해 디비 작업을 수행하고, 완료 or 예외 발생시 트랜잭션을 커밋 or 롤백 한다고 했다.
- 그렇다면 현재 요청 스레드에서 여러개의 코루틴 워커 스레드를 만들고 그 안에서 여러번의 쓰기 작업이 일어났는데 제 각각 다른 트랜잭션으로 관리하고 싶다면?
  - 각각의 코루틴 잡들이 하나의 커넥션에 묶여서 처리되면서 각각 잡 별로 트랜잭션을 따로 관리하고 싶어도 하나의 커넥션에 묶여 하나의 트랜잭션으로 처리될 수 있음
  - 롤백되면 안되는 것 들 까지 롤백 될 수 있음
  - 이런 경우도 별도의 트랜잭션을 정의해서 사용해야한다.
    - ex) transactionTemplate 활용
