---
title: Spring Basic[Lombok 사용하기]
date: 2023-02-21 21:12:00 +0800
categories: [Spring-Basic, Lombok]
tags: [Spring]
---

# Lombok

## Lombok이란?
Lombok[롬복]은 자바 프로젝트에서 반복적인 코드를 줄이기 위한 라이브러리 입니다.         
주로 Getter, Setter, toString(), equals()등 메소드를 자동으로 생성해주는 어노테이션을 제공하여,         
개발자가 더 간결하고 가독성이 높은 코드를 작성할 수 있도록 돕습니다.

## 적용 방법
1. [Spring.starter.io](https://start.spring.io/)로 이동하여 Spring Project를 생성 합니다.
- 생성 방법은 [Poster](https://ljw22222.github.io/posts/spring-basic-two/#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%83%9D%EC%84%B1)의 포스터를 참고하시면 됩니다.<br/>
 ![Spring Project Lombok png](/assets/img/spring/springprojectlombok.png){: width="700" height="600" }<br/>
- 설정을 마치신 다음 Dependencies에서 Lombok을 검색하여 추가해줍니다.       
- Lombok을 추가하셨다면 GENERATE버튼을 눌러 zip파일을 생성 합니다.      

2. zip파일을 압축해제 하여 Tools(InteliJ 기준)을 사용해 파일을 열어줍니다.

 ![Spring Project Lombok Gradle png](/assets/img/spring/springprojectgradlesetting.png){: width="500" height="400" }
3. 왼쪽의 파일 구조에서 build.gradle파일에 들어가시면 위와 같이 라이브러리가 추가되어 있는 모습이 확인 가능합니다.

4. build.gradle파일을 마우스 오른쪽 누르고 디스크에 로드 버튼을 눌러 줍니다.

 ![Spring Project Lombok Gradle Reload png](/assets/img/spring/springprojecgrablereload.png){: width="500" height="400" }
5. 그런 다음 위의 사진처럼 오른쪽에 있는 Gradle을 누르고, reload버튼을 눌러주면,        
build.gradle파일이 프로젝트에 적용이 되면서, 설정되어있는 라이브러리파일을 다운 받기 시작합니다.

 ![Spring Project Lombok Annotation enable png](/assets/img/spring/springlombokannotation.png){: width="500" height="400" }
6. 라이브러리파일 다운이 끝났으면, 왼쪽위의 점3개 버튼을 클릭하여 아래의 절차를 따라가면 됩니다.
- 파일 -> 설정 -> 빌드,실행,배포 -> 컴파일러 -> 어노테이션 프로세서
- 파일 -> 설정 이 부분을 한번에 들어갈수 있는 단축키는
    - Ctrl + Alt + S [ Windows 기준 ]
그러면 위의 사진과 같은 창이 뜨는데, 여기서 어노테이션 처리를 활성화 시켜주면 됩니다.

그럼 이제 Lombok이 활성화가 되고, 사용하실 수 있는 환경이 설정되었습니다.

Lombok의 적용 전과 후의 예제를 간단하게 살펴보겠습니다.

## 적용 전
```java
public class PersonWithoutLombok {
    private String name;
    private int age;

    public PersonWithoutLombok(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "PersonWithoutLombok{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
```



## 적용 후 
Lombok의 어노테이션중 하나인 @Data라는 어노테이션을 적용한 후의 모습입니다.
```java
import lombok.Data;

@Data
public class PersonWithLombok {
    private String name;
    private int age;
}
```
- Data라는 어노테이션에는 아래와 같은 기능이 포함되어 있습니다.
    - @toString()
    - @equals
    - @getter
    - @setter
    등 여러가지가 포함 되어있습니다.

## 결론
적용 전과 적용 후의 코드를 비교하면, Lombok을 적용한 후에는 불필요한 코드를 효과적으로 제거할 수 있어 코드가 간결해지고 가독성이 향상됩니다. @Data 어노테이션을 사용함으로써 게터, 세터, toString 등의 메서드를 자동으로 생성할 수 있습니다. 이를 통해 개발자는 핵심 로직에 더 집중할 수 있고, 불필요한 반복 코드 작성을 피할 수 있습니다. Lombok은 Java 코드 작성을 간소화하여 생산성을 높여주는 유용한 도구 중 하나입니다.