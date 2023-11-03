---
title: Section 18 자바 Array, ArrayList
date: 2023-11-02
categories: [blog]
tags: [java]
---

<style>
  /* 기본 스타일 */
  .center-table {
    margin: 0 auto; /* 가운데 정렬을 위한 스타일 */
    text-align: center; 
    width : 50%;
    
  }
  .center-table td {
    width : 25%;
  }
    .center-table td:last-child {
    width : 30%;
  }
  div.content .table-wrapper>table {
    min-width: 90%;
    }
  .table-wrapper{
            width:100%;
            margin: 0;
            padding:0;
    }
  .table-wrap {
    display: flex;
    width: 100%;
    justify-content: space-around;
    flex-wrap: nowrap;
    flex-direction: row;           
  }

    .title-cell{
        background-color : orange;
    }

  @media (max-width: 768px) {
    .center-table {
      width: 100%; 
    }
    .table-wrap {
      flex-direction : column;
      justify-content: center; 
    }
  }
</style>

```sh
jshell>     int mark1 = 100;
   ...>     int mark2 = 75;
   ...>     int mark3 = 60;
   ...>     int sum = mark1 + mark2 + mark3;
mark1 ==> 100
mark2 ==> 75
mark3 ==> 60
sum ==> 235

jshell>     int[] marks = { 75, 60, 56 };
marks ==> int[3] { 75, 60, 56 }

jshell> for(int mark:marks){
   ...>          sum += mark;
   ...>        }

jshell> sum
sum ==> 191
```

## Array 선언해보기

```sh
jshell> int[] marks2 = new int[5];
marks2 ==> int[5] { 0, 0, 0, 0, 0 } // 배열의 값을 지정하지 않으면 0 으로 초기화 됩니다.
```


```java
jshell> int[] marks2 = new int[5];
marks2[0] = 1;
marks2[1] = 2;
marks2[2] = 3;
marks2[3] = 4;
marks2[4] = 5;
```

```java
int[] marks2 = new int[5];
marks2[0] = 1;

for (int i = 1; i < 5; i++) {
    marks2[i] = i + 1;
}
```

```java
int[] marks2 = {1, 2, 3, 4, 5};
```

```sh
jshell> marks2.length
$2 ==> 5
```

String.length() 메서드
Array.length 는 속성(property)

연습문제 직접 풀어봄

```java
int[] marksPractice = new int[8];
for(int mark:marksPractice){
  System.out.println(mark);
}

int[] marksPractice = new int[8];
int count = 0;
while(count < marksPractice.length){
  System.out.println(marksPractice[count]);
  count++;
}
```

```sh
double[] doubleArray = new double[5]; // 기본값은 0.0 으로 채워진다.
boolean[] booleanArray = new boolean[5]; // 기본값은 false 로 채워진다.

class Person{};

Person[] peopleArray = new Person[5]; // 객체의 배열을 만들 경우 초기값은 null 입니다.


int[5] marks; // Error : 배열을 선언할때 구문의 왼쪽(타입 선언 부분)에는 요소의 갯수가 올 수 없다.

int[] marks = new int[]; // Error : 배열의 규모를 알 수 없기 때문에 안된다.


jshell> int[] marksPractice = new int[8];
marksPractice ==> int[8] { 0, 0, 0, 0, 0, 0, 0, 0 }

jshell> marksPractice
marksPractice ==> int[8] { 0, 0, 0, 0, 0, 0, 0, 0 }

jshell> System.out.println(marksPractice) // syso 로 배열을 불러보면 메모리 주소가 나온다.
[I@5387f9e0 

jshell> System.out.println(Arrays.toString(marksPractice)
)
[0, 0, 0, 0, 0, 0, 0, 0] 
// 표현식(representaiton)을 print 하고 싶으면 Arrays 의 static 메서드를 쓰면 된다.

```

아래 처럼 배열 반복문을 쓰는 걸 Enhanced for loop 라고 부른다.

```java
int[] marks = {100, 99, 95, 96, 100};

for(int mark:marks){
    System.out.println(mark);
}
```


`Arrays.fill`

```sh
jshell> int[] marks = new int[5];
   ...> 
   ...> Arrays.fill(marks,5);
marks ==> int[5] { 0, 0, 0, 0, 0 }
```

`Arrays.equals`

```sh
jshell> int[] marks1 = {1,2,3,4,5};
   ...> int[] marks2 = {1,2,3,4,5};
   ...> Arrays.equals(marks1,marks2);
marks1 ==> int[5] { 1, 2, 3, 4, 5 }
marks2 ==> int[5] { 1, 2, 3, 4, 5 }
$29 ==> true
```

`Arrays.sort()` // Arrays 에 역순 정렬은 없다.

```sh
jshell> int[] marks = {4,3,5};
marks ==> int[3] { 4, 3, 5 }

jshell> Arrays.sort(marks);

jshell> marks
marks ==> int[3] { 3, 4, 5 }
```

`Refactor > Inlining `



```sh
jshell> ArrayList arrayList = new ArrayList();
arrayList ==> []

jshell> arrayList.add("apple");
|  Warning:
|  unchecked call to add(E) as a member of the raw type java.util.ArrayList
|  arrayList.add("apple");
|  ^--------------------^
$2 ==> true

jshell> arrayList.add("bat");
|  Warning:
|  unchecked call to add(E) as a member of the raw type java.util.ArrayList
|  arrayList.add("bat");
|  ^------------------^
$3 ==> true

jshell> arrayList.add("cat");
|  Warning:
|  unchecked call to add(E) as a member of the raw type java.util.ArrayList
|  arrayList.add("cat");
|  ^------------------^
$4 ==> true

jshell> arrayList;
arrayList ==> [apple, bat, cat]

jshell> arrayList.remove("cat");
$6 ==> true

jshell> arrayList;
arrayList ==> [apple, bat]

jshell> arrayList.add(1); // 우리가 ArrayList 에 타입을 안정해줘서 문자열, 숫자 다 넣을 수 있다. 하지만 리스트에는 모두 같은 타입의 값을 넣는게 권장된다.
|  Warning:
|  unchecked call to add(E) as a member of the raw type java.util.ArrayList
|  arrayList.add(1);
|  ^--------------^
$8 ==> true

// 그래서 나온게 아래의 문법이다. 제네릭이라고 부른다.
jshell> ArrayList<String> stringArrayList = new ArrayList<String>();
stringArrayList ==> []

jshell> stringArrayList.add("apple");
$11 ==> true

jshell> stringArrayList.add("bat");
$12 ==> true

jshell> stringArrayList.add("cat");
$13 ==> true

jshell> stringArrayList.remove("cat");
$14 ==> true

jshell> stringArrayList.remove(0); // remove의 인자로 인덱스 가능
$15 ==> "apple"

```