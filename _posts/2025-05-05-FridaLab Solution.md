---
title: "FridaLab Solution"
date: 2025-05-05 18:10:00 +0900
author: aestera
categories: [Mobile]
tags: [Mobile, Android, FridaLab]
description: My FridaLab Solution 01 ~ 08
math: true
image: /assets/img/preview/FridaLab.png
---

# **Intro**

안드로이드 공부해보고 싶어서 안드로이드 후킹 툴인 `Frida`를 좀 익혀보기 위해 **FriaLab** 워게임(?)을 풀어보았다.


> - Android 에뮬레이터 버전: **Android 13.0 ("Tiramisu") \| arm64** <br>
> - Frida Version: **16.7.14** <br>
> - **Python 3.11.12**
{: .prompt-info }

> Frida Server 16.7.14 버전은 python 3.12 버전 이상에서 오류가 발생할 때가 있다. 파이썬 버전을 3.11 이하로<br>
다운그레이드해서 사용하는 것을 추천한다.
{: .prompt-warning }


# **Challenge 01**
---

```java
package uk.rossmarks.fridalab;

public class challenge_01 {
    static int chall01;

    public static int getChall01Int() {
        return chall01;
    }
}
```

`Mainactivity`의 `onCreate`메소드를 보면, `getChall01Int`의 리턴값을 **1**로 만들어야 한다.

```js
setImmediate(function() {
    Java.perform(function(){
        var ClassChall01 = Java.use("uk.rossmarks.fridalab.challenge_01");
        ClassChall01.chall01.value = 1;
        console.log("[+] Challenge_01 Solved");
    })
})
```
`challenge_01`의 경우에는 인스턴스화되지 않은 클래스이기 때문에 `Java.use()`로 클래스를 가져와 내부 변수 값을 1로 바꿔주었다.

# **Challenge 02**
---

```java
private void chall02() {
    this.completeArr[1] = 1;
}
```
`MainActivity`클래스 내부에 존재하는 method인 `chall02()`를 실행만 하면 된다.

```js
setImmediate(function() {
    Java.choose("uk.rossmarks.fridalab.MainActivity", {
        onMatch : function (instance) {
            instance.chall02();
        },
        onComplete : function () {
            console.log("[+] Challenge_02 Solved");
        }
    })
})
```
`MainActivity`는 안드로이드가 실행되면 자동으로 인스턴스화 된다. 그래서 `Java.choose()`로 `MainActivity`의 인스턴스를 가져와 `chall02()`를 실행시키면 된다.


# **Challenge 03**
---

```java
public boolean chall03() {
    return false;
}
````
`MainActivity`클래스 내부에 존재하는 method인 `chall03()`의 리턴값을 true로 바꿔주면 된다.

```js
setImmediate(function() {
    Java.choose("uk.rossmarks.fridalab.MainActivity", {
        onMatch : function (instance) {
            instance.chall03.implementation = function() {
                return true;
            }
        },
        onComplete : function () {
            console.log("[+] Challenge_03 Solved");
        }
    })
})
```
인스턴스를 가져온 후 `chall03`메소드를 `implementation`로 오버라이드 하여 무조건 `true`를 리턴하도록 했다.


# **Challenge 04**
---

```java
public void chall04(String str) {
    if (str.equals("frida")) {
        this.completeArr[3] = 1;
    }
}
```
`MainActivity`클래스 내부에 존재하는 method인 `chall04()`를 호출할 때 **frida**라는 문자열을 넘겨주면 된다.

```js
setImmediate(function() {
    Java.choose("uk.rossmarks.fridalab.MainActivity", {
        onMatch : function (instnace) {
            instnace.chall04("frida");
        },
        onComplete : function () {
            console.log("[+] Challenge_04 Solved");
        }
    })
})
```
인스턴스를 가져와 `chall04`를 가져온 후 **frida**를 전달하면 된다.


# **Challenge 05**
---

```java
public void chall05(String str) {
    if (str.equals("frida")) {
        this.completeArr[4] = 1;
    } else {
        this.completeArr[4] = 0;
    }
}
```
4번문제와 비슷하지만 약간 다르다. `onCreate`의 `onClick`메소드를 보면 Confirm 버튼을 클릭할 때 마다 **notfrida!**를 전달한다. 따라서 매 요청 때 마다 **firda**를 전달하도록 해야한다.

```js
setImmediate(function() {
    Java.perform(function() {
        let MainClass = Java.use('uk.rossmarks.fridalab.MainActivity');
        MainClass.chall05.implementation = function() {
            this.chall05("frida");
            console.log("[+] Challenge_05 Solved");
        }
    })
})
```
`Java.choose`로 MainActivity의 인스턴스를 가져오는 방식은 한번만 실행되기 때문에 프로세스 실행 중에 버튼을 한번 더 클릭하면 **notfrida!**가 전달된다.<br>
따라서 `Java.use`로 MainActivity 클래스를 가져와 `chall05` 자체를 오버라이드 하여, Frida 프로세스가 돌아가고 있는 한 `chall05("frida")`가 실행되도록 하면 문제를 해결할 수 있다.

# **Challenge 06**
---

```java
package uk.rossmarks.fridalab;

public class challenge_06 {
    static int chall06;
    static long timeStart;

    public static void startTime() {
        timeStart = System.currentTimeMillis();
    }

    public static boolean confirmChall06(int i) {
        return i == chall06 && System.currentTimeMillis() > timeStart + 10000;
    }

    public static void addChall06(int i) {
        chall06 += i;
        if (chall06 > 9000) {
            chall06 = i;
        }
    }
}
```
`confirmChall06`의 리턴값을 1로 만들어야 한다.

```java
// onCreate
challenge_06.startTime();
challenge_06.addChall06(new Random().nextInt(50) + 1);
new Timer().scheduleAtFixedRate(new TimerTask() {
    @Override
    public void run() {
        int nextInt = new Random().nextInt(50) + 1;
        challenge_06.addChall06(nextInt);
        Integer.toString(nextInt);
    }
}, 0L, 1000L);
```

`onCreate`에서 위 코드가 실행된다. 이와 함께 문제 코드를 분석해보자.
1. startTime를 호출하여 현재 시간을 `challenge_06`클래스의 `timeStart`변수에 저장
2. 10초마다 0 ~ 50 사이의 정수를 `challenge_06`클래스의 `chall06`변수에 더함

<br>
문제를 해결하기 위해서는 아래 두 개의 조건을 맞춰야 한다.
- 10초 후 `chall06`의 값을 맞춘다.
- `confirmChall06`을 실행할 때의 시간이 `timeStart`변수에 저장된 시간보다 10초 후여야 한다.

<br>
그렇다면 `setTimeout(function(){}, 10000)`로 10초를 기다린 후 `chall06`변수 값을 후킹해 가져오는 방법과
`timeStart`변수를 후킹해 조작하는 방법 이렇게 두 가지가 있을 수 있다. 두번째 방법으로 풀어보겠다.

```js
setImmediate(function(){
    Java.perform(function(){
        Java.choose("uk.rossmarks.fridalab.MainActivity", {
            onMatch : function(instnace){
                let ClassChall06 = Java.use("uk.rossmarks.fridalab.challenge_06")
                ClassChall06.timeStart.value = -10000
                ClassChall06.addChall06.implementation = function(i) {
                    this.addChall06(i);
                }
                instnace.chall06(Number(ClassChall06.chall06.value));
            },
            onComplete : function(){
                console.log("[+] Challenge_06 Solved");
            }
        })
    })
});
```


# **Challenge 07**
---

```java
package uk.rossmarks.fridalab;

public class challenge_07 {
    static String chall07;

    public static void setChall07() {
        chall07 = BuildConfig.FLAVOR + (((int) (Math.random() * 9000.0d)) + 1000);
    }

    public static boolean check07Pin(String str) {
        return str.equals(chall07);
    }
}
```

`chall07`의 값을 맞춰야 풀리는 문제다. `BuildConfig.FLAVOR`는 빈 문자열이기에 신경 쓸 필요 없고, `(((int) (Math.random() * 9000.0d)) + 1000);`의 값은 1000 ~ 9999 사이의 값이니 브루트포싱하여 해결하자.

```js
setImmediate(function() {
    Java.perform(function() {
        Java.choose("uk.rossmarks.fridalab.MainActivity", {
            onMatch: function (instance){
                let ClassChall07 = Java.use("uk.rossmarks.fridalab.challenge_07")
                for (let i=1000; i<=9999; i++){
                    if(ClassChall07.check07Pin(i.toString())){
                        instance.chall07(String(i));
                        console.log("[+] Challenge_07 Solved : " + i);
                        break;
                    }
                }
            },
            onComplete : function(){}
        })
    })
})
```
브루트포싱하여 알맞은 값을 확인한 후 `chall07`을 그 값과 함께 실행했다.


# **Challenge 08**
---

```java
public boolean chall08() {
    return ((String) ((Button) findViewById(R.id.check)).getText()).equals("Confirm");
}
```
마지막 문제다. 현재 확인버튼 이름인 `Check`를 `Confirm`으로 변경해주면 된다.

```js
setImmediate(function() {
    Java.perform(function() {
        Java.choose("uk.rossmarks.fridalab.MainActivity", {
            onMatch: function (instance){
               let button = Java.use("android.widget.Button")
			   // public static final int check = 0x7f07002f;
               let check = instance.findViewById(0x7f07002f)
               let checkButton = Java.cast(check, button)
               let String = Java.use("java.lang.String")
               checkButton.setText(String.$new("Confirm"))
            },
            onComplete : function(){
                console.log("[+] Challenge_08 Solved");
            }
        })
    })
})
```

위 코드를 실행하면 Check 버튼이 Confirm 버튼으로 바뀐다.

---


# **All Solved**

이렇게 모든 문제를 해결했다. `FridaLab`은 Frdia 툴을 사용하기 위한 기초적인 문제이기에 다른 워게임들을 풀어보고 다양한 APK를 분석해봐야 할 것 같다.

<div style="display: flex; gap: 1em; align-items: flex-start;">
	<img src="/assets/img/post_images/FridaLab/FridaLab_Capture.png"
		alt="FridaLabCapture"
		width="400" height="100" 
		style="border-radius: 8px;"
	/>
	<img src="/assets/img/post_images/FridaLab/FridaLab_solved.png"
		alt="FridaLabAllSolved"
		width="400" height="100"
		style="border-radius: 8px;"
	/>
</div>