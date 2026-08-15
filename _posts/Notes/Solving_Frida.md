------------

### Solving Frida Challenges

------------

- [Labs](http://github.com/DERE-ad2001/Frida-Labs/)
- Hooking Method
```js
//Hooking methods
Java.perform(function() {
    var mainActivity = Java.use('com.ad2001.frida0x1.MainActivity');
    //Textview
    mainActivity.get_random.overload().implementation = function() {
        console.log('[+] Hooked get_random');
        return 1;
    }
})
```
- Lab2(Hooking Static Method)-> Since the  function ain't getting clicked, you have to call it yourself

```java
public static void get_flag(int a) {
        if (a == 4919) {
            try {
                SecretKeySpec secretKeySpec = new SecretKeySpec("HILLBILLWILLBINN".getBytes(), "AES");
                Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
                IvParameterSpec iv = new IvParameterSpec(new byte[16]);
                cipher.init(2, secretKeySpec, iv);
                byte[] decryptedBytes = cipher.doFinal(Base64.decode("q7mBQegjhpfIAr0OgfLvH0t/D0Xi0ieG0vd+8ZVW+b4=", 0));
                String decryptedText = new String(decryptedBytes);
                t1.setText(decryptedText);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

```

- Script-:

```js
Java.perform(function() {
    var check = Java.use("com.ad2001.frida0x2.MainActivity");
    check.get_flag(4919);
})
```
- Changing the values of a variable-:

```
Java.perform(function() {
    var checker =  Java.use('com.ad2001.frida0x3.Checker');
    checker.code.value = 512;
})
```

- Creating a class instance-:

```java
//Vulnerable code
package com.ad2001.frida0x4;

/* JADX INFO: loaded from: classes3.dex */
public class Check {
    public String get_flag(int a) {
        if (a == 1337) {
            byte[] decoded = new byte["I]FKNtW@]JKPFA\\[NALJr".getBytes().length];
            for (int i = 0; i < "I]FKNtW@]JKPFA\\[NALJr".getBytes().length; i++) {
                decoded[i] = (byte) ("I]FKNtW@]JKPFA\\[NALJr".getBytes()[i] ^ 15);
            }
            return new String(decoded);
        }
        return "";
    }
}
```
- Solve-:

```js
Java.perform(function() {
    var cls =  Java.use('com.ad2001.frida0x4.Check');
    var obj = cls.$new();
    console.log(obj.get_flag(1337));
})
```

- Challenge 0x5(Invoking methods on an existing instance)-:

```java
public class MainActivity extends AppCompatActivity {
    TextView t1;

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.t1 = (TextView) findViewById(R.id.textview);
    }

    public void flag(int code) {
        if (code == 1337) {
            try {
                SecretKeySpec secretKeySpec = new SecretKeySpec("WILLIWOMNKESAWEL".getBytes(), "AES");
                Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
                IvParameterSpec iv = new IvParameterSpec(new byte[16]);
                cipher.init(2, secretKeySpec, iv);
                byte[] decodedEnc = Base64.getDecoder().decode("2Y2YINP9PtJCS/7oq189VzFynmpG8swQDmH4IC9wKAY=");
                byte[] decryptedBytes = cipher.doFinal(decodedEnc);
                String decryptedText = new String(decryptedBytes);
                this.t1.setText(decryptedText);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```
```js
Java.perform(function() {
    Java.choose("com.ad2001.frida0x5.MainActivity",{
        onMatch(ins){
            console.log("[+] Instance Found");
            ins.flag(1337);
        },
        onComplete(){

        }
    })
})
```

- [Frida 0x6]Invoking a method with an object argument-:

```js
Java.perform(function() {
    var cls =  Java.use('com.ad2001.frida0x6.Checker');
    var clsinstance = cls.$new();
    clsinstance.num1.value =  1234;
    clsinstance.num2.value = 4321;

    Java.choose('com.ad2001.frida0x6.MainActivity',{
        onMatch(ins){
            console.log("[+] Instance found->")
            ins.get_flag(clsinstance);
        },
        onComplete(){


        }
    })

})
```
- Hooking the constructor( Frida 0x7)-:

```java
package com.ad2001.frida0x7;

/* JADX INFO: loaded from: classes3.dex */
public class Checker {
    int num1;
    int num2;

    Checker(int a, int b) {
        this.num1 = a;
        this.num2 = b;
    }
}
```

```js
Java.perform(function() {
    var clsChecker =  Java.use('com.ad2001.frida0x7.Checker');
    clsChecker.$init.implementation =  function(a,b){
        this.num1.value = 600;
        this.num2.value = 600;
    }
})
```

