-------------

### Hooking Android Methods with Frida

------------


- Building patched apks with apktool-:

```
apktool b revme/ -o revme1.apk
```

- You have to sign it afterwards, you knowww that.

```powershell
keytool -genkey -v -keystore debug.keystore -alias androiddebugkey -keyalg DSA -sigalg  SHA1withDSA -keysize 1024 -validity 10000
jarsigner -keystore debug.keystore -verbose -storepass "password" -sigalg SHA1withDSA -digestalg SHA1 c:\Users\HP\Downloads\revme1.apk androiddebugkey
```
- [Apk file](https://github.com/IR0NBYTE/Reverse-Engineering-Practice/blob/main/revme.apk)
- Solve-: 

```js
//Running it-:  frida -U -f com.example.basic_rev -l "C:\Users\HP\Downloads\revme\hook.js"
Java.perform(function(){
    Java.scheduleOnMainThread(function() {
      console.log("[+] Starting solve script");

      var targetClass =  Java.use("com.example.basic_rev.MainActivity");
      var makeFlag = targetClass.makeFlag;
      makeFlag.implementation = function(seed) {
        console.log("[+] Make flag called with seed "+ seed);
        var result =  makeFlag.call(this,seed);
        console.log("[+] Makeflag result::" + result);
        return result;
      };
    })
})
```
- Hooking shared libraries (.so) files-:
- Checking for modules and base address-:

```js
//Format-: frida -U -n boxy -l hook.js
var libboxy = Process.getModuleByName("libboxy.so");
console.log("[+] Library base address" + libboxy.base);
//Exported functions
var exportedFunctions =  libboxy.enumerateExports();
exportedFunctions.forEach(function(exp) {
    console.log("[+] Exported functions")
    console.log("[+]" + exp.name + " at " + exp.address);
})
 //Imported functions
var importedFunctions =  libboxy.enumerateImports();
importedFunctions.forEach(function(imp) {
    console.log("[+] Imported functions")
    console.log(" [+] " + imp.module + imp.name + " at " + imp.address);
})
```
- Hook script( interceptor)


```js
Java.perform(function() {
    var target  = Java.use("com.example.boxy.MainActivity$1");
    const libboxy = Process.getModuleByName('libboxy.so', 'Java_com_example_boxy_MainActivity_checkCred');
    target.onClick.implementation = function(view) {
        console.log("[+] Onclick is called");
        Interceptor.attach(libboxy.getExportByName('read'), {
            onEnter(args) {
                console.log("[+] Java_com_example_boxy_MainActivity_checkCred Intercepted (onEnter)");
            },
            onLeave(retval) {
               retval.replace(0);
               console.log("[+] The return value is " + retval);
            }
       });
       this.onClick(view);
    }
});
```
- AndroGoat-:

```js
Java.perform(function() {
    var target =  Java.use('owasp.sat.agoat.RootDetectionActivity');
    var target2 = Java.use('okhttp3.CertificatePinner');
    var target3  = Java.use('owasp.sat.agoat.EmulatorDetectionActivity');
    var target4 =  Java.use('owasp.sat.agoat.BinaryPatchingActivity');
    //Trying to patch isAdmin
    //Accessing class instance values with Java.choose
    Java.choose("owasp.sat.agoat.BinaryPatchingActivity", {
        onMatch: function(instance) {
            console.log("[+] Current value: " + instance.isAdmin.value);
            instance.isAdmin.value = true;
        },
        onComplete: function() {}
    })
    target.isRooted.implementation = function(){
        console.log("[+] IsRooted() hooked")
        var result =  this.isRooted();
        console.log('[+] Result: ' + result);
        return false;
    }
    target2.check.overload.implementation = function(hostname, peerCertificates) {
        result = this.check(hostname, peerCertificates);
        console.log(result);
    }
    target3.isEmulator.implementation = function() {
        console.log("[+] Hooked isEmulator() function");
        return true;

    }

})
```
------------

### Installing Keytool| Jar signer

------------

- It is installed by default in java bins `C:\Program Files\Java\jre1.8.0_141\bin\`-:
- Both available in jdks

-----------------

### Hooking arm based native functions

-----------------

- On most emulators, it is sort of a challenge to hook arm based binaries e.g libil2cpp.so, libapp.so. That's where `LDplayer` comes in to save the day.

<img width="697" height="461" alt="image" src="https://github.com/user-attachments/assets/357b8f86-d6b9-4c3c-9577-c2fa55fbf562" />

- Then, use the LD player multi instance manager to download an `Android 64bit` image.

<img width="343" height="147" alt="image" src="https://github.com/user-attachments/assets/2f672d26-8016-45c4-a19f-896686c808d2" />

- Setup frida, the first thing is to get the process identifier for the app process with-:

```
Frida-ps -Uia
```
<img width="361" height="197" alt="image" src="https://github.com/user-attachments/assets/cba7b2cf-35c5-43ae-83d9-9162146d3066" />


- Then, run this to hook it-:

```
frida -U --realm=emulated -p [pid] -l hook.js
```
- Script

```js
const libil2cpp = Process.getModuleByName("libil2cpp.so");
if (libil2cpp !== null){
    console.log("[+]Hooked libil2cpp.so found");
}
```
<img width="468" height="183" alt="image" src="https://github.com/user-attachments/assets/6cb54805-94a9-490f-90a5-648854dbe686" />

-------------

### Hooking Native functions

--------------

- Challenge 8-:

```js
Java.perform(()=>{
    const libfrida = Process.findModuleByName("libfrida0x8.so");
    if (libfrida){
        console.log("[+] Lib frida found");
    }
    Interceptor.attach(libfrida.getExportByName('Java_com_ad2001_frida0x8_MainActivity_cmpstr'),{
        onEnter(args){
            console.log(`[+] Found function: Java_com_ad2001_frida0x8_MainActivity_cmpstr::`,args[1].readCString());
        },
        onLeave(retVal){
            retVal.replace(ptr(1));
        }
    })
})
```

- Dealing with java types in Frida-:

```js
const HashMap =  Java.use('java.util.HashMap');
const javaString = Java.use('java.lang.String');
const javaBoolean = Java.use('java.lang.Boolean');
//Creating a new map
var javaHashMap =  HashMap.$new()
javaHashMap.put(javaString.$new("isJailed"), javaBoolean.FALSE.value);
```

--------------
