---
title : Houseplant CTF Reversing Solution
categories : [Hacking, CTF]
tags : [Houseplant CTF, Reversing]
---

## 1. Fragile
```
Can you help me move my stuff? This one's fragile!

Dev: Sri
 fragile.java 8a3966a1a07bf03681a4da4deb2d12ca
```
```java
// fragile.java

import java.util.*;

public class fragile
{
    public static void main(String args[]) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter flag: ");
        String userInput = scanner.next();
        String input = userInput.substring("rtcp{".length(),userInput.length()-1);
        if (check(input)) {
            System.out.println("Access granted.");
        } else {
            System.out.println("Access denied!");
        }
    }
    
    public static boolean check(String input){
        boolean h = false;
        String flag = "h1_th3r3_1ts_m3";
        String theflag = "";
        if(input.length() != flag.length()){
            return false;
        }
        for(int i = 0; i < flag.length(); i++){
            theflag += (char)((int)(flag.charAt(i)) + (int)(input.charAt(i)));
        }
        return theflag.equals("ÐdØÓ§åÍaèÒÁ¡");
    }
}
```
```check```를 보면 입력한 값에 ```flag```값을 더한 값이 ```ÐdØÓ§åÍaèÒÁ¡```와 같으면 true를 리턴하므로 ```ÐdØÓ§åÍaèÒÁ¡```에서 ```flag```값을 뺀 값이 flag임.  
```python
cmp1="h1_th3r3_1ts_m3"
cmp2="ÐdØÓ§åÍaèÒÁ¡"
flag=""
for i in range(0,len(cmp2)):
    flag+=chr(ord(cmp2[i])-ord(cmp1[i]))
print(flag)

# result : h3y_1ts_n0t_b4d 
```
```
flag : rtcp{h3y_1ts_n0t_b4d}
```
## 2. Breakable
```
Okay...this one's better, but still be careful!

Dev: Sri
 breakable.java 825676b8c563a3b6f6b394ce338bfae3
```
```java
// breakable.java

import java.util.*;

public class breakable
{
    public static void main(String args[]) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter flag: ");
        String userInput = scanner.next();
        String input = userInput.substring("rtcp{".length(),userInput.length()-1);
        if (check(input)) {
            System.out.println("Access granted.");
        } else {
            System.out.println("Access denied!");
        }
    }
    
    public static boolean check(String input){
        boolean h = false;
        String flag = "k33p_1t_in_pl41n";
        String theflag = "";
        int i = 0;
        if(input.length() != flag.length()){
            return false;
        }
        for(i = 0; i < flag.length()-2; i++){
            theflag += (char)((int)(flag.charAt(i)) + (int)(input.charAt(i+2)));
        }
        for(i = 2; i < flag.length(); i++){
            theflag += (char)((int)(flag.charAt(i)) + (int)(input.charAt(i-2)));
        }
        String[] flags = theflag.split("");
        for(; i < (int)((flags.length)/2); i++){
            flags[i] = Character.toString((char)((int)(flags[i].charAt(0)) + 20));
        }
        return theflag.equals("ÒdÝ¾¤¤¾ÙàåÐcÝÆ¥ÌÈáÏÜ¦aã");
    }
}
```
리턴할 때 ```theflag```값과 비교하므로 그 부분만 보면 다음과 같음. ```flag 길이 : 16```  
```
첫 번째 for문 : i=0 ~ i=13까지 반복 -> 길이 : 14

-> flag[0]+input[2], flag[1]+input[3], ..., flag[13]+input[15]

-> theflag[0]~theflag[13] 
```
```
두 번째 for문 : i=2 ~ i=15까지 반복 -> 길이 : 14

-> flag[2]+input[0], flag[3]+input[1], ..., flag[15]+input[13]

-> theflag[14]~theflag[27]
```
따라서 ```ÒdÝ¾¤¤¾ÙàåÐcÝÆ¥ÌÈáÏÜ¦aã```에서 ```i=13```까지는 ```flag[0]~flag[13]```을 빼면 ```input[2]~input[15]```까지 구할 수 있으므로, 나머지 ```input[0],input[1]```은 ```i=14, 15```에서 ```flag[2], flag[3]```을 빼면 됨. 코드를 짜면 다음과 같음.
```python
cmp1="ÒdÝ¾¤¤¾ÙàåÐcÝÆ¥ÌÈáÏÜ¦aã"
cmp2="k33p_1t_in_pl41n"
flag=""

flag+=chr(ord(cmp1[14])-ord(cmp2[2]))
flag+=chr(ord(cmp1[15])-ord(cmp2[3]))

for i in range(0,14) :
    flag+=chr(ord(cmp1[i])-ord(cmp2[i]))

print(flag)

# result : 0mg_1m_s0_pr0ud_
```
```
flag : rtcp{0mg_1m_s0_pr0ud_}
```
## 3. Bendy
```
I see you've found my straw collection...
(this is the last excessive for loop one i swear)

Dev: Sri
 bendy.java 962c7f3c4606ff7b6cd72729c98be91a
```
```java
public static boolean check(String input){
    boolean h = false;
    String flag = "r34l_g4m3rs_eXclus1v3";
    String theflag = "";
    int i = 0;
    if(input.length() != flag.length()){
        return false;
    }
    if(!input.substring(0,2).equals("h0")){
        return false;
    }
    if(input.charAt(7) != 'u'){
        return false;
    }
    for(i = 0; i < flag.length()-14; i++){
        theflag += (char)((int)(flag.charAt(i)) + (int)(input.charAt(i+8)));
    }
    for(i = 10; i < flag.length()-6; i++){
        theflag += (char)((int)(flag.charAt(i)) + (int)(input.charAt(i-8)));
    }
    for(; i < flag.length(); i++){
        theflag += (char)((int)(flag.charAt(i-3)) + (int)(input.charAt(i)));
    }
    //ÒdÝ¾¤¤¾ÙàåÐcÝÆ¥ÌÈáÏÜ¦aã
    String[] flags = theflag.split("");
    for(i=0; i < (int)((flags.length)/2); i++){
        flags[i] = Character.toString((char)((int)(flags[i].charAt(0)) + 20));
    }
    theflag = theflag.substring(flags.length/2);
    for(int k = 0; k < ((flags.length)/2); k++){
        theflag += flags[k];
    }
    return theflag.equals("ÄÑÓ¿ÂÒêáøz§è§ñy÷¦");
}
```
코드는 다음과 같음.  
```
input 길이 = flag 길이 = 21

input[0], input[1] = "h", "0" -> flag : "h0"+ 나머지 19글자

input[7] = "u" -> flag : "h0"+xxxxx"+"u"+ 나머지 13글자
```
```
첫 번째 for문 : i=0 ~ i=6까지 반복

-> flag[0]+input[8], flag[1]+input[9], ..., flag[6]+input[14]

-> theflag[0] ~ theflag[6] -> input[8] ~ input[14] 까지 구할 수 있음.
```
```
두 번째 for문 : i=10 ~ i=14까지 반복

-> flag[10]+input[2], flag[11]+input[3], ..., flag[14]+input[6]

-> theflag[7] ~ theflag[11] -> input[2] ~ input[6] 까지 구할 수 있음.
```
```
세 번째 for문 : i=15 ~ i=20까지 반복

-> flag[12]+input[15], flag[13]+input[16], ..., flag[17]+input[20]

-> theflag[12] ~ theflag[17] -> input[15] ~ input[20] 까지 구할 수 있음.
```
```theflag``` 길이가 18인 이유는 ```input[0,1,7]``` 값이 빠졌기 때문임.  
```
네 번째 for문 : i=0 ~ i=8까지 반복

flags[0] ~ flags[8] 까지 각각의 자리에 20씩 더한 아스키 값
```
```
java substring 함수 

인자값이 하나인 경우 ex) String str="0123456789"; str.substring(5) -> "56789"

인자값이 두 개인 경우 ex) String str="0123456789"; str.substring(1,5) -> "1234"
```
```
마지막 부분은 요약하면 이렇게 됨.

theflag=flags[9] ~ flags[17] + flags[0] ~ flags[8] -> ÄÑÓ¿ÂÒêá + øz§è§ñy÷¦

-> 0번째부터 해주면 -> øz§è§ñy÷¦ÄÑÓ¿ÂÒêá
```
우선 ```theflag```의 인덱스 0 ~ 8까지의 값에서 20씩 뺀 값을 구하고 위의 내용을 토대로 코드를 짜면 다음과 같음. 
```python
cmp1="øz§è§ñy÷¦ÄÑÓ¿ÂÒêá"    # 순서 바꿔서 넣은 값
cmp2="r34l_g4m3rs_eXclus1v3"
flag="h0"

cmp1=list(cmp1)
for i in range(0,9) :
    cmp1[i]=chr(ord(cmp1[i])-20)
cmp1=''.join(cmp1)

for i in range(10,15) :
    flag+=chr(ord(cmp1[i-3])-ord(cmp2[i]))

flag+="u"

for i in range(0,7):
    flag+=chr(ord(cmp1[i])-ord(cmp2[i]))

for i in range(12,18) :
    flag+=chr(ord(cmp1[i])-ord(cmp2[i]))

print(flag)

# result : h0p3_y0ur3_h4v1ng_fun
```
```
flag : rtcp{h0p3_y0ur3_h4v1ng_fun}
```

## 4. EZ
```
I made a password system, bet you can't get the flag

Dev: William
 Hint! Just a series of nice and relatively simple Python reverse engineering!
 pass0.py 2c5b3d8284ce1758feef1cc777c2c67d
```
```python
print("rtcp{tH1s_i5_4_d3c0Y_fL4g_s0_DoNt_sUbm1T_1t!}")

##they will never suspect a thing if i hide it here :)
##print("rtcp{tH1s_i5_4_r3aL_fL4g_s0_Do_sUbm1T_1t!}") <- 이게 flag
```

## 5. PZ
```
Ok, I think I made it slightly better. Now you won't get the flag this time!

Dev: William
 pass1.py 22302b4223436f2f6c1a490b4049c928
```
```python
def checkpass():
  userinput = input("Enter the password: ")
  if userinput == "rtcp{iT5_s1mPlY_1n_tH3_C0d3}":
      return True
  else:
      return False
    
def catcheckpass():
  userinput = input("pwease enter youwr password... uwu~ nya!!: ")
  if userinput == "rtcp{iT5_s1mPlY_1n_tH3_C0d3}":
      return True
  else:
      return False

def main():
    access = checkpass()
    if access == True:
        print("Unlocked. The flag is the password.")
        print("b-but i wunna show off my catswpeak uwu~... why wont you let me do my nya!!")
        exit()
    else:
        print("Incorrect password!")
        print("sowwy but now you gunnu have to listen to me spweak in cat giwrl speak uwu~")
        catmain()

def catmain():
    access = catcheckpass()
    if access == True:
        print("s-senpai... i unwocked it fowr you.. uwu~")
        print("t-the fwlag is... the password.. nya!")
        exit()
    else:
        print("sowwy but that wasnt quite rwight nya~")
        catmain()

access = False
main()
```
이름만 비슷하게 한 함수들이 여러개 있는 말장난 같은 문제여서 플래그가 있는 함수만 남겨놓음.  
풀이할게 없는, 코드만 잘 보면 푸는 문제.
```
flag : rtcp{iT5_s1mPlY_1n_tH3_C0d3}
```

## 6. LEMON
```
Fine. I made it a bit more secure by not just leaving it directly in the code.

Dev: William
 pass2.py 737e9ec98282f6831084dfe0b2eef879
```
```python
def checkpass():
  userinput = input("Enter the password: ")
  if userinput[0:4] == "rtcp":
        if userinput[10:13] == "tHi":
            if userinput[22:25] == "cuR":
                if userinput[4:7] == "{y3":
                    if userinput[16:19] == "1nT":
                        if userinput[7:10] == "4H_":
                            if userinput[13:16] == "S_a":
                                if userinput[19:22] == "_sE":
                                    if userinput [25:27] == "3}":
                                        return True # rtcp{y34H_tHiS_a1nT_sEcuR3}
  else:
    return False
def main():
    access = checkpass()
    if access == True:
        print("Unlocked. The flag is the password.")
        print("redacted") # 너무 길어서 지움.
        exit()
    else:
        print("Incorrect password!")
        print("sowwy but now you gunnu have to listen to me spweak in cat giwrl speak uwu~")
        catmain()

def catmain() : 
    redacted # 필요 없어서 지움
def catcheckpass() :
    redacted # 필요 없어서 지움

access = False
main()
```
```PZ```와 마찬가지로 코드만 보면 풀 수 있는 문제
```
flag : rtcp{y34H_tHiS_a1nT_sEcuR3}
```

## 7. SQUEEZY
```
Ok this time, you aren't getting anywhere near anything.

Dev: William
 pass3.py 397e98b651afba7856167ccb46497ee4
```
```python
import base64
def checkpass():
  userinput = input("Enter the password: ")
  key = "meownyameownyameownyameownyameownya"
  a = woah(key,userinput)
  b = str.encode(a)
  result = base64.b64encode(b, altchars=None)
  if result == b'HxEMBxUAURg6I0QILT4UVRolMQFRHzokRBcmAygNXhkqWBw=':
      return True
  else:
      return False

def main():
    access = checkpass()
    if access == True:
        print("Unlocked. The flag is the password.")
        print("pwease let me do my nya~ next time!!")
        exit()
    else:
        print("Incorrect password!")
        print("sowwy but now you gunnu have to listen to me spweak in cat giwrl speak uwu~")
        catmain()

def woah(s1,s2):
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

def catmain():
    redacted # 필요 없음.
def catcheckpass():
    redacted # 필요 없음. 

access = False
main()
```
```XOR```연산이므로 코드의 반대로 해주기만 하면 되므로 코드로 나타내면 다음과 같음.
```python
import base64

def woah(s1,s2):
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

result='HxEMBxUAURg6I0QILT4UVRolMQFRHzokRBcmAygNXhkqWBw='
b=base64.b64decode(result)

cmp1=b.decode('utf-8')  # str.decode는 py-3에서 없다고 함. 
cmp2="meownyameownyameownyameownyameownya"
flag=woah(cmp1,cmp2)

print(flag)   # result : rtcp{y0u_L3fT_y0uR_x0r_K3y_bEh1nD!}
```
```
flag : rtcp{y0u_L3fT_y0uR_x0r_K3y_bEh1nD!}
```

## 8. thedanzman
```
Fine. I made it even harder. It is now no longer "ez", "pz", "lemon" or "squeezy".
You will never get the flag this time.

Dev: William
 Hint! This should be no problem if you look at the previous ones.
 pass4.py 602543fd23f2383806cb8ef880199f8b
```
```python
import base64
import codecs
def checkpass():
  userinput = input("Enter the password: ")
  key = "nyameowpurrpurrnyanyapurrpurrnyanya"
  key = codecs.encode(key, "rot_13")
  a = nope(key,userinput)
  b = str.encode(a)
  c = base64.b64encode(b, altchars=None)
  c = str(c)
  d = codecs.encode(c, 'rot_13')
  result = wow(d)
  if result == "'=ZkXipjPiLIXRpIYTpQHpjSQkxIIFbQCK1FR3DuJZxtPAtkR'o":
      return True
  else:
      return False

def main():
    access = checkpass()
    if access == True:
        print("Unlocked. The flag is the password.")
        print("pwease let me do my nya~ next time!!")
        exit()
    else:
        print("Incorrect password!")
        print("redacted") # 길어서 자름.
        catmain()

def nope(s1,s2):
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

def wow(x):
  return x[::-1]

def catmain():
    redacted # 필요 없음
def catcheckpass():
    redacted # 필요 없음

access = False
main()
```
힌트에 있는 말처럼 이전 문제와 동일.  
```python
import base64
import codecs

def nope(s1,s2):
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

def wow(x):
  return x[::-1]

result=wow("'=ZkXipjPiLIXRpIYTpQHpjSQkxIIFbQCK1FR3DuJZxtPAtkR'o")
a=codecs.decode(result,'rot_13')
b=a.replace('b',"",1) # decode 시 에러가 발생해서 byte를 표식을 없앰. 
c=base64.b64decode(b)

cmp1=c.decode('utf-8')
cmp2="alnzrbjcheecheealnalncheecheealnaln"
flag=nope(cmp1,cmp2)

print(flag)   # result : rtcp{n0w_tH4T_w45_m0r3_cH4lL3NgiNG}
```
```
flag : rtcp{n0w_tH4T_w45_m0r3_cH4lL3NgiNG}
```

## 9. Tough 1,237 Points
```
You would have to drop this one a few times before it breaks. (all ASCII chars)

Dev: Sri
 Hint! If it doesn't work, use this is unicode for thefinalflag: 
 157, 157, 236, 168, 160, 162, 171, 162, 165, 199, 169, 169, 160, 194, 235, 207, 227, 210, 157, 203, 227, 104, 212, 202
 tough.java 3bb39afa5804ac57ef6324bb3d98c190
```
```java
import java.util.*;

public class tough
{
    public static int[] realflag = {9,4,23,8,17,1,18,0,13,7,2,20,16,10,22,12,19,6,15,21,3,14,5,11};
    public static int[] therealflag = {20,16,12,9,6,15,21,3,18,0,13,7,1,4,23,8,17,2,10,22,19,11,14,5};
    public static HashMap<Integer, Character> theflags = new HashMap<>();
    public static HashMap<Integer, Character> theflags0 = new HashMap<>();
    public static HashMap<Integer, Character> theflags1 = new HashMap<>();
    public static HashMap<Integer, Character> theflags2 = new HashMap<>();
    public static boolean m = true;
    public static boolean g = false;
    
    public static void main(String args[]) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter flag: ");
        String userInput = scanner.next();
        String input = userInput.substring("rtcp{".length(),userInput.length()-1);
        if (check(input)) {
            System.out.println("Access granted.");
        } else {
            System.out.println("Access denied!");
        }
    }
    
    public static boolean check(String input){
        boolean h = false;
        String flag = "ow0_wh4t_4_h4ckr_y0u_4r3"; // length : 24
        createMap(theflags, input, m);  
        createMap(theflags0, flag, g);
        createMap(theflags1, input, g);
        createMap(theflags2, flag, m);  
        String theflag = "";
        String thefinalflag = "";
        int i = 0;
        if(input.length() != flag.length()){
            return h;
        }
        if(input.charAt(input.length()-2) != 'o'){
            return false;
        }
        if(!input.substring(2,4).equals("r3") || input.charAt(5) != '_' || input.charAt(7) != '_'){
            return false;
        }
        //rtcp{h3r3s_a_fr33_fl4g!}
        for(; i < input.length()-3; i++){
            theflag += theflags.get(i);
        }
        for(; i < input.length();i++){
            theflag += theflags1.get(i);
        }
        for(int p = 0; p < theflag.length(); p++){
            thefinalflag += (char)((int)(theflags0.get(p)) + (int)(theflag.charAt(p)));
        }
        for(int p = 0; p < theflag.length(); p++){
            if((int)(thefinalflag.charAt(p)) > 146 && (int)(thefinalflag.charAt(p)) < 157){
                thefinalflag = thefinalflag.substring(0,p) + (char)((int)(thefinalflag.charAt(p)+10)) + thefinalflag.substring(p+1);
            }
        }
        return thefinalflag.equals("ì¨ ¢«¢¥Ç©© ÂëÏãÒËãhÔÊ");
    }
    public static void createMap(HashMap owo, String input, boolean uwu){
        if(uwu){
            for(int i = 0; i < input.length(); i++){
                owo.put(realflag[i],input.charAt(i));
            }
        } else{
            for(int i = 0; i < input.length(); i++){
                owo.put(therealflag[i],input.charAt(i));
            }
        }
    }
}
```
우선 ```Hashmap```에 대한 설명은 여기 : <a href="https://codechacha.com/ko/java-map-hashmap/" target="_blank">Here</a>  
코드의 내용을 정리하면 다음과 같음.
```
  {key, value} of theflags                        {key, value} of theflags0
       {0, input[7]},                                      {0, 4}, 
       {1, input[5]},                                      {1, 4}, 
       {2, input[10]},                                     {2, y}, 
       {3, input[20]},                                     {3, t}, 
       {4, input[1]},                                      {4, c}, 
       {5, input[22]},                                     {5, 3}, 
       {6, input[17]},                                     {6, w}, 
       {7, input[9]},                                      {7, h}, 
       {8, input[3]},                                      {8, r}, 
       {9, input[0]},                                      {9, _}, 
       {10, input[13]},                                    {10, 0}, 
       {11, input[23]},                                    {11, h4, 
       {12, input[15]},                                    {12, 0}, 
       {13, input[8]},                                     {13, _}, 
       {14, input[21]},                                    {14, r}, 
       {15, input[18]},                                    {15, h}, 
       {16, input[12]},                                    {16, w}, 
       {17, input[4]},                                     {17, _}, 
       {18, input[6]},                                     {18, _}, 
       {19, input[16]},                                    {19, _}, 
       {20, input[11]},                                    {20, o}, 
       {21, input[19]},                                    {21, 4}, 
       {22, input[14]},                                    {22, u}, 
       {23, input[2]}                                      {23, k}
```
```
   {key, value} of theflags1                
       {0, input[9]},                                      
       {1, input[12]},                                  
       {2, input[17]},                                 
       {3, input[7]},                                   
       {4, input[13]},                                 
       {5, input[23]},                             
       {6, input[4]},                                   
       {7, input[11]},                                   
       {8, input[15]},                                    
       {9, input[3]},                                     
       {10, input[18]},                                     
       {11, input[21]},                                    
       {12, input[2]},                                      
       {13, input[10]},                               
       {14, input[22]},                               
       {15, input[5]},                                
       {16, input[1]},                                    
       {17, input[16]},                                   
       {18, input[8]},                                   
       {19, input[20]},                                    
       {20, input[0]},                                     
       {21, input[6]},                                    
       {22, input[19]},                                   
       {23, input[14]}                                    
```
```
realflag, therealflag에는 0~23까지의 인덱스 숫자가 들어있음.

input 길이 : 24 
input[2], input[3] = "r", "3"
input[5], input[7] = "_"
input[22] = 'o'
```
```
첫 번째 for문 : i=0 ~ i=20까지 반복

theflag[0] ~ theflag[20] : theflags의 key가 0 ~ 20까지의 value 
```
```
두 번째 for문 : i=21 ~ i=23까지 반복

theflag[21] ~ theflag[23] : theflags1의 key가 21~23까지의 value 
```
```
   {index, value} of theflag
       {0, input[7]},
       {1, input[5]},
       {2, input[10]},
       {3, input[20]},
       {4, input[1]},
       {5, input[22]},
       {6, input[17]},
       {7, input[9]},
       {8, input[3]},
       {9, input[0]},
       {10, input[13]},
       {11, input[23]},
       {12, input[15]},
       {13, input[8]},
       {14, input[21]},
       {15, input[18]},
       {16, input[12]},
       {17, input[4]},
       {18, input[6]},
       {19, input[16]},
       {20, input[11]},
       {21, input[6]},
       {22, input[19]},
       {23, input[14]} 
```
```
세 번째 for문 : p=0 ~ p=23까지 반복

thefinalflag[0] ~ thefinalflag[23] : theflags0의 key 0의 value + theflag[0] 부터 23까지
```
```
네 번째 for문 : p=0 ~ p=23까지 반복

thefinalflag에서 아스키값이 147~156에 해당하는 값이 있다면 그 값에 +10을 해주겠다는 소리. 
```
정리하는건 힘든데 해놓고보니 해야될 게 별로 없는 문제였음;;    
우선 최종 ```thefinalflag``` 값에서 아스키 값이 147~156에 해당하는 값을 찾아내야함.
```python
realflag=[9,4,23,8,17,1,18,0,13,7,2,20,16,10,22,12,19,6,15,21,3,14,5,11]
therealflag=[20,16,12,9,6,15,21,3,18,0,13,7,1,4,23,8,17,2,10,22,19,11,14,5]
flag = "ow0_wh4t_4_h4ckr_y0u_4r3"

theflags={}
theflags0={}
theflags1={}

for i in range(0,24) :
    theflags0[therealflag[i]]=flag[i]
    theflags[realflag[i]]="input[{}]".format(i)
    theflags1[therealflag[i]]="input[{}]".format(i)

    
thefinalflag=[157, 157, 236, 168, 160, 162, 171, 162, 165, 199, 169, 169, 160, 194, 235, 207, 227, 210, 157, 203, 227, 104, 212, 202]

for i,j in enumerate(thefinalflag) :
    if (j > 156) and (j < 167) :
        thefinalflag[i]=j-10

flag1=[]

for i in range(0,24) :
    flag1.append(chr(thefinalflag[i]-ord(theflags0[i])))

flag1[5]="o" # input[22]
flag1[8]="3" # input[3]

print(''.join(flag1))

# result : __s43o403hyufcygls4lt4__
# 순서대로 정렬하면 -> h3[2]3s_4_c0stly_fl4g_4you
```
```input[2]="r"```이므로 적용하면 ```h3r3s_4_c0stly_fl4g_4you ```  
```
flag : rtcp{h3r3s_4_c0stly_fl4g_4you} 
```
