---
title : Houseplant CTF Crypto Solution
categories : [Hacking, CTF]
tags : [Houseplant CTF, Crypto]
---

## 1. Broken Yolks 
```
Fried eggs are the best.
Oh no! I broke my yolk... well, I guess I have to scramble it now.

Ciphertext: smdrcboirlreaefd -> 조합해보면 scramble_or_fried

Dev: Delphine
 Hint! words are separated with underscores
```
## 2. Sizzle -> Bacon Cipher
```
Due to the COVID-19 outbreak, we ran all out of bacon, 
so we had to use up the old stuff instead. 
Sorry for any inconvenience caused...

Dev: William

 Hint! Wrap your flag with rtcp{}, use all lowercase, and separate words with underscores.
 Hint! Is this really what you think it is?
 encode.txt
 ```
 ```
 ....- ..... ...-. .--.- .--.. ....- -..-- -..-. ..--. -.... .-... .-.-. .-.-. ..-.. ...-- ..... .--.. ...-- .-.-- .--.- -.... -...- .-... ..-.- .-... ..-.. ...-- 
 ```
 암호문이 모스부호처럼 생겼지만 모스부호는 알파벳은 4자리를 넘지않고 숫자는 5자리를 넘지않는 다는 점  
 **Tip** : 모스부호를 복호화 했을 때, $, #, % 이외의 비스무리한 것들이 나온다면 잘못 복호화 된 것이므로 다른 종류의 모스 코드(Fractionated Morse)로 복호화 시도를 해보면 좋음.      
 Bacon Cipher는 보통 a,b 두 개의 문자로 알파벳이나 숫자들을 치환하고 그리고 5비트 길이의 2진수 형태로도 치환함. 이러한 특성을 이용하여 모스부호 형태로 되어 있는 암호문을 치환해보면 다음과 같음.  
 ```
 Letter    Encoded Set Types        Letter    Encoded Set Types
  A         aaaaa   00000   .....    N         abbab   01101   .--.-
  B         aaaab   00001   ....-    O         abbba   01110   .---. 
  C         aaaba   00010   ...-.    P         abbbb   01111   .----
  D         aaabb   00011   ...--    Q         baaaa   10000   -....
  E         aabaa   00100   ..-..    R         baaab   10001   -...-
  F         aabab   00101   ..-.-    S         baaba   10001   -..-.
  G         aabba   00110   ..--.    T         baabb   10011   -..--
  H         aabbb   00111   ..---    U         babaa   10011   -.-..
  I         abaaa   01000   .-...    V         babab   10101   -.-.-
  J         abaab   01001   .-..-    W         babba   10110   -.--.
  K         ababa   01010   .-.-.    X         babbb   10111   -.---
  L         ababb   01011   .-.--    Y         bbaaa   11000   --...
  M         abbaa   01100   .--..    Z         bbaab   11000   --..-
```
따라서 위의 내용으로 암호문을 복호화 하면 ```baconbutgrilledandmorsified``` 이 되며 문제에서 나온대로 해주면 ```rtcp{bacon_but_grilled_and_morsified}```  

## 3. fences are cool unless they're taller than you -> Rail fence cipher
```
“They say life's a roller coaster, but to me, it's just jumping over fences.”

tat_uiwirc{s_iaaotrc_ahn}pkdb_esg -> cyberchef로 해서 key=3, offset=3 -> rtcp{ask_tida_about_rice_washing}
```
전치암호의 종류중 하나로 여기 참고  
<a href="https://bpsecblog.wordpress.com/2016/08/12/amalmot_3/" target="_blank">https://bpsecblog.wordpress.com/2016/08/12/amalmot_3/</a>

## 4. Returning Stolen Archives -> RSA 
```
So I was trying to return the stolen archives securely, 
but it seems that I had to return them one at a time, 
and now it seems the thieves stole them back! 
Can you help recover them once and for all? 
It seems they had to steal them one at a time...

Dev: William
 Hint! Well you sure as hell ain't going to solve this one through factorization.
 intercepted.txt 17bbbcf788131eeda1c728f5bada7899
 returningstolenarchives.py a8d0b9393d5013ce54d5a1a7c3fb17e6
```
```
intercepted.txt

n = 54749648884874001108038301329774150258791219273879249601123423751292261798269586163458351220727718910448330440812899799 
e = 65537
ct = [52052531108833646741308670070505961165002560985048445381912028939564989677616205955826911335832917245744890104862186090,24922951057478302364724559167904980705887738247412638765127966502743153757232333552037075100099370197070290632101808468,31333127727137796897042309238173536507191002247724391776525004646835609286736822503824661004274731843794662964916495223,37689731986801363765434552977964842847326744893755747412237221863834417045591676371189948428149435230583786704331100191,10128169466676555996026197991703355150176544836970137898778443834308512822737963589912865084777642915684970180060271437,31333127727137796897042309238173536507191002247724391776525004646835609286736822503824661004274731843794662964916495223,32812400903438770915197382692214538476619741855721568752778494391450400789199013823710431516615200277044713539798778715,48025916179002039543667066543229077043664743885236966440148037177519549014220494347050632249422811334833955153322952673,52052531108833646741308670070505961165002560985048445381912028939564989677616205955826911335832917245744890104862186090,32361547617137901317806379693272240413733790836009458796321421127203474492226452174262060699920809988522470389903614273,4363489969092225528080759459787310678757906094535883427177575648271159671231893743333971538008898236171319923600913595,47547012183185969621160796219188218632479553350320144243910899620916340486530260137942078177950196822162601265598970316,32361547617137901317806379693272240413733790836009458796321421127203474492226452174262060699920809988522470389903614273,33230176060697422282963041481787429356625466151312645509735017885677065049255922834285581184333929676004385794200287512,32315367490632724156951918599011490591675821430702993102310587414983799536144448443422803347161835581835150218650491476,6693321814134847191589970230119476337298868688019145564978701711983917711748098646193404262988591606678067236821423683,32710099976003111674253316918478650203401654878438242131530874012644296546811017566357720665458366371664393857312271236,49634925172985572829440801211650861229901370508351528081966542823154634901317953867012392769315424444802884795745057309,50837960186490992399835102776517955354761635070927126755411572132063618791417763562399134862015458682285563340315570436]
```
```python
returningstolenarchives.py

p = [redacted]
q = [redacted]
e = 65537
flag = "[redacted]"

def encrypt(n, e, plaintext):
  print("encrypting with " + str(n) + str(e))
  encrypted = []
  for char in plaintext:
    cipher = (ord(char) ** int(e)) % int(n)
    encrypted.append(cipher)
  return(encrypted)

n = p * q
ct = encrypt(n, e, flag)
print(ct)
```
코드를 보면 이 암호화 알고리즘은 **RSA 알고리즘**인걸 알 수 있음.  
공개키 ```{n, e}```가 있고 개인키 ```{p, q}```가 있음.  
```
RSA 암호화 알고리즘

서로 다른 소수 p, q를 정한 뒤 n, φ(n), e, d를 구한다. 

n = p * q 
φ(n)=(p-1)(q-1)
e : gcd(e, φ(n)) = 1, 1 < e < φ(n)인 e를 정한다.
d : ed=1 mod (φ(n)), 즉 φ(n)에 대한 e의 역원을 구한다. 1 < d < φ(n)

C : Ciphertext, M : Plaintext

encrypt : C ≡ M^e mod (n)
decrypt : M ≡ C^d mod (n)
```
문제에서 factorization, 즉, 소인수분해를 사용하지 말라 했으므로 p, q를 구할 수 없음.    
따라서 ct는 flag의 글자들을 각각 e의 승을 하고 mod n으로 나눈 값이므로 모든 아스키값들을 brute-force를 통해 구해서 비교를 하는 방법으로 구해야 함.  
여기서 pow함수를 사용.  
```python
# pow(x,y) is equal to x^y
# pow(x,y,z) is equal to x^y % z == x^y mod (z)

n = [redacted] 
e = 65537
ct = [redacted]
flag=""

for i in ct :
    for j in range(32,127) :
        if pow(j,e,n) == i :
            flag+=chr(j)
            break
print(flag)

# rtcp{cH4r_bY_Ch@R!}
```

## 5. Rivest Shamir Adleman -> RSA 
```
A while back I wrote a Python implementation of RSA, but Python's really slow at maths. Especially generating primes.

Dev: Tom
 Hint! There are two possible ways to get the flag ;-)
 chall.7z 36ed4035eb6a0c7a9b2aaa18a5c427ae

chall.7z에는 다음의 파일들이 있음.
encrypt.py, decrypt.py, component.txt, generate_keys.py, public-key.json, secrets.txt.enc, requirements.txt
```
문제에서 두 가지 방법으로 풀 수 있다고 했는데 구하는 과정만 다를 뿐, 결국엔 p, q를 구하여 푸는 방법임.  

### 5-1. Solution 1.
```
public-key.json

{"n": 5215102981058174620100754813213017625443626121109099133656454487932754235228856710661075956048331662593471061936196995326042367228980357932444477256496372200491821105922086202549125972429240337409176104237690646206864286971669895986447543904638596421264915837230690039800948447210554706127145724519079487023930504508462885777797916915752532472831523596571484341342780877665593787078959178539369282442522815729401991936772080063808078804309866694041173404657777517753433918322041736500126265865045225739241983004392226366771900174432875800986183772576663590650132115754645829772406067103501861326445534174181231077263, "e": 5}

secrets.txt.enc

0x20ba6aee3bd1c1b751082bfcb667bad8b632504336f3994606594f4ab756f66e3a24f9782da3a07280aa67cd875e6e33f2c573abf7b7901e5cd428ab8ceb6738b13536fee35a90dac7c2175e41eea5977dfbaff6e68f5b1f6fa3673cba64923b02bff899e2535f7d09afecae6774260ce8be4867f45e63571a2055c645a03dd05d9dd596eec273e1ef4352d712deffc658745d17853cbe5c3bc138574703c994be5374e3ac73279f51f23ec7e55b25b6ab904e06562025c380ce4c4d5ddffc2d649fbd1421b82090d01f24c70254187f1f435e64d7b2bf8395915da3cfdd8680187566b6a51e48146b4a40f08aebdedca8a08557ea3dc5efc2c50377b5764a8c
```
공개키가 주어졌으므로 <a href="http://factordb.com/" target="_blank">tool</a>을 이용하여 ```{p, q}```를 구하여 푸는 방법으로 플래그를 얻을 수 있음.  
```python  
from Crypto.Util.number import *
n=[redacted]
e=5
p=[redacted]
q=[redacted]
phi_n=(p-1)*(q-1)
d = inverse(e,phi_n)

#open_ciphertext=open('secrets.txt.enc','r')
#read_ciphertext=read(open_ciphertext)
#ciphertext=int(read_ciphertext,16)

secret_enc="[redacted]"
ciphertext=int(secret_enc,16)
plaintext=pow(ciphertext,d,n)
flag=long_to_byte(plaintext)

# Result : b'VERIFICATION-UpTheCuts-END\n .--.\n/.-. \'----------.\n\\\'-\' .--"--""-"-\'\n \'--\'\n\nrtcp{f1xed_pr*me-0r_low_e?}'

# flag : rtcp{f1xed_pr*me-0r_low_e?}
```
Crypto Module DOC : <a href="https://pycryptodome.readthedocs.io/en/latest/src/util/util.html#module-Crypto.Util.number" target="_blank">here</a>

### 5-2. Solution 2
문제에도 써있지만 prime 생성이 느리다고 했고, encrypt, decrypt 코드에서 별로 얻을 게 없어서 generate_keys.py 중 p, q 생성 코드를 보면
```python
# Step 1: Generate primes
if not os.path.exists("primes.json"):
    print("Generating primes")

    primes = []

    for i in range(2):
        print(f"Searching for prime {i+1}")
        while True:
            print("Generating new prime - ", end="")
            p = prime_candidate(1024)
            print(str(p)[:10] + "... - ", end="")
            if new_primality_test(p):
                break
            print("not prime")
        print("prime!")
        primes.append(p)

    primes[0] = int(open("component.txt").read())
    with open("primes.json", "w") as f:
        json.dump({"p": primes[0], "q": primes[1]}, f)
    print("Written to file")
else:
    print("Loading predefined primes")
    with open("primes.json") as f:
        t = json.load(f)
        primes = []
        primes.append(t["p"])
        primes.append(t["q"])

p = primes[0]
q = primes[1]
```
보면 소수를 생성해놓고 p를 ```component.txt```에 있는 값으로 다시 설정함.  
```
component.txt

88761620475672281797897005732643499821690688597370440945258776182910533850401433150065043871978311565287949564292158396906865512113015114468175188982916489347656271125993359554057983487741599275948833820107889167078943493772101668339096372868672343763810610724807588466294391846588859523658456534735572626377
```
따라서 p값을 구했으므로 q 값을 구할 수 있고 그 뒤는 sol 1이랑 같음.  
```python
# import json
from Crypto.Util.number import *

# c=int(open('secrets.txt.enc').read()[2:],16)
secret_enc="redacted"
# key=json.loads(open('public-key.json').read())
n=redacted
# p=int(open('component.txt').read())
p=redacted
# q=key['n'] //p
q=n/p

phi_n=(p-1)*(q-1)
d = inverse(e,phi_n)

plaintext=pow(secret_enc,d,n)
flag=long_to_byte(plaintext)
```
다음과 같은 메세지를 얻을 수 있음.  
![](https://masrt200.github.io/hacker-blog/Snips/HOUSEPLANT/RSA3.png){: .align-center}

## 6. Rainbow Vomit
```
o.O What did YOU eat for lunch?!

The flag is case insensitive. // 플래그는 대소문자 구분 x

Dev: Tom
 Hint! Replace spaces in the flag with { or } depending on their respective places within the flag.
 Hint! Hues of hex
 Hint! This type of encoding was invented by Josh Cramer.
 output.png 707fe5d5aa9109efcc7d4193d5b0e83f
```
<img src="https://static.wixstatic.com/media/ea085c_b4765c6fd7bc4f23907ed093b5940e35~mv2.png/v1/fill/w_740,h_445,al_c,q_90,usm_0.66_1.00_0.01/ea085c_b4765c6fd7bc4f23907ed093b5940e35~mv2.webp">
<img src="https://static.wixstatic.com/media/ea085c_d78cbbc613494889a19d40b74297f767~mv2.png/v1/fill/w_740,h_257,al_c,q_90,usm_0.66_1.00_0.01/ea085c_d78cbbc613494889a19d40b74297f767~mv2.webp" alt="output.png">  
  
플래그는 마지막 줄에 있어서 수동적으로 하여 풀 수 있음.
```
rtcp{SHOULD,FL5G4,B3,ST1CKY,OR,N0T}

코드를 짜서 풀면 결과는 이렇다고 함.
there is such as thing as a tomcat but have you ever heard of a tomdog. this is the most important question of our time, and unfortunately one that may never be answered by modern science. the definition of tomcat is a male cat, yet the name for a male dog is max. wait no. the name for a male dog is just dog. regardless, what would happen if we were to combine a male dog with a tomcat. perhaps wed end up with a dog that vomits out flags, like this one rtcp should,fl5g4,b3,st1cky,or,n0t
```

## 7. Post-Homework Death
```
My math teacher made me do this, so now I'm forcing you to do this too.

Flag is all lowercase; replace spaces with underscores.

Dev: Claire
 Hint! When placing the string in the matrix, go up to down rather than left to right.
 Hint! Google matrix multiplication properties if you're stuck.
 posthomeworkdeaths.txt 1459c156daa68f0d7bfcc698a18e93e8
```
```
posthomeworkdeaths.txt

Decoding matrix:

1.6  -1.4  1.8
2.2  -1.8  1.6
-1     1    -1

String:

37 36 -1 34 27 -7 160 237 56 58 110 44 66 93 22 143 210 49 11 33 22
```
string(21개 숫자)을 Decoding matrix에 곱해야 하므로 ```3x3 * 3x7```이나 ```7x3 * 3x3```로 만들어줘야 함. 
힌트에서 string을 행렬로 나타낼 때, 위에서 아래로 하라 했으므로 ```3x3 * 3x7```으로 만들어주면  
```
1.6  -1.4  1.8     37  34  160   58   66  143  11
2.2  -1.8  1.6     36  27  237  110   93  210  33
-1     1    -1     -1  -7   56   44   22   49  22
```
```numpy``` module을 이용하여 풀면
```python
import numpy as np
decoding_mat = np.matrix([[1.6,-1.4,1.8],[2.2,-1.8,1.6],[-1,1,-1]])
string = np.matrix([[37,34,160,58,66,143,11],[36,27,237,110,93,210,33],[-1,-7,56,44,22,49,22]])
output = decoding_mat*string
print(output)
```
```
output

[[7.00000000e+00 4.00000000e+00 2.50000000e+01 1.80000000e+01
  1.50000000e+01 2.30000000e+01 1.10000000e+01]
 [1.50000000e+01 1.50000000e+01 1.50000000e+01 1.42108547e-14
  1.30000000e+01 1.50000000e+01 7.10542736e-15]
 [0.00000000e+00 0.00000000e+00 2.10000000e+01 8.00000000e+00
  5.00000000e+00 1.80000000e+01 0.00000000e+00]]

바꿔주면 다음과 같음.

 7   4  25  18  15  23  11
15  15  15   0  13  15   0
 0   0  21   8   5  18   0

string처럼 나타내주면 715041502515211808151352315181100

이는 A1Z26 Cipher이여서 복호화해주면 godoyourhomework 
```
```
flag : rtcp{go_do_your_homework}
```

## 8. Parasite -> SKATS
```
paraSite Killed me A liTtle inSide

Flag: English, case insensitive; turn all spaces into underscores


Dev: Claire
 Hint! Make sure you keep track of the spacing- it's there for a reason
 Parasite.txt ce126723dee2e0ae63ffda0cf2b73202
```
```
parasite.txt

.---  -..  ..-    --  .  -.-     -.-  -..  ..-.     .--.  ..-  ..-.     .--.  -  -.-    .---  .  ..-.    .-..  ..-    --.  .  ..-  -.-    -.-.  ....  -.-    -.-  ..-  .--   ..-.  ..-    -...  .
```
모스부호를 해독하면 ```JDUMEKKDFPUFPTKJEFLUGEUKCHKKUWFUBE```임.  
문제 설명에서도 강조를 해준 알파벳은 SKATS임.  
검색해보니 ```Standard Korean Alphabet Transliteration System```라고 함.  
해독하면 ```희망은진정한기생충입니다``` 이며 이를 영어로 바꾸면 ```Hope is a true parasite```
```
flag : rtcp{Hope_is_a_true_parasite}
```
## 9. 11 -> Sebald Cipher
```
I wrote a quick script, would you like to read it? - Delphine

(doorbell rings)
delphine: Jess, I heard you've been stressed, you should know I'm always ready to help!
Jess: Did you make something? I'm hungry...
Delphine: Of course! Fresh from the bakery, I wanted to give you something, after all, you do so much to help me all the time!
Jess: Aww, thank you, Delphine! Wow, this bread smells good. How is the bakery?
Delphine: Lots of customers and positive reviews, all thanks to the mention in rtcp!
Jess: I am really glad it's going well! During the weekend, I will go see you guys. You know how much I really love your amazing black forest cakes.
Delphine: Well, you know that you can get a free slice anytime you want.
(doorbell rings again)
Jess: Oh, that must be Vihan, we're discussing some important details for rtcp.
Delphine: sounds good, I need to get back to the bakery!
Jess: Thank you for the bread! <3

Dev: Delphine

edit: This has been a source of confusion, so: the code in the first hint isn't exactly the method to solve, but is meant to give you a starting point to try and decode the script. Sorry for any confusion.
 Hint! I was eleven when I finished A Series of Unfortunate Events.
 Hint! Flag is in format: rtcp{.*}

add _ (underscores) in place of spaces.
 Hint! Character names count too
```
Link : <a hre="https://snicket.fandom.com/wiki/Sebald_Code" target="_blank">Sebald Cipher</a>
```
Sebald Cipher

ring이라는 신호가 언급되면 그 다음 단어가 첫번째 코드화된 메세지임.
첫번째 코드화 메세지부터 해서 그 뒤 11번째 단어들이 코드화된 메세지임.

Instructions for Usage

The beginning of a coded passage is signaled by the ringing, 
or mention of the ringing, of a bell. 
The first word to come after this signal is the first word of the coded message.
Every eleventh word after this first word is another part of the coded message, 
making it so that ten uncoded words fall between every coded word. 
This pattern continues until the first bell stops ringing, a second bell rings, 
or a bell's ringing is again mentioned.

Here is a simple example of the Sebald Code:

(A doorbell rings.)
Gertrude: This is a very pleasant surprise! Please come in, Bob!
Bob: How is Ebenezer?
Gertrude: He is very ill. I have to give him an injection every hour.
Bob: That's a shame. He once was an example of good health.
Gertrude: For comfort he looks at old photos of when he was healthier. 
I took him to see Dr. Sebald, but it seemed as if he was just talking in code.
(An alarm clock rings.)
Gertrude: He must need another injection. Coming, Ebenezer!
(All leave.)

When decoded, the message in this passage is: "This is an example of Sebald Code".
```
위의 예시처럼 해보면 완성된 문장이 굉장히 어색함.  
edit를 보면 script를 해독할 수 있는 출발점을 주기 위한 것이라고 함.  
따라서 다음과 같이 정리가 가능함.    
![](https://static.wixstatic.com/media/ea085c_68a67712c76547579ebabe4a1414bdd0~mv2.png/v1/fill/w_740,h_293,al_c,q_90,usm_0.66_1.00_0.01/ea085c_68a67712c76547579ebabe4a1414bdd0~mv2.webp){: .align-center}
```
flag : rtcp{i'm_hungry_give_me_bread_and_i_will_love_you}
```

## 10. .... .- .-.. ..-. 
```
Ciphertext: DXKGMXEWNWGPJTCNVSHOBGASBTCBHPQFAOESCNODGWTNTCKY

Dev: Sri
 Hint! All letters must be capitalized
 Hint! The flag must be in the format rtcp{.*}
```
제목을 해독하면 ```HALF```임. 하지만 이 말은 **Fractionated Morse Code**로 연결이 가능함. 
```
해독하면 TW0GALLONSOFH4LFMAK3WH0LEM1LK

flag : rtcp{TW0GALLONSOFH4LFMAK3WH0LEM1LK}
```

## 11. Solution 출처
<a href="https://medium.com/zh3r0/houseplant-ctf-write-up-part-2-2abc1621510a" target="_blank">https://medium.com/zh3r0/houseplant-ctf-write-up-part-2-2abc1621510a</a>
<a href="https://masrt200.github.io/hacker-blog/houseplant-ctf" target="_blank">https://masrt200.github.io/hacker-blog/houseplant-ctf</a>
<a href="https://www.zsquare.org/post/houseplant-ctf-2020-crypto-writeups" target="_blank">https://www.zsquare.org/post/houseplant-ctf-2020-crypto-writeups</a>
