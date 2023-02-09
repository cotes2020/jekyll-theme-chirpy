---
layout: post
title: "[FIA] Cryptography Technical Test"
summary: "Mật mã"
author: technical
date: '2023-02-08 9:00:00'
category: CTF
thumbnail: assets/img/thumbnail/crypto.jpeg
keywords: Hacking, CTF
permalink: /blog/Cryptography_techtest/
usemathjax: true
---

# Cryptography Technical Test

# Bory Cipher
###### Description: 
*Bory has created his own super secure cipher. To create this masterpiece, he worked lots of hours at the University with his consultant Mr Vignere. His mentor give him the advice. Mix as many as Cipher you known and a little bit magic. That is called **"FIA"**.*

At the start, we have a text file BoryCipher.txt with the content:
```
簴簴簴簴簴簴簴簴簴簴籤籇簴籇簴簴簴籇簴簴簴簴簴簴簴籇簴簴簴簴簴簴簴簴簴簴籅籅籅籅簶籦籇籇籇簴簴簴簴簴簴簴簴簴簴簴簴簴簷籅簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籇簶簶簶簶簶簶簶簶簶簶簶簶簶簷簶簶簶簶簷籇簴簷籅籅簷籇籇簴簴簴簷簴簴簷簶簶簶簶簶簶簶簷籅簴簴簷簴簴簴簴簴簴簴簴簴簴簷籅簴簴簷籇簴簴簴簴簴簴簴簴簴簴簷籅簶簶簶簷籇籇簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籅籅簴簷籇簴簴簴簴簴簴簴簴簴簴簴簴簷簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簷簶簶簶簶簶簶簷籅簴簴簴簴簴簷籇簴簴簴簴簴簴簴簴簴簴簷籅簶簶簶簶簶簶簷籇簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簷籇簶簷簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簷籅籅簴簴簷籇籇簷簴簴簷籅簴簴簴簴簴簴簴簴簴簴簴簴簷籇簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籅簶簶簶簶簶簶簶簷籇簶簶簶簶簶簶簶簶簶簶簶簷籅簴簴簴簴簴簴簴簴簴簴簴簷籅簴簷籇簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簷籇簴簴簴簴簴簴簴簴簴簴簴簷籅簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籅簶簶簶簷籇簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籇簶簶簷籅簶簶簷籅簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簴簷籇簴簴簷籇簴簴簷籅簶簶簶簷籅簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簶簷簶簶簶簷簴簴簴簴簴簴簴簴簴簴簴簴簴簷
```

Let's check its cipher type with Cipher Identifier (https://www.dcode.fr/cipher-identifier)

<p><img class="article-img" src="/assets/img/CTF/2023TechnicalTest/identify1.png" alt="Header.png" width="1053" height="606"></p>

And we got "ROT8000 Cipher". Let's decode it then we receive a text with many symbols that look a bit messy.
```
++++++++++[>+>+++>+++++++>++++++++++<<<<-]>>>+++++++++++++.<+++++++++++++++++++.>-------------.----.>+.<<.>>+++.++.-------.<++.++++++++++.<++.>++++++++++.<---.>>+++++++++++++++++++++.<<+.>++++++++++++.----------------.------.<+++++.>++++++++++.<------.>---------------.>-.-------------------.<<++.>>.++.<++++++++++++.>++++++++++++++++++++.<-------.>-----------.<+++++++++++.<+.>----------------.>+++++++++++.<+++++++++++++++.<---.>+++++++++++++++.>--.<--.<+++++++++++++++++++++++.>++.>++.<---.<--------------------.---.+++++++++++++.
```

If you've researched, it's easy to see that it is BrainFuck - code that was designed to hurt. But if you haven't heard of it, no problem, let's keep using Cipher Identifier.

<p><img class="article-img" src="/assets/img/CTF/2023TechnicalTest/identify2.png" alt="Header.png" width="1053" height="606"></p>

Not so surprised. Let's run it. Then we got the output.
```
S1FBe1hjcDN3X0x1dTN6X0Iwd2dfUzNoY3IzX0gxeGgzd30=
```

It seems to be base64 encoded, but let's check it for sure.

<p><img class="article-img" src="/assets/img/CTF/2023TechnicalTest/identify3.png" alt="Header.png" width="1053" height="606"></p>

Continue decoding it. We got
```
KQA{Xcp3w_Luu3z_B0wg_S3hcr3_H1xh3w}
```
It's quite similar to the flag pattern but not correct, the head must be FIA. Continue to identify it. We got a list of cipher types, but it seems to be not good.

<p><img class="article-img" src="/assets/img/CTF/2023TechnicalTest/identify4.png" alt="Header.png" width="1053" height="606"></p>

Read the description again and we got **Mr Vignere**. It is like a type of cipher in the list above. Let's try it. After run **Automatic decryption**, we also can't get the flag. What is wrong? The Vignere Cipher needs a key/password to encrypt/decrypt. **Automatic decryption** is a brute force attack to find the exact key/password but it is unreliable, and if you have researched it, the success rate of it is very low. Check the description again. What is special? **FIA** in double quote. It may be the key. Check it. 

<p><img class="article-img" src="/assets/img/CTF/2023TechnicalTest/result.png" alt="Header.png" width="1053" height="606"></p>

Successfully, we got the flag.
```
FIA{Sup3r_Dup3r_B0ry_S3cur3_C1ph3r}
```


# Strange Base
###### Description
*Do you know that Base encode method have many type ? From the most common one is Base64. Some others Base can mention like: Base32, Base45, Base58, Base62, Base85, etc. Furthermore, each type of Base have a mount of ways to implement its. You should have a little bit research about this.
Can you break down this Strange Base challenge ?*

At the start, we have a python file with a dict, an ecoded function and encoded string. Now we analysic how encode fuction work. 

- Firstly, it converts each character in the input string to binary 8 bit then concentrate all of it and store to **s**. 
- Secondly, the code check the length of **s** string, if it modulo 6 equal 4 then assign string "=" to **pad** variable and add "11" to **s**. Vice versa if it divides by 6 leaving the remainder 4 then assign "\=\=" to **pad** and add "1111" to **s**. 
- Thirdly, it divides the string with the length of each equal to 6 and uses the **base** dict to get the value of each key as the string had divided. Then concentrate all part into **ret** string variable.
- Finally, add the **pad** to **ret** had been reversed and return it.

Now we need to decode the encoded string to get the flag.
```
TTcilRoPcTnOd3sMlTsvdlcvh7sp9ToKdPnPc5dyZ5dSERu1BAh=
```

What must we do? Reverse the encoded function and we can get the decode function.
The code for this can seem be like that:
```
#!/usr/bin/python3
from chall import base
encoded = "TTcilRoPcTnOd3sMlTsvdlcvh7sp9ToKdPnPc5dyZ5dSERu1BAh="
# Swap key and value of base dict
base = dict(zip(base.values(), base.keys()))

import re

def decode(string):
    equal_count = len(re.findall("=", string))
    string = re.sub(r"=.*", "", string)
    encoded_rev = string[::-1]

    ret = ""
    for i in encoded_rev:
        ret += base[i]

    if equal_count == 1:
        ret = ret[:-1]

    bin_text = ""
    flag = ""
    for i in ret:
        if len(bin_text) == 8:
            flag += chr(int(bin_text, 2))
            bin_text = ""
        bin_text += i

    print(flag)

decode(encoded)
```

After run this code, we get the flag
```
FIA{Z64_b4S3_3ncrYpt_1S_supp3r_s3cUR3}
```


# AES and RSA
###### Description
Split your brain into two parts and decrypt these two encryption alg.

At the start, we have a python file. Let's analysis it.

- Firstly, it divides the flag into 2 part. 
- Secondly, convert the first part to long **c1**
	- **cipher1 = c1 ** e % n**
	- Its mean is: **c1 to the power of e divided by n get remainder**
	- It  is a subproblem of RSA encrypt.
- Thirdly, AES encrypt second part with 
```
key = b'borylikesunshine'
iv = b'someoneisunshine'
```
then encode with base64.

Now, try to solve it,
The first part - subproblem of RSA encrypt, we have to find 2 prime numbers (p, q) whose multiplication is equal to n. But it is too big number, let's use factordb (http://factordb.com)

After found p and q, run this code.
```
from Crypto.Util.number import long_to_bytes
from Crypto.Util.number import inverse


N = 5891773551966962003993799117509614436456325750025065237059

# http://factordb.com/index.php?query=882564595536224140639625987659416029426239230804614613279163
p = 1 // p get from factordb
q = 1 // q get from factordb

e = 65537

phi = (p - 1) * (q - 1)

d = inverse(e, phi)  # d = e^(-1) MOD phi

c = 2672883185209710100976590695140651434215965781515258009988

# uses his private key (N,d) to compute m = c^d mod N

plain = pow(c, d, N)

print(long_to_bytes(plain).decode())

```

And we get the first part of flag
```
4r3_Fun!!!!}
```

AES encrypt, we've got everything we need, just decrypt it.
```
import base64
from Crypto.Cipher import AES

key = b'borylikesunshine'
iv = b'someoneisunshine'

c2 = AES.new(key, AES.MODE_CBC, iv)

cipher2 = "zNl9vs+tZaM7IIYL2/EvdRLT0VehWU+/26CGJCaAmA8="

msg2 = c2.decrypt(base64.b64decode(cipher2))

print(msg2.decode())
```
We get the result
```
FIA{RSA_4nd_AES_►►►►►►►►►►►►►►►►
```

The flag is
```
FIA{RSA_4nd_AES_4r3_Fun!!!!}
```

### Author: nquangit