---
title : "CTFlearn - [Crypto] Vigenere Cipher"
categories: [Wargame, CTFlearn]
tags: [Crypto, Vigenere Cipher]
---

## Vigenere Cipher
<hr style="border-top: 1px solid;">

<br>

![image](https://user-images.githubusercontent.com/52172169/152632490-cccd8fc9-d49d-4204-9871-ee50fd5dc3b9.png)

<br>

```
key : blorpy

Cipher : gwox{RgqssihYspOntqpxs}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;">

<br>

```python
c='gwox{RgqssihYspOntqpxs}'
key='blorpy'
idx=0

pt=''

for i in c:
	if i == '{' or i == '}':
		pt+=i
		continue
	
	capital = i.isupper()
	diff = ord(key[idx]) - ord('a')
	k=key[idx]
	idx=(idx+1)%len(key)

	decode = ord(i)-diff
	if decode < 97 and capital != 1: 
		decode+=26

	pt+=chr(decode)

print(pt)
```

<br><br>
