---
title : "CTFlearn - [Programming] An Old Image"
categories: [Wargame, CTFlearn]
tags: [Programming, An Old Image, CTFlearn, PIL, PILLOW]
---

## An Old Image
<hr style="border-top: 1px solid;">

Link : <a href="https://ctflearn.com/challenge/1133" taget="_blank">An Old Image</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/152629511-5e2ca5f1-9f2a-4765-99de-7aa72c1610ca.png)

<br>

old image.png : 

![old_image](https://user-images.githubusercontent.com/52172169/152629547-9983fa75-c35d-47f6-8e60-333237f17c81.png)

<br>

+ 이미지를 테이블로 저장하였고 이 테이블로 이미지를 새로 만들었는데 컬럼들이 서로 섞임.

+ 컬럼에는 ```x, y red, green``` 총 4개의 컬럼이 있음.


<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;">

PIL : <a href="https://ind2x.github.io/posts/ctf_pymodule/#pillow" target="_blank">ind2x.github.io/posts/ctf_pymodule/#pillow</a>

<br>

PIL을 이용해본건 처음이었는데 PIL 라이브러리를 이용해서 풀 수 있음. 

문제에서 컬럼들이 섞였다 했으므로 섞였을 때의 모든 경우의 수들로 이미지를 만들어서 확인해보았음.

<br>

```python
from PIL import Image as ig
import numpy as np

img1 = ig.open('old_image.png')

img = img1.copy()
np_img = np.array(img) # np_img[0][0:256] ~ np_img[255][0:256] -> [r,g,b], 65536개

pixel=img.load() # pixel[x,y] = (r,g,b)

for x in range(0,256) :
	  for y in range(0,256) :
		    pixel[np_img[x][y][1], np_img[x][y][0]] = (x,y,0) # 컬럼의 순서 변경

img.save('a.png')
```

<br>

컬럼의 순서들의 경우의 수들을 코드로 정리하면 아래와 같음. 

<br>

```python
[x,y] = (r, g) -> 현재 값

[x,y] = (g, r)	-> pixel[x,y] = (np_img[x][y][1],np_img[x][y][0],0)  

[y, x] = (r, g) -> pixel[y,x] = (np_img[x][y][0],np_img[x][y][1],0)

[y, x] = (g, r) -> pixel[y,x] = (np_img[x][y][1],np_img[x][y][0],0) 

[x, r] = (y, g) or (g,y) -> pixel[x,np_img[x][y][0]] = (y,np_img[x][y][1],0) or (np_img[x][y][1],y,0)

[r, x] = (y, g) or (g, y) -> pixel[np_img[x][y][0],x] = (y,np_img[x][y][1],0) or (np_img[x][y][1],y,0)

[x, g] = (y, r) or (r, y) -> pixel[x,np_img[x][y][1]] = (y,np_img[x][y][0],0) or (np_img[x][y][0],y,0)

[g, x] = (y, r) or (r, y) -> pixel[np_img[x][y][1],x] = (y,np_img[x][y][0],0) or (np_img[x][y][0],y,0)

[y,r] = (x,g) or (g, x) -> pixel[y,np_img[x][y][0]] = (x,np_img[x][y][1],0) or (np_img[x][y][1],x,0)

[r, y] = (x,g) or (g, x) -> pixel[np_img[x][y][0],y] = (x,np_img[x][y][1],0) or (np_img[x][y][1],x,0)

[y,g] = (x,r) or (r,x) -> pixel[y,np_img[x][y][1]] = (x,np_img[x][y][0],0) or (np_img[x][y][0],x,0)

[g,y] = (x,r) or (r,x) -> pixel[np_img[x][y][1],y] = (x,np_img[x][y][0],0) or (np_img[x][y][0],x,0)

[r, g] = (x, y) or (y, x) -> pixel[np_img[x][y][0], np_img[x][y][1]] = (x,y,0) or (y,x,0)

[g, r] = (x, y) or (y, x) -> pixel[np_img[x][y][1], np_img[x][y][0]] = (x,y,0) or (y,x,0)
```

<br>

따라서 위에서부터 코드를 바꿔주면서 이미지를 확인하다보면 마지막 ```[r,g] = (x,y) or (y,x)```과 ```[g, r] = (x, y) or (y, x)``` 부분에서 QR Code가 보이게 됨.

이 QR Code를 <a href="https://products.aspose.app/barcode/recognize" target="_blank">online decoder</a>를 통해 확인하면 FLAG가 나옴.

<br>

+ Flag : ```CTFlearn{how_can_swapping_columns_hide_a_qr_code}```

<br><br>
