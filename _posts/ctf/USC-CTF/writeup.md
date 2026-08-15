----------------

### CTF: USC-CTF 2024

----------------

![image](https://github.com/user-attachments/assets/8100ab24-3d91-40ad-9444-187bd84e076a)

----------------

### CHALLENGES-:

- Beginner
  - Colors
  - Weird Traffic
  - Irobots
- Web
  - Tommy's Adventures
  - Tictactocket
- Forensics
  - Think_twice

-----------------------

### Colors-:

![image](https://github.com/user-attachments/assets/363fdb80-9b17-4d30-8a73-932dfa731458)

-----------------------

- The challenge file contains this base64 encoded string.

```MzAgMzEgMzAgMzAgMzAgMzAgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzAgMzAgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzAgMzAgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzAgMzAgMzAgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzEgMzAgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzAgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzEgMzAgMzAgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzAgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzEgMzEgMjAgMzAgMzAgMzEgMzEgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzEgMzEgMzEgMjAgMzAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzEgMzEgMzEgMzAgMzE=```

- After I decoded it ,I got  base 16 encoded value which after decoding reveals the next string in binary.I got the flag after decoding the binary text.

```python3
#! /usr/bin/env python3
import base64

base64_encoded: str = base64.b64decode("MzAgMzEgMzAgMzAgMzAgMzAgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzAgMzAgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzAgMzAgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzAgMzAgMzAgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzEgMzAgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzAgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzEgMzAgMzAgMzEgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzEgMzEgMzAgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzEgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzEgMzEgMjAgMzAgMzAgMzEgMzEgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzEgMzAgMzAgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMzEgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzEgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzAgMzEgMzEgMzEgMzEgMzEgMjAgMzAgMzEgMzAgMzAgMzAgMzEgMzEgMzEgMjAgMzAgMzAgMzEgMzEgMzAgMzAgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzEgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzAgMzAgMzEgMzAgMzAgMjAgMzAgMzEgMzEgMzEgMzEgMzEgMzAgMzE=")
print(f"[+] Base16: {base64_encoded.decode()}")
result: list =  base64_encoded.decode().split(" ")

def solve_base16(i: str):
    base2 = str(chr(int(i,16)))
    return base2
def solve_base2(i: str) -> str:
    return chr(int(i,2))

base2 = ''.join(solve_base16(str(i)) for i in result).split(' ')
print(f"[+] Base2: {base2}")
flag = ''.join(solve_base2(i) for i in base2)
print(flag)
```

- Flag-:```CYBORG{tR0jans_love_C4rdinal_@nd_G0ld}```

![image](https://github.com/user-attachments/assets/a514cf3b-b3b1-44cd-acfe-1c62cb207577)

--------------------

### Weird Traffic

--------------------

![image](https://github.com/user-attachments/assets/ec7e0554-5942-4e67-8ca8-456898d2e555)

--------------------

- The challenge requires analyzing a pcapng file. The required tool for this challenge is wireshark.
- Flag-:```CYBORG{hping3-is-a-cool-tool}```

![image](https://github.com/user-attachments/assets/e5784792-0568-4464-b624-7cc991cdfc68)

-----------------

### Web: iRobots

-----------------

![image](https://github.com/user-attachments/assets/8ac57d6f-ade0-455f-9726-1d492fd61966)

----------------

- As the name implies, my first stop was the `robots.txt`.The flag's path was disallowed in the robots.txt disallowed entry.

![image](https://github.com/user-attachments/assets/a1dfb010-2768-4f10-af36-b9eee16fd489)

- Flag-:```CYBORG{robots_txt_is_fun}```

![image](https://github.com/user-attachments/assets/7b7c7f45-efed-4e0d-bafc-0117c8f605d2)

-----------------

### Tommy's ArtAdventures

-----------------

![image](https://github.com/user-attachments/assets/6a034677-2688-4611-8afb-31bd0e0eedcd)

-----------------

- The challenge's secret key fo signing Flask's session was given.This is a major vulnerability because flask session's key is used to sign cookies for a flask web application.

![image](https://github.com/user-attachments/assets/2c6b3b72-4b21-4448-a498-4490f921f648)

- After checking the source code of the app, I noticed an endpoint `/curate` that gives the fla only to the admin user.

![image](https://github.com/user-attachments/assets/e304d48f-850f-403f-8445-a1932fd61f74)

- I used `flask-unsign` to sign a cookie with user `admin`.

![image](https://github.com/user-attachments/assets/3b0079a9-56e1-4386-9682-c37bb5b1fe01)

- Flag-:```CYBORG{oce4n5_auth3N71ca7i0N}```

------------------

### Tictatocket

------------------

![image](https://github.com/user-attachments/assets/15902355-20ae-4649-a68c-fa907ed9d8ba)

------------------

- This is a tictactoe game that applies websockets to transmit data.

![image](https://github.com/user-attachments/assets/ca311f70-16cc-4e48-9806-213060e6be39)

- I discovered that players can play in the computer position if they spam clicks in the position the computer is about to place its square.

![image](https://github.com/user-attachments/assets/eb08cc77-96b4-4947-886c-9a801377f95a)

- Flag-:```CYBORG{S3RVER_W45_0VERTRUST1NG}```

---------------------

### Think_Twice

---------------------

![image](https://github.com/user-attachments/assets/8c0d7d57-4e13-4d4e-9235-6144099453e7)

---------------------

- The challenge file contains a png file `metadata.png`.
- I checked for the exifdata with `exiftool` and got a base6 encoded string.

![image](https://github.com/user-attachments/assets/f262f1ea-a1cb-411e-a9ad-4c3ea34cd810)

- I double decoded it with base64 and got the flag.

![image](https://github.com/user-attachments/assets/bd8acb9a-d9b0-4742-bcd9-530887dbcea7)

- Flag-:```Cyb0rg{McCarthy}```


--------------------

### THANKS FOR READING !!!!

--------------------


