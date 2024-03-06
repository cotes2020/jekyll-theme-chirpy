
# REST API for User
This is a RESTful API that supports 4 operations:
- adding a user,
- authenticating,
- retrieving user details,
- and logging out.

```
code
│   README.md
└───code3
│   │   __init__.py
│   |   auth.py
│   |   main.py
│   |   models.py
│   |   db.sqlite
│   └───templates
│       │   index.html
│       │   login.html
│       │   profile.html
│       └───signup.html
│
│
└───code-noAuth (another RESTful API that supports some operations)
```

---

## Endpoints

> `/adduser` **adding a user**


`GET /adduser`
- GET request to get the adduser.html
- Request data: NIL
- Response data: a JSON or redirect url to adduser.html
- Response Status Code for success:
  - 200 OK
- Response Status Code for failure:
  - 404 Not Found

`POST /adduser`
- POST request to create a new user (Database CREATE).
- Request data: a new user (JSON)
- Response data: the created item (JSON)
- Response Status Code for success:
  - 201 Created
- Response Status Code for failure:
  - 400 Bad Request (inval username)


> `/login` **authenticating**

`GET /login`
- GET request to get the login.html
- Request data: NIL
- Response data: a JSON or redirect url to login.html
- Response Status Code for success:
  - 20O OK
- Response Status Code for failure:
  - 404 Not Found

`POST /login`
- GET request to authenticate a new user
- Request data: a username and password (JSON)
- Response data: the login status and user info (JSON)
- Response Status Code for success:
  - 200 logged in
- Response Status Code for failure:
  - 401 Unauthorized



> `/logout` **logged out**

`GET /logout`
- logout the current user
- Request data: NIL
- Response data: the login status (JSON)
- Response Status Code for success:
  - 200 logged in
- Response Status Code for failure:
  - 403 No Permissions



> `/userinfo` **retrieving user details**

`GET /userinfo`
- retrieving the current user information
- Request data: NIL
- Response data: the current login user info (JSON)
- Response Status Code for success:
  - 200 current user logged in
- Response Status Code for failure:
  - 401 Unauthorized



---


## Steps to run app and tests:

1. Install Python

2. Create virtualenv

3. required modules:

```bash
$ pip3 install dataset
$ pip3 install flask
$ pip3 install flask-bcrypt
$ pip3 install flask_sqlalchemy
$ pip3 install flask_login
$ pip3 install werkzeug
```

4. Go to root folder of the project, run app:

```bash
# the current database db.sqlite already have the test data

# -------------- if you want to create your own database (optional) --------------
$ python3
>>> from code3 import db, create_app
>>> db.create_all(app=create_app())
>>> exit()


# -------------- under the code folder --------------
$ export FLASK_APP=code3
$ flask run


# insert the test data, if you delete the current exsisted database db.sqlite (optional)
$ echo '{"username": "gh0st", "Firstname Lastname": "William L. Simon", "password": "", "Mother’s Favorite Search Engine": "Searx"}' | http POST https://127.0.0.1:5000/signup
$ echo '{"username":"jet-setter", "Firstname Lastname":"Frank Abignale","password":"r0u7!nG", "Mother’s Favorite Search Engine":"Bing"}' | http POST https://127.0.0.1:5000/signup
$ echo '{"username":"kvothe", "Firstname Lastname":"Patrick Rothfuss","password":"3##Heel7sa*9-zRwT", "Mother’s Favorite Search Engine":"Duck Duck Go"}' | http POST https://127.0.0.1:5000/signup
$ echo '{"username":"tpratchett", "Firstname Lastname":"Terry Pratchett","password":"Thats Sir Terry to you!", "Mother’s Favorite Search Engine":"Google"}' | http POST https://127.0.0.1:5000/signup
$ echo '{"username":"lmb", "Firstname Lastname":"Lois McMaster Bujold","password":"null", "Mother’s Favorite Search Engine":"Yandex"}' | http POST https://127.0.0.1:5000/signup
$ echo '{"username":"a",  "Firstname Lastname":"x", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/signup




# -------------- test the /adduser, add user --------------
$ echo '{"username":"a",  "Firstname Lastname":"x", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/adduser
# HTTP/1.0 200 OK
# Content-Length: 73
# Content-Type: application/json
# Date: Mon, 21 Dec 2020 08:43:13 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {
#     "status": 200,
#     "new_user" : {
#                     "username":"a",
#                     "Firstname Lastname":"x",
#                     "password":"123",
#                     "Mother’s Favorite Search Engine":"c"
#                     }
# }
$ echo '{"username":"ab",  "Firstname Lastname":"doubleuser", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/adduser
# HTTP/1.0 400 BAD REQUEST
# Content-Length: 73
# Content-Type: application/json
# Date: Mon, 21 Dec 2020 08:43:13 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {
#     "Add user": false,
#     "Error Message": "username not available or already exsisted",
#     "status": 400
# }
$ echo '{"username":"newusernotindatabseyet",  "Firstname Lastname":"x", "password":"123sufueiwbryilsdifbe",  "Mother’s Favorite Search Engine":"bingo"}' | http POST https://127.0.0.1:5000/adduser





# -------------- test the /login, authenticate the user --------------
echo '{"username":"a", "password":"123"}' | http POST https://127.0.0.1:5000/login
# HTTP/1.0 200 OK
# Content-Length: 75
# Content-Type: application/json
# Date: Mon, 21 Dec 2020 08:41:09 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# Set-Cookie: session=.eJwdzjsOwjAMANC7ZGaIHdtxepkq_kSwtnRC3J2K6a3vU_Z15Pks2_u48lH2V5StGJioaG2QudTIV1Y1r8I0ktEGkXlAq7NPX-SygqmRAyfpYtOAQUmzoo5-2zCAu2ln5mhOMCdghqqzKAGM6ANB5kQyJNByR64zj_9Gvj9qzC5q.X-BfpQ.tL0UcpeDR2YxnFFAPuA17bhNUso; HttpOnly; Path=/
# Vary: Cookie
# {
#     "Successful login": true,
#     "current_user.is_authenticated": true,
#     "status": 200,
#     "userinfo": {
#                     "Firstname Lastname": "x",
#                     "Mother’s Favorite Search Engine": "c",
#                     "username": "a"
#                 }
# }
echo '{"username":"a", "password":"wrongpasswd"}' | http POST https://127.0.0.1:5000/login
# HTTP/1.0 401 UNAUTHORIZED
# Content-Length: 53
# Content-Type: application/json
# Date: Mon, 21 Dec 2020 08:41:28 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {
#     "reason": "Username or Password Error",
#     "status": 401
# }




# -------------- test the /logout, logging out --------------
$ http GET https://127.0.0.1:5000/logout
# redirect url to the main.index




# -------------- test the /userinfo, retrieving user details --------------
$ http GET https://127.0.0.1:5000/userinfo
# HTTP/1.0 201 CREATED
# Content-Length: 108
# Content-Type: application/json
# Date: Mon, 21 Dec 2020 08:46:46 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {
#     "Firstname Lastname": "William L. Simon",
#     "Mother’s Favorite Search Engine": "Searx",
#     "username": "gh0st"
# }
```


---
