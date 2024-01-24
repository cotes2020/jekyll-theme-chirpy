
# REST API for User
This is a RESTful API that supports 4 operations:
- adding a user,
- authenticating,
- retrieving user details,
- and logging out.

```
movie-bag
│   app.py
|   Pipfile
|   Pipfile.lock
└───database
    │   db.py
    └───models.py
```

## Active endpoints

### `/api/userlist/`


`GET /api/userlist/`
- GET request to **list ALL the users** (Database READ).
- Request data: NIL
- Response data: a list of users (JSON)
- Response Status Code for success:
  - 200 OK
- Response Status Code for failure:
  - 404 Not Found

`POST /api/userlist/`
- POST request to **create a new user** (Database CREATE).
- Request data: `a new user (JSON)`
- Response data: URL of the created item, or the created item (JSON)
- Response Status Code for success:
  - 201 Created
- Response Status Code for failure:
  - 400 Bad Request


### `/api/user/<username>`

`GET /api/user/<username>`
- GET request to **list ONE user with username** (Database READ).
- Request data: NIL
- Response data: a user (JSON)
- Response Status Code for success:
  - 20O OK
- Response Status Code for failure:
  - 400 Bad Request

`PUT /api/user/<username>`
- PUT request to **update ONE user with username** (Database UPDATE)
- Request data: selected fields of a user (JSON)
- Response data: URL of the updated item, or the updated item (JSON)
- Response Status Code for success:
  - 200 OK
- Response Status Code for failure:
  - 400 Bad Request

`DELETE /api/user/<user_name>`
- DELETE request to **delete ONE user with username** (Database DELETE).
- Response Status Code for success:
  - 204: no content for the user anymore.


---

## Steps to run app and tests:
1. Install Python
2. Create virtualenv
3. Go to root folder of the project, install required modules:

```bash
$ pip3 install dataset
$ pip3 install flask
$ pip3 install flask-bcrypt
```

4. To run app:

> 4.1. the user data is in a text file (actully pre-defined inside the python code)

```bash
# run the api
$ python api.py

# check all userlist
$ http GET https://127.0.0.1:5000/api/userlist
# add a user to userlist
$ echo '{"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/api/userlist
http GET https://127.0.0.1:5000/api/userlist/

# check userlist with username
$ http GET https://127.0.0.1:5000/api/userlist/a
# update userlist with username
$ echo '{"username":"a",  "Firstname Lastname":"cd", "password":"345",  "Mother’s Favorite Search Engine":"d"}' | http PUT https://127.0.0.1:5000/api/userlist/a
# delete userlist with username
$ http DELETE https://127.0.0.1:5000/api/userlist/a
```

> 4.2. the user data is in a sqlite db

```bash
# run the api
$ python apidb.py

# create the default user database
$ http GET https://127.0.0.1:5000/api/creatdefaultuser

# check all userlist
$ http GET https://127.0.0.1:5000/api/userlist
# add a user to userlist
$ echo '{"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/api/userlist
http GET https://127.0.0.1:5000/api/userlist/

# check userlist with username
$ http GET https://127.0.0.1:5000/api/userlist/a
# update userlist with username
$ echo '{"username":"a",  "Firstname Lastname":"cd", "password":"345",  "Mother’s Favorite Search Engine":"d"}' | http PUT https://127.0.0.1:5000/api/userlist/a
# delete userlist with username
$ http DELETE https://127.0.0.1:5000/api/userlist/a
```

> You can also use the Postman to test the api
