---
title: Python Crash
date: 2019-10-11 11:11:11 -0400
description:
categories: [00CodeNote, PythonNote]
img: /assets/img/sample/rabbit.png
tags: [Python]
---

[toc]

---


# RESTful Web Services API

REST is a programming pattern which describes how data should be transfer between client and server over the network.
- REST specifies a set of design constraints that leads to higher performance and better maintainability.
- These constraints are: client-server, stateless, cacheable, layer system, uniform interface and code-on-demand.

If your web services conform to the REST constraints, and can be used in standalone (i.e., does not need an UI), then you have a RESTful Web Service API (RESTful API). RESTful API works like a regular API, but delivers through a web server.

RESTful API is typically accessible through a URI.
- The most common scheme is to use the various HTTP request method to perform `CRUD (Create-Read-Update-Delete) database` operations.


For example:


`GET /api/user/ `
- GET request to **list ALL the users** (Database READ).
- Request data: NIL
- Response data: a list of users (JSON)
- Response Status Code for success: 200 OK
- Response Status Code for failure:
  - 401 Unauthorized (login required),
  - 403 No Permissions (role required) - these status codes are applicable to all requests.

`POST /api/user/ `
- POST request to **create a new user** (Database CREATE).
- Request data: a new user (JSON)
- Response data: URL of the created item, or auto-increment ID, or the created item (JSON)
- Response Status Code for success: 201 Created
- Response Status Code for failure:
  - 400 Bad Request (invalid or missing input),
  - 401 Unauthorized,
  - 403 No Permissions

`GET /api/user/<id>`
- GET request to **list ONE user with id** (Database READ).
- Request data: NIL
- Response data: a user (JSON)
- Response Status Code for success: 20O OK
- Response Status Code for failure:
  - 404 Not Found,
  - 401 Unauthorized,
  - 403 No Permissions

`PUT / PATCH /api/user/<id>`
- PUT or PATCH request to **update ONE user with id** (Database UPDATE)
- Request data: selected fields of a user (JSON)
- Response data: URL of the updated item, or the updated item (JSON)
- Response Status Code for success: 200 OK
- Response Status Code for failure:
  - 400 Bad Request (invalid or missing input),
  - 404 Not Found,
  - 401 Unauthorized,
  - 403 No Permissions

`DELETE /api/user/<id>`
- DELETE request to **delete ONE user with id** (Database DELETE).
- Request data: NIL
- Response data: NIL
- Response Status Code for success: 204 No Content
- Response Status Code for failure:
  - 404 Not Found,
  - 401 Unauthorized,
  - 403 No Permissions

`GET /api/user/me `
- to list the current user (Database READ).

`GET /api/course/<code>/student/ `
- to list all students of the given course code.

`POST /api/course/<code>/student/<id> `
- to add a student to the given course code.


Collections are identified by a trailing slash to give a directory representation.
- can also include a version number in the URI, `e.g., /api/1.2/user/`,
- so client can choose a suitable version,
- whereas `/api/user/` uses the latest version.

The request data (for POST, PATCH or PUT) and the response data (for GET) could use 'transport format' of `text, HTML/XML, JSON, or other formats`.
- JSON has become the most common data format, for its simplicity in representing objects (over XML) and its close ties to the client-side JavaScript programming language.

The HTTP requests could be `synchronous or asynchronous (AJAX)`.
- Again, AJAX is becoming popular for `Single-Page Architecture (SPA)`.



---


# simple simple

```py
from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def hello():
   return "Hello World!"

if __name__ == '__main__':
    app.run()
```

run

```bash
$ http GET https://127.0.0.1:5000/hello
# HTTP/1.0 200 OK
# Content-Length: 12
# Content-Type: text/html; charset=utf-8
# Date: Sun, 20 Dec 2020 21:10:30 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3

# Hello World!
```

```py
from flask import Flask, json
from flask import make_response, jsonify, request

api = Flask(__name__)

user1 = { "username": "gh0st",
          "Firstname Lastname": "William L. Simon",
          "password": "",
          "Mother’s Favorite Search Engine": "Searx"}

user2 = {  "username":"jet-setter",
          "Firstname Lastname":"Frank Abignale",
          "password":"r0u7!nG",
          "Mother’s Favorite Search Engine":"Bing"}

user3 = {  "username":"kvothe",
          "Firstname Lastname":"Patrick Rothfuss",
          "password":"3##Heel7sa*9-zRwT",
          "Mother’s Favorite Search Engine":"Duck Duck Go"}

user4 = { "username":"tpratchett",
          "Firstname Lastname":"Terry Pratchett",
          "password":"Thats Sir Terry to you!",
          "Mother’s Favorite Search Engine":"Google"}

user5 = { "username":"lmb",
          "Firstname Lastname":"Lois McMaster Bujold",
          "password":"null",
          "Mother’s Favorite Search Engine":"Yandex"}

userlist = {"gh0st":user1, "jet-setter":user2, "kvothe":user3, "tpratchett":user4, "lmb":user5}


# get the user information
@api.route('/api/userlist', methods=['GET'])
def get_userlist():
    return make_response(jsonify(userlist), 200)
#   return json.dumps(userlist)
# test:
# $ http GET https://127.0.0.1:5000/api/userlist
# HTTP/1.0 200 OK
# Content-Length: 730
# Content-Type: application/json
# Date: Sun, 20 Dec 2020 21:30:46 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {...}



# creat a new user
@api.route('/api/userlist', methods=['POST'])
def post_userlist():
    # ensure we get the response form the request
    # request:
    # {"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}
    content = request.json
    user_username = content['username']
    userlist[user_username] = content
    user_new = userlist.get(user_username, {})
    return make_response(jsonify(user_new), 201)
#   return json.dumps({"success": True}), 201
# test:
# $ echo '{"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/api/userlist
# HTTP/1.0 201 CREATED
# Content-Length: 103
# Content-Type: application/json
# Date: Sun, 20 Dec 2020 21:33:47 GMT
# Server: Werkzeug/1.0.1 Python/3.8.3
# {...}




@api.route('/api/userlist/<username>', methods=['GET'])
def get_user(user_username):
    user = userlist.get(user_username, {})
    return make_response(jsonify(user), 200)
    # if user:
    #     return make_response(jsonify(user), 200)
    # else:
    #     return make_response(jsonify({"Get failure"}), 404)
# http GET https://127.0.0.1:5000/api/userlist/a

@api.route('/api/userlist/<username>', methods=['PUT'])
def update_user(user_username):
    content = request.json
    userlist[user_username] = content
    user = userlist.get(user_username, {})
    return make_response(jsonify(user), 200)
# $ echo '{"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/api/userlist
# $ echo '{"username":"a",  "Firstname Lastname":"cd", "password":"345",  "Mother’s Favorite Search Engine":"d"}' | http PUT https://127.0.0.1:5000/api/userlist/a


@api.route('/api/userlist/<username>', methods=['DELETE'])
def delete_user(user_username):
    if user_username in userlist.keys():
        del userlist[user_username]
        return make_response(jsonify({}), 204)
    else:
        return make_response(jsonify({"DELETE failure"}), 404)

if __name__ == '__main__':
    api.run()

```

---

# simple PY flask


```py
from flask import Flask, json

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

# initialize Flask
api = Flask(__name__)

# declare a route for endpoint.
# When a consumer visits /companies using a GET request, the list of two companies will be returned.
@api.route('/companies', methods=['GET'])
def get_companies():
  return json.dumps(companies)
  # status code wasn’t required because 200 is Flask’s default.

@api.route('/companies', methods=['POST'])
def post_companies():
  return json.dumps({"success": True}), 201

if __name__ == '__main__':
    api.run()
```

run


```bash
$ python flaskapi.py
 * Serving Flask app "flaskapi" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on https://127.0.0.1:5000/ (Press CTRL+C to quit)
127.0.0.1 - - [20/Dec/2020 13:05:01] "GET /usrlist HTTP/1.1" 200 -
127.0.0.1 - - [20/Dec/2020 13:06:23] "POST /usrlist HTTP/1.1" 201 -
```

---

# Python + Flask


## setup

```bash
# Download the dataset from the Employees
# https://www.sqlitetutorial.net/sqlite-sample-database/


# extract in your project folder named 'python_rest'.
# Database name is "chinook.db"
# make a file named server.py


# create a basic virtual environment
# and install the packages after it's activation.
$ virtualenv venv
$ source venv/bin/activate
$ pip install flask flask-jsonpify flask-sqlalchemy flask-restful
$ pip freeze
```


## create the basic GET API

```py
# 1. connect yourself to database.
$ python_rest sqlite3 chinook.db

# server.py
# exposing employees data
# tracks data from database
# also add a query operator on employees where employee's details is searched and fetched by EmployeeID.
from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.ext.jsonpify import jsonify

db_connect = create_engine('sqlite:///chinook.db')
class Employees(Resource):
    def get(self):
        # connect to database
        conn = db_connect.connect()
        # performs query and returns json result
        query = conn.execute("select * from employees")
        # Fetches first column that is Employee ID
        return {'employees': [i[0] for i in query.cursor.fetchall()]}

class Tracks(Resource):
    def get(self):
        conn = db_connect.connect()
        query = conn.execute("select trackid, name, composer, unitprice from tracks;")
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)

class Employees_Name(Resource):
    def get(self, employee_id):
        conn = db_connect.connect()
        query = conn.execute("select * from employees where EmployeeId =%d "  %int(employee_id))
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)


app = Flask(__name__)
api = Api(app)
# Route_1
api.add_resource(Employees, '/employees')
# Route_2
api.add_resource(Tracks, '/tracks')
# Route_3
api.add_resource(Employees_Name, '/employees/<employee_id>')


if __name__ == '__main__':
     app.run(port='5002')


# It is simple to create a API. You can also add support to PUT,POST and DELETE on data too.
```


There will be three routes created :
- `https://127.0.0.1:5002/employees` shows ids of all the employees in database
- `https://127.0.0.1:5002/tracks` shows tracks details
- `https://127.0.0.1:5002/employees/8` shows details of employee whose employeeid is 8







---

# Marshmallow + Flask

## 1.1  Marshmallow
- using Python package [marshmallow](https://marshmallow.readthedocs.org/en/latest/) for object serialization/deserialization and field validation.

To install marshmallow:

```bash
# Activate your virtual environment
(venv)$ pip install marshmallow
(venv)$ pip show --files marshmallow
Name: marshmallow
Version: 2.12.2
Location: .../venv/lib/python3.5/site-packages
Requires:
```


```py
from datetime import date
from pprint import pprint
from marshmallow import Schema, fields

class ArtistSchema(Schema):
    name = fields.Str()

class AlbumSchema(Schema):
    title = fields.Str()
    release_date = fields.Date()
    artist = fields.Nested(ArtistSchema())

bowie = dict(name="David Bowie")
album = dict(artist=bowie, title="Hunky Dory", release_date=date(1971, 12, 17))

schema = AlbumSchema()
result = schema.dump(album)
pprint(result, indent=2)
# { 'artist': {'name': 'David Bowie'},
#   'release_date': '1971-12-17',
#   'title': 'Hunky Dory'}
```

marshmallow schemas can be used to:
- `Validate input data`.
- `Deserialize input data` to app-level objects.
- `Serialize app-level objects` to primitive Python types. The serialized objects can then be rendered to standard formats such as JSON for use in an HTTP API.

---

## 1.2 Flask RESTful API


### Example 1: Handling GET Request

```py
# resteg1_get.py:
# HTTP GET request with JSON response

import simplejson as json  # Needed to jsonify Numeric (Decimal) field
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy  # Flask-SQLAlchemy
from marshmallow import Schema

app = Flask(__name__)
app.config['SECRET_KEY'] = 'YOUR-SECRET'  # Needed for CSRF
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://testuser:xxxx@localhost:3306/testdb'
db = SQLAlchemy(app)

# Define Model mapped to table 'cafe'
class Cafe(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    category = db.Column(db.Enum('tea', 'coffee', name='cat_enum'), nullable=False, default='coffee')
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Numeric(precision=5, scale=2), nullable=False)
    # 'json' does not support Numeric; need 'simplejson'

    def __init__(self, category, name, price):
        # Constructor: id is auto_increment
        self.category = category
        self.name = name
        self.price = price

# Drop, re-create all the tables and insert records
db.drop_all()
db.create_all()
db.session.add_all([Cafe('coffee', 'Espresso', 3.19),
                    Cafe('coffee', 'Cappuccino', 3.29),
                    Cafe('coffee', 'Caffe Latte', 3.39),
                    Cafe('tea', 'Green Tea', 2.99),
                    Cafe('tea', 'Wulong Tea', 2.89)])
db.session.commit()

# We use marshmallow Schema to serialize our database records
class CafeSchema(Schema):
    class Meta:
        fields = ('id', 'category', 'name', 'price')  # Serialize these fields

item_schema = CafeSchema()            # Single object
items_schema = CafeSchema(many=True)  # List of objects

@app.route('/api/item/', methods=['GET'])
@app.route('/api/item/<int:id>', methods=['GET'])
def query(id = None):
    if id:
        item = Cafe.query.get(id)
        if item is None:
            return jsonify({'err_msg': ["We could not find item '{}'".format(id)]}), 404
        else:
            result = item_schema.dump(item)  # Serialize object
                # dumps() does not support Decimal too
                # result:
                # MarshalResult(data={'name': 'Espresso', 'id': 1, 'price': Decimal('3.19'), 'category': 'coffee'}, errors={})
            return jsonify(result.data)  # Uses simplejson
    else:
        items = Cafe.query.limit(3)  # don't return the whole set
        result = items_schema.dump(items)  # Serialize list of objects
            # Or, item_schema.dump(items, many=True)
        return jsonify(result.data)

if __name__ == '__main__':
    # Turn on debug only if launch from command-line
    app.config['SQLALCHEMY_ECHO'] = True
    app.debug = True
    app.run()
```

> Notes: In order to jsonify Decimal (or Numeric) field
> need to use `simplejson` to replace the json of the standard library.
> To install simplejson
> pip install simplejson
> import simplejson as json
> the jsonify() invokes simplejson.
> Marshmallow's dumps() does not support Decimal field too.

Try these URLs and observe the JSON data returned.
- Trace the `request/response` messages using web browser's developer web console.

```
GET request: https://localhost:5000/api/item/
GET request: `https://localhost:5000/api/item/1`
GET request: https://localhost:5000/api/item/6
```


### Example 2: AJAX/JSON

```py
# resteg2_ajax.py:
# AJAX request with JSON response
import simplejson as json  # Needed to support 'Decimal' field
from flask import Flask, jsonify, request, render_template, abort
from flask_sqlalchemy import SQLAlchemy
from marshmallow import Schema

app = Flask(__name__)
app.config['SECRET_KEY'] = 'YOUR-SECRET'  # Needed for CSRF
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://testuser:xxxx@localhost:3306/testdb'
db = SQLAlchemy(app)

# Define Model mapped to table 'cafe'
class Cafe(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    category = db.Column(db.Enum('tea', 'coffee', name='cat_enum'), nullable=False, default='coffee')
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Numeric(precision=5, scale=2), nullable=False)

    def __init__(self, category, name, price):
        # Constructor: id is auto_increment
        self.category = category
        self.name = name
        self.price = price

# Drop, re-create all the tables and insert records
db.drop_all()
db.create_all()
db.session.add_all([Cafe('coffee', 'Espresso', 3.19),
                    Cafe('coffee', 'Cappuccino', 3.29),
                    Cafe('coffee', 'Caffe Latte', 3.39),
                    Cafe('tea', 'Green Tea', 2.99),
                    Cafe('tea', 'Wulong Tea', 2.89)])
db.session.commit()

# We use marshmallow Schema to serialize our database records
class CafeSchema(Schema):
    class Meta:
        fields = ('id', 'category', 'name', 'price')  # Serialize these fields

item_schema = CafeSchema()            # Single object
items_schema = CafeSchema(many=True)  # List of objects

@app.route("/api/item/", methods=['GET'])
@app.route("/api/item/<int:id>", methods=['GET'])
def query(id = None):
    if id:
        item = Cafe.query.get(id)
        if request.is_xhr:  # AJAX?
            # Return JSON
            if item is None:
                return jsonify({"err_msg": ["We could not find item '{}'".format(id)]}), 404
            else:
                result = item_schema.dump(item)
                return jsonify(result.data)

        else:
            # Return a web page
            if item is None:
                abort(404)
            else:
                return render_template('resteg2_ajax_query.html')

    else:  # if id is None
        items = Cafe.query.limit(3)  # don't return the whole set
        if request.is_xhr:
            # Return JSON
            result = items_schema.dump(items)
            return jsonify(result.data)
        else:
            # Return a web page
            return render_template('resteg2_ajax_query.html')

if __name__ == '__main__':
    # Debug only if running through command line
    app.config['SQLALCHEMY_ECHO'] = True
    app.run(debug=True)
```

```html
<!-- templates/resteg2_ajax_query.html -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Cafe Query</title>
</head>
<body>
  <h1>Cafe Query</h1>
  <ul id="items"></ul>

<script src="https://code.jquery.com/jquery-1.11.2.min.js"></script>
<script>
// Send AJAX request after document is fully loaded
$(document).ready(function() {
   $.ajax({
      // default url of current page and method of GET
   })
      .done( function(response) {
         $(response).each(function(idx, elm) {
            $("#items").append("<li>" + elm['category'] + ", " + elm['name'] + ", $" + elm['price'] + "</li>");
         })
      });
});
</script>
</body>
</html>
```

> There are two requests for each URL. The initial request is a regular HTTP GET request, which renders the template, including the AJAX codes, but without any items.
> When the page is fully loaded, an AJAX request is sent again to request for the items, and placed inside the document.
> Turn on the web browser's developer console to trace the `request/response` messages.
> Also, try using firefox's plug-in 'HttpRequester' to trigger a AJAX GET request.




## 1.3  Flask-RESTful Extension
Reference: [Flask-RESTful](https://flask-restful-cn.readthedocs.org/en/0.3.4/).

Flask-RESTful is an extension for building REST APIs for Flask app, which works with your existing ORM.

**Installing Flask-RESTful**
```bash
# Activate your virtual environment
(venv)$ pip install flask-restful
Successfully installed aniso8601-1.2.0 flask-restful-0.3.5 python-dateutil-2.6.0 pytz-2016.10

(venv)$ pip show flask-restful
Name: Flask-RESTful
Version: 0.3.5
Summary: Simple framework for creating REST APIs
Requires: Flask, aniso8601, pytz, six
```

**Flask-Restful Example 1: Using Flask-Restful Extension**

```py
# frestful_eg1:
# Flask-Restful Example 1 - Using Flask-Restful Extension
from flask import Flask, abort
from flask_restful import Api, Resource

class Item(Resource):
    # For get, update, delete of a particular item via URL /api/item/<int:item_id>.
    def get(self, item_id):
        return 'reading item {}'.format(item_id), 200

    def delete(self, item_id):
        return 'delete item {}'.format(item_id), 204  # No Content

    def put(self, item_id):  # or PATCH
        # Request data needed for update
        return 'update item {}'.format(item_id), 200

class Items(Resource):
    # For get, post via URL /api/item/, meant for list-all and create new.
    def get(self):
        return 'list all items', 200

    def post(self):
        # Request data needed for create
        return 'create a new post', 201  # Created

app = Flask(__name__)
api_manager = Api(app)
# Or,
#api_manager = Api()
#api_manager.init_app(app)

api_manager.add_resource(Item, '/api/item/<item_id>', endpoint='item')
api_manager.add_resource(Items, '/api/item/', endpoint='items')
# endpoint specifies the view function name for the URL route


if __name__ == '__main__':
    app.run(debug=True)
```

define two URLs:
- `/api/item/` for get-all and create-new via GET and POST methods;
- and `/api/item/<item_id>` for `get, update, delete via GET, PUT, and DELETE` methods.

We extend the Resource class to support all these methods.
In this example, we did not use an actual data model.

To send POST/PUT/DELETE requests, you can use
- the command-line curl (which is rather hard to use);
- or browser's extension such as Firefox's HttpRequester,
- or Chrome's Advanced REST client, user-friendly graphical interface.

For example, use Firefox's HttpRequester to send the following requests:
- GET request to `https://localhost:5000/api/item/` to list all items.
- GET request to `https://localhost:5000/api/item/1` to list one item.
- POST request to `https://localhost:5000/api/item/` to create a new item.
- PUT request to `https://localhost:5000/api/item/1` to update one item.
- DELETE request to `https://localhost:5000/api/item/1` to delete one item.


**Sending AJAX-POST/PUT/PATCH/DELETE HTTP Requests**
When you enter a URL on a web browser, an HTTP GET request is sent.
You can send a POST request via an HTML Form.
There are a few ways to test PUT/PATCH/DELETE/Ajax-POST requests:

1. Via web browser's plug-in such as Firefox's HttpRequester.
2. Via client-side script in JavaScript/jQuery/AngularJS
3. Via Flask client, e.g.,

4. Via the curl command, e.g.,

```bash
# Show manual page
$ man curl
... manual page ...
# Syntax is: $ curl options url

# Send GET request
# reading item 1
$ curl --request GET https://localhost:5000/api/item/1

# Send DELETE request. To include the response header
$ curl --include --request DELETE https://localhost:5000/api/item/1
# HTTP/1.0 204 NO CONTENT
# Content-Type: application/json
# Content-Length: 0
# Server: Werkzeug/0.11.15 Python/3.5.2
# Date: Thu, 16 Mar 2017 02:44:11 GMT

# Send PUT request, with json data and additional header
# update item 1
$ curl --include --request PUT --data '{"price":"9.99"}'
       --Header "Content-Type: application/json" https://localhost:5000/api/item/1
# HTTP/1.0 200 OK
# Content-Type: application/json
# Content-Length: 16
# Server: Werkzeug/0.11.15 Python/3.5.2
# Date: Thu, 16 Mar 2017 03:00:43 GMT
```

---





ref
- [Python Developing Web Applications with Flask](https://www3.ntu.edu.sg/home/ehchua/programming/webprogramming/Python3_Flask.html#zz-8.)
- [Build a Python REST API Server for Quick Mocking](https://stoplight.io/blog/python-rest-api/)
- [good: youtube](https://www.youtube.com/watch?v=iQVvpnRYl-w&list=PLVUt86LpSCWgEe5EZsRomCTcwUH5dP-2x&index=3&ab_channel=MVPEngineer)
- [authentication](https://github.com/MaBlaGit/REST_API_Flask)
- [RESTful Setup Instructions](https://www.se.rit.edu/~swen-344/03/projects/rest-setup/)
- [RESTful API Project](https://www.se.rit.edu/~swen-344/03/projects/rest/)
- [Designing a RESTful API to interact with a simple SQLite database](https://subscription.packtpub.com/book/application_development/9781786462251/1/ch01lvl1sec7/designing-a-restful-api-to-interact-with-a-simple-sqlite-database)
- [mobile](https://github.com/OWASP/owasp-mstg/blob/master/Document/0x05d-Testing-Data-Storage.md)
- [final!!!](https://www.youtube.com/watch?v=MAt-mRJk4rw&list=PLS1QulWo1RIZ6OujqIAXmLR3xsDn_ENHI&index=2&ab_channel=ProgrammingKnowledge)
- [youtubcc](https://www.youtube.com/c/PrettyPrintedTutorials/search?query=flask)
- [Flask-Login: Remember Me and Fresh Logins](https://www.youtube.com/watch?v=CRvV9nFKoPI&ab_channel=PrettyPrinted)
- [Password Hashing with Flask Tutorial](https://pythonprogramming.net/password-hashing-flask-tutorial/)
- [youtu-6!!!](https://www.youtube.com/watch?v=8PPvgexhmYg&list=PLS1QulWo1RIZ6OujqIAXmLR3xsDn_ENHI&index=6&ab_channel=ProgrammingKnowledge)
- [Flask Rest API -Part:2- Better Structure with Blueprint and Flask-restful #flask #python #mongodb #beginners](https://dev.to/paurakhsharma/flask-rest-api-part-2-better-structure-with-blueprint-and-flask-restful-2n93)
- [youtuve Python Projects | Flask REST API with Sqlite Database in 100 lines](https://www.youtube.com/watch?v=Sf-7zXBB_mg&list=PLVUt86LpSCWgEe5EZsRomCTcwUH5dP-2x&index=2&ab_channel=MVPEngineer)










.
