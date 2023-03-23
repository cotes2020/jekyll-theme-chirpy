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


# use python to working with APIs

Communication with external services is an integral part of any modern system. Whether it's a payment service, authentication, analytics or an internal one - **systems need to talk to each other**.

**In this short article we are going to implement a module for communicating with a made-up payment gateway, step by step.**


```py
# todo.py
import requests

def _url(path):
    return 'https://todo.example.com' + path

def get_tasks():
    return requests.get(_url('/tasks/'))

def describe_task(task_id):
    return requests.get(_url('/tasks/{:d}/'.format(task_id)))

def add_task(summary, description=""):
    return requests.post(_url('/tasks/'), json={'summary': summary,'description': description,})

def task_done(task_id):
    return requests.delete(_url('/tasks/{:d}/'.format(task_id)))

def update_task(task_id, summary, description):
    url = _url('/tasks/{:d}/'.format(task_id))
    return requests.put(url, json={'summary': summary,'description': description,})
```

to use it

```py
import todo

resp = todo.add_task("Take out trash")
if resp.status_code != 201:
    raise ApiError('Cannot create task: {}'.format(resp.status_code))
print('Created task. ID: {}'.format(resp.json()["id"]))

resp = todo.get_tasks()
if resp.status_code != 200:
    raise ApiError('Cannot fetch all tasks: {}'.format(resp.status_code))
for todo_item in resp.json():
    print('{} {}'.format(todo_item['id'], todo_item['summary']))




# GET, != 200, raise ApiError()

# Post, != 201, raise ApiError()



def refund(transaction_id):
    # Refund charged transaction.
    # transaction_id (str):
    # Transaction id to refund.
    # Raises:
    # Return (RefundResponse)

    try:
        data = make_payment_request('/refund', {
            'uid': str(uuid.uuid4()),
            'transaction_id': transaction_id,
        })

    except requests.HTTPError as e:
        # TODO: Handle refund remote errors


    except (requests.ConnectionError, requests.Timeout) as e:
        raise errors.Unavailable() from e

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error = e.response.json()
            code = error['code']
            message = error['message']

        if code == 1:
            raise errors.Refused(code, message) from e
        elif code == 2:
            raise errors.Stolen(code, message) from e
        else:
            raise errors.PaymentGatewayError(code, message) from e

        logger.exception("Payment service had internal error.")
        raise errors.Unavailable() from e

```




---

## The External Service

Let's start by defining an imaginary payment service.

To charge a credit card we need a `credit card token`, an `amount to charge` and `unique ID by client`:

```html
POST
{
    token: <string>,
    amount: <number>,
    uid: <string>,
}
```


If the charge was successful we get a `200 OK status` with the `data from our request`, an `expiration time for the charge` and a `transaction ID`:

```html
200 OK
{
    uid: <string>,
    amount: <number>,
    token: <string>,
    expiration: <string, isoformat>,
    transaction_id: <number>
}
```


If the charge was not successful we get a 400 status with an `error code` and an `informative message`:

```html
400 Bad Request
{
    uid: <string>,
    error: <number>,
    message: <string>
}
```


There are two error codes we want to handle

```
- 1 = refused,
2 = stolen.
```



## Native Implementation

To get the ball rolling, we start with a naive implementation and build from there:

```py
# payments.py

import uuid
import requests

PAYMENT_GATEWAY_BASE_URL = 'https://gw.com/api'
PAYMENT_GATEWAY_TOKEN = 'topsecret'

def charge(amount,token,timeout=5):

    # Charge.
    # amount (int):
    #     Amount in cents to charge.
    # token (str):
    #     Credit card token.
    # timeout (int):
    #     Timeout in seconds.
    # Returns (dict):
    # New payment information.

    headers = {"Authorization": "Bearer " + PAYMENT_GATEWAY_TOKEN}

    payload = {
        "token": token,
        "amount": amount,
        "uid": str(uuid.uuid4()),
    }

    response = requests.post(
        PAYMENT_GATEWAY_BASE_URL + '/charge',
        json=payload,
        headers=headers,
        timeout=timeout,
    )

    # POST
    # {
    #     token: <string>,
    #     amount: <number>,
    #     uid: <string>,
    # }

    response.raise_for_status()

    return response.json()
```


## Handling Errors

There are two types of errors we need to handle:
* HTTP errors such as connection errors, timeout or connection refused.
* Remote payment errors such as refusal or stolen card.

Our decision to use `requests` is an internal implementation detail. The consumer of our module shouldn't have to be aware of that.

**To provide a complete API our module must communicate errors.**



Let's start by defining `custom error classes`:


```py
# errors.py

class Error(Exception):
    pass

class Unavailable(Error):
    pass

class PaymentGatewayError(Error):
    def __init__(self, code, message):
        self.code = code
        self.message = message

class Refused(PaymentGatewayError):
    pass

class Stolen(PaymentGatewayError):
    pass
```

add exception handling and logging to our function:

```py
import logging
from . import errors

logger = logging.getLogger('payments')

def charge(amount,token,timeout=5):

    # ...

    try:
        response = requests.post(
            PAYMENT_GATEWAY_BASE_URL + '/charge',
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

    except (requests.ConnectionError, requests.Timeout) as e:
        raise errors.Unavailable() from e

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error = e.response.json()
            code = error['code']
            message = error['message']

        if code == 1:
            raise errors.Refused(code, message) from e
        elif code == 2:
            raise errors.Stolen(code, message) from e
        else:
            raise errors.PaymentGatewayError(code, message) from e

        logger.exception("Payment service had internal error.")
        raise errors.Unavailable() from e
```


Great! Our function no longer raises `requests` exceptions.
- Important errors such as stolen card or refusal are raised as custom exceptions.

---


## Defining the Response

Our function returns a dict. A dict is a great and flexible data structure, but when you have a defined set of fields you are better off using a more targeted data type.

In every OOP class you learn that everything is an object. While it is true in Java land, Python has a lightweight solution that works better in our case - [**namedtuple**]

A namedtuple is just like it sounds, a tuple where the fields have names. You use it like a class and it consumes less space (even compared to a class with slots).


define a namedtuple for the charge response:

```py
from collections import namedtuple
ChargeResponse = namedtuple('ChargeResponse', ['uid','amount','token','expiration','transaction_id',])
```


If the charge was successful, we create a `ChargeResponse` object:

```py
from datetime import datetime

# ...

def charge(amount,token,timeout=5):

    # ...

    data = response.json()

    charge_response = ChargeResponse(
        uid=uuid.UID(data['uid']),
        amount=data['amount'],
        token=data['token'],
        expiration=datetime.strptime(data['expiration'], "%Y-%m-%dT%H:%M:%S.%f"),
        transaction_id=data['transaction_id'],
    )

    return charge_response
```


Our function now returns a `ChargeResponse` object. Additional processing such as casting and validations can be added easily.

In the case of our imaginary payment gateway, we convert the expiration date to a datetime object. The consumer doesn't have to guess the date format used by the remote service (when it comes to date formats I am sure we all encountered a fair share of horrors).

By using a custom "class" as the return value we reduce the dependency in the payment vendor‘s serialization format. If the response was an XML, would we still return a dict? That's just awkward.



---



## Using a Session

To skim some extra milliseconds from API calls we can use a session. [Requests session] uses a connection pool internally.
- Requests to the same host can benefit from that.
- We also take the opportunity to add useful configuration such as blocking cookies:


```py
import http.cookiejar

# A shared requests session for payment requests.
class BlockAll(http.cookiejar.CookiePolicy):
    def set_ok(self, cookie, request):
        return False

payment_session = requests.Session()
payment_session.cookies.policy = BlockAll()

# ...

def charge(amount,token,timeout=5):
    # ...
    response = payment_session.post( ... )
    # ...
```


---



## More Actions

Any external service, and a payment service in particular, has more than one action.

- The first section of our function takes care of `authorization, the request and HTTP errors`.
- The second part handle `protocol errors and serialization specific to the charge action`.
- The first part is relevant to all actions while the second part is specific only to the charge.


Let's split the function so we can reuse the first part:

```py
import uuid
import logging
import requests
import http.cookiejar
from datetime import datetime

logger = logging.getLogger('payments')

class BlockAll(http.cookiejar.CookiePolicy):
    def set_ok(self, cookie, request):
        return False

payment_session = requests.Session()
payment_session.cookies.policy = BlockAll()

def make_payment_request(path, payload, timeout=5):
    # Make a request to the payment gateway.
    # path (str): Path to post to.
    # payload (object): JSON-serializable request payload.
    # timeout (int): Timeout in seconds.
    # Raises
    #     Unavailable
    #     requests.exceptions.HTTPError
    #     Returns (response)

    headers = {
        "Authorization": "Bearer " + PAYMENT_GATEWAY_TOKEN,
    }

    try:
        response = payment_session.post(
            PAYMENT_GATEWAY_BASE_URL + path,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        raise errors.Unavailable() from e

    response.raise_for_status()
    return response.json()


def charge(amount, token):
    # Charge credit card.
    # amount (int): Amount to charge in cents.
    # token (str): Credit card token
    # Raises
    #     Unavailable
    #     Refused
    #     Stolen
    #     PaymentGatewayError
    # Returns (ChargeResponse)

    try:
        data = make_payment_request('/charge', {
            'uid': str(uuid.uuid4()),
            'amount': amount,
            'token': token,
        })

    except requests.HTTPError as e:
        if e.response.status_code == 400:
            error = e.response.json()
            code = error['code']
            message = error['message']

            if code == 1:
                raise Refused(code, message) from e

            elif code == 2:
                raise Stolen(code, message) from e

            else:
                raise PaymentGatewayError(code, message) from e

        logger.exception("Payment service had internal error")
        raise errors.Unavailable() from e

    return ChargeResponse(
        uid=uuid.UID(data['uid']),
        amount=data['amount'],
        token=data['token'],
        expiration=datetime.strptime(data['expiration'], "%Y-%m-%dT%H:%M:%S.%f"),
        transaction_id=data['transaction_id'],
    )
```


**This is the entire code.**

There is a clear separation between "transport", serialization, authentication and request processing. We also have a well defined interface to our top level function `charge`.

To add a new action we define a new return type, call `make_payment_request` and handle the response the same way:


```py
RefundResponse = namedtuple('RefundResponse', ['transaction_id','refunded_transaction_id',])

def refund(transaction_id):
    # Refund charged transaction.
    # transaction_id (str):
    # Transaction id to refund.
    # Raises:
    # Return (RefundResponse)

    try:
        data = make_payment_request('/refund', {
            'uid': str(uuid.uuid4()),
            'transaction_id': transaction_id,
        })

    except requests.HTTPError as e:
        # TODO: Handle refund remote errors

    return RefundResponse(
        'transaction_id': data['transaction_id'],
        'refunded_transaction_id': data['refunded_transaction_id'],
    )
```

---

## Testing

The challenge with external APIs is that you can't (or at least, shouldn't) make calls to them in automated tests. I want to focus on **testing code that uses our payments module** rather than testing the actual module.

Our module has a simple interface so it's easy to mock. Let's test a made up function called `charge_user_for_product`:


```py
# test.py

from unittest import TestCase
from unittest.mock import patch

from payment.payment import ChargeResponse
from payment import errors

def TestApp(TestCase):

    @mock.patch('payment.charge')
    def test_should_charge_user_for_product(self, mock_charge):
        mock_charge.return_value = ChargeResponse(
            uid='test-uid',
            amount=1000,
            token='test-token',
            expiration=datetime.datetime(2017, 1, 1, 15, 30, 7),
            transaction_id=12345,
        )
        charge_user_for_product(user, product)
        self.assertEqual(user.approved_transactions, 1)

    @mock.patch('payment.charge')
    def test_should_suspend_user_if_stolen(self, mock_charge):
        mock_charge.side_effect = errors.Stolen
        charge_user_for_product(user, product)
        self.assertEqual(user.is_active, False)
```

Pretty straight forward - no need to mock the API response. The tests are contained to data structures we defined ourselves and have full control of.



---



## Note About Dependency Injection

Another approach to test a service is to provide two implementations: the real one, and a fake one. Then for tests, inject the fake one.

This is of course, how dependency injection works. Django doesn't do DI but it utilizes the same concept with "backends" (email, cache, template, etc). For example you can test emails in django by using a test backend, test caching by using in-memory backend, etc.

This also has other advantages in that you can have multiple "real" backends.

Whether you choose to mock the service calls as illustrated above or inject a "fake" service, you must have a proper interface.


---


## Summary

We have an external service we want to use in our app.
- We want to implement a module to communicate with that external service and make it robust, resilient and reusable.

We worked the following steps:
1. **Naive implementation** - Fetch using requests and return a json response.
2. **Handled errors** - Defined custom errors to catch both transport and remote application errors. The consumer is indifferent to the transport (HTTP, RPC, Web Socket) and implementation details (requests).
3. **Formalize the return value** - Used a `namedtuple` to return a class-like type that represents a response from the remote service. The consumer is now indifferent to the serialization format as well.
4. **Added a session** - Skimmed off a few milliseconds from the request and added a place for global connection configuration.
5. **Split request from action** - The request part is reusable and new actions can be added more easily.
6. **Test** - Mocked calls to our module and replaced them with our own custom exceptions.



---


# Example: to-do list API


```json
// GET /tasks/
// Return a list of items on the to-do list, in the following format:
{
    "id": "<item_id>",
    "summary": "<one-line summary>"
}


// GET /tasks/<item_id>/
// Fetch all available information for a specific to-do item, in the following format:
{
    "id": "<item_id>",
    "summary": "<one-line summary>",
    "description" : "<free-form text field>"
}


// POST /tasks/
// Create a new to-do item. The POST body is a JSON object with two fields:
// “summary” (must be under 120 characters, no newline),
// and “description” (free-form text field).
// On success,
// the status code is 201,
// and the response body is an object with one field: the id created by the server (for example, { "id": 3792 }).



// DELETE /tasks/<item_id>/
// Mark the item as done: strike it off the list so that GET /tasks/ will not show it.
// The response body is empty.



// PUT /tasks/<item_id>/
// Modify an existing task.
// The PUT body is a JSON object with two fields:
// summary (must be under 120 characters, no newline), and description (free-form text field).


// Note: Unless otherwise noted, all actions return 200 on success; those referencing a task ID return 404 if the ID is not found. The response body is empty unless specified otherwise. All non-empty response bodies are JSON. All actions that take a request body are JSON (not form-encoded).
```


---



## code

use `HTTP library: Kenneth Reitz’ requests`

- primary tool for writing Python code to use REST APIs
- or any service exposed over HTTP

```bash
# Step one for every Python app that talks over the web
$ pip install requests
```

---


### get a list of action items, via the `GET /tasks/ endpoint`:

```py
import requests

# function called to get the HTTP GET.
resp = requests.get('https://todolist.example.com/tasks/')

# This means something went wrong.
if resp.status_code != 200:
    raise ApiError('GET /tasks/ {}'.format(resp.status_code))

# and transforms it into a Python list of dictionaries by json.loads().
for todo_item in resp.json():
    print('{} {}'.format(todo_item['id'], todo_item['summary']))
```


---


### create a new task: add something to my to-do list.


In our API, this requires an `HTTP POST`. I start by creating a Python dictionary with the required fields, “summary” and “description”, which define the task.


```py
# start by creating a Python dictionary with the required fields, “summary” and “description”
# Convert that into a JSON representation string by json.dumps()
task = {"summary": "Take out trash", "description": "" }

# post function takes a json argument
resp = requests.post('https://todolist.example.com/tasks/', json=task)

if resp.status_code != 201:
    raise ApiError('POST /tasks/ {}'.format(resp.status_code))

print('Created task. ID: {}'.format(resp.json()["id"]))


# If you are using something other than JSON (some custom format, XML, or everybody’s favorite, YAML) then you need to do this manually, which is a bit more work. Here’s how it looks:

# The shortcut
resp = requests.post('https://todolist.example.com/tasks/', json=task)
# The equivalent longer version
resp = requests.post('https://todolist.example.com/tasks/', data=json.dumps(task), headers={'Content-Type':'application/json'}）
```




---

## Constructing an API Library

- If you are doing anything more than a few API calls
- if you are the one providing the API and want to develop that library so others can easily use your service.

The structure of the library depends on how the API authenticates, if it does at all.
- For the moment, let’s ignore authentication, to get the basic structure.
- Then we’ll look at how to install the auth layer.


```py
# Let’s start with the simplest thing that could possibly work.
# todo.py
import requests

def _url(path):
    return 'https://todo.example.com' + path

def get_tasks():
    return requests.get(_url('/tasks/'))

def describe_task(task_id):
    return requests.get(_url('/tasks/{:d}/'.format(task_id)))

def add_task(summary, description=""):
    return requests.post(_url('/tasks/'), json={'summary': summary,'description': description,})

def task_done(task_id):
    return requests.delete(_url('/tasks/{:d}/'.format(task_id)))

def update_task(task_id, summary, description):
    url = _url('/tasks/{:d}/'.format(task_id))
    return requests.put(url, json={'summary': summary,'description': description,})
```

to use it

```py
import todo

resp = todo.add_task("Take out trash")
if resp.status_code != 201:
    raise ApiError('Cannot create task: {}'.format(resp.status_code))
print('Created task. ID: {}'.format(resp.json()["id"]))

resp = todo.get_tasks()
if resp.status_code != 200:
    raise ApiError('Cannot fetch all tasks: {}'.format(resp.status_code))
for todo_item in resp.json():
    print('{} {}'.format(todo_item['id'], todo_item['summary']))
```






---

ref:
- [Working With APIs the Pythonic Way](https://hakibenita.com/working-with-apis-the-pythonic-way)
