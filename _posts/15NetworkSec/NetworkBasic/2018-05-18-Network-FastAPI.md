---
title: NetworkSec - Fast API
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec, NetworkBasic]
tags: [NetworkSec, TCP]
math: true
# pin: true
toc: true
---

ref:
- https://github.com/tiangolo/fastapi/blob/master/README.md
- https://fastapi.tiangolo.com/tutorial/

---

# NetworkSec - Fast API


<p align="center">
  <a href="https://fastapi.tiangolo.com"><img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI"></a>
</p>
<p align="center">
    <em>FastAPI framework, high performance, easy to learn, fast to code, ready for production</em>
</p>

---

**Documentation**: https://fastapi.tiangolo.com

**Source Code**: https://github.com/tiangolo/fastapi

---

## overall

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

---

## Requirements

Python 3.7+

FastAPI stands on the shoulders of giants:

- [Starlette](https://www.starlette.io/) for the web parts.
- [Pydantic](https://pydantic-docs.helpmanual.io/) for the data parts.

---

## Installation


```bash
$ pip install fastapi
```

You will also need an ASGI server, for production such as [Uvicorn](https://www.uvicorn.org) or [Hypercorn](https://github.com/pgjones/hypercorn).


```bash
$ pip install "uvicorn[standard]"
```

---

## Example

### Create it

* Create a file `main.py` with:

```Py
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

<details markdown="1">

<summary>Or use <code>async def</code>...</summary>

If your code uses `async` / `await`, use `async def`:

```Py
from typing import Union

from fastapi import FastAPI

app = FastAPI()

# http://127.0.0.1:8000/
@app.get("/")
def read_root():
    return {"Hello": "World"}

# http://127.0.0.1:8000/items/5?q=somequery.
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

**Note**:

If you don't know, check the _"In a hurry?"_ section about <a href="https://fastapi.tiangolo.com/async/#in-a-hurry" target="_blank">`async` and `await` in the docs</a>.

</details>



---

### Run it

Run the server with:

```bash
$ uvicorn main:app --reload

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [28720]
INFO:     Started server process [28722]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```


<details markdown="1">
<summary>About the command <code>uvicorn main:app --reload</code>...</summary>

The command `uvicorn main:app` refers to:

* `main`: the file `main.py` (the Python "module").
* `app`: the object created inside of `main.py` with the line `app = FastAPI()`.
* `--reload`: make the server restart after code changes. Only do this for development.

</details>

---

### Check it

Open your browser at http://127.0.0.1:8000/items/5?q=somequery.

You will see the JSON response as:

```JSON
{"item_id": 5, "q": "somequery"}
```

You already created an API that:

* Receives HTTP requests in the _paths_ `/` and `/items/{item_id}`.
* Both _paths_ take `GET` operations (HTTP _methods_).
* The _path_ `/items/{item_id}` has a _path parameter_ `item_id` that should be an `int`.
* The _path_ `/items/{item_id}` has an optional `str` _query parameter_ `q`.

---

## API docs

### Interactive API docs

http://127.0.0.1:8000/docs
- automatic interactive API documentation (provided by Swagger UI):

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

---

### Alternative API docs

http://127.0.0.1:8000/redoc
- alternative automatic documentation (provided by ReDoc):

![ReDoc](https://fastapi.tiangolo.com/img/index/index-02-redoc-simple.png)

---

## Example upgrade

Now modify the file `main.py` to receive a body from a `PUT` request.

Declare the body using standard Python types, thanks to Pydantic.

```Py
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

# http://127.0.0.1:8000/
@app.get("/")
def read_root():
    return {"Hello": "World"}

# http://127.0.0.1:8000/items/5?q=somequery.
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
```

The server should reload automatically (because you added `--reload` to the `uvicorn` command above).

**FastAPI** will:

* Validate that `item_id` in the path for `GET` and `PUT` requests.
* Validate that `item_id` is of type `int` for `GET` and `PUT` requests.
  * If it is not, the client will see a useful, clear error.

* Check if there is an optional query parameter named `q` for `GET` requests.
  * As the `q` parameter is declared with `= None`, it is optional.
  * Without the `None` it would be required (as is the body in the case with `PUT`).

* For `PUT` requests to `/items/{item_id}`, Read the body as JSON:
  * Check that it has a required attribute `name` that should be a `str`.
  * Check that it has a required attribute `price` that has to be a `float`.
  * Check that it has an optional attribute `is_offer`, that should be a `bool`, if present.
  * All this would also work for deeply nested JSON objects.
* Convert from and to JSON automatically.

* Document everything with OpenAPI, that can be used by:
  * Interactive documentation systems.
  * Automatic client code generation systems, for many languages.
* Provide 2 interactive documentation web interfaces directly.


---


### Interactive API docs upgrade

http://127.0.0.1:8000/docs

* The interactive API documentation will be automatically updated, including the new body:

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-03-swagger-02.png)

* "Try it out": fill the parameters and directly interact with the API:

![Swagger UI interaction](https://fastapi.tiangolo.com/img/index/index-04-swagger-03.png)

* "Execute": communicate with your API, send the parameters, get the results and show them on the screen:

![Swagger UI interaction](https://fastapi.tiangolo.com/img/index/index-05-swagger-04.png)

---

### Alternative API docs upgrade

http://127.0.0.1:8000/redoc

* The alternative documentation will also reflect the new query parameter and body:

![ReDoc](https://fastapi.tiangolo.com/img/index/index-06-redoc-02.png)

---

### command

For example, for an `int`:

```Py
item_id: int
```

or for a more complex `Item` model:

```Py
item: Item
```

...and with that single declaration you get:

* Editor support, including:
  * Completion.
  * Type checks.
* Validation of data:
  * Automatic and clear errors when the data is invalid.
  * Validation even for deeply nested JSON objects.
* <abbr title="also known as: serialization, parsing, marshalling">Conversion</abbr> of input data: coming from the network to Python data and types. Reading from:
  * JSON.
  * Path parameters.
  * Query parameters.
  * Cookies.
  * Headers.
  * Forms.
  * Files.
* <abbr title="also known as: serialization, parsing, marshalling">Conversion</abbr> of output data: converting from Python data and types to network data (as JSON):
  * Convert Python types (`str`, `int`, `float`, `bool`, `list`, etc).
  * `datetime` objects.
  * `UUID` objects.
  * Database models.
  * ...and many more.
* Automatic interactive API documentation, including 2 alternative user interfaces:
  * Swagger UI.
  * ReDoc.
