---
title: Python Package
date: 2019-10-11 11:11:11 -0400
description:
categories: [00CodeNote, PythonNote]
img: /assets/img/sample/rabbit.png
tags: [Python]
---

- [Python Package](#python-package)
  - [1 example](#1-example)


---


# Python Package


```
main.py
mypackage/
    __init__.py
    mymodule.py
    myothermodule.py
```



...a `mymodule.py` like this...

```py
#!/usr/bin/env python3

# Exported function
def as_int(a):
    return int(a)

# Test function for module
def _test():
    assert as_int('1') == 1

if __name__ == '__main__':
    _test()
```


## 1 example

- `main.py`: good

```py
#!/usr/bin/env python3

from mypackage.myothermodule import add

def main():
    print(add('1', '1'))

if __name__ == '__main__':
    main()

```

- `myothermodule.py`: fail

```py
#!/usr/bin/env python3

from .mymodule import as_int

# Exported function
def add(a, b):
    return as_int(a) + as_int(b)

# Test function for module
def _test():
    assert add('1', '1') == 2

if __name__ == '__main__':
    _test()
```


- The way you're supposed to run it is...
- `python3 -m mypackage.myothermodule`


solution:
- assuming the name `mymodule` is globally unique
  - would be to avoid using relative imports, and just use...
  - `from mymodule import as_int`

- if it's not unique, or your package structure is more complex,
  - include the directory containing your package directory in PYTHONPATH, and do it like this...
  - `from mypackage.mymodule import as_int`



.
