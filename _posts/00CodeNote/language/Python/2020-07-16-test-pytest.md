---
title: Python Pytest
date: 2020-07-16 11:11:11 -0400
description:
categories: [00CodeNote, PythonNote]
img: /assets/img/sample/rabbit.png
tags: [Python]
---

- [pytest](#pytest)
  - [overview](#overview)
    - [运行 pytest](#运行-pytest)
  - [basic](#basic)
    - [first test](#first-test)
    - [pytest 跳过](#pytest-跳过)
    - [pytest 标记](#pytest-标记)
    - [Pytest 参数化测试](#pytest-参数化测试)
    - [pytest 夹具](#pytest-夹具)
    - [testing Exceptions](#testing-exceptions)
    - [context-sensitive comparisons](#context-sensitive-comparisons)
  - [Pytest 布局](#pytest-布局)
    - [综合测试](#综合测试)
    - [外部测试](#外部测试)
- [pytest-mock](#pytest-mock)
  - [unittest.mock](#unittestmock)
  - [常见用法](#常见用法)
    - [定义返回值 `mock.return_value`](#定义返回值-mockreturn_value)
    - [定义变化的返回值 `side_effect=xxx`](#定义变化的返回值-side_effectxxx)
    - [定义抛出异常 `side_effect = KeyError`](#定义抛出异常-side_effect--keyerror)
    - [mock 变量/属性](#mock-变量属性)
    - [mock dict](#mock-dict)
  - [常用检查方法](#常用检查方法)
    - [检查调用次数](#检查调用次数)
    - [`.called`: 是否被调用过](#called-是否被调用过)
    - [`.call_count`: 获取调用次数](#call_count-获取调用次数)
    - [.assert_called():](#assert_called)
    - [`.assert_called_once()`: 确保调用过一次](#assert_called_once-确保调用过一次)
    - [`.assert_not_called()`:](#assert_not_called)
  - [检查调用时使用的参数¶](#检查调用时使用的参数)
    - [.call_args: 最后调用时的参数](#call_args-最后调用时的参数)
    - [`.assert_called_once_with(*args, **kwargs): `](#assert_called_once_withargs-kwargs-)
    - [`.assert_any_call(*args, **kwargs)`: 检查某次用特定参数进行过调用](#assert_any_callargs-kwargs-检查某次用特定参数进行过调用)
    - [`.assert_called_with(*args, **kwargs)`: 检查最后一次调用时使用的参数](#assert_called_withargs-kwargs-检查最后一次调用时使用的参数)
    - [.call_args_list: 所有调用时使用的参数列表](#call_args_list-所有调用时使用的参数列表)
    - [.assert_has_calls(calls, any_order=False):](#assert_has_callscalls-any_orderfalse)
    - [.method_calls: mock 对象的方法调用记录](#method_calls-mock-对象的方法调用记录)
    - [.mock_calls:](#mock_calls)
  - [手动重置 mock 调用记录](#手动重置-mock-调用记录)

---

# pytest


## overview


Pytest 是用于测试 Python 应用的 Python 库。 它是鼻子测试和单元测试的替代方法。


使用以下命令安装 Pytest：

    $ pip install pytest

### 运行 pytest

pytest 不带任何参数，将查看当前工作目录（或其他一些预配置的目录）以及测试文件的所有子目录，并运行找到的测试代码。

```bash
# 运行当前目录中的所有测试文件。
$ pytest

# 通过指定名称作为参数来运行特定的测试文件。
$ pytest min_max_test.py

# 通过在`::`字符后提供其名称来运行特定功能。
$ pytest min_max_test.py::test_min

# 标记可用于对测试进行分组。 然后使用`pytest -m`运行一组标记的测试。
$ pytest -m smoke

# 使用表达式来运行与测试函数和类的名称匹配的测试。
$ pytest -k <expression>
```

---

## basic

### first test

```py
# $ vim text_1.py
def func(x):
    return x+1

def test_answer():
    assert func(3) == 5

# ---------
# $ py.test test_1.py
# =================== FAILURES ===================
# _________________ test_answer __________________

#     def test_answer():
# >       assert func(3) == 5
# E       assert 4 == 5
# E        +  where 4 = func(3)

# tests/test_1.py:5: AssertionError
# =========== short test summary info ============
# FAILED tests/test_1.py::test_answer - assert ...
# ============== 1 failed in 0.05s ===============
```


```py
# algo.py
def max(values):
  _max = values[0]
  for val in values:
      if val > _max:
          _max = val
  return _max
def min(values):
  _min = values[0]
  for val in values:
      if val < _min:
          _min = val
  return _min

# min_max_test.py
#!/usr/bin/env python3
import algo
def test_min():
    values = (2, 3, 1, 4, 6)
    val = algo.min(values)
    assert val == 1
def test_max():
    values = (2, 3, 1, 4, 6)
    val = algo.max(values)
    assert val == 6

# ---
# $ pytest min_max_test.py
# == test session starts ====
# platform win32 -- Python 3.7.0, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
# rootdir: C:\Users\Jano\Documents\pyprogs\pytest
# collected 2 items
# min_max_test.py  [100%]
# == 2 passed in 0.03 seconds ==

```



---

### pytest 跳过

使用跳过装饰器，我们可以跳过指定的测试。
- 跳过测试有多种原因。
- 例如，数据库/在线服务目前不可用，或者我们跳过了 Windows 上针对 Linux 的特定测试。

```py
# skipping.py

    #!/usr/bin/env python3
    import algo
    import pytest

    @pytest.mark.skip
    def test_min():
        values = (2, 3, 1, 4, 6)
        val = algo.min(values)
        assert val == 1

    def test_max():
        values = (2,
        val = algo.max(values)
        assert val == 6

# 在示例中，`test_min()`被跳过。

#     $ pytest min_max_test.py
#     ===== test session starts ====
#     platform win32 -- Python 3.7.0, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
#     rootdir: C:\Users\Jano\Documents\pyprogs\pytest
#     collected 2 items
#     min_max_test.py s.   [100%]
#     = 1 passed, 1 skipped in 0.04 seconds =
```
在测试文件名后面的输出中，s 代表跳过的和。 通过。


---


### pytest 标记

使用标记将测试组织为单元。

```py
# marking.py

    #!/usr/bin/env python3
    # pytest -m a marking.py
    # pytest -m b marking.py

    import pytest

    @pytest.mark.a
    def test_a1():
        assert (1) == (1)

    @pytest.mark.a
    def test_a2():
        assert (1, 2) == (1, 2)

    @pytest.mark.a
    def test_a3():
        assert (1, 2, 3) == (1, 2, 3)

    @pytest.mark.b
    def test_b1():
        assert "falcon" == "fal" + "con"

    @pytest.mark.b
    def test_b2():
        assert "falcon" == f"fal{'con'}"

# 两组由标记 a 和 b 标识的测试。
# 这些单元由`pytest -m a marking.py`和`pytest -m b marking.py`运行。
```

---

### Pytest 参数化测试

- 通过参数化测试，我们可以向断言中添加多个值。
- 使用`@pytest.mark.parametrize`标记。

```py
# parametrized.py

    #!/usr/bin/env python3

    import algo
    import pytest

    @pytest.mark.parametrize("data, expected", [
        ((2, 3, 1, 4, 6), 1),
        ((5, -2, 0, 9, 12), -2),
        ((200, 100, 0, 300, 400), 0)])
    def test_min(data, expected):
        val = algo.min(data)
        assert val == expected

    @pytest.mark.parametrize("data, expected", [
        ((2, 3, 1, 4, 6), 6),
        ((5, -2, 0, 9, 12), 12),
        ((200, 100, 0, 300, 400), 400)])
    def test_max(data, expected):
        val = algo.max(data)
        assert val == expected

# 使用多个输入数据测试这两个功能。

    @pytest.mark.parametrize("data, expected", [
        ((2, 3, 1, 4, 6), 1),
        ((5, -2, 0, 9, 12), -2),
        ((200, 100, 0, 300, 400), 0)])
    def test_min(data, expected):
        val = algo.min(data)
        assert val == expected

# 我们将两个值传递给测试函数：数据和期望值。 在我们的例子中，我们用三个数据元组测试`min()`函数。

    # $ pytest parametrized.py
    # ==== test session starts ====
    # platform win32 -- Python 3.7.0, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
    # rootdir: C:\Users\Jano\Documents\pyprogs\pytest
    # collected 6 items
    # parametrized.py ..... [100%]
    # = 6 passed in 0.03 seconds ==

Pytest 输出告知有六次运行。
```


---

### pytest 夹具

- 测试需要在一组已知对象的背景下进行。 这组对象称为测试夹具。

Fixtures
- the purpose of test fixtures is to provide a fixed baseline
    - upon which tests can reliably and repeatedly execute.
- fixtures have explicit names
- and are activated by declaring their use from test function, modules, classes or whole projects. <- dependency injection

```py
# algo.py
# 向`algo.py`模块添加了一个选择排序算法。

    def sel_sort(data):
      if not isinstance(data, list):
          vals = list(data)
      else:
          vals = data
      size = len(vals)

      for i in range(0, size):
          for j in range(i+1, size):
              if vals[j] < vals[i]:
                  _min = vals[j]
                  vals[j] = vals[i]
                  vals[i] = _min
      return vals
    ...


# `fixtures.py`
# 我们用夹具测试选择排序。

    #!/usr/bin/env python3
    import algo
    import pytest

    @pytest.fixture
    def data():
        return [3, 2, 1, 5, -3, 2, 0, -2, 11, 9]
    # 我们的测试装置仅返回一些测试数据。
    # 请注意，我们通过其名称引用此灯具：`data`。

    def test_sel_sort(data):
        sorted_vals = algo.sel_sort(data)
        assert sorted_vals == sorted(data)
    # 在`test_sel_sort()`函数中，我们将数据夹具作为函数参数传递。

#     $ pytest fixtures.py
#     ===== test session starts =====
#     platform win32 -- Python 3.7.0, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
#     rootdir: C:\Users\Jano\Documents\pyprogs\pytest
#     collected 1 item
#     fixtures.py  [100%]
#     == 1 passed in 0.02 seconds ===
```

```py
# vim test_4.py
import pytest
class Person:
    def greet(self):
        return "hello, there!"

@pytest.fixture
def person():
    return Person()

def test_greet(person):
    greeting = person.greet()
    assert greeting == "hi, there!"
# pytest see test_greet needs a function argument named person
# find matching fixture-marked function named person.
# person() is called to create an instance
# test_greet(<Person instance) is called.

# ---------------------------
# $ py.test tests/test_4.py
# =================== FAILURES ===================
#     def test_greet(person):
#         greeting = person.greet()
# >       assert greeting == "hi, there!"
# E       AssertionError: assert 'hello, there!' == 'hi, there!'
# E         - hi, there!
# E         ?  ^
# E         + hello, there!
# E         ?  ^^^^
# tests/test_4.py:13: AssertionError
# =========== short test summary info ============
# FAILED tests/test_4.py::test_greet - Assertio...
# ============== 1 failed in 0.07s ===============
```

---

### testing Exceptions

```py
# $ vim test_sysexit.py
import pytest
def f():
    raise SystemExit(1)
def test_mytest():
    with pytest.raises(SystemExit):
        f()  # see if this will raise the Exception
# if raise the same Exception
# this test will pass.

# ---------------------------
# $ py.test tests/test_2.py
# tests/test_2.py .                        [100%]
# ============== 1 passed in 0.01s ===============
```

---

### context-sensitive comparisons

will tell the different.

```py
# $ vim text_3.py
def test_answer():
    assert set(['0','1','2']) == set(['0','2','3'])
    assert 'foo1' == 'foo2'
    assert {'a':0,'b':1,'c':0} == {'a':0,'b':0,'d':0}

# ---------------------------
# $ py.test tests/test_3.py
# =================== FAILURES ===================
# _________________ test_answer __________________
#     def test_answer():
# >       assert set(['0','1','2']) == set(['0','2','3'])
# E       AssertionError: assert {'0', '1', '2'} == {'0', '2', '3'}
# E         Extra items in the left set:
# E         '1'
# E         Extra items in the right set:
# E         '3'
# E         Use -v to get the full diff

# E       AssertionError: assert 'foo1' == 'foo2'
# E         - foo2
# E         ?    ^
# E         + foo1
# E         ?    ^

# tests/test_3.py:5: AssertionError
# =========== short test summary info ============
# FAILED tests/test_3.py::test_answer - Asserti...
# ============== 1 failed in 0.05s ===============
# (pgbackup) pgbackup[master !?] $
```

---

## Pytest 布局

Python 测试可以多种方式组织。 测试可以集成在 Python 包中，也可以放在包外。

### 综合测试

在 Python 包中运行测试。

```py

# 这种包装布局。 测试与软件包一起位于`tests`子目录中。
    # setup.py
    # utils
    # │   algo.py
    # │   srel.py
    # │   __init__.py
    # │
    # └───tests
    #         algo_test.py
    #         srel_test.py
    #         __init__.py

# `setup.py`

    #!/usr/bin/env python3
    from setuptools import setup, find_packages
    setup(name="utils", packages=find_packages())


# `utils/algo.py`

    def sel_sort(data):
        if not isinstance(data, list):
            vals = list(data)
        else:
            vals = data
        size = len(vals)
        for i in range(0, size):
            for j in range(i+1, size):
                if vals[j] < vals[i]:
                    _min = vals[j]
                    vals[j] = vals[i]
                    vals[i] = _min
        return vals

    def max(values):
        _max = values[0]
        for val in values:
            if val > _max:
                _max = val
        return _max

    def min(values):
        _min = values[0]
        for val in values:
            if val < _min:
                _min = val
        return _min



# `utils/srel.py`

    def is_palindrome(val):
        return val == val[::-1]


# 我们还有另一个模块，其中包含一个测试单词是否为回文的功能。

# `tests/algo_test.py`

    #!/usr/bin/env python3
    import utils.algo
    import pytest

    @pytest.fixture
    def data():
        return [3, 2, 1, 5, -3, 2, 0, -2, 11, 9]

    def test_sel_sort(data):
        sorted_vals = utils.algo.sel_sort(data)
        assert sorted_vals == sorted(data)

    def test_min():
        values = (2, 3, 1, 4, 6)
        val = utils.algo.min(values)
        assert val == 1

    def test_max():
        values = (2, 3, 1, 4, 6)
        val = utils.algo.max(values)
        assert val == 6

# 这些是`utils.algo`模块的测试。 注意，我们使用完整的模块名称。


# `tests/srel_test.py`

    #!/usr/bin/env python3
    import utils.srel
    import pytest
    @pytest.mark.parametrize(
        "word, expected",
        [('kayak', True), ('civic', True), ('forest', False)])
    def test_palindrome(word, expected):
        val = utils.srel.is_palindrome(word)
        assert val == expected

# 这是对`is_palindrome()`功能的测试。

# `utils/__init__.py`
# `utils/tests/__init__.py`
# 两个`__init__.py`文件均为空。

    # $ pytest --pyargs utils
    # === test session starts ===
    # platform win32 -- Python 3.7.0, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
    # rootdir: C:\Users\Jano\Documents\pyprogs\pytest\structure
    # collected 6 items
    # utils\tests\algo_test.py .. [ 50%]
    # utils\tests\srel_test.py .. [100%]
    #  6 passed in 0.06 seconds =

# 我们使用`pytest --pyargs utils`命令运行测试。
```


### 外部测试

下一个示例显示了应用源布局，其中测试未集成在包内。

```py
    # setup.py
    # src
    # └───utils
    # │       algo.py
    # │       srel.py
    # tests
    #     algo_test.py
    #     srel_test.py

# 在这种布局中，我们在源代码树之外进行测试。 请注意，不需要`__init__.py`文件。
    $ set PYTHONPATH=src
    $ pytest

# 我们设置`PYTHONPATH`并运行 pytest。
```

---

# pytest-mock
- This plugin provides a mocker fixture
- a thin-wrapper around the patching API provided by the mock package.

---

## unittest.mock

Python3.3 新增用来在单元测试的时候进行 mock 操作的 unittest.mock 模块。

---

## 常见用法

### 定义返回值 `mock.return_value`

```py
from unittest.mock import MagicMock
mock = MagicMock(return_value=3)
# or
mock = MagicMock()
mock.return_value = 3

>>> mock()
3
```

### 定义变化的返回值 `side_effect=xxx`

```py
mock = MagicMock(side_effect=[1, 2, 3])
# or
mock = MagicMock()
mock.side_effect = [1, 2, 3]

>>> mock()
1
>>> mock()
2
>>> mock()
3
>>> mock()
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/xxx/lib/python3.6/unittest/mock.py", line 939, in __call__
#     return _mock_self._mock_call(*args, **kwargs)
#   File "/xxx/lib/python3.6/unittest/mock.py", line 998, in _mock_call
#     result = next(effect)
# StopIteration



def side_effect(arg=1):
    return arg
m = MagicMock(side_effect=side_effect)

>>> m()
1
>>> m(1)
1
>>> m(2)
2
```

---

### 定义抛出异常 `side_effect = KeyError`

```py
m.side_effect = KeyError

m()
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/xxx/python3.6/unittest/mock.py", line 939, in __call__
#     return _mock_self._mock_call(*args, **kwargs)
#   File "/xxx/python3.6/unittest/mock.py", line 995, in _mock_call
#     raise effect
# KeyError
```

---

### mock 变量/属性

```py
# example.py:
class FooBar:
    def __init__(self):
        self.msg = 'test'
    def hello(self):
        return 'hello'

foo = {}

def bar():
    return foo
    # return {}

def foobar():
    return FooBar().hello()
    # return 'hello'

fb = FooBar()

def hello():
    return fb.msg
    # self.msg = 'test'
```

```py
from unittest.mock import patch
m = MagicMock()
m.test = MagicMock(return_value=233)

>>> m()
# <MagicMock name='mock()' id='4372854824'>
>>> m.test
# <MagicMock name='mock.test' id='4372854768'>
>>> m.test()
233


import example
>>> example.foo
{}
>>> example.hello()
'test'
>>> with patch.object(example, 'foo', {'lalala': 233}):
        example.foo
{'lalala': 233}

>>> example.foo
{}
>>> with patch.object(example.fb, 'msg', 666):
        example.hello()
666
>>> example.hello()
'test'
```

---

### mock dict
```py
>>> foo = {'a': 233}
>>> foo['a']
233

>>> with patch.dict(foo, {'a': 666, 'b': 222}):
        print(foo['a'])   # 666
        print(foo['b'])   # 222

>>> foo['a']
233

>>> 'b' in foo
False
```

---

## 常用检查方法
- mock 的对象拥有一些可以用于单元测试的检查方法，
- 可以用来测试 mock 对象的调用情况。

---

### 检查调用次数

```py
# 待检查的 mock 对象:
m = MagicMock()

>>> m(1)
<MagicMock name='mock()' id='4372904760'>
>>> m(2)
<MagicMock name='mock()' id='4372904760'>
>>> m(3)
<MagicMock name='mock()' id='4372904760'>
>>>
```

---

### `.called`: 是否被调用过

```py
>>> m.called
True
```

---

### `.call_count`: 获取调用次数

```py
>>> m.call_count
3
```

---

### .assert_called():
- 检查是否被调用过
- 如果没有被调用过，则会抛出 AssertionError 异常

```py
>>> m.assert_called()
>>>
```

### `.assert_called_once()`: 确保调用过一次
- 如果没调用或多于一次，否则抛出 AssertionError 异常

```py
>>> m.assert_called_once()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/xxx/lib/python3.6/unittest/mock.py", line 795, in assert_called_once
    raise AssertionError(msg)
AssertionError: Expected 'mock' to have been called once. Called 3 times.
```

### `.assert_not_called()`:
- 确保没被调用过，否则抛出 AssertionError 异常

```py
>>> m.assert_not_called()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/xxx/lib/python3.6/unittest/mock.py", line 777, in assert_not_called
    raise AssertionError(msg)
AssertionError: Expected 'mock' to not have been called. Called 3 times.
```

---

## 检查调用时使用的参数¶

待检查的 mock 对象:
```py
m = MagicMock()

>>> m(1, 2, foo='bar')
<MagicMock name='mock()' id='4372980792'>
```

### .call_args: 最后调用时的参数
- 最后一次调用时使用的参数，未调用则返回 None

```py
>>> m.call_args
call(1, 2, foo='bar')
```

### `.assert_called_once_with(*args, **kwargs): `

确保只调用过一次，并且使用特定参数调用

```py
>>> m.assert_called_once_with(1, 2, foo='bar')

>>> m(2)
<MagicMock name='mock()' id='4372980792'>

>>> m.assert_called_once_with(1, 2, foo='bar')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/xxx/lib/python3.6/unittest/mock.py", line 824, in assert_called_once_with
    raise AssertionError(msg)
AssertionError: Expected 'mock' to be called once. Called 2 times.
```

### `.assert_any_call(*args, **kwargs)`: 检查某次用特定参数进行过调用

```py
>>> m.assert_any_call(1, 2, foo='bar')
>>>
```

### `.assert_called_with(*args, **kwargs)`: 检查最后一次调用时使用的参数

```py
>>> m.assert_called_with(2)
>>>

>>> m.assert_called_with(1, 2, foo='bar')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/xxx/lib/python3.6/unittest/mock.py", line 814, in assert_called_with
    raise AssertionError(_error_message()) from cause
AssertionError: Expected call: mock(1, 2, foo='bar')
Actual call: mock(2)
>>>
```

### .call_args_list: 所有调用时使用的参数列表

```py
>>> m.call_args_list
[call(1, 2, foo='bar'), call(2)]

>>> m(3)
<MagicMock name='mock()' id='4372980792'>

>>> m.call_args_list
[call(1, 2, foo='bar'), call(2), call(3)]
```


### .assert_has_calls(calls, any_order=False):
- 检查某几次调用时使用的参数
- any_order 为 False 时必须是挨着的调用顺序, 可以是中间的几次调用
- any_order 为 True 时 calls 中的记录可以是无序的

```py
>>> from unittest.mock import call

>>> m.call_args_list
[call(1, 2, foo='bar'), call(2), call(3)]

>>> m.assert_has_calls([call(2), call(3)])
>>>

>>> m.assert_has_calls([call(3), call(2)])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/xxx/lib/python3.6/unittest/mock.py", line 846, in assert_has_calls
    ) from cause
AssertionError: Calls not found.
Expected: [call(3), call(2)]
Actual: [call(1, 2, foo='bar'), call(2), call(3)]

>>> m.assert_has_calls([call(3), call(2)], any_order=True)
>>>
```


### .method_calls: mock 对象的方法调用记录

```py
>>> m.test_method(2, 3, 3)
<MagicMock name='mock.test_method()' id='4372935456'>

>>> m.method_calls
[call.test_method(2, 3, 3)]
```

### .mock_calls:
- 记录 mock 对象的所有调用
- 包含方法、magic method 以及返回值 mock

```py
>>> m.mock_calls
[call(1, 2, foo='bar'), call(2), call(3), call.test_method(2, 3, 3)]

>>> m.call_args_list
[call(1, 2, foo='bar'), call(2), call(3)]
```

## 手动重置 mock 调用记录
可以使用 .reset_mock() 重置 mock 对象记录的调用记录:

```py
>>> m.mock_calls
[call(1, 2, foo='bar'), call(2), call(3), call.test_method(2, 3, 3)]

>>> m.call_args_list
[call(1, 2, foo='bar'), call(2), call(3)]

>>> m.reset_mock()

>>> m.call_args_list
[]
>>> m.mock_calls
[]
>>>
```




.
