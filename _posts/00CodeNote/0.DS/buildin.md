
- [`__repr__()`](#__repr__)
- [`__radd__`](#__radd__)
- [`__iadd__`](#__iadd__)

## `__repr__()`

`__repr__()` function returns the object representation. It could be any valid python expression such as tuple, dictionary, string etc.
- This method is called when repr() function is invoked on the object, in that case, `__repr__()` function must return a String otherwise error will be thrown.

Difference between `__str__` and `__repr__` functions
1. `__str__` must return string object, `__repr__` can return any python expression.
2. If `__str__` implementation is missing then `__repr__` function is used as fallback. There is no fallback if `__repr__` function implementation is missing.
3. If `__repr__` function is returning String representation of the object, we can skip implementation of `__str__` function.


```py
class Person:
    name = ""
    age = 0

    def __init__(self, personName, personAge):
        self.name = personName
        self.age = personAge

    def __repr__(self):
        return {'name':self.name, 'age':self.age}

    def __str__(self):
        return 'Person(name='+self.name+', age='+str(self.age)+ ')'

p = Person('Pankaj', 34)

# __str__() example
print(p)               # Person(name=Pankaj, age=34)
print(p.__str__())     # Person(name=Pankaj, age=34)
s = str(p)
print(s)               # Person(name=Pankaj, age=34)


# __repr__() example
print(p.__repr__())          # {'name': 'Pankaj', 'age': 34}
print(type(p.__repr__()))    # <class 'dict'>
print(repr(p))               # TypeError: __repr__ returned non-string (type dict)
```



## `__radd__`

These functions `__radd__` are only called if the left operand does not support the corresponding operation and the operands are of different types. For example,

```py
class X:
  def __init__(self, num):
    self.num = num
class Y:
  def __init__(self, num):
    self.num = num

  def __radd__(self, other_obj):
    return Y(self.num+other_obj.num)

  def __str__(self):
    return str(self.num)
>>> x = X(2)
>>> y = Y(3)
>>> print(x+y)
5
>>>
>>> print(y+x)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-60-9d7469decd6e> in <module>()
----> 1 print(y+x)

TypeError: unsupported operand type(s) for +: 'Y' and 'X'
```


## `__iadd__`

```py
# +=

def __iadd__(self, other):
    self.number += other.number
    return self
```
