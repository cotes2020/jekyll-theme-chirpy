---
title: Global Interpreter Lock (GIL) in Python
date: 2023-06-05 02:17 +0900
category: [Language]
tag: [Python, Multiprocessing]
---

### GIL

Global Interpreter Lock (GIL)
: 파이썬 바이트 코드를 하나의 프로세스에서 한 번에 하나씩만 실행할 수 있게 제한하는 뮤텍스.

파이썬은 GIL을 채택하고 있다. GIL은 파이썬의 악명높은 토픽 중 하나인데 단점이 있어도 제거할 수 없는 요소이기 때문이다.

### GIL의 단점

**멀티스레딩으로 성능향상이 불가능**하다. `threading` 모듈을 이용하면 파이썬에서 멀티스레드를 구현할 수 있지만 하나의 스레드가 여러 포인트를 돌아가며 실행시키는 것일 뿐 동시에 실행되지는 않는다.

* **멀티프로세싱을 사용하면 성능향상이 가능**하다. 프로세스끼리는 완전히 독립적으로 실행되기 때문에 서로 GIL의 영향을 받지 않는다.

* CPU-bound process는 대부분의 시간을 파이썬 바이트 코드를 실행하며 보내기 때문에 위에서 말한 바와 같이 멀티스레딩으로 성능향상이 불가능하다. 하지만 **I/O-bound process는 외부 프로세스에 명령을 내려놓고 대부분의 시간을 대기하기 때문에 이런 경우에는 멀티스레딩으로 성능향상이 가능**하다.

### GIL을 제거할 수 없는 이유

Garbage Collecting 문제
: 파이썬은 Garbage Collector로 메모리를 관리한다. 모든 객체마다 참조횟수를 카운팅 하면서 0이 되면 접근할 수 없는 객체로 보고 메모리를 해제한다. 만약 이때 멀티스레딩으로 인해 참조횟수가 잘못 연산되면 메모리 유출이나 세그먼테이션 오류가 날 수 있다.

C기반 라이브러리와의 호환 문제
: 파이썬 초기에 다양한 C기반 라이브러리와의 호환이 중요했다. 하지만 상당수의 라이브러리가 Thread-Safe하지 않았고, 간단하게 호환할 수 있게 하는 방식이 GIL이었다. 또한 파이썬 초기에는 멀티스레딩의 개념이 없었다.

> GIL을 없애면 멀티스레드 환경에서는 더 빨라지지만 싱글스레드 환경에서는 더 느려지기 때문에 GIL을 없애려는 노력이 있어왔지만 채택되지 않았다.
{: .prompt-tip}

> 모든 파이썬 구현이 GIL을 채택한 것은 아니다. GIL은 CPython, PyPy에는 있고 Jython, Iron Python에는 없다. Cython은 평소에는 있지만 with문 내부에서는 일시적으로 없어진다.
{: .prompt-tip}

### 실험

CPU-bound process와 I/O-bound process에 대하여 single thread, multithreading, multiprocessing의 시간을 측정해보았다. GIL이 채택된 파이썬 구현인 CPython으로 실험하였다. 실행 결과는 제일 아래에 정리하였다.

필요한 모듈을 불러오고 CPU-bound process와 I/O-bound process를 대표하는 함수를 선언한다.
```python
from time import time
import threading
import multiprocessing
import requests

def cpu_bound_process(n):
    while n > 0:
        n -= 1

def io_bound_process(n):
    for _ in range(n):
        requests.get('http://numbersapi.com/42')
```
cpu_bound_process는 파이썬 바이트 코드만 실행하도록 구성하였고, io_bound_process는 서버와의 통신을 많이 하도록 구성하였다. numbersapi는 서버와의 통신을 연습할 때 유용하게 쓰이는 사이트이다.

single thread, multi thread, multi process에 해당하는 함수를 선언한다.
```python
def single_thread(process, count):
    start = time()
    process(count)
    print('single thread', time() - start)

def multi_thread(process, count):
    start = time()
    t1 = threading.Thread(target=process, args=[count // 2])
    t2 = threading.Thread(target=process, args=[count // 2])
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print('multi_thread', time() - start)

def multi_process(process, count):
    pool = multiprocessing.Pool(2)
    start = time()
    r1 = pool.apply_async(process, [count // 2])
    r2 = pool.apply_async(process, [count // 2])
    pool.close()
    pool.join()
    print('multiprocessing', time() - start)
```

multi_thread와 multi_process에서 각각 생성되는 스레드와 프로세스는 2개이다.

실행해본다.
```python
single_thread(cpu_bound_process, 50000000)
multi_thread(cpu_bound_process, 50000000)
multi_process(cpu_bound_process, 50000000)
single_thread(io_bound_process, 10)
multi_thread(io_bound_process, 10)
multi_process(io_bound_process, 10)
```

|             |CPU-bound process|I/O-bound process|
|:------------|:----------------|:----------------|
|single thread|2.115            |7.553            |
|multi thread |1.987            |1.897            |
|multi process|1.130            |1.914            |

* CPU-bound process에서 multi thread의 경우 single thread와 비슷한 시간이 걸렸다. GIL 때문이다.

* I/O-bound process에서 multi thread의 경우 single thread보다 확실히 적은 시간이 걸렸다. 대부분이 대기하는 시간이기 때문에 GIL의 영향을 거의 받지 않는다.

* multi process의 경우 프로세스의 종류에 상관없이 적은 시간이 걸렸다. 프로세스는 독립적으로 실행되기 때문에 GIL의 영향을 받지 않는다.

> PyTorch의 `torch.utils.data.DataLoader`에서 num_worker인자를 조절하여 데이터를 공급하는 과정을 가속화할 때 멀티스레딩이 아니라 멀티프로세싱으로 수행된다.

### Ref.

[#realpython](https://realpython.com/python-gil/) [#dabeaz](http://www.dabeaz.com/python/GIL.pdf) [#dabeaz2](http://dabeaz.blogspot.com/2010/01/python-gil-visualized.html) [#numbersapi](http://numbersapi.com/#42)
