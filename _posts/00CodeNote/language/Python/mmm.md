
### cc

如何将多线程应用到网络爬虫当中

---

#### 1 网络爬虫基础

##### 1.1 HTML
- HTML,  **Hypertext Markup Language** , 是开发网页和网页应用的标准语言之一。
- HTML中, 文本由tags包围和分割, 如`<p>, <img>, <i>`等

##### 1.2 HTTP请求/ HTTP Requests

- **客户/浏览器** 是HTTP请求的发出者,  **被访问的网站** 是HTTP请求的接收者, 并在一定条件下, 对客户发出HTTP的相应信息。
- 请求的主要模式包括：`GET, POST, PUT, HEAD, DELET`等
  - 其中, GET和POST是最主要的方式。
  - GET就是 **单纯** 从服务器中 **拿** 一个数据, 
  - POST则是把一个数据 **添加至** 服务器的 **数据库** 中。

- 简单例子就是,  **在金融的程序化交易中** , 如果我们想从交易所 **取得行情信息** , 那么需要发出 **GET** 类型的请求, 如果我们希望向交易所 **下单** , 那么需要发出 **POST** 类型的请求。

![pic](https://pic1.zhimg.com/v2-feba8e2ecbdb80511b9abb079199159c_b.jpg)


##### 1.3 HTTP的状态码/ Status Code

主要分为5大类：

| HTTP Status Code   | 含义                                       |
| ------------------ | ------------------------------------------ |
| 1xx(100, 102, ...) | 服务器已经接受了HTTP请求, 并正在处理       |
| 2xx(200, 202, ...) | 服务器成功收到并处理了HTTP请求             |
| 3xx(300, 301, ...) | 用户需要额外的请求, 才能正确处理该HTTP请求 |
| 4xx(400, 404, ...) | 报错：用户的问题                           |
| 5xx(500, 504, ...) | 报错：服务器的问题                         |

---

#### 2 Python的request模块

用request模块向Bing发出HTTP请求。

```py
import requests  
url = "https://www.bing.com/?mkt=zh-CN"  
res = requests.get(url)  
print(res.status_code) 
print(res.headers)
``` 

![pic](https://pic3.zhimg.com/v2-816bc569bf4400db3a791d9c77e60ac6_b.png)
 
request模块:
- requests.get(url) 代表用户向Bing发送了一个 **GET** 请求
- 返回的HTTP状态码是200, 说明HTTP请求成功
- 返回的header里面有更加详细的信息, 将此Dict转换为Pandas的DataFrame可以更好的阅读：

response的header中的信息:
![pic](https://pic2.zhimg.com/v2-34a267d4c13669034c7c483a75ea1051_b.jpg)
 

---

#### 3 使用多线程进行HTTP请求 

![pic](https://pic2.zhimg.com/v2-da71a6940a3db4cf37c3e04b91a06049_b.jpg)
  
- **每一个HTTP请求, 一般来讲, 是相互独立的。** 
- 尝试用多线程来加快多个HTTP请求的速度。

通过继承之前提及的threading模块中的Thread类, 来编写符合需求的Class。

```py
import threading 
import requests 
import time  

class MyThread(threading.Thread): 
    def __init__(self, url):     
        super().__init__()     
        self.url = url     
        self.result = None  
    
    def run(self):     
        res = requests.get(url=self.url)     
        self.result = f"{self.url}:{res.text}"
```
            
- 其中,  **_run_** 这个方法是进行override。

使用线程的基本操作模式
- 先基于所有的input来instantiate这个MyThread类
- 接着依次对每一个实例进行 **_start_** 和 **_join_** 。

```py
if __name__ == "__main__": 
    urls = [     
        'http://httpstat.us/200',    
        'http://httpstat.us/400',     
        'http://httpstat.us/404',     
        'http://httpstat.us/408',     
        'http://httpstat.us/500',     
        'http://httpstat.us/524' 
    ]  
    start = time.time()  
    threads = [MyThread(url) for url in urls] 
    
    for thread in threads:     
        thread.start() 
    for thread in threads:     
        thread.join()  
    for thread in threads:     
        print(thread.result)  

    print(f'Took {time.time() - start : .2f} seconds')  
    print('Done.')
```
            
运行结果如下：
![pic](https://pic1.zhimg.com/v2-0995b2cfd39aaf470dc9a97a78f9de74_b.jpg)

- 其实将多线程用在网络爬虫中, 主要的操作模式是和其他方面的应用没有区别的, 依旧是先自定义一个Thread的类型, 再把需要process的函数(如果爬虫)应用到该class的run方法中来。
 
