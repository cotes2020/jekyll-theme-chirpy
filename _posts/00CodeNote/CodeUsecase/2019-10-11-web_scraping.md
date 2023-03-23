---
title: Web scraping
date: 2019-10-11 11:11:11 -0400
description:
categories: [00CodeNote, CodeUsecase]
img: /assets/img/sample/rabbit.png
tags: [Python, webscrap]
---

- [Web scraping](#web-scraping)
  - [Code](#code)
  - [test](#test)
    - [HTMLParser 解析 html](#htmlparser-解析-html)


---


# Web scraping

---

## Code

---


## test

Target

```py
# python main.py

url: xx.com

'APT', 'Beds / Baths', 'Rent Starting from * /month', 'Deposit', 'Sq. Ft', 'Limited Time Offer', 'Valid Through', 'Available'

['Urban with Kitchen Bar', '1 bd / 1 ba', '$1,921', '$300', '605', '8 Weeks Free on Select Homes and Move-In Dates!', 'Oct 27, 2022 - Jan 31, 2023', '4']

['Urban with L or Galley Kitchen', '1 bd / 1 ba', '$1,940', '$300', '578', 'Oct 27, 2022 - Jan 31, 2023', '8 Weeks Free on Select Homes and Move-In Dates!' 'Available Feb 09, 2023']

```


---


### HTMLParser 解析 html


- HTMLParser 是 Python 自带的一个类，主要用来解析 HTML 和 XHTML 文件。

HTMLParser 常用方法
- handle_starttag(tag, attrs)：找到开始标签时调用，attrs 是（名称，值）对的序列
- handle_startendtag(tag, attrs)：使用空标签时调用，默认分开处理开始和结束标签
- handle_endtag(tag)：找到结束标签时调用
- handle_data(data)：使用文本数据时调用
- handle_charred(ref)：当使用 ref;形式的实体引用时调用
- handle_comment(data)：注释时调用，只对注释内容调用
- handle_decl(decl)：声明<!...>形式时调用


使用 HTMLParser 非常简单，只需要在你的 python 文件中导入 HTMLParser 类，创建一个新类来继承它，并且对其 handle_starttag、handle_data 等事件处理方法进行重写从而解析出需要的 HTML 数据。


实例 1：抓取网页内容


```py
# coding=utf-8

from HTMLParser import HTMLParser

# 创建类MyHTMLParser并继承HTMLParser
class MyHTMLParser(HTMLParser):
    #重写handle_starttag方法
    def handle_starttag(self, tag, attrs):
        print "Start tag:", tag
        for attr in attrs:
            print "     attr:", attr

    #重写handle_endtag方法
    def handle_endtag(self, tag):
        print "End tag  :", tag

    #重写handle_data方法
    def handle_data(self, data):
        print "Data     :", data

    #重写handle_comment方法
    def handle_comment(self, data):
        print "Comment  :", data

    #重写handle_charref方法
    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print "Num ent  :", c

    def handle_decl(self, data):
        print "Decl     :", data


if __name__ == "__main__":

    parser = MyHTMLParser()
    parser.feed('''
    <html>
        <head>
        </head>
        <body>
            <p class = "aa" >
                Some&nbsp;
                <a href=\"#123\">html</a>
                parser&#62;
                <!-- comment -->
                <br>END
            </p>
        </body>
    </html>''')
```


实例 1：抓取网页内容

```py
# coding=utf-8
class Scraper(HTMLParser):
    def handle_starttag(self, tag, attrs):
            print "StartedTag: ", tag

    def handle_endtag(self, tag):
        print "EndTag: ", tag

    def handle_data(self, data):
            print "Data: ", data

if __name__=="__main__":
    content = urlopen("http://www.baidu.com").read()
    sc = Scraper()
    sc.feed(content)
```


实例 2：对生成的 HTML 测试报告统计 Pass 和 Fail 个数


```py
# coding=utf-8
class AnalyzeReport(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.ps = 0
        self.fail = 0

    def handle_starttag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        '''
        # 打印测试用例详细执行结果
        if self.lasttag == 'font' or data.__contains__("用例："):
            print data
        '''
        if data == "Fail":
            self.fail += 1
        elif data == "Pass":
            self.ps += 1
        else:
            pass

    def get_ps(self):
        return self.ps

    def get_fail(self):
        return self.fail

def read_html(filepath):
    '''
    :param filepath: 要解析的html文件路径
    :return: 返回文件内容
    '''
    try:
        f = open(filepath)
    except IOError, e:
        print e
    else:
        content = f.read()
    return content

if __name__ == "__main__":
    content = read_html("TestReport.html")
    pars = AnalyzeReport()
    pars.feed(content)
    print "*" * 100
    print "失败测试用例数：", pars.fail
    print "成功测试用例数：", pars.ps
```
