---
title: Meow's Testing Tools - Postman
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

- [Postman](#postman)
  - [Postman basic](#postman-basic)
  - [Postman的操作环境](#postman的操作环境)
  - [Postman install](#postman-install)
  - [Postman安装](#postman安装)
  - [Navigating Postman](#navigating-postman)
    - [main modes:](#main-modes)
    - [Find and replace](#find-and-replace)
    - [History](#history)
    - [Tabs](#tabs)
    - [Next steps](#next-steps)
  - [接口请求流程](#接口请求流程)
    - [GET 请求](#get-请求)
    - [POST请求](#post请求)
  - [管理用例—Collections](#管理用例collections)
  - [身份验证Authentication\*\*](#身份验证authentication)
  - [extension](#extension)

---

# Postman

---

## Postman basic

- 网页调试工具
- 用户在开发或者调试网络程序/网页B/S模式的程序的时候是需要一些方法来`跟踪网页请求`，一些网络的监视工具比如著名的Firebug等网页调试工具。
- postman 可以
  - 调试简单的css、html、脚本等简单的网页基本信息
  - 可以发送几乎所有类型的HTTP请求, Chrome插件类产品中的代表产品之一。

特点：
- 创建 + 测试：创建和发送任何的HTTP请求，请求可以保存到历史中再次执行
- Organize: 使用 `Collections` 为更有效的测试 及 集成工作流管理和组织APIs
- document: 依据创建的 `Clollections` 自动生成API文档,并将其发布成规范的格式
- collarorate: 通过同步 连接你的team和你的api，以及权限控制，API库


Postman provides a variety of views and controls for managing your API projects.

[![Postman app](https://assets.postman.com/postman-docs/app-overview-console-open.jpg)](https://assets.postman.com/postman-docs/app-overview-console-open.jpg)

---

## Postman的操作环境

postman适用于不同的操作系统，Postman Mac、Windows X32、Windows X64、Linux系统，还支持postman 浏览器扩展程序、postman chrome应用程序等。

[官方英文文档](https://www.getpostman.com/docs/v6/)  '


---

## Postman install

下载地址：

1. [Postman for MAC](https://app.getpostman.com/app/download/osx64?utm_source=site&utm_medium=apps&utm_campaign=macapp&_ga=2.21151352.2119858274.1527039878-1088353859.1527039878)

2. [Postman for windows X64](https://app.getpostman.com/app/download/win64?_ga=2.201562513.1250696341.1530543681-1582181135.1530543681)

3. [Postman for windows X86](https://app.getpostman.com/app/download/win32?_ga=2.21151352.2119858274.1527039878-1088353859.1527039878)

4. [Postman for linux X64](https://app.getpostman.com/app/download/linux64?_ga=2.96050783.2119858274.1527039878-1088353859.1527039878)

5. [Postman for Linux X86](https://app.getpostman.com/app/download/linux32?_ga=2.96050783.2119858274.1527039878-1088353859.1527039878)

6. [官网下载地址](https://www.getpostman.com/apps)


---

## Postman安装

postman的安装方法分好几种:

1. chrome postman 插件安装
   - postman谷歌浏览器的安装插件，所以说它的使用前提是你的电脑上得安装谷歌浏览器才行
   <!-- - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181030002023904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz,size_16,color_FFFFFF,t_70) -->

2. Postman电脑客户端安装
   1. macOS安装
   2. Windows安装
   3. Linux安装
      - [ubuntu安装postman](https://blog.csdn.net/qianmosolo/article/details/79353632)
      - [Ubuntu16.04上安装Postman应用程序](https://blog.bluematador.com/posts/postman-how-to-install-on-ubuntu-1604/?utm\_source=hootsuite&utm\_medium=twitter&utm\_campaign=)

---

## Navigating Postman

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523232921542?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->



---

### main modes:

- **Build** and **Browse** using the control at the bottom-right of the app, or the top-left on the web.
- **Build** mod: the primary interface for working with API requests. 快速创建几乎所有的请求

> HTTP请求的4部分: URL，request method，headers，body。

[![general layout](https://assets.postman.com/postman-docs/59046674.jpg)](https://assets.postman.com/postman-docs/59046674.jpg)

* **sidebar** provides access to your request history, collections, and APIs.
* **header**
  - create and import requests and collections,
  - to access the Collection Runner,
  - to move and invite collaborators to workspaces,
  -  access the Interceptor, view sync status / notifications,
  - open your Settings, account, and Postman plan.

*  The status bar along the bottom allows you to show/hide the sidebar, find and replace, and open the console on the left.
* On the right you can launch the **Bootcamp**, toggle between **Build** and **Browse** mode, toggle pane view, open keyboard shortcuts, and access help resources.


resize the panes in the Postman UI.
- declutter your workspace by collapsing panes using the arrow button—click the collapsed section to reopen it.
- hide the sidebar and toggle between single and two pane view.

![Resizing panes](https://assets.postman.com/postman-docs/resizing-panes-app.gif)

---

### Find and replace

- search the Postman workspace by clicking **Find and Replace** at the bottom-left of Postman
- the keyboard shortcuts `Command + SHIFT + F` / `Control + SHIFT + F`.
- Enter your search string and optionally choose which entities to return, entering replacement text if necessary.
- Postman will search tabs, collections, and variables.
- click directly from the search results to open an entity.

![Find and replace](https://assets.postman.com/postman-docs/find-and-replace-tab.jpg)


---

### History

- access a history of the requests you've made in Postman in **History** on the left of Postman.
- history will sync across devices.
- Click a request to open it again.
- Click **+** to save the request to a collection.
- Toggle **Save Responses** to save request responses so that you can view what was returned by a request when you open it from your history.
- The **View more actions** menu allows you to `save, monitor, document, or mock a request`.
- Use the **delete (trash icon)** or **Clear all** options to remove requests from your history.
- multi-select requests by pressing `Command` or `Control` and clicking the requests.


![History](https://assets.postman.com/postman-docs/history-in-app.jpg)

---

### Tabs

- send requests in Postman by opening tabs—click **+** in the center of the screen, or press `Command/Control + T`.

![Tabs](https://assets.postman.com/postman-docs/open-unsaved-tab-options.jpg)

> If you open a request and do not edit or send it, then open another, the first tab will be replaced by the second. While the tab is in _Preview_ mode it will display in italics.

- can have multiple tabs open at the same time as you work, and can drag tabs to rearrange them. Use the **...** button to manage tabs and access recent tabs.

> Duplicating a tab does not mean creating a second request to the same endpoint—when you duplicate a tab any edits you make will affect the original request.

Postman will display a dot on any tabs with unsaved changes.
- A tab may indicate a conflict if you or a collaborator changes it in another tab or workspace.
- Postman will prompt you to resolve any conflicts that occur.

> You can toggle whether Postman opens requests in new tabs or not in the **Settings**, as well as configuring whether Postman prompts you when closing tabs with unsaved changes.


---


### Next steps


The best way to get to know Postman is by firing up the **Bootcamp** on the bottom-right and working through the lessons.

![Bootcamp](https://assets.postman.com/postman-docs/bootcamp-overview-app.jpg)

You can also access Bootcamp together with other resources for getting started and staying up to date by opening Launchpad—you'll see a button when you have no open tabs.

![Launchpad](https://assets.postman.com/postman-docs/launchpad-open-app.jpg)



---

## 接口请求流程


Each API request uses an HTTP method. The most common methods are `GET, POST, PATCH, PUT, and DELETE`.
- `GET` methods retrieve data from an API.
- `POST` sends new data to an API.
- `PATCH` and `PUT` methods update existing data.
- `DELETE` removes existing data.

In Postman you can make API requests and examine the responses without using a terminal or writing any code.
- When you create a request and click Send, the API response appears inside the Postman user interface.

![anatomy-of-a-request](https://i.imgur.com/fKW8rh0.png)

![first-request-sent](https://i.imgur.com/9ue5Tgf.png)

```
postman-echo.com/get

{
    "args": {},
    "headers": {
        "x-forwarded-proto": "http",
        "x-forwarded-port": "80",
        "host": "postman-echo.com",
        "x-amzn-trace-id": "Root=1-5fbc10c4-2b1fe7686b5d211975ddd574",
        "user-agent": "PostmanRuntime/7.26.8",
        "accept": "*/*",
        "cache-control": "no-cache",
        "postman-token": "8f289d40-e514-4b91-a923-de4b63c1860c",
        "accept-encoding": "gzip, deflate, br"
    },
    "url": "http://postman-echo.com/get"
}
```


---

### GET 请求

```
GET请求：
点击Params，输入参数及value，可输入多个，即时显示在URL链接上，
GET请求的请求头与请求参数如在接口文档中无特别声明时，可以不填。
```

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523233825152?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

> 这里会有请求的响应状态码，响应时间，以及响应大小

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523234132434?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

> 响应体示例： 响应的格式可以有多种, 一般情况下，自定义接口是 json格式的响应体

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523234247147?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

---


### POST请求

1. **POST请求一：表单提交**

下图示例中设置了请求方法，请求URL，请求参数，但是没有设置请求头
在我的使用过程中，请求头是根据请求参数的形式自动生成的
请求头中的Content-Type与请求参数的格式之间是有关联关系，比如：


<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524000345232?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->


<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523234739215?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180523234748383?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

2. **POST请求二：json提交**

下图中，当我们选择JSON(application/json) 是会自动帮我们设置 headers 为 application/json
在这里就不截图 举例了，朋友们可以自行去查看

<!-- ![这里写图片描述](https://img-blog.csdn.net/2018052400054291?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->


3. **POST请求三：xml提交**

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524000901598?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->


4. **POST请求四：二进制文件提交**

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524001010654?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

其它请求方式如PUT,DELETE 大致流程和GET,POST 差不多，这里就不一一举例说明了


---


## 管理用例—Collections

在POST基础功能那里有一张图片大致说了一下Collections 的作用， 这里我们再详细说明一下

Collections集合：也就是将多个接口请求可以放在一起，并管理起来。
- 接口请求可以放在同一个collection里
- 一个工程一个Collection，这样方便查找及统一处理数据。

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524001252769?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

<!-- ![这里写图片描述](https://img-blog.csdn.net/2018052400150515?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

1. 创建Collections > + > Name:”abc” > Description:”demo” > Create Collections.

2. 在Collections里添加请求

在右侧准备好接口请求的所有数据，并验证后，点击save按钮。
<!-- ![这里写图片描述](https://img-blog.csdn.net/2018052400243724?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

保存好之后就可以在这里看到啦，之后要再次调用时可以点击这里，方便快捷

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524002002823?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

collections 管理精细化， 这里我们针对不同的请求方式做分组

添加子文件夹
<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524002857320?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524002953353?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->

精细化划分之后的结果:

<!-- ![这里写图片描述](https://img-blog.csdn.net/20180524003120219?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Z4YmluMTIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) -->



---


## 身份验证Authentication**

1. Basic Auth
   1. 是基础的验证，所以会比较简单
   2. 会直接把用户名、密码的信息放在请求的 Header 中

2. Digest Auth
   1. 要比Basic Auth复杂的多。
   2. 使用当前填写的值生成authorization header。
   3. 所以在生成header之前要确保设置的正确性。
   4. 如果当前的header已经存在，postman会移除之前的header。

3. OAuth 1.0
   1. postman的OAuth helper让你签署支持OAuth
   2. 1.0基于身份验证的请求。OAuth不用获取access token,你需要去API提供者获取的。
   3. OAuth 1.0可以在header或者查询参数中设置value。

4. OAuth 2.0
   1. postman支持获得OAuth 2.0 token并添加到requests中。


---

## extension


```bash

# install jython
brew install jython

echo "export JYTHON_HOME=/usr/local/Cellar/jython/2.7.2/libexec" | tee -a ~/.bash_profile; source ~/.bash_profile

JAVA_HOME = /Library/Java/JavaVirtualMachines/jdk1.8.0_181.jdk/Contents/Home


```

.
