# PHP 教程

[toc]

## PHP
一种创建动态交互性站点的强有力的服务器端脚本语言。

PHP 是免费的，并且使用非常广泛。同时，对于像微软 ASP 这样的竞争者来说，PHP 无疑是另一种高效率的选项。

### PHP初学者的学习线路：
1. 熟悉HTML/CSS/JS等网页基本元素，完成阶段可自行制作简单的网页，对元素属性相对熟悉。
2. 理解动态语言的概念和运做机制，熟悉基本的PHP语法。
3. 学习如何将PHP与HTML结合起来，完成简单的动态页面。
4. 接触学习MySQL，开始设计数据库。
5. 不断巩固PHP语法，熟悉大部分的PHP常用函数，理解面向对象编程，MySQL优化，以及一些模板和框架。
6. 最终完成一个功能齐全的动态站点。

- 任何网站都是由网页组成的，也就是说想完成一个网站，必须先学会做网页，掌握静态网页的制作技术是学习开发网站的先决条件。学习HTML，为今后制作网站打下基础。

- 假设你已经可以完成一个静态页面了，开始了解动态语言，代码不是作为直接输出的，而是要经过处理的，HTML是经过HTML解析器，而PHP也要通过PHP解析器，跟学习HTML一样的道理，想让任何的解析器工作，就必须使用它专用的语法结构。

- 学习PHP，搞清楚HTML和PHP的概念，你现在完全可以让PHP给你算算一加一等于几，然后在浏览器输出。

- 学习数据库，MySQL可以说是PHP的黄金搭档，在理解了数据库的概念之后，尝试通过PHP来连接数据库，进而用PHP成功的插入，删除和更新数据。

- 这个时候，你可能会处于这种状态：你会HTML吗？会，我能编好几个表格排板的网页呢！你会PHP吗？会，我会把一加一的运算写在函数里，然后调用！你会MySQL吗？会，我可以把数据库里的数据插入删除啦！

- 那接下来该做什么呢？尝试着做个小的留言本吧，这同样是新手面临的一道关卡。花了一段时间，你终于学会把表单的数据插入数据库，然后显示出来了，应该说一个程序的雏形已经诞生了。但是，你可能会看人家这个编论坛，那个开发CMS，我什么时候可以写一个呢？不要急，再巩固一下知识，熟悉了PHP和MySQL开发的要领后，再回头看你写的那个留言本，你也许会怀疑那真的是你写的吗？这个时候，你可以完善一下你写的留言本。留言本应该加入注册以及分页的功能，可以的话，UI也可以加强。

- OOP,模板和框架. PHP框架提供了一个用以构建web应用的基本框架，从而简化了用PHP编写web应用程序的流程。可以节省开发时间、有助于建立更稳定的应用。所以说，PHP框架是一个可以用来节省时间并强化自己代码的工具。当你第一次选择PHP框架时，建议多尝试几个，每个框架都有自己的长处和短处，例如Zend框架由于多样的功能、并且有一个广泛的支持系统，流行了很长时间。而CakePHP是一个晚于Zend的PHP框架，相应的支持系统也比较少，但是更为方便和易于使用。

- 了解了面向对象和框架后，接触XML。学会了PHP，那么再学其它语言，肯定速成，反过来也一样，如果你之前学过其它的语言，那么学PHP肯定快。

- 多借鉴别人成功的代码，绝对是有益无害，所以要多看那些经过千锤百炼凝出来的经典代码，是进步的最好方法。另外，要强调的是，学习一项技术过程中可能会遇到困难，可能会迷茫，你也许学了一半的PHP，又开始打C#的主意，或者有人说Java很好，这个时候你绝对不能动摇，要坚持到底，彻底学会。祝你顺利学成PHP，开发自己想要的网站。


## PHP开发工具
PHP 开发工具其实包括以下四种：
- PHP服务器组件。
- PHP IDE(Integrated Development Environment,集成开发环境)。
- MySql管理工具
- 文本编辑器

### PHP服务器组件
PHP服务器组件非常多有WampServer、XAMPP、AppServ、phpStudy、phpnow等。
推荐：
WampServer，这也是目前window平台上使用最广泛的，操作也非常简单。
WampServer, 内部还集成了PhpMyAdmin 数据库管理工具。
下载地址：http://www.wampserver.com/en/#download-wrapper

### PHP IDE(Integrated Development Environment,集成开发环境)
PHP IDE 也是非常多有Zend Studio、Eclipse for PHP、EasyEclipse等。
推荐：easyeclipse for php
下载地址：http://www.easyeclipse.org/site/distributions/php.html

### MySql管理工具
MySql管理工具常用的有：Navicat for Mysql、PhpMyAdmin。
推荐：Navicat for Mysql，
Navicat for MySQL是一套专为MySQL设计的强大数据库管理及开发工具。它可以用于任何3.21或以上的MySQL数据库服务器，并支持大部份MySQL最新版本的功能，包括触发器、存储过程、函数、事件、检索、权限管理等等。
下载地址：http://www.navicat.com.cn/download/navicat-for-mysql

### 文本编辑器
如果你已经能够熟练掌握PHP的语法，那你可以逐渐抛弃那些笨重的IDE，使用文本编辑器来编写PHP代码。
常用的编辑器有：Notepad++、editplus、ultraedit等。
推荐：Notepad++


# PHP 简介
PHP 是服务器端脚本语言。

您应当具备的基础知识
在继续学习之前，您需要对以下知识有基本的了解：

HTML
CSS
如果您希望首先学习这些项目，请在我们的 首页 访问这些教程。

PHP 是什么？
PHP（全称：PHP：Hypertext Preprocessor，即"PHP：超文本预处理器"）是一种通用开源脚本语言。
PHP 脚本在服务器上执行。
PHP 可免费下载使用。
lamp	PHP 对初学者而言简单易学。

PHP 也为专业的程序员提供了许多先进的功能。

PHP 文件是什么？
PHP 文件可包含文本、HTML、JavaScript代码和 PHP 代码
PHP 代码在服务器上执行，结果以纯 HTML 形式返回给浏览器
PHP 文件的默认文件扩展名是 ".php"
PHP 能做什么？
PHP 可以生成动态页面内容
PHP 可以创建、打开、读取、写入、关闭服务器上的文件
PHP 可以收集表单数据
PHP 可以发送和接收 cookies
PHP 可以添加、删除、修改您的数据库中的数据
PHP 可以限制用户访问您的网站上的一些页面
PHP 可以加密数据
通过 PHP，您不再限于输出 HTML。您可以输出图像、PDF 文件，甚至 Flash 电影。您还可以输出任意的文本，比如 XHTML 和 XML。

为什么使用 PHP？
PHP 可在不同的平台上运行（Windows、Linux、Unix、Mac OS X 等）
PHP 与目前几乎所有的正在被使用的服务器相兼容（Apache、IIS 等）
PHP 提供了广泛的数据库支持
PHP 是免费的，可从官方的 PHP 资源下载它： www.php.net
PHP 易于学习，并可高效地运行在服务器端








































.# PHP 简介
PHP 是服务器端脚本语言。

您应当具备的基础知识
在继续学习之前，您需要对以下知识有基本的了解：

HTML
CSS
如果您希望首先学习这些项目，请在我们的 首页 访问这些教程。

PHP 是什么？
PHP（全称：PHP：Hypertext Preprocessor，即"PHP：超文本预处理器"）是一种通用开源脚本语言。
PHP 脚本在服务器上执行。
PHP 可免费下载使用。
lamp	PHP 对初学者而言简单易学。

PHP 也为专业的程序员提供了许多先进的功能。

PHP 文件是什么？
PHP 文件可包含文本、HTML、JavaScript代码和 PHP 代码
PHP 代码在服务器上执行，结果以纯 HTML 形式返回给浏览器
PHP 文件的默认文件扩展名是 ".php"
PHP 能做什么？
PHP 可以生成动态页面内容
PHP 可以创建、打开、读取、写入、关闭服务器上的文件
PHP 可以收集表单数据
PHP 可以发送和接收 cookies
PHP 可以添加、删除、修改您的数据库中的数据
PHP 可以限制用户访问您的网站上的一些页面
PHP 可以加密数据
通过 PHP，您不再限于输出 HTML。您可以输出图像、PDF 文件，甚至 Flash 电影。您还可以输出任意的文本，比如 XHTML 和 XML。

为什么使用 PHP？
PHP 可在不同的平台上运行（Windows、Linux、Unix、Mac OS X 等）
PHP 与目前几乎所有的正在被使用的服务器相兼容（Apache、IIS 等）
PHP 提供了广泛的数据库支持
PHP 是免费的，可从官方的 PHP 资源下载它： www.php.net
PHP 易于学习，并可高效地运行在服务器端
