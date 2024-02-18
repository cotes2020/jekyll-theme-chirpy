---
title : Django 작성 List
categories : [Programming, Django]
tags : [Django]
---

## 참고자료 출처 및 주의사항
Link : <a href="https://ssungkang.tistory.com/entry/Django-02-Django-시작-Hello-World-출력?category=320582" target="_blank">Django-02-Django-시작-Hello-World-출력</a>
```
아래 2번부터는 정해진 순서는 없고 해보니까 이렇게 하는게 편하다는 것.  
보고 까먹은건 참고에서 다시 확인  
```

## 1. models.py
```
models.py에는 내가 만들 앱에 적용시킬 class를 만드는 곳.   
models를 db에 연결해줘야 함.  
```
```
python3 manage.py makemigrations 
: 앱 디렉토리 밑에 migrations 폴더를 생성하여 db와 소통시킴.  
```
```
python3 manage.py migrate
: db에 최신 models (변경사항) 적용
단, app 디렉토리에 있는 admin.py에 등록.
```    
ex) 블로그 생성을 위한 blog 객체 생성
```python
from django.db import models

class Blog(models.Model):
	title = models.CharField(max_length = 200)
  
	writer = models.CharField(max_length = 30, default="anonymous", null=True) 
  # title과 field가 곂쳐서 오류가 나서 수정한 부분. 오류 내용은 까먹음
  
	pub_date = models.DateField('date published')
  
	body = models.TextField()
  
	def __str__(self) : # admin page에서 blog 객체 생성 시 제목을 보여주게끔 설정
		return self.title
    
	def summary(self) :
		return self.body[:100]
```
```
Field는 3개가 사용됬는데

짧은 문장을 담는 CharField
날짜를 담는 DateField
긴 글을 담는 TextField
```

## 2. views.py
```
views.py에는 구체적인 기능을 하는 함수들을 작성함.

ex) blog에서 구현할 login, logout, register, modify 등등
```
```python
from django.shortcuts import render, get_object_or_404, redirect # render 제외 2개 추가
from django.utils import timezone
from .models import Blog    ## models.py에 있는 Blog 객체 사용
from django.contrib.auth.models import User # login, logout, register를 위해 추가
from django.contrib import auth # login, logout, register를 위해 추가 

def login(request) :
	if request.method == 'POST' :
		user_id = request.POST['user_id']
		user_pw = request.POST['user_pw']
		user = auth.authenticate(request, username=user_id, password=user_pw)	
		if user is not None :
			auth.login(request, user)	
			return redirect('home')
		else :
			return render(request, 'login.html', {'error': 'ID or PW is incorrect'})
	else :
		return render(request,'login.html')
	
def register(request) :
	if request.method == "POST" :
		if request.POST['user_pw1'] == request.POST['user_pw2'] :
			user = User.objects.create_user( username=request.POST['user_id'], password=request.POST['user_pw1'],
		        				 email=request.POST['user_email'])
			auth.login(request,user)
			return redirect('home')
		else :
			return render(request, 'register.html', {'error' : 'Confirm PW'})	
	return render(request, 'register.html')
	
def logout(request) :
	auth.logout(request)
	return render(request, 'login.html')
	
def home(request):
	blogs = Blog.objects
	return render(request, 'home.html', {'blogs' : blogs})

def detail(request, blog_id):
	blog_detail = get_object_or_404(Blog, pk=blog_id)
	return render(request, 'detail.html', {'blog' : blog_detail})

def write(request) :
	return render(request, 'write.html')

def create(request) :
	blog = Blog()
	blog.title = request.GET['title']
	blog.writer = request.user.get_username()
	blog.body = request.GET['body']
	blog.pub_date = timezone.datetime.now()
	blog.save()
	return redirect('home')

def modify(request, blog_id):
	blog = get_object_or_404(Blog, pk=blog_id)
	if request.method == "POST" :
		blog.title= request.POST['title']
		blog.body = request.POST['body']
		blog.pub_date= timezone.datetime.now()
		blog.save()
		return redirect('/blog/'+str(blog.id)) 
	else :
		return render(request, 'modify.html', {'blog' : blog}) 
		
def delete(request, blog_id):
	blog = Blog.objects.get(pk=blog_id)
	blog.delete()
	return redirect('home')	
``` 

### pk, path converter, get_object_or_404
Link : <a href="https://ssungkang.tistory.com/entry/Django-06pk-path-converter-getobjector404%EB%9E%80?category=320582" target="_blank">ssungkang.tistory.com/entry/Django-06pk-path-converter-getobjector404%EB%9E%80?category=320582</a>

### render와 redirect 차이점
Link : <a href="https://ssungkang.tistory.com/entry/Django-render-와-redirect-의-차이?category=320582" target="_blank">ssungkang.tistory.com/entry/Django-render-와-redirect-의-차이?category=320582</a>
```python
render(request, template_name, context=None, content_type=None, status=None, using=None)

#request : default 값으로 request 사용하면 됨.
#template_name : 불러오고 싶은 template(html 등) 파일
#context : views.py에서 사용한 변수를 불러올 template에 인자로 넘겨줄 수 있음.
#          dict형으로 key 값이 template에서 사용할 변수명
#          value가 넘겨줄 파이썬 변수
```
```python
redirect(to, permanent=False, *args, **kwargs)

#to : 이동할 url 주소로 상대, 절대 url 모두 가능함. 
#     urls.py 에 name 을 정의하고 이를 많이 사용함. 
#     단순히 URL로 이동하는 것으로 render 처럼 context 값을 넘기지는 못함.

```

## 3. urls.py
```
url를 지정하는 파일

해당 url로 접근 시, views.py에 있는 함수를 실행시킴.
즉, 그 함수가 html 파일을 화면에 보여주는 원리 (using render, redirect)

--> 앱 폴더에서 urls.py를 작성 
--> 그 내용을 프로젝트 폴더에 있는 urls.py에도 작성
```

### 3-1. 프로젝트 폴더 안에 있는 urls.py
```python
"""work URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
	path('admin/', admin.site.urls) 
	path('앱이름/', include('앱이름.urls')), 
]
```
위에 예제들이 있고 그 중 3번째를 사용


### 3-2. 앱 안에 있는 urls.py
```python
from django.urls import path
import 앱이름.views  
# 각각의 app에 만든 하위 urls.py는 path에서 VIEW의 정보를 argument로 받기 때문에, 
# 반드시 views.py를 import 해야함!!

urlpatterns = [
	# path('admin/', admin.site.urls)  
    # 기본 urls.py에만 있으면 됨. 거기에 없으면 오류 메세지 출력 
	
  path('',앱이름.views.함수, name='index'),  
    # 경로가 ''이면 접속 시 기본적으로 index.html을 불러오는 것. 
]
```
```python
# Example code

from django.urls import path
import blog.views

urlpatterns = [
	path('', blog.views.login, name = 'login'), 
    # 기본 페이지가 login 페이지, views.py에 있는 login 함수를 통해 login.html 페이지를 불러옴.
	path('blog/register', blog.views.register, name = 'register'),
	path('blog/board', blog.views.home, name='home'),
	path('blog/<int:blog_id>', blog.views.detail, name = 'detail'),
	path('blog/write', blog.views.write, name = 'write'),
	path('blog/create', blog.views.create, name = 'create'),
	path('blog/<int:blog_id>/modify', blog.views.modify, name = 'modify'),
	path('blog/<int:blog_id>/delete', blog.views.delete, name = 'delete'),
	path('blog/logout', blog.views.logout, name = 'logout'),
]

# <int:blog_id>는 path converter임.
```

## 4. 웹 페이지 상에 보여줄 html 파일 작성
앱 폴더 안에 templates 폴더 생성 후 (앱과 동일한 이름 폴더 생성) html 파일 작성 (views.py에 사용될 html파일)  
ex) login.html, detail.html, register.html 등등..  

### 4-1. 템플릿 언어
<a href="https://ssungkang.tistory.com/entry/Django-%ED%85%9C%ED%94%8C%EB%A6%BF-%EC%96%B8%EC%96%B4%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90?category=320582" target="_blank">여기서 템플릿 언어라는 것이 사용됨.</a>   
템플릿 언어를 통해 더 많은 기능을 구현 가능  

### 4-2. 쿼리셋과 메소드
<a href="https://ssungkang.tistory.com/entry/Django-05-queryset-%EA%B3%BC-method?category=320582" target="_blank">참고</a>  

## 5. admin.py
```
python3 manage.py createsuperuser -> admin 계정 생성
```
```python
from django.contrib import admin
from .models import 클래스 이름

admin.site.register(클래스이름)

''' ex)
from django.contrib import admin
from .models import Blog

admin.site.register(Blog) # admin 페이지에 Blog 객체 추가
'''
```
