---
title: Django 기초
author: hyun geun
date: 2023-09-05
categories:
    - Django
tags:
    - [Django]
math: true
mermaid: true
image: 
comments: true
---
## 1. Django 설치 및 실행

- 설치 : pip install django
- 프로젝트 생성 : django-admin startproject webproj
- 프로젝트 파일 확인 : ls
- 프로젝트 폴더로 이동 : cd webproj
- 서버 실행 : python manage.py runserver
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/44af7c30-c3d6-4b58-bda1-32e95285c185/%EC%BA%A1%EC%B2%98.png)
    
- 접속 확인

## 2. Django 구성 요소

- 파일들
    - manage.py : 장고를 실제로 실행하는 파일 →runserver를 통해 이 파일을 실행
    - __ init__.py : django 폴더가 파이썬 모듈로써 인식되게 해줌
    - asgi.py, wsgi.py : 서버에서 django 프로젝트 파일
    - setting.py : 환경설정
        - Secret key :
        - Debug
        - Allowed Hosts : 화이트리스트
        - Root_URLCONF : URL 담당하는 곳 주소
        - Templates : 화면 구성 요소
        - Databases : 데이터베이스 저장 관련
        - Static_URL : 정적 요소 저장 경로 표시
    - urls.py

```python
urlpatterns = [
    path('', index),  # 127.0.0.1
    path("admin/", admin.site.urls),  # 127.0.0.1/admin
]
```

- 한 프로젝트는 여러 App(블로그, 카페, 메일 등등과 같이)으로 구성되어 있다.

## 3. Django App 생성

- App 생성 : django-admin startapp <app_name>
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97ffcc88-d289-4362-9992-1db360200ecc/%EC%BA%A1%EC%B2%98.png)
    
    (주의! project 폴더 내로 이동해서(cd) 할 것!)
    
- 파일 확인
    - admin.py : admin 페이지
    - apps.py : app 설정
    - **models.py : database 스키마 작성 부분**
    - tests.py : 테스트용
    - views.py : view 관리

## 4. Django 디자인 패턴→MVT Pattern

![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d5ac01-8a26-4704-9b48-ee152f88e92c/%EC%BA%A1%EC%B2%98.png)

- MVT Pattern
    - user Req → Django
        - URL(urls.py)에서 요청 주소가 있으면 **View(views.py**)로 전달
        - 데이터베이스를 처리해야 하는 요청이면 Model(models.py → ORM으로 관리)에서 처리
        - 웹페이지를 보여줘야 한다면 Template에서 관리(.html + template(for, if 가능) 언어)

## 5. View로 Request Handling 하기

- Req 처리 → views.py에서 / 어떤 경로로 오는 Req에 대해서 특정 처리를 진행 할 건가? 프로젝트 폴더의 urls.py에서

```python
#app 폴더 내 views.py
def index(request):
    return HttpResponse("Hello World!")
    #return render(request, 'index.html', {})
```

```python

#프로젝트 내 동일 이름 폴더 내 urls.py
from homepage.views import index
#실행 경로 명시
urlpatterns = [
    path('', index),  # 127.0.0.1
    path("admin/", admin.site.urls),  # 127.0.0.1/admin
]
```

- 프로젝트 폴더 내 settings.py에서 새로 생성한 App의 이름 추가해서 App 인식시키기

```python
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    'homepage', #여기에 추가!
]
```

- 서버 실행해서 정상 작동 확인

```python
python [manage.py](http://manage.py/) runserver
```

```python
python [manage.py](http://manage.py/) runserver
```

- admin 페이지 들어가기 : http://127.0.0.1:8000/admin/
    
    → 데이터베이스 관리 등을 진행
    
- 관리자 계정 생성(CLI 에서 진행)
    - admin 데이터베이스는 자동 생성되므로 이를 마이그레이션을 진행(commit 처럼)
    
    ```python
    python [manage.py](http://manage.py/) migrate
    ```
    
    - 관리자 계정 생성
    
    ```python
    python [manage.py](http://manage.py/) createsuperuser
    ```
    
    - 접속 확인
        
        ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45a50805-3525-4b60-b4d4-9deb4ce55ec7/%EC%BA%A1%EC%B2%98.png)
        

## 6. Template로 보여줄 화면 구성

- App 내 urls.py
    - render 함수 → (request, ‘.html’, {딕셔너리 형태})
        
                                요청 사항, 보여줄 파일, 기타 인자들
        
    - 문서 위치 지정
        - 프로젝트 내 settings.py 의 TEMPLATS 딕셔너리 내 “DIRS” 에서 진행
        
        ```python
        TEMPLATES = [
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [
                    os.path.join(BASE_DIR, "homepage", "template"),
                ],
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.debug",
                        "django.template.context_processors.request",
                        "django.contrib.auth.context_processors.auth",
                        "django.contrib.messages.context_processors.messages",
                    ],
                },
            },
        ]
        ```
        
        - 
        - 

## 7. Model

- Django에서 데이터베이스를 담당하는 곳(models.py)
- SQL을 이용하지 않고, ORM(Object Relational Mapping)을 통해서도 관리 가능

- 데이터베이스 생성

```python
class <모델 이름>(modles.Model):
```

- Attribute 생성
    - 속성 값
        - default : 기본 값
        - null : null값 허용?

```python
class Coffee(models.Model):
	#문자열 Attribute는 max_length 반드시 지정!
    name = models.CharField(default="",null=False)
    price = models.IntegerField()
    is_ice = models.BooleanField()
		'''
    문자열 : CharField
    숫자 : IntegerField, SmallIntegerField,...
    논리형 : BooleanField
    시간 : DateTimeField 등등...
    '''

```

- admin.py 에서 model 연동

```python
#모델 위치 지정
from .models import Coffee
# Register your models here.
#모델 등록
admin.site.register(Coffee)
```

- Database 변동 사항을 settings.py에 기입
    - django에서는 git의 commit 처럼 migration 단위로 관리
        
        → DB 필드 정보를 수정하더라도, 바로 수정되지 않고, migration 진행 후 반영
        
        *python manage.py migrate : 클래스로 만든 모델을 실제로 객체로 만드는 과정
        
        - (git add 와 같은/ 마이그레이션 생성)
        
        ```python
         python manage.py makemigrations homepage
        ```
        
        - git push와 같은/ 생성한 마이그레이션을 실제 db에 반영)
        
        ```python
        python [manage.py](http://manage.py) migrate
        ```
        
    - ㅏ
        
        
- 적용 확인
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afd47dba-66d8-43ee-baff-7529954e2e23/%EC%BA%A1%EC%B2%98.png)
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b05a215b-c400-4f54-be03-3b67ae3fdfd3/%EC%BA%A1%EC%B2%98.png)
    
    Coffee Column 이름 바꿔주자
    
    ```python
    class Coffee(models.Model):
    		#객체 생성할 때마다 이름을 바꿔준다
        def __str__(self):
            return self.name
        name = models.CharField(default="",null=False,max_length=30)
    ```
    

- 만든 model을 Template 에 보여주기
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d5ac01-8a26-4704-9b48-ee152f88e92c/%EC%BA%A1%EC%B2%98.png)
    
    <다시 한번 보는 Django 모델 구조>
    
    - 모델에서 템플릿으로 db를 전달하기 위해서는 무조건 View를 거쳐야 함
    
    ```python
    #view.py
    #모델 가져오기
    from .models import Coffee
    
    # Create your views here.
    
    def introduce(request):
        return render(request,'index.html',{})
    
    def coffee_view(request):
        #Coffee db의 모든 행을 전부 가져오기(SELECT * FROM Coffee)
        coffee_all = Coffee.objects.all()
        #key-value 형태로 전달
        return render(request, 'coffee.html',{"coffee_list":coffee_all})
    
    #urls.py
    from django.contrib import admin
    from django.urls import path,include
    from homepage.views import introduce, coffee_view
    app_name ='posts'
    
    urlpatterns = [
        path('',introduce),#127.0.0.1/
        path('coffee/',coffee_view), #127.0.0.1/coffee
        path('admin/', admin.site.urls),#127.0.0.1/admin
    ]
    ```
    
    ```html
    <!DOCTYPE html>
    <html>
        <head>
            <title>Coffee List</title>
        </head>
        
        <body>
            <h1>My Coffee List</h1>
           
            <p>{{coffee_list}}</p>
        </body>
    </html>
    ```
    
    - 여기에 쓰인 {{변수}}는 템플릿 변수로, 뷰에서 템플릿으로 객체를 전달할 수 있다.
        
        자세한 것은 여기로
        
        [[Django] 템플릿 언어](https://velog.io/@hidaehyunlee/Django-템플릿-언어)
        
- 탬플릿 변수로 해당 데이터 베이스를 탬플릿에 보여주기

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Coffee List</title>
    </head>
    
    <body>
        <h1>My Coffee List</h1>
       

        {%for coffee in coffee_list %}
            <p>{{coffee.name}},{{coffee.price}}</p>
        {%endfor %}
    </body>
</html>
```

하지만 맨날 admin으로 들어가서 DB를 관리하는 것은 매우 귀찮은 일이다.

POST 요청을 통해서 DB를 수정해보자

## 8. Form으로 Template에서 Model 수정

- form.py 생성

```python
#forms.py
from django import forms
from .models import Coffee #Model 호출

#모델 관련 form은 ModelForm을 상속받아서 생성한다
class CoffeeForm(forms.ModelForm):
    #어떤 모델이 쓰여야하는지 지정해주어야 함
    class Meta:
        model = Coffee
        fields = ('name', 'price', 'is_ice')
```

- 탬플릿으로 전달하는 함수 작성

```python
#views.py
from .models import Coffee
from .forms import CoffeeForm
# Create your views here.

def introduce(request):
    return render(request,'index.html',{})

def coffee_view(request):
    #Coffee db의 모든 행을 전부 가져오기(SELECT * FROM Coffee)
    coffee_all = Coffee.objects.all()#.get(),.filter(),....
    #클래스로 객체 생성
    form = CoffeeForm()
    #key-value 형태로 전달
    return render(request, 'coffee.html',{"coffee_list":coffee_all,"coffee_form":form})
```

- form에서 전달한 객체를 탬플릿에서 보여주기
    
    ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2670141e-29b9-4e9b-b7e2-71ab9c7fb56b/%EC%BA%A1%EC%B2%98.png)
    

앗! 하지만 db에 입력하는 상호작용이 없다

- 버튼 만들기

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Coffee List</title>
    </head>
    
    <body>
        <h1>My Coffee List</h1>  
        {% for coffee in coffee_list %}
            <p>{{coffee.name}},{{coffee.price}}</p>
        {% endfor %}
        <form method="POST">
            {{ coffee_form.as_p }}
            <button type="submit">Save</button>
        </form>
    </body>
</html>
```

- 버튼 내용 설명
    - form에서 받아온 내용을 GET하기 위해서는 POST
    - 받아오는 작업 : type=submit
    - as_p : 출력될 때마다 개행
        
        ![캡처.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4cae3c2b-97c4-4a2a-bf53-becb7888b41b/%EC%BA%A1%EC%B2%98.png)
        
    
    form의 경우에는 기본적으로 보안 옵션을 작성해주어야 함
    
    →CSRF 토큰을 form안에 삽입해주어야 한다
    
    ```html
    <form method="POST">{% csrf_token %}
                {{ coffee_form.as_p }}
                <button type="submit">Save</button>
    ```
    
- POST 요청에 대한 동작
    
    ```python
    #views.py
    def coffee_view(request):
        #Coffee db의 모든 행을 전부 가져오기(SELECT * FROM Coffee)
        coffee_all = Coffee.objects.all()#.get(),.filter(),....
        #if req = POST
            #POST를 바탕으로 Form을 완성하고
            #Form이 유효하면 저장
        if request.method == "POST":
            form = CoffeeForm(request.POST)# 완성된 Form
            if form.is_valid(): # 채워진 Form이 유효하다면
                form.save() # Form 내용을 Model에 저장
    
        #클래스로 객체 생성
        form = CoffeeForm()
        #key-value 형태로 전달
        return render(request, 'coffee.html',{"coffee_list":coffee_all,"coffee_form":form})
    ```
    
- ㅇㅇ
***
[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}