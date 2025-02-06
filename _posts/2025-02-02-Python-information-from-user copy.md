---
title: ورودی از کاربر
date: 2025-02-02
category: [اموزش پایتون]
priority: 2
---
>اهداف یادگیری\
>پس از این بخش، شما خواهید دانست:
>- چگونه برنامه‌ای بنویسید که از ورودی کاربر استفاده کند.
>- چگونه از متغیرها برای ذخیره ورودی و چاپ آن استفاده کنید.
>- چگونه رشته‌ها را ترکیب کنید.
{: .prompt-tip }
## مقدمه

ورودی به هر اطلاعاتی اشاره دارد که کاربر به برنامه می‌دهد. به طور خاص، دستور `input` در پایتون یک خط ورودی که توسط کاربر تایپ شده است را می‌خواند. همچنین می‌توان از آن برای نمایش پیام به کاربر و درخواست ورودی خاص استفاده کرد.

برنامه زیر نام کاربر را با دستور `input` می‌خواند و سپس آن را با دستور `print` چاپ می‌کند:

```python
name = input("What is your name? ")
print("Hi there, " + name)
```

اجرای این برنامه ممکن است به این شکل باشد (ورودی کاربر به رنگ قرمز):

```
What is your name? Paul Python
Hi there, Paul Python
```

چیزی که این برنامه چاپ می‌کند تا حدی به ورودی کاربر بستگی دارد. این یعنی اجرای برنامه می‌تواند به این شکل نیز باشد:

```
What is your name? Paula Programmer
Hi there, Paula Programmer
```

کلمه `name` در این برنامه یک متغیر است. در زمینه برنامه‌نویسی، متغیر محلی برای ذخیره مقداری مانند یک رشته یا عدد است. این مقدار می‌تواند بعداً استفاده شود و همچنین می‌تواند تغییر کند.

# نام‌گذاری متغیرها

در اصل، متغیرها می‌توانند به طور آزادانه نام‌گذاری شوند، البته در محدوده‌های مشخص شده توسط زبان پایتون.

یک روش رایج بین‌المللی در برنامه‌نویسی این است که متغیرها را به زبان انگلیسی نام‌گذاری کنند، اما ممکن است با کدهایی مواجه شوید که در آن‌ها متغیرها به زبان‌های دیگر، مانند زبان مادری برنامه‌نویس، نام‌گذاری شده‌اند. نام متغیر تأثیر مستقیمی بر محتوای آن ندارد، بنابراین از این نظر نام مهم نیست. با این حال، نام‌گذاری منطقی و به زبان انگلیسی می‌تواند در درک عملکرد کد کمک کند.

# ارجاع به یک متغیر

یک متغیر می‌تواند بارها در یک برنامه مورد ارجاع قرار گیرد:

```python
name = input("What is your name? ")

print("Hi, " + name + "!")
print(name + " is quite a nice name.")
```

اگر کاربر نام `Paul Python` را وارد کند، این برنامه خروجی زیر را چاپ می‌کند:

```
What is your name? Paul Python
Hi, Paul Python!
Paul Python is quite a nice name.
```

بیایید نگاهی دقیق‌تر به نحوه استفاده از دستور `print` در بالا بیندازیم. داخل پرانتزهای دستور، هم متن داخل نقل‌قول‌ها و هم نام متغیرهایی که به ورودی کاربر اشاره می‌کنند وجود دارد. این‌ها با عملگر `+` ترکیب شده‌اند که دو رشته را به یک رشته واحد تبدیل می‌کند.

رشته‌ها و متغیرها می‌توانند به طور آزادانه ترکیب شوند:

```python
name = input("What is your name? ")

print("Hi " + name + "! Let me make sure: your name is " + name + "?")
```

اگر کاربر نام `Ellen Example` را وارد کند، این برنامه خروجی زیر را چاپ می‌کند:

```
What is your name? Ellen Example
Hi Ellen Example! Let me make sure: your name is Ellen Example?
```

# بیش از یک ورودی

یک برنامه می‌تواند بیش از یک ورودی درخواست کند. توجه کنید که چگونه هر دستور `input` مقدار دریافتی را در یک متغیر متفاوت ذخیره می‌کند.

```python
name = input("What is your name? ")
email = input("What is your email address? ")
nickname = input("What is your nickname? ")

print("Let's make sure we got this right")
print("Your name: " + name)
print("Your email address: " + email)
print("Your nickname: " + nickname)
```

این برنامه ممکن است خروجی زیر را چاپ کند:

```
What is your name? Frances Fictitious
What is your email address? frances99@example.com
What is your nickname? Fran
Let's make sure we got this right
Your name: Frances Fictitious
Your email address: frances99@example.com
Your nickname: Fran
```

اگر از همان متغیر برای ذخیره بیش از یک ورودی استفاده شود، هر مقدار جدید جایگزین مقدار قبلی می‌شود. به عنوان مثال:

```python
address = input("What is your address? ")
print("So you live at address " + address)

address = input("Please type in a new address: ")
print("Your address is now " + address)
```

یک مثال از اجرای برنامه:

```
What is your address? Python Path 101, Flat 3D
So you live at address Python Path 101, Flat 3D
Please type in a new address: New Road 999
Your address is now New Road 999
```

این بدان معناست که اگر از همان متغیر برای ذخیره دو ورودی متوالی استفاده شود، پس از جایگزینی مقدار اول با مقدار دوم، هیچ راهی برای دسترسی به مقدار اول وجود ندارد:

```python
address = input("What is your address? ")
address = input("Please type in a new address: ")

print("Your address is now " + address)
```

مثال دیگری از خروجی برنامه:

```
What is your address? Python Path 10
Please type in a new address: Programmer's Walk 23
Your address is now Programmer's Walk 23
```