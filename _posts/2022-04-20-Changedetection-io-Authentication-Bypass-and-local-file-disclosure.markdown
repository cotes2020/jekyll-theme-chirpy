---
layout:	post
title:	"Changedetection CSRF to Local File Disclosure"
date:	2022-04-20
categories: [Hacking, Code Review]
image: /img/changedetection-io-3.PNG
tags: [Hacking, Code Review, Django, Web]
---


# INTRODUCTION
Changedetection.io is a famous open source self hosted website change detection monitoring and notification service with over 4k github stars and over 1m+ docker pull. In this writeup, i will show you a bug that i found, that allows an unauthenticated user to bypass the authentication and also achieve a local file disclosure post authentication. For this attack to be successful, the victim should have selenium installed

# CODE REVIEW
The application is made in flask. Routing in flask is done using the decorator `route()`. You can learn more about flask auditing in <https://github.com/tomorroisnew/Code-Review-Notes/blob/main/Python/Flask.md>    
While going through all route, i found the route for the settings page.
```python
    @app.route("/settings", methods=['GET', "POST"])
    @login_required
    def settings_page():
        #CODE
```
Here, you can see that it allows both GET and POST method.
```python
def settings_page():
        #Unimportant Snippet
        if request.method == 'GET':
            #Unimportant Snippet

            # Password unset is a GET, but we can lock the session to always need the password
            if not os.getenv("SALTED_PASS", False) and request.values.get('removepassword') == 'yes': #Check for removepassword query parameter
                from pathlib import Path
                datastore.data['settings']['application']['password'] = False # REMOVE PASSWORD
                flash("Password protection removed.", 'notice')
                flask_login.logout_user()
                return redirect(url_for('settings_page'))
```
Here, you can see that if we have `removepassword`, as a query parameter, it will remove the password. And it is also done in a get request making it vulnerable to CSRF attacks.

# CSRF POC
I set up a local instance of changedetection.io to test it up.     
I logged in and setup up a password on my instance. Then, i visited `http://<host>/settings?removepassword=yes`, and now, my password is removed
![](/img/changedetection-io-1.gif)
With the password gone, we now have access to all authenticated functions

# Local File Disclosure
While checking the code, it seems like there is no scheme check when supplying a url allowing us to use the `file:///` scheme and read local files. The class responsible for fetching a site is the Fetcher class which is an abstract class
```python
class Fetcher():
    error = None
    status_code = None
    content = None
    headers = None

    fetcher_description ="No description"

    #Unimportant Snippet

    @abstractmethod
    def run(self, url, timeout, request_headers, request_body, request_method):
        # Should set self.error, self.status_code and self.content
        pass

    #Unimportant Snippet
```
The most important function is run. Two classes implements this Abstract class, these are `html_requests` and `html_webdriver`. `html_requests` uses `requests` for fetching sites while `html_webdriver` uses selenium. This is the run function of html_requests    
When using `html_requests`, the server fetches a site using these lines of code
```python
class html_requests(Fetcher):
    def run(self, url, timeout, request_headers, request_body, request_method):

        r = requests.request(method=request_method,
                         data=request_body,
                         url=url,
                         headers=request_headers,
                         timeout=timeout,
                         verify=False)
``` 
Unfortunately for us, by default, `requests` doesnt allow the file scheme so we cant do any much about it. But we can still fetch local ips like `169.254.169.254`. 
```sh
>>> r = requests.request(method="GET", url="file:///etc/passwd")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Brandon\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\api.py", line 61, in request
    return session.request(method=method, url=url, **kwargs)
  File "C:\Users\Brandon\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\sessions.py", line 529, in request
    resp = self.send(prep, **send_kwargs)
  File "C:\Users\Brandon\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\sessions.py", line 639, in send
    adapter = self.get_adapter(url=request.url)
  File "C:\Users\Brandon\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\sessions.py", line 732, in get_adapter
    raise InvalidSchema("No connection adapters were found for {!r}".format(url))
requests.exceptions.InvalidSchema: No connection adapters were found for 'file:///etc/passwd'
```
`html_webdriver` on the other hand allows any scheme, which is good for us. So, i anbled selenium, and fetched `/etc/passwd` and it works

# POC
![](/img/changedetection-io-2.gif)

Thanks for reading.