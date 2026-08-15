--------------

### CTF-: PwnMe2025

-------------

![image](https://github.com/user-attachments/assets/3b86d90a-3657-4080-b77a-343cf530da5e)

------------

### Challenges

------------

- Web
  - Profile-Editor

------------

### Profile Editor

------------

![image](https://github.com/user-attachments/assets/f5d46cca-4da6-49cb-8410-c6daa81846e2)

------------

- Vulnerable source code-:

```python3
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not session.get('username'):
        return redirect('/login')
    
    profiles_file = 'profile/' + session.get('username')
    profiles_file = profiles_file if profiles_file.endswith('.html') else profiles_file + '.html'

    if commonpath((app.root_path, abspath(profiles_file))) != app.root_path:
        return render_template('error.html', msg='Error processing profile file!', return_to='/profile')

    if request.method == 'POST':
        with open(profiles_file, 'w') as f:
            f.write(request.form.get('profile'))
        return redirect('/profile')
    
    profile=''
    if exists(profiles_file):
        with open(profiles_file, 'r') as f:
            profile = f.read()

    return render_template('profile.html', username=session.get('username'), profile=profile)
@app.route('/show_profile', methods=['GET', 'POST'])
def show_profile():
    if not session.get('username'):
        return redirect('/login')
    
    profiles_file = 'profile/' + session.get('username')

    if commonpath((app.root_path, abspath(profiles_file))) != app.root_path:
        return render_template('error.html', msg='Error processing profile file!', return_to='/profile')

    profile = ''
    if exists(profiles_file):
        with open(profiles_file, 'r') as f:
            profile = f.read()

    return render_template('show_profile.html', username=session.get('username'), profile=profile)
```

---------------

### Explanation-:

---------------

- The chain for exploitation is `Arbitrary file write and read to Remote Code Execution`.The code picks the username and use it to create an html file which we can see as a `jinja template`.The username query is not filtered which allows us to control the path and rewrite other files e.g templates in the `templates` directory.The mode of exploitation revoles around "controlling the path to one of the templates with the username e.g `../templates/error.html` which will be read,later we can inject the template with code like `config.__init.\_\_globals\_\_.os.popen('ls').read()`.

```python3
profiles_file = 'profile/' + session.get('username')
    profiles_file = profiles_file if profiles_file.endswith('.html') else profiles_file + '.html'

    if commonpath((app.root_path, abspath(profiles_file))) != app.root_path:
        return render_template('error.html', msg='Error processing profile file!', return_to='/profile')

    if request.method == 'POST':
        with open(profiles_file, 'w') as f:
            f.write(request.form.get('profile'))
        return redirect('/profile')
    
    profile=''
    if exists(profiles_file):
        with open(profiles_file, 'r') as f:
            profile = f.read()
```

- This vulnerable can only be exploited if the templates get reloaded which is set to `True` as seen in the code.

```python3
app.config['TEMPLATES_AUTO_RELOAD'] = True
```

----------

### Exploiting it

------------

- Create an account with username `../templates/error.html`

![image](https://github.com/user-attachments/assets/714d58e2-dd30-45b6-bf57-63725a9af3de)

- We have write access to the file,I update the template with the code shown in the image.

![image](https://github.com/user-attachments/assets/01ab5ca9-0c3a-4834-b4cd-8e1a63b77a27)

- In order to get the flag,we need to find a way to trigger `error.html`.I was able to trigger it by registering an already registered user.Code-:

```python3
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if username in users:
            return render_template('error.html', msg='Username already taken!', return_to='/register')  ### Template gets triggered here

```

- Flag-: ```PWNME{afb7daa3a595522882f6a9efae30c4ec}```

![image](https://github.com/user-attachments/assets/3be9954a-3957-4a75-98bc-96df0555c7b3)

-------------

### Thanks for Reading

-------------

