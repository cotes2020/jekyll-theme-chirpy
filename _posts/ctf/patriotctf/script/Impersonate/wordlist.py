#! /usr/bin/env python3
from hashlib import sha256
import os
static_part= "secret_key_2024092200"
wordlist = open("brute.txt","w")
for i in range(0,9000):
    if len(str(i)) == 1:
       secret = static_part + "000" + str(i)
       secret = sha256(secret.encode()).hexdigest()
       secret = secret + "\n"
       wordlist.write(secret)
    elif len(str(i)) == 2:
         secret = static_part + "00" + str(i) 
         secret = sha256(secret.encode()).hexdigest()
         secret = secret + "\n"
         wordlist.write(secret)
    elif len(str(i)) == 3:
         secret = static_part + "0" + str(i) 
         secret = sha256(secret.encode()).hexdigest()
         secret = secret + "\n"
         wordlist.write(secret)
    elif len(str(i)) == 4:
         secret = static_part + str(i) 
         secret = sha256(secret.encode()).hexdigest()
         secret = secret + "\n"
         wordlist.write(secret)
    elif len(str(i)) == 5:
         secret = static_part + "0" + str(i) 
         secret = sha256(secret.encode()).hexdigest()
         secret = cookiez(secret) + "\n"
         wordlist.write(secret)
    else:
         secret = static_part + str(i)
         secret = sha256(secret.encode()).hexdigest()
         secret = cookiez(secret) + "\n"
         wordlist.write(secret)
    print(f"Writing {secret}")

wordlist.close()
        


