#! /usr/bin/env python3

output: str = open("output.txt","r").read()
flag = ''.join(chr(ord(i) ^ 3) for i in output)
print(flag)
