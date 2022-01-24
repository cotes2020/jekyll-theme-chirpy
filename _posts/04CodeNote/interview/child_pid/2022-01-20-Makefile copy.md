---
title: Interview note
date: 2022-01-20 11:11:11 -0400
categories: [04CodeNote]
tags: []
math: true
image:
---
- [Interview note](#interview-note)
  - [GG](#gg)

---

# Interview note

---

## GG

get all the child pid


```py
# file.txt:
# parent pid, child pid, exe
# 110,111,chrome
# 111,222,target
# 333,444,other.exe
# 110,123,firefox
# 123,223,target

# output.txt
# 222
# 223

def child_pid(file):
    child_list = []
    broswer_list = ['chrome', 'firefox']
    with open(file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            p_pid = lines[i].split(",")[2].strip()
            if(p_pid in broswer_list):
                c_pid = lines[i+1].split(",")[1]
                child_list.append(c_pid)
    f = open('output.txt', 'w')
    for pid in child_list:
        f.write(pid + "\n")
    f.close()
        
file = 'file.txt'
child_pid(file)
```






.