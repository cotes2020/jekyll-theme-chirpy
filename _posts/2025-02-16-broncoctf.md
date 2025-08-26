---
title: BroncoCTF 2025 - Writeups
date: 2025-02-18 15:00:30 
categories: [Write-ups, BroncoCTF2025]
tags: [misc, crypto, forensic, OSINT]     
description: not web chall xD
math: true
---

Firstly, i am thrilled to announce that RaptX has secured the 6th position in BroncoCTF 2025. Honesly, challenges were not that hard, the host seems dont have deeply knowledge in web then there are only 3 web challenge (all of them are easy too). There are only some misc and rev challenges make me think a lot. This time, i solved 6 challenges including crypto, forensic, OSINT and misc and lastly i spend 5+ hours on a rev challenge. So, imma rush this writeup really quick. (CHATGPT is my best friend when on those categories)

![bronco](/commons/ctfs/broncoCTF/bronco.png)

## Crypto

### Mid PRNG (386 points, 7th solve/122 solves)

![midprng](/commons/ctfs/broncoCTF/midprng.png)

Source code:

```python
import bad_prng
import random

flag = ""

output = []
random = bad_prng.generate_seed()
for c in flag:
    random = bad_prng.rand_word()
    output.append(random ^ ord(c))

print(bytes(output).hex())

```

Yes, i know this is bad crypto, but have nothing to do, so i came up with this chall. Overall, it use a bad prng to generate a random number, and then xor with the flag. Firsly, i thought that we can use generated output, xor with first `"bronco{"` to take the randomed numbers, then use it with some tool to recover the seed => Got the flag. But we dont know WTF is `import bad_prng`

After a short time in the internet, i found that possible LCG (idk what it is actually).

![midprng2](/commons/ctfs/broncoCTF/midprng1.png)
So i told chatgpt give me crack solution with this idea and algorithm. Here is the script:

```python

from z3 import *

numbers = [21, 63, 189, 55, 165, 239, 205] # XORed number with "bronco{" to produce the output
output = bytes.fromhex("774dd259c680b6575ec0ece3b61083f721a09a85d69b795a") # generated output

def find_lcg_params():
    s = Solver()
    a = BitVec('a', 32)
    c = BitVec('c', 32)
    
    for i in range(len(numbers)-1):
        s.add((numbers[i] * a + c) % 256 == numbers[i+1])
    
    if s.check() == sat:
        m = s.model()
        return m[a].as_long(), m[c].as_long()
    return None

# Get LCG parameters
a, c = find_lcg_params()
print(f"Found parameters: a={a}, c={c}")

# Generate the full sequence of random numbers
def generate_sequence(length):
    sequence = numbers.copy()  # Start with known numbers
    current = sequence[-1]     # Start from last known number
    
    # Generate remaining numbers using LCG formula: next = (a * current + c) % 256
    while len(sequence) < length:
        next_num = (a * current + c) % 256
        sequence.append(next_num)
        current = next_num
    
    return sequence

# Generate enough random numbers to cover the output length
random_sequence = generate_sequence(len(output))

# Recover the flag
flag = ""
for i in range(len(output)):
    flag += chr(output[i] ^ random_sequence[i])

print("Recovered flag:", flag)

```
> bronco{0k_1ts_n0t_gr34t}


## Misc

### It's a bird (448 points, 2nd solve/83 solves)

![bird](/commons/ctfs/broncoCTF/brid.png)

To be honest, i hate steganography, so i just use prebuilt tool to solve this chall. I read the resource from BroncoCTF then found `Aperi'Solve`. Here is the link to the output after i put the image into the tool [https://www.aperisolve.com/865866ed39b0cb208da18f571420664d](https://www.aperisolve.com/865866ed39b0cb208da18f571420664d)

Then we got a csv file
![csv](/commons/ctfs/broncoCTF/csv.png)

Open the csv file with visible tool, we notice that the column `R` has a discrete value `098, 0114, 0111, 0110, 0099, 0111, ...`. Seems familiar right? yes its our "bronco" in ascii number.

So, we just need to convert the number into ascii, and we got the flag. Here is my script:

```python

import csv
flag = ""
def extract_column_r(filename):
    r_values = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if row has enough columns and column R (index 17) exists
            if len(row) > 17:
                r_value = row[17]
                if r_value:  # Only append non-empty values
                    r_values.append(r_value)
    
    return r_values

# Use the function
filename = 'birb.csv'
r_values = extract_column_r(filename)

# Print all values from column R
print("Values from column R:")
for value in r_values:
    print(value)

# If you want to see them as potential ASCII characters:
print("\nTrying to convert to ASCII:")
for value in r_values:
    flag += chr(int(value))

print(flag)

```

> bronco{i<3planes}

### Flag Saber (490 points, 18th solve/37 solves)

![saber](/commons/ctfs/broncoCTF/saber.png)

YESSS, i really enjoyed this challenge. First look at this, idk what to do. The intended solution was put the file into the saber game, but i havent play this game before. So all my things is on the terminal. Here is the overall of extracted files:

![zip](/commons/ctfs/broncoCTF/zip.png)

The Hard.dat is our main file, that contain every movement of saber, here is the content:
```
{"_version":"2.0.0","_events":[],"_notes":[{"_time":2,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":2,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":7,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":7,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":12,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":12,"_lineIndex":3,"_lineLayer":2,"_type":0,"_cutDirection":5},{"_time":17,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":17,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":22,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":22,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":27,"_lineIndex":3,"_lineLayer":1,"_type":0,"_cutDirection":3},{"_time":27,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":32,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":32,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":37,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":37,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":42,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":42,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":47,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":47,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":52,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":52,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":57,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":57,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":62,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":62,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":67,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":67,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":72,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":72,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":77,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":77,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":82,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":82,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":87,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":87,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":92,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":92,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":97,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":97,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":102,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":102,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":107,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":107,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":112,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":112,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":117,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":117,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":122,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":122,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":127,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":127,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":132,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":132,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":137,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":137,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":142,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":142,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":147,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":147,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":152,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":152,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":157,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":157,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":162,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":162,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":167,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":167,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":172,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":172,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":177,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":177,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":182,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":182,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":187,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":187,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1}],"_obstacles":[],"_customData":{"_bookmarks":[]}}

```

Firstly, i thought that there are binary hidden, the `_type: 0 = red block, 1 = blue block`, concatinate them together and split each 8bits, then convert to ascii. But it seems not the case.

Then i thought that maybe there are some missed cut on blocks, i analyzed them with the file info.dat, but all is correct cut:

```
==================================================
Total blocks analyzed: 76
Correct natural cuts: 76
Non-natural cuts: 0
Natural flow percentage: 100.00%

Sequence of non-natural cuts (might be significant):
Non-natural cut sequence: []
Unusual cuts as ASCII:
```
Not that case too, then the right thought came up to me, i used chatgpt to visualize the block for me, here is my script:

```python
import json

def analyze_positions():
    with open('Hard.dat', 'r') as f:
        data = json.load(f)
    
    # Grid visualization from behind (reversed left-right)
    print("\nGrid visualization from behind (R=Red, B=Blue):")
    for time in sorted(set(note['_time'] for note in data['_notes'])):
        notes_at_time = [n for n in data['_notes'] if n['_time'] == time]
        
        grid = [[' ' for _ in range(4)] for _ in range(3)]
        for note in notes_at_time:
            x = note['_lineIndex']
            y = note['_lineLayer']
            symbol = 'R' if note['_type'] == 0 else 'B'
            # Place in reversed x position (3-x instead of x)
            grid[y][3-x] = symbol
        
        print(f"\nTime {time}:")
        for row in reversed(grid):  # Reverse to show top row first
            print(row)

analyze_positions()
```
![grid](/commons/ctfs/broncoCTF/grid.png)



Yes, at the time i see that, i remembered a challenge from another CTF, that use semaphore to encode the flag. This is our familiar `"bronco"` in semaphore. Then i used `http://www.semaphorify.info/` and [https://bobbiec.github.io/semaphore-decoder.html](https://bobbiec.github.io/semaphore-decoder.html) to decode the grid, and i got the flag.

> bronco 0hit5th4tk1nd0ff1ag

## OSINT

### Secure Copy Shenanigans (455 points, 15th solve/77 solves)

![scp](/commons/ctfs/broncoCTF/scp.png)

Actually, my teammate have the idea first, but they got confused when they cant search the name of 2nd SCP, i have that thought too, but after i search each SCP name on CHATGPT, ask them about what its name, concatinate them together, i got the flag.

![scp1](/commons/ctfs/broncoCTF/scp1.png)

> i forgot the flag

### Filling Some Data (366 points, 24th solve/132 solves)

![fill](/commons/ctfs/broncoCTF/fill.png)

CHATGPT is my best friend, i just ask them abt that, he tells me should search on [California Secretary of State Business Search portal](https://bizfileonline.sos.ca.gov/search/business), put the school name on that, view the history and find the restat:

![fill2](/commons/ctfs/broncoCTF/fill1.png)

> bronco{A0693181}

## Forensic
### Logged (446 points, 28th solve/117 solves)

![logged](/commons/ctfs/broncoCTF/logged.png)

You can download the log file [here](https://github.com/RaptX/CTF-Writeups/blob/main/commons/ctfs/broncoCTF/logged.txt)

I putted some lines of log on CHATGPT, then know its x11 key definition, you can see the source code [here](https://github.com/D-Programming-Deimos/libX11/blob/master/c/X11/keysymdef.h)

Then make a python translator to keys, replay it and we got the flag, here is the script:

```python
import csv

with open("./keys.log", "r") as f:
    reader = csv.reader(f)
    dat = list(reader)

dat2 = []
for i in dat:
    if i[0].startswith("KeyPress") or i[0].startswith("KeyRelease"):
        dat2.append(i)

info = dat2
c_min = len(info[0])
c_max = len(info[0])
for i in info:
    if len(i) > c_max:
        c_max = len(i)
    if len(i) < c_min:
        c_min = len(i)
print(f"Colum nb: max {c_max}, min {c_min}")
print([i for i in info if i[-1].strip() != "same_screen YES"])

info2 = [
    ["P" if i[0].startswith("KeyPress") else "R", i[-2].split()[-1].strip()[:-1]]
    for i in info
]
info3 = [[i[0], int(i[1], 16)] for i in info2]


def s2t(info):
    sym = info[1]
    char = None
    spec = True
    if sym <= 127 and sym >= 0:
        char = chr(sym)
        spec = False
    elif sym == 0xFFE1:
        char = "<XK_Shift_L>"
    elif sym == 0xFFE2:
        char = "<XK_Shift_R>"
    elif sym == 0xFF0D:
        char = "<XK_Return>"
    elif sym == 0xFF1B:
        char = "<XK_Escape>"
    elif sym == 0xFF08:
        char = "<XK_BackSpace>"
    elif sym == 0xFFEB:
        char = "<XK_Super_L>"
    else:
        char = f"<0x{sym:x}>"
    # ret part
    if spec == False and info[0] == "P":
        return ["P", char, spec]
    elif spec == True:
        return [info[0], char, spec]
    else:
        return None


info4 = [s2t(i) for i in info3]
info5 = [i for i in info4 if i is not None]
# automata to merge string
info6 = []
merge = ""
for info in info5:
    if info[2] == True:
        if len(merge) > 0:
            info6.append(["T", merge, False])
        info6.append(info)
        merge = ""
    else:
        merge += info[1]

info_fn = info6
with open("result.log", "w") as f:
    for i in info_fn:
        f.write(f"{i[0]}: {i[1]}\n")
```

Then we just replay the process, and we got the flag:

> bronco{l0gg1ng_ftw}

## Rev
### Desmos Destroyer (500 points, 6 solves) 

#### ***NOTE***: I almost solve that TvT

![desmos](/commons/ctfs/broncoCTF/desmos.png)

[Link to the challenge](https://www.desmos.com/calculator/pfsf4sl1id)

Here is the instruction:
```
Only allow valid orders, which are the following:
u - move up
d - move down
l - move left
r - move left
ul - move up and left
ur - move up and right
dl - move down and left
dr - move down and right
g - send out survivors to gather food
a - do the action of whatever square you are on
c - camp out on the current space
f - fortify the current space

Types:
0: basic - Nothing
1: Cave - Action provides a single turn of safety, then can't be used again
2: Mountain - Action gives you a one-time bonus to combat
3: Swamp - Slows down entities while they are in it
5: Desert - Action finds new people at the cost of some energy, also increase energy drain
6: Collapsed cave - Nothing
```
Generally, im trying to collect the mountain first and beat the zombie, BUT i lack of 1 TURN, TvT bad for me.

I just captured a little equation and overall for you to better understanding:

![desmos1](/commons/ctfs/broncoCTF/desmos1.png)

There are a bunch of math equation, basically, you have to find the pattern main character, the win condition is pass the turn 23 and kill the zombie; otherwise, hungry, tiredness or catched by zombie will make you lose.

$$A_{eat}=V_{hunger}\to V_{hunger}-\operatorname{ceil}\left(V_{effectivePopulation}\cdot.25\right)\ \ \ +\left(L_{expeditions}.\operatorname{length}-L_{nextExpeditions}.\operatorname{length}\right)\cdot4V_{expeditionSize}$$

I used a glitch inside this equation, that significant increase our food but i didnt know, this caused by in the right side of the map, when we use "g" before, the expedition will gap bigger, then we can get food bigger.

![desmos2](/commons/ctfs/broncoCTF/desmos2.png)
![desmos3](/commons/ctfs/broncoCTF/desmos3.png)

Then i notice that the more energy we have, the more energy we gain if we use `"c"`, it can cause the infinite energy for us.

Combine that, also i made a brilliant move that can make the `zombie got swamped twice`, then i failed on last turn to collect the mountain for beating the zombie (actually i forgo beating the zombie is a win condition, before that i just tried to pass turn 23), TvT.

Here is my moves:

```
g,c,d,g,c,dr,dr,dr,a,c,ur,c,c,u,u,ul,ul,c,ul,l,ul,a
```

![desmos4](/commons/ctfs/broncoCTF/desmos4.png)

Sadly, i have only one hour before the CTF end.

## Wrapping Up

Overall, i really enjoyed this CTF, im really enjoy the challenges, especially the Desmos Destroyer (very unique challenge from [***shwhale***](https://github.com/MrShwhale)). Lastly congrats to RaptX with first time in top 10, i hope next time i can do better.


