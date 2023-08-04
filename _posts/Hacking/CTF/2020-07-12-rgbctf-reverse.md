---
title : rgbCTF Reversing Writeup
categories : [Hacking, CTF]
tags : [rgbCTF, Reversing]
---

## Beginner - Pieces
```
My flag has been divided into pieces :( Can you recover it for me?


~Quintec#0689
Main.java Size: 0.60 KBMD5: 0ef37e478675de04e468fb1496a5258f
```
```java
import java.io.*;
public class Main {
	public static void main(String[] args) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		String input = in.readLine();
		if (divide(input).equals("9|2/9/:|4/7|8|4/2/1/2/9/")) {
			System.out.println("Congratulations! Flag: rgbCTF{" + input + "}");
		} else {
			System.out.println("Incorrect, try again.");
		}
	}
	
	public static String divide(String text) {
		String ans = "";
		for (int i = 0; i < text.length(); i++) {
			ans += (char)(text.charAt(i) / 2);
			ans += text.charAt(i) % 2 == 0 ? "|" : "/";
		}
		return ans;
	}
}
```
```python
def rev_divide(text) :
    rev_ans=""
    for i in range(0, len(text),2) :
        next=i+1
        if text[i+1] == "|" :
            rev_ans+=chr(ord(text[i])*2)
        else :
            rev_ans+=chr(ord(text[i])*2+1)
            
    return rev_ans



ans="9|2/9/:|4/7|8|4/2/1/2/9/"
flag=rev_divide(ans)
print(flag)
```
```
flag : rgbCTF{restinpieces}
```
