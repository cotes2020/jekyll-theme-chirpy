---
title: File Encryption with GPG
author: Tai Le
date: 2021-10-12
tags: [Back-end Engineering]
---

A week ago, I had a task related to file encryption, the only technology available was GnuPrivacy Guard (GPG) so I took a day to research it. So today I want to summarize what I have learned from this topic.

## 1. Introduction

GPG is a complete and free implementation of the OpenPGP standard, which is the most widely used email encryption standard. It helps us to protect our privacy when communicating on the Internet. Just imagine a hacker hacks the financial report of your company and there is no encryption method applied to it, this would be the worst situation that nobody ever wants.


## 2. How it works

Generally, GPG is two-way encryption in which we can encrypt the data on one side and decrypt it on another side. The basic workflow consists of 3 steps:
1. Generate a public and private key using the GPG command-line tool
2. Give the public key to the others, have them import the key
3. Encrypt the data using the public key and send it back to us
3. Decrypt the encrypted data using our private key

Besides file protection, the digital signature is another application of GPG, but it is out of the scope of this post. You can find out more in the [document](https://www.gnupg.org/gph/en/manual/x135.html).

In the following section, I will demonstrate and explain each step using Python. Don't worry if you don't know Python, you just need to understand the core idea in the code.


#### a. Setup

Let's go and install the library. From the description, you need Python 2.4 or higher to use it.

```bash
pip install python-gnupg
```


#### b. Generate a public and private keypair

 ```python
import gnupg

# Initialize the GPG instance with a root folder, where your keys will be stored.
gpg = gnupg.GPG(gnupghome="/any/folder")
# A passphrase is needed when you decrypt the data using your private key
# It serves as another authentication layer to verify the private key's owner
passphrase = "the_decryption_password"
input_data = gpg.gen_key_input(
    # Associate the keypair with owner information
    # This is just for identification, not for verification
    name_real="Tai Le",
    name_email="ltquoctaidn98@gmail.com", 
    # Specify our passphrase here
    passphrase=passphrase,
    # Encryption method like RSA, DSA, ....
    # Each type has its own strength
    key_type="RSA",
    # The longer the key length the more secure it is against brute-force attacks
    # But it has its cost; the more secure, the lower performance
    key_length=1024,
    # We can also define an expiration date
    expire_date="365d",
)
# Generate the key
key = gpg.gen_key(input_data)
# Dump the key into the variable
public_key = gpg.export_keys(str(key))
private_key = gpg.export_keys(str(key), True, passphrase=passphrase)

# Export key to files
with open("public_key.asc", "w") as f:
    f.write(public_key)

with open("private_key.asc", "w") as f:
    f.write(private_key)
```


#### c. Import the public key in another computer

Assume we already gave the public key to another person. You can follow the code below to import it.

``` python
import os
import gnupg

gpg = gnupg.GPG(gnupghome="/any/folder2")

file_data = open("public_key.asc", "r").read()
input_data: gnupg.ImportResult
# Import the file
input_data = gpg.import_keys(file_data)
# Remember to set trust level to "TRUST_ULTIMATE" to the key. If not, we cannot encrypt files using the key
gpg.trust_keys(input_data.fingerprints, "TRUST_ULTIMATE")
# Check whether the key was imported
print(gpg.list_keys())
```

Below is the output of the `print` statement:

```
[{'type': 'pub', 'trust': 'u', 'length': '1024', 'algo': '1', 'keyid': '3B5294A0FE1941E0', 'date': '1634195983', 'expires': '1665731983', 'dummy': '', 'ownertrust': 'u', 'sig': '', 'cap': 'escaESCA', 'issuer': '', 'flag': '', 'token': '', 'hash': '', 'curve': '', 'compliance': '', 'updated': '', 'origin': '0', 'uids': ['Tai Le <ltquoctaidn98@gmail.com>'], 'sigs': [], 'subkeys': [], 'fingerprint': 'E64FD57C36AE65C34377FE423B5294A0FE1941E0'}
```


#### d. Encrypt the data

I will encrypt a PDF file to see if the library works for files other than text.

```python
import gnupg
import time

gpg = gnupg.GPG(gnupghome="/any/folder2")

path = "/home/user/Documents/High performance MySQL optimization, backups, and replication.pdf"
start = time.time()

# We should always use "read binary" here, because it is the input of decrypt_file method
with open(path, "rb") as file_stream:
    status = gpg.encrypt_file(
        file_stream,
        # Convert the output file into text for friendly delivering via email and other methods
        armor=True,
        # Recipient here is to find the appropriate public key among many keys imported
        # We can define a string or an array of string here. If multiple recipients are passed, the library will attach multiple keyprints of the public keys to the metadata of the encrypted file. So we can have one encrypted file for multiple public keys
        recipients=["ltquoctaidn98@gmail.com"],
        # Output file
        output=f"{path}.encrypted",
    )

print(status.ok)
print(status.stderr)
print("End time:", time.time() - start)
```

Output:
```
True
gpg: WARNING: unsafe permissions on homedir '/home/tai/.gnupg2'
[GNUPG:] KEY_CONSIDERED E64FD57C36AE65C34377FE423B5294A0FE1941E0 0
[GNUPG:] KEY_CONSIDERED E64FD57C36AE65C34377FE423B5294A0FE1941E0 2
[GNUPG:] BEGIN_ENCRYPTION 2 9
[GNUPG:] END_ENCRYPTION

End time: 0.5904629230499268
```

File output:

![/assets/img/2021-10-12/encryption-result.png](/assets/img/2021-10-12/encryption-result.png)

As you can see, the file gets encrypted into a lengthy text file. One notable point is that we can encrypt any file extensions, even zip files. I demonstrated with the zipped Webstorm application (~500MB) and it only cost 12 seconds, which is not a considerable amount of time for a heavy file.


#### e. Decrypt the data

When decrypting, the software will use the fingerprints in the encrypted file header to compare. We need to import the private key first (same steps as importing the public key), then we decrypt the file.

```python
import gnupg
import time

gpg = gnupg.GPG(gnupghome="/any/folder")

# Import the private key
file_data = open("private_key.asc", "rb").read()
input_data: gnupg.ImportResult
input_data = gpg.import_keys(file_data)
gpg.trust_keys(input_data.fingerprints, "TRUST_ULTIMATE")

# Make sure the result of list_keys method is not empty
print(gpg.list_keys(True))

# Specify the encrypted file for decrypting
input_path = "/home/tai/Documents/High performance MySQL optimization, backups, and replication.pdf.encrypted"
output_path = "/home/tai/Documents/High performance MySQL optimization, backups, and replication2.pdf"
passphrase = "the_decryption_password"

start = time.time()
# We should always use "read binary" here, because it is the input of decrypt_file method
with open(input_path, "rb") as f:
    # As I already told, we need to detail the passphrase when decrypting
    status = gpg.decrypt_file(f, output=output_path, passphrase=passphrase)

print(status.ok)
print(status.stderr)
print("End time:", time.time() - start)
```

Output:
```
True
gpg: WARNING: unsafe permissions on homedir '/home/tai/.gnupg2'
[GNUPG:] ENC_TO 3B5294A0FE1941E0 1 0
[GNUPG:] KEY_CONSIDERED E64FD57C36AE65C34377FE423B5294A0FE1941E0 0
[GNUPG:] KEY_CONSIDERED E64FD57C36AE65C34377FE423B5294A0FE1941E0 0
[GNUPG:] DECRYPTION_KEY E64FD57C36AE65C34377FE423B5294A0FE1941E0 E64FD57C36AE65C34377FE423B5294A0FE1941E0 u
[GNUPG:] KEY_CONSIDERED E64FD57C36AE65C34377FE423B5294A0FE1941E0 0
gpg: encrypted with 1024-bit RSA key, ID 3B5294A0FE1941E0, created 2021-10-14
      "Tai Le <ltquoctaidn98@gmail.com>"
[GNUPG:] BEGIN_DECRYPTION
[GNUPG:] DECRYPTION_INFO 2 9
[GNUPG:] PLAINTEXT 62 1634199713 
[GNUPG:] DECRYPTION_OKAY
[GNUPG:] GOODMDC
[GNUPG:] END_DECRYPTION

End time: 1.7071013450622559
```

## 3. Conclusion

This post already covers the basic implementation of the GPG encryption method. So what is next? You can delve into digital signatures, the algorithms (RSA, DSA, ...), cryptographic concepts, and so on.

So I have achieved my goal, which is writing a summary for GPG encryption. I don't know when the next post of mine will be released, but I am sure that it must be an interesting topic and worth summarizing.


## 4. References:

- [GPG document](https://www.gnupg.org/gph/en/manual.html)
- [Python implementation by Paul Mahon](https://www.youtube.com/watch?v=9NiPwvLCDpM&ab_channel=PracticalPythonSolutions-ByPaulMahon)

