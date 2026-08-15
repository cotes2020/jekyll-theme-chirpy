---
title: Reversing Flutter Notes
date: 2026-8-12 13:10:30 +0000
categories: [notes]
tags: [notes]
math: true
mermaid: true
---

-----------

### Reversing Flutter like a noob

-----------

- Dealing with libapp.so
- Dart code runs in an “isolate”, which is a structure which contains a heap, references to objects etc. Isolates are isolated 😏 and can’t access each other, except one special isolate, the “VM” isolate that everybody can access.

- The code is basically in-:

```
_kDartIsolateSnapshotInstruction
```
- Code in `\misc-code\flutter`:

```
https://github.com/cryptax/misc-code.git
```

- Usage-:

```
python3 flutter-header.py -i libapp.so
```

- Finding the offset-:

```
readelf -Ws  libapp.so | grep kDartIsolateSnapshotInstructions
```

<img width="1174" height="81" alt="image" src="https://github.com/user-attachments/assets/5ecb9c9e-6c4f-4a06-a54a-f489bd54276d" />



<img width="806" height="175" alt="image" src="https://github.com/user-attachments/assets/cdc978ac-4a60-4c60-8b41-c7dc38c26644" />

- Use blutter-termux to dump the source code-:
```
https://github.com/dedshit/blutter-termux/blob/main/README.md
```

```
docker/run.sh morbb ./lib_dump
```
<img width="1487" height="321" alt="image" src="https://github.com/user-attachments/assets/1552f10e-7123-4c98-8833-b573bd249192" />

