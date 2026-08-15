---
title: HTTPlib Python CRLF
date: 2026-4-11 13:10:30 +0000
categories: [Security Research, CVES]
tags: [Security Research, CVES]
math: true
mermaid: true
media_subpath: /assets/posts/cves/httplib_crlf
image:
  path: preview.jpeg
---
-------------

### Python's cython httplib `set_tunnel` CRLF issue

--------------

- Original Advisory:

 <img width="1754" height="883" alt="image" src="https://github.com/user-attachments/assets/c26f6ba1-81a3-4837-8d0a-6dcf72c4b8b8" />

- Affected code in http.client._tunnel()-:

```python
#Line 984
 for header, value in self._tunnel_headers.items():
            headers.append(f"{header}: {value}\r\n".encode("latin-1"))
        headers.append(b"\r\n")
```

- It directly passes headers values without filtering input for \r\n leading to injection of new headers through a header's value.
- Proof-of-concept-:
<img width="1102" height="153" alt="image" src="https://github.com/user-attachments/assets/2fa77f5b-1820-4735-b31f-c052093b8f9b" />

- Result-:

<img width="607" height="193" alt="image" src="https://github.com/user-attachments/assets/98db9867-9b76-4a6c-81da-1b1bf175ab27" />

-------------
