---
title: "[Troubleshooting] conda 가상환경 상에서 pip install로 설치한 모듈을 찾을 수 없는 경우"
author: kwon
date: 2025-02-12T23:00:00 +0900
categories: [toubleshooting]
tags: [python, conda]
math: true
mermaid: false
---

# 🚫 현상

- 분명 `conda activate` 이후 가상환경 안에서 `pip install`을 통해 모듈을 설치했음에도 불구하고 `not found module`이 발생하는 경우가 있다.
---


# 💡원인

- pip의 경로를 확인해보면 conda 환경의 경로가 아님을 확인할 수 있다. 이 때문에 global 환경에 설치가 되어 가상환경 내에서 사용할 수 없었던 것
    
    ```bash
    $ which pip
    /home/user/.local/bin/pip
    ```
---


# 🛠 해결책

- pip를 현재 가상환경의 것으로 사용하도록 명시한다.
    
    ```bash
    python -m pip install <module_name>
    ```
---


# 🤔 회고


---


# 📚 Reference